"""Streaming benchmark: feeds the fixed 24 kHz clips through moshi.server's per-frame path
(mimi.encode -> lm_gen.step -> mimi.decode -> host copy), one 80 ms frame at a time, and times
each stage with CUDA events. Model code is not changed: timers wrap moshi's own CUDA-graphed
callables (temporal transformer = graphed_main, depth transformer = graphed_depth) from outside.

Stages per frame (GPU time, ms):
  encode    mimi.encode of 1920 samples
  temporal  7B temporal transformer step (lm.forward_text, graphed)
  text      text-token sampling and cache bookkeeping inside lm_gen.step, outside the two graphs
  depth     8 sequential depth-transformer steps + sampling (lm_gen.depformer_step, graphed)
  decode    mimi.decode of the 8 audio tokens
  host      pcm .cpu() copy and text token .item(), as the server does
  frame     whole frame on the GPU timeline; wall = CPU perf_counter around the frame (the real-time test)
"""
import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import ROOT, make_lm_gen, seed_all, write_result
from quant.int8 import MODES, compile_temporal, load_quantized, weight_bytes

BUDGET_MS = 80.0
STAGES = ["encode", "temporal", "text", "depth", "decode", "host", "frame", "wall"]


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


class Timed:
    """Wraps a callable and records CUDA events around each call into the current frame."""

    def __init__(self, fn, name, rec):
        self.fn, self.name, self.rec = fn, name, rec

    def __call__(self, *args):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = self.fn(*args)
        e.record()
        self.rec.append((self.name, s, e))
        return out


def pct(a):
    a = np.asarray(a, dtype=np.float64)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3), "mean": round(float(a.mean()), 3),
            "max": round(float(a.max()), 3)}


def gpu_telemetry():
    """SM clock, power, temperature and active throttle reasons, to catch clock/thermal effects."""
    import subprocess
    q = "clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    try:
        out = subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                      text=True).strip().split(", ")
        return dict(zip(["sm_mhz", "mem_mhz", "power_w", "temp_c", "throttle"], out))
    except Exception as e:
        return {"error": str(e)[:80]}


@torch.no_grad()
def run(args):
    seed_all(args.seed)
    if args.mem_cap_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.mem_cap_gb * 1e9 / total)
    info, mimi, tok, lm = load_quantized(args.quant)
    if args.compile_temporal:
        compile_temporal(lm)
    if args.kv != "sdpa":
        from kernels.kv_attn import patch_temporal_attention
        assert patch_temporal_attention(lm, args.kv) == 32
    load_peak = torch.cuda.max_memory_allocated()
    lm_gen = make_lm_gen(info, lm, batch_size=args.batch)
    B = args.batch
    mimi.streaming_forever(B)
    lm_gen.streaming_forever(B)
    if args.graph_mimi:
        from bench.mimi_graph import patch_mimi_encode
        patch_mimi_encode(mimi)
    frame_n = int(mimi.sample_rate / mimi.frame_rate)

    rec = []  # (stage, start_event, end_event) for the current frame
    st = lm_gen._streaming_state
    if args.compile_temporal:
        # moshi's CUDAGraphed captures on the 2nd call; a freshly compiled forward can still recompile
        # then, and dynamo touches the CUDA RNG while tracing, which is illegal during capture.
        # Run it uncaptured a few more times so every recompile happens before capture.
        st.graphed_main.warmup_steps = 4
    st.graphed_main = Timed(st.graphed_main, "temporal", rec)
    st.graphed_depth = Timed(st.graphed_depth, "depth", rec)

    clips = [read_wav(p) for p in sorted((ROOT / "audio").glob("clip*.wav"))]
    assert len(clips) == 5, "run scripts/make_clips.py first"
    n_frames = min(len(c) for c in clips) // frame_n
    # stream b plays clip (b + offset) so a batch holds different conversations
    def batch_audio(offset):
        x = np.stack([clips[(b + offset) % len(clips)][: n_frames * frame_n] for b in range(B)])
        return torch.from_numpy(x).cuda()[:, None]

    rows, texts, telemetry = [], [], [gpu_telemetry()]
    total = 0
    for ci in range(len(clips)):
        mimi.reset_streaming(); lm_gen.reset_streaming()
        audio = batch_audio(ci)
        text = []
        next_t = time.perf_counter()
        for f in range(n_frames):
            rec.clear()
            ev = {k: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for k in ("frame", "encode", "lm", "decode", "host")}
            torch.cuda.synchronize()
            if args.paced:  # live audio: a new frame only arrives every 80 ms
                if total < args.warmup or f == 0:
                    next_t = time.perf_counter()  # no schedule debt from warm-up or graph capture
                next_t += BUDGET_MS / 1000
                time.sleep(max(0.0, next_t - time.perf_counter()))
            w0 = time.perf_counter()
            ev["frame"][0].record()
            ev["encode"][0].record()
            codes = mimi.encode(audio[..., f * frame_n:(f + 1) * frame_n])
            ev["encode"][1].record()
            if f == 0:
                mimi.reset_streaming()  # server.recv_loop drops the first frame's encoder state
            ev["lm"][0].record()
            tokens = lm_gen.step(codes[:, :, :1])
            ev["lm"][1].record()
            decoded = tokens is not None
            if decoded:
                ev["decode"][0].record()
                pcm = mimi.decode(tokens[:, 1:])
                ev["decode"][1].record()
                ev["host"][0].record()
                pcm_cpu = pcm.cpu()
                tt = tokens[0, 0, 0].item()
                ev["host"][1].record()
            ev["frame"][1].record()
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            wall = 1000 * (t_end - w0)
            # paced only: how far past its 80 ms deadline the frame finished (lateness carries over)
            late = max(0.0, 1000 * (t_end - (next_t + BUDGET_MS / 1000))) if args.paced else 0.0
            total += 1
            if decoded and tt not in (0, 3):
                text.append(tok.id_to_piece(tt).replace("▁", " "))
            if total <= args.warmup or not decoded:
                continue
            g = {k: dict((n, s.elapsed_time(e)) for n, s, e in rec).get(k, 0.0) for k in ("temporal", "depth")}
            lm_ms = ev["lm"][0].elapsed_time(ev["lm"][1])
            rows.append({
                "clip": ci, "frame": f,
                "encode": ev["encode"][0].elapsed_time(ev["encode"][1]),
                "temporal": g["temporal"], "depth": g["depth"],
                "text": lm_ms - g["temporal"] - g["depth"],
                "decode": ev["decode"][0].elapsed_time(ev["decode"][1]),
                "host": ev["host"][0].elapsed_time(ev["host"][1]),
                "frame": ev["frame"][0].elapsed_time(ev["frame"][1]),
                "wall": wall, "late": late,
            })
        texts.append("".join(text).strip())
        telemetry.append(gpu_telemetry())
        print(f"clip {ci + 1}: {len(rows)} frames measured so far; moshi said: {texts[-1][:120]!r}", flush=True)

    stats = {k: pct([r[k] for r in rows]) for k in STAGES}
    over = int(sum(r["wall"] > BUDGET_MS for r in rows))
    free_b, total_b = torch.cuda.mem_get_info()  # whole device, incl. CUDA context and kernels
    wb = weight_bytes(lm)
    if args.quant != "none":  # a quantized LM must store fewer bytes than bf16 (15.4 GB)
        assert wb < 0.8 * 15.375e9, f"weight_bytes did not see quantized storage: {wb / 1e9:.2f} GB"
    res = {
        "what": "per-frame latency budget on the moshi.server path, CUDA events per stage, ms",
        "config": {"batch": B, "dtype": "bf16", "quant": args.quant, "compile_temporal": args.compile_temporal,
                   "mem_cap_gb": args.mem_cap_gb, "paced": args.paced, "kv": args.kv,
                   "graph_mimi": args.graph_mimi,
                   "pacing": "restart after warm-up and at each clip" if args.paced else None, "cuda_graphs": not _env_on("NO_CUDA_GRAPH"),
                   "torch_compile": not _env_on("NO_TORCH_COMPILE"),
                   "warmup_frames": args.warmup, "clips": len(clips), "frames_per_clip": n_frames,
                   "hf_repo": "kyutai/moshiko-pytorch-q8" if args.quant == "q8bnb" else "kyutai/moshiko-pytorch-bf16"},
        "frames_measured": len(rows),
        "frames_over_80ms": over,
        "frames_over_80ms_pct": round(100 * over / len(rows), 3),
        "realtime_strict": over == 0,  # strictest criterion: not a single frame over 80 ms
        "realtime_p99": stats["wall"]["p99"] < BUDGET_MS,
        "paced_late_ms": pct([r["late"] for r in rows]) if args.paced else None,
        # per-stage breakdown of every frame over budget (tail attribution), capped at 100
        "late_frames": [{k: round(r[k], 2) for k in ("clip", "frame", "encode", "temporal", "text", "depth",
                                                      "decode", "host", "wall")}
                        for r in rows if r["wall"] > BUDGET_MS][:100],
        "stages_ms": stats,
        "per_clip_wall_p99": [round(float(np.percentile([r["wall"] for r in rows if r["clip"] == c], 99)), 3)
                              for c in range(len(clips)) if any(r["clip"] == c for r in rows)],
        "peak_mem_gb": {"allocated": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                        "reserved": round(torch.cuda.max_memory_reserved() / 1e9, 3),
                        "after_load": round(load_peak / 1e9, 3),
                        "device_used_now": round((total_b - free_b) / 1e9, 3)},
        "lm_weight_gb": round(wb / 1e9, 3),
        "gpu_telemetry": telemetry,
        "moshi_text": texts,
    }
    write_result(args.tag, res, seed=args.seed)
    return res


def _env_on(name):
    import os
    return os.environ.get(name, "").lower() not in {"", "0", "no", "n"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline_l4_bf16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42424242)
    ap.add_argument("--quant", choices=MODES, default="none")
    ap.add_argument("--compile-temporal", action="store_true")
    ap.add_argument("--mem-cap-gb", type=float, default=0.0)
    ap.add_argument("--paced", action="store_true", help="feed one frame every 80 ms like live audio")
    ap.add_argument("--kv", default="sdpa", choices=("sdpa", "bf16", "bf16-len", "int8", "int8-len"),
                    help="temporal attention: moshi SDPA or kernels/kv_attn.py fused kernel")
    ap.add_argument("--graph-mimi", action="store_true", help="CUDA-graph the rest of Mimi's encode")
    args = ap.parse_args()
    try:
        res = run(args)
    except torch.OutOfMemoryError as e:  # record OOM instead of raising
        write_result(args.tag, {"what": "out of memory", "oom": True, "error": str(e)[:300],
                                "config": {"batch": args.batch, "quant": args.quant, "kv": args.kv,
                                           "compile_temporal": args.compile_temporal,
                                           "mem_cap_gb": args.mem_cap_gb, "paced": args.paced},
                                "peak_mem_gb": {"allocated": round(torch.cuda.max_memory_allocated() / 1e9, 3)}},
                     seed=args.seed)
        print("OOM at batch", args.batch)
        return
    s = res["stages_ms"]
    for k in STAGES:
        print(f"{k:>9}  p50 {s[k]['p50']:7.2f}  p95 {s[k]['p95']:7.2f}  p99 {s[k]['p99']:7.2f}  max {s[k]['max']:7.2f}")
    print("frames over 80 ms:", res["frames_over_80ms"], "of", res["frames_measured"], "| peak", res["peak_mem_gb"])
    # self-check: enough frames, ordered percentiles, stages add up to (at most) the frame
    assert res["frames_measured"] >= 500
    for k in STAGES:
        assert s[k]["p50"] <= s[k]["p95"] <= s[k]["p99"] <= s[k]["max"] + 1e-6
    parts = sum(s[k]["p50"] for k in ("encode", "temporal", "text", "depth", "decode", "host"))
    assert 0.8 * s["frame"]["p50"] <= parts <= 1.05 * s["frame"]["p50"], (parts, s["frame"]["p50"])
    assert s["wall"]["p50"] >= s["frame"]["p50"] * 0.98
    if args.paced and s["wall"]["p50"] < BUDGET_MS:  # pacing held when frames fit the budget
        assert res["paced_late_ms"]["p50"] < BUDGET_MS, res["paced_late_ms"]


if __name__ == "__main__":
    main()
