"""Profile N steady-state frames of the moshi.server path with the PyTorch profiler (CUPTI).
Answers: launch-bound or compute/bandwidth-bound? Writes a summary (never the raw trace) to
profiles/<tag>.json: kernels per frame, GPU busy vs wall per frame, CPU launch API counts,
top kernels by GPU time, and achieved weight bandwidth per transformer.
Raw chrome trace stays on the VM in profiles/raw/."""
import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import ROOT, load, make_lm_gen, meta, seed_all
from bench.stream_bench import read_wav

L4_PEAK_GBS = 300.0  # NVIDIA L4 datasheet memory bandwidth


def param_bytes(mod):
    return sum(p.numel() * p.element_size() for p in mod.parameters())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="profile_l4_bf16")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=60)
    args = ap.parse_args()
    seed_all()
    info, mimi, tok, lm = load()
    lm_gen = make_lm_gen(info, lm)
    mimi.streaming_forever(1); lm_gen.streaming_forever(1)
    fn = int(mimi.sample_rate / mimi.frame_rate)
    audio = torch.from_numpy(read_wav(sorted((ROOT / "audio").glob("clip*.wav"))[0])).cuda()[None, None]

    def frame(f):
        with record_function("encode"):
            codes = mimi.encode(audio[..., f * fn:(f + 1) * fn])
        with record_function("lm_step"):
            tokens = lm_gen.step(codes[:, :, :1])
        if tokens is not None:
            with record_function("decode"):
                pcm = mimi.decode(tokens[:, 1:])
            with record_function("host"):
                pcm.cpu(); tokens[0, 0, 0].item()

    for f in range(args.warmup):
        frame(f)
    torch.cuda.synchronize()

    walls = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for f in range(args.warmup, args.warmup + args.frames):
            t0 = time.perf_counter()
            with record_function("FRAME"):
                frame(f)
            torch.cuda.synchronize()
            walls.append(1000 * (time.perf_counter() - t0))
    raw = ROOT / "profiles" / "raw"; raw.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(raw / f"{args.tag}.trace.json"))

    kern = defaultdict(lambda: [0, 0.0])  # name -> [count, total_us]
    api = defaultdict(int)
    busy_us = 0.0
    ranges = {"FRAME", "encode", "lm_step", "decode", "host"}  # our own labels, not GPU work
    for e in prof.events():
        if e.name in ranges or getattr(e, "is_user_annotation", False):
            continue
        if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA:
            # kernels, memcpy and memset on the GPU timeline (one stream, so durations add)
            us = e.time_range.elapsed_us()
            k = kern[e.name]; k[0] += 1; k[1] += us; busy_us += us
        elif e.name.startswith("cuda") or e.name.startswith("cu"):
            api[e.name] += 1
    n = args.frames
    top = sorted(kern.items(), key=lambda kv: -kv[1][1])[:20]
    wall = np.array(walls)

    # weight bytes per top-level LM submodule: at batch 1 each is read once per frame
    groups = defaultdict(int)
    for name, p in lm.named_parameters():
        groups[name.split(".")[0]] += p.numel() * p.element_size()
    dep_bytes = sum(v for k, v in groups.items() if k.startswith("depformer") or k == "linears")
    tr_bytes = groups["transformer"]
    out = {
        "meta": meta(),
        "what": "PyTorch profiler summary over steady-state frames, single stream, bf16",
        "frames": n,
        "graphs_off": os.environ.get("NO_CUDA_GRAPH", "") not in ("", "0"),
        "wall_ms": {"p50": round(float(np.percentile(wall, 50)), 3), "mean": round(float(wall.mean()), 3)},
        "gpu_busy_ms_per_frame": round(busy_us / 1000 / n, 3),
        "gpu_busy_fraction": round(busy_us / 1000 / wall.sum(), 4),
        "gpu_ops_per_frame": round(sum(v[0] for v in kern.values()) / n, 1),
        "cpu_api_calls_per_frame": {k: round(v / n, 2) for k, v in sorted(api.items(), key=lambda kv: -kv[1])[:12]},
        "top_gpu_ops": [{"name": k[:140], "per_frame": round(v[0] / n, 2), "us_per_frame": round(v[1] / n, 1),
                         "avg_us": round(v[1] / max(1, v[0]), 2), "share": round(v[1] / busy_us, 4)} for k, v in top],
        "weights_gb": {"temporal": round(tr_bytes / 1e9, 3), "depth": round(dep_bytes / 1e9, 3),
                       "lm_total": round(param_bytes(lm) / 1e9, 3), "mimi": round(param_bytes(mimi) / 1e9, 3),
                       "lm_groups": {k: round(v / 1e9, 4) for k, v in sorted(groups.items(), key=lambda kv: -kv[1])}},
        "l4_peak_gbs": L4_PEAK_GBS,
        "notes": "busy fraction near 1 = GPU never waits on the CPU (not launch-bound). "
                 "Achieved bandwidth = weight bytes / GEMV time, see scripts/summarize_ablation.py.",
    }
    path = ROOT / "profiles" / f"{args.tag}.json"
    import json
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("wall_ms", "gpu_busy_ms_per_frame", "gpu_busy_fraction",
                                           "gpu_ops_per_frame", "weights_gb")}, indent=1))
    for t in out["top_gpu_ops"][:10]:
        print(f"{t['us_per_frame']:9.1f} us/frame  x{t['per_frame']:6.1f}  {t['name'][:90]}")
    print("api/frame:", out["cpu_api_calls_per_frame"])
    assert out["gpu_ops_per_frame"] > 10 and 0 < out["gpu_busy_fraction"] <= 1.05
    assert sum(t["share"] for t in out["top_gpu_ops"]) <= 1.0001
    assert out["weights_gb"]["temporal"] > 10, "temporal transformer should hold most of the 7B weights"


if __name__ == "__main__":
    main()
