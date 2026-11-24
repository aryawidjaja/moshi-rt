"""Correctness test + microbenchmark for kernels/kv_attn.py at Moshi's temporal shapes
(H=32 heads, D=128, capacity=context=3000). Runs on the GPU VM.

Test: reference = moshi's own RingKVCache.complete + the exact mask and SDPA from
StreamingMultiheadAttention.forward. Random q/k/v, more steps than the ring holds (wrap-around),
one stream idle on some steps (exec_mask False), and a reset of one stream mid-run.
Bench: ms per frame for all 32 temporal layers (attention part only: cache write + attend),
stock vs each fused mode, at B = 1, 4, 8 and used length 375 (30 s) vs 3000 (full ring).
Writes results/kernel_kv_attn.json."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import seed_all, write_result
from kernels.kv_attn import FusedRingKV

H, D, LAYERS = 32, 128, 32


def ref_attend(cache, q, k, v, offset, exec_mask, context):
    """moshi: RingKVCache.complete, then the causal/context mask and SDPA, T=1."""
    keys, values, pos_k = cache.complete(k, v, exec_mask)
    pos_k = pos_k[:, None]
    pos_q = offset.view(-1, 1, 1) + torch.arange(1, device=q.device).view(-1, 1)
    delta = pos_q - pos_k
    bias = (pos_k >= 0) & (delta >= 0) & (delta < context)
    return F.scaled_dot_product_attention(q, keys, values, bias[:, None], dropout_p=0.0)


@torch.no_grad()
def test(mode, B=3, cap=300, steps=700, tol=None):
    from moshi.modules.transformer import RingKVCache
    seed_all(0)
    dev, dt = "cuda", torch.bfloat16
    int8, length = mode.startswith("int8"), mode.endswith("-len")
    ref = RingKVCache(B, H, D, cap, respect_exec_mask=True, device=dev, dtype=dt)
    fus = FusedRingKV(B, H, D, cap, dev, dt, int8, length)
    offset = torch.zeros(B, device=dev, dtype=torch.long)
    worst = 0.0
    for t in range(steps):
        exec_mask = torch.ones(B, device=dev, dtype=torch.bool)
        if t % 7 == 3:
            exec_mask[1] = False  # stream 1 idle on some steps
        if t == 450:  # reset stream 2 mid-run (moshi resets end_offset and the layer offset)
            rm = torch.tensor([False, False, True], device=dev)
            ref.reset(rm); fus.reset(rm)
            offset = torch.where(rm, torch.zeros_like(offset), offset)
        q, k, v = (torch.randn(B, H, 1, D, device=dev, dtype=dt) for _ in range(3))
        want = ref_attend(ref, q, k, v, offset, exec_mask, cap)
        fus.write(k, v, exec_mask)
        got = fus.attend(q, offset, cap)
        live = exec_mask  # outputs of idle streams are discarded by moshi
        err = (got[live].float() - want[live].float()).abs().max().item()
        worst = max(worst, err)
        offset = torch.where(exec_mask, offset + 1, offset)
    tol = tol or (0.08 if int8 else 0.02)
    assert worst < tol, f"{mode}: max abs err {worst:.4f} >= {tol}"
    return worst


@torch.no_grad()
def bench(mode, B, used, cap=3000, iters=100):
    """Attention time for one frame (32 layers), steady state with `used` tokens in the ring."""
    from moshi.modules.transformer import RingKVCache
    dev, dt = "cuda", torch.bfloat16
    caches = []
    for _ in range(LAYERS):
        if mode == "sdpa":
            c = RingKVCache(B, H, D, cap, respect_exec_mask=True, device=dev, dtype=dt)
        else:
            c = FusedRingKV(B, H, D, cap, dev, dt, mode.startswith("int8"), mode.endswith("-len"))
        c.end_offset.fill_(used)
        caches.append(c)
    q, k, v = (torch.randn(B, H, 1, D, device=dev, dtype=dt) for _ in range(3))
    offset = torch.full((B,), used, device=dev, dtype=torch.long)
    em = torch.ones(B, device=dev, dtype=torch.bool)

    def frame():
        for c in caches:
            if mode == "sdpa":
                ref_attend(c, q, k, v, offset, em, cap)
                c.end_offset.fill_(used)  # keep the ring at a fixed fill level
            else:
                c.write(k, v, em)
                c.attend(q, offset, cap)
                c.end_offset.fill_(used)

    # CUDA-graph the whole frame, as moshi does, so launch overhead is not what we measure
    for _ in range(3):
        frame()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        frame()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    kv_mb = sum((c.nbytes() if mode != "sdpa" else c.cache.numel() * c.cache.element_size()) for c in caches) / 1e6
    return round(s.elapsed_time(e) / iters, 3), round(kv_mb, 1)


def main():
    errs = {m: round(test(m), 5) for m in ("bf16", "bf16-len", "int8", "int8-len")}
    print("max abs err vs moshi RingKVCache+SDPA:", errs)
    rows = []
    for B in (1, 4, 8):
        for used in (375, 3000):
            base = None
            for mode in ("sdpa", "bf16", "bf16-len", "int8", "int8-len"):
                ms, mb = bench(mode, B, used)
                base = base or ms
                rows.append({"batch": B, "used_slots": used, "mode": mode, "ms_per_frame_32_layers": ms,
                             "speedup_vs_sdpa": round(base / ms, 2), "kv_cache_mb": mb})
                print(rows[-1], flush=True)
    write_result("kernel_kv_attn", {
        "what": "fused ring-KV decode attention vs moshi RingKVCache+mask+SDPA; 32 temporal layers, "
                "H=32, D=128, capacity 3000; CUDA-graphed frame; attention part only",
        "max_abs_err_vs_moshi": errs, "bench": rows})
    assert all(r["ms_per_frame_32_layers"] > 0 for r in rows)


if __name__ == "__main__":
    main()
