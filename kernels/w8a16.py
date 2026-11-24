"""INT8-weight x bf16-activation linear (W8A16) as one Triton kernel, for the temporal transformer.

At batch 1-16 the 7B step is a weight-streaming GEMV, so time ~ bytes read. torchao's
Int8WeightOnly in eager dequantizes to a bf16 copy first (reads int8, writes and re-reads bf16),
which measured SLOWER than bf16 (results/int8wo_eager.json). This kernel reads only the int8
weights, converts tiles to bf16 in registers and multiplies on tensor cores (batch padded to 16
rows, free because the step is bandwidth-bound). No compiler involved; CUDA-graph safe.

Quantization: symmetric per output channel, scale = absmax(row) / 127 (same math as torchao
Int8WeightOnly). Activations, KV cache and accumulation stay bf16/fp32.
"""
import torch
import triton
import triton.language as tl
from torch import nn


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 512}, num_warps=4, num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _w8a16_kernel(X, W, S, Y, M, N, K, s_xm, s_wn, s_ym,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_m[:, None] * s_xm + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(W + offs_n[:, None] * s_wn + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))  # int8 -> bf16 is exact for |q| <= 127
    s = tl.load(S + offs_n, mask=offs_n < N, other=0.0)
    y = acc * s[None, :]
    tl.store(Y + offs_m[:, None] * s_ym + offs_n[None, :], y.to(Y.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


MAX_M = 16


def w8a16_linear(x, qweight, scale):
    """x [..., K] bf16, qweight [N, K] int8, scale [N] fp32 -> [..., N] bf16."""
    *lead, K = x.shape
    N = qweight.shape[0]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if M > MAX_M:  # not used in streaming decode; keep correct for anything larger
        return torch.nn.functional.linear(x2, qweight.to(x.dtype) * scale.to(x.dtype)[:, None]).reshape(*lead, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _w8a16_kernel[grid](x2, qweight, scale, y, M, N, K, x2.stride(0), qweight.stride(0), y.stride(0),
                        BLOCK_M=MAX_M)
    return y.reshape(*lead, N)


def _quantize_rows(w):
    w = w.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


class W8Linear(nn.Module):
    """Plain module (not an nn.Linear), so moshi's ActivationGating takes its generic path and
    calls our forward instead of reading .weight directly."""

    def __init__(self, linear: nn.Linear, device):
        super().__init__()
        assert linear.bias is None
        q, s = _quantize_rows(linear.weight.data.to(device))
        self.register_buffer("qweight", q)
        self.register_buffer("scale", s)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        return w8a16_linear(x, self.qweight, self.scale)


class W8LinearAttn(nn.Linear):
    """For attention projections: moshi's attention reads in_proj.weight.dtype/.device to size the
    KV cache, so keep an nn.Linear with an EMPTY bf16 weight for that check; data lives in buffers."""

    def __init__(self, linear: nn.Linear, device):
        nn.Module.__init__(self)
        assert linear.bias is None
        q, s = _quantize_rows(linear.weight.data.to(device))
        self.weight = nn.Parameter(torch.empty(0, device=device, dtype=linear.weight.dtype), requires_grad=False)
        self.bias = None
        self.register_buffer("qweight", q)
        self.register_buffer("scale", s)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        return w8a16_linear(x, self.qweight, self.scale)


def quantize_temporal_w8(lm, device="cuda"):
    """Swap every Linear in the temporal transformer's attention and FFN for the W8A16 kernel.
    Depth transformer, embeddings, text/audio heads and Mimi stay bf16. Returns #layers swapped."""
    n = 0
    for mod in lm.transformer.modules():
        name = type(mod).__name__
        if name == "StreamingMultiheadAttention":
            for lst in (mod.in_projs, mod.out_projs):
                for i, lin in enumerate(lst):
                    lst[i] = W8LinearAttn(lin, device); n += 1
        elif name == "ActivationGating":
            mod.linear_in = W8Linear(mod.linear_in, device)
            mod.linear_out = W8Linear(mod.linear_out, device)
            n += 2
    torch.cuda.empty_cache()
    return n


if __name__ == "__main__":
    # correctness at moshi's temporal shapes, M = 1, 4, 16, vs bf16 matmul on the dequantized weight
    torch.manual_seed(0)
    rows = []
    for N, K in ((12288, 4096), (4096, 4096), (22528, 4096), (4096, 11264)):
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        q, s = _quantize_rows(w)
        wd = (q.float() * s[:, None]).to(torch.bfloat16)
        for M in (1, 4, 16):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            ref = (x.float() @ wd.float().t())
            got = w8a16_linear(x, q, s).float()
            rel = ((got - ref).norm() / ref.norm()).item()
            assert rel < 1e-2, (N, K, M, rel)
        # bandwidth at M=1. Rotate through enough weight copies (>= 256 MB, > 5x the L4's 48 MB L2)
        # so every call streams from DRAM, like the real 7B step; timing one 46-50 MB matrix
        # repeatedly mostly hits L2 and overstates bandwidth.
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
        copies = max(2, -(-256 * 2 ** 20 // (N * K)))
        qs = [q.clone() for _ in range(copies)]
        ws = [w.clone() for _ in range(copies)]
        del w, wd
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def timed(fn, reps=100):
            for i in range(2 * copies):
                fn(i % copies)
            torch.cuda.synchronize()
            st.record()
            for i in range(reps):
                fn(i % copies)
            en.record(); torch.cuda.synchronize()
            return st.elapsed_time(en) / reps

        ms = timed(lambda i: w8a16_linear(x, qs[i], s))
        ms_bf16 = timed(lambda i: torch.nn.functional.linear(x, ws[i]))
        del qs, ws
        torch.cuda.empty_cache()
        rows.append({"N": N, "K": K, "w8a16_us": round(ms * 1e3, 1), "w8a16_int8_gbs": round(N * K / ms / 1e6, 1),
                     "bf16_us": round(ms_bf16 * 1e3, 1), "bf16_gbs": round(2 * N * K / ms_bf16 / 1e6, 1),
                     "speedup": round(ms_bf16 / ms, 2), "rotated_copies": copies})
        print(rows[-1], flush=True)
        assert rows[-1]["w8a16_int8_gbs"] < 320 and rows[-1]["bf16_gbs"] < 320, "faster than DRAM: L2 is leaking in"
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.common import write_result
    write_result("kernel_w8a16", {"what": "W8A16 Triton linear vs bf16 F.linear at M=1, moshi temporal shapes, "
                                          "weights rotated through >=256 MB so reads come from DRAM",
                                  "max_rel_err_ok": True, "shapes": rows})
    print("w8a16 self-check ok")
