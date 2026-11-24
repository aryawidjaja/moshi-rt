"""Fused decode attention for Moshi's ring KV cache (temporal transformer, one query per step).

moshi's StreamingMultiheadAttention, per layer per frame:
  1. writes k, v into a fixed-capacity ring (RingKVCache, capacity = context = 3000),
  2. builds a [B, 1, 1, 3000] boolean mask from slot positions with ~10 small elementwise kernels,
  3. runs PyTorch SDPA (mem-efficient kernel) over all 3000 slots, valid or not.
Fixed shapes are what make it CUDA-graph safe; the cost is reading the whole bf16 cache
(1.57 GB per stream per frame over 32 layers) plus the mask kernels.

This module replaces steps 2-3 (and optionally the cache storage) with one split-K Triton kernel
plus a small merge kernel. Same shapes every call, so it stays CUDA-graph safe. Modes:
  sdpa      moshi stock (no patch)
  bf16      bf16 cache, fused kernel, scans all slots (isolates the kernel itself)
  bf16-len  + reads only written slots (count read on device; helps until the ring fills at 240 s)
  int8      int8 K/V with one fp32 scale per (token, head): half the cache bytes at every context length
  int8-len  both
Slot validity is computed in-kernel with exactly moshi's RingKVCache arithmetic.
"""
import math

import torch
import triton
import triton.language as tl

MODES = ("sdpa", "bf16", "bf16-len", "int8", "int8-len")


@triton.jit
def _split_kernel(Q, K, KS, V, VS, LAST, ENDA, POSQ, ACC, M, L,
                  s_qb, s_qh, s_kb, s_kh, s_kn, s_sb, s_sh, s_ab, s_ah, s_as, s_mb, s_mh,
                  H, N, CONTEXT, sm_scale,
                  D: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr,
                  IS_INT8: tl.constexpr, LEN_AWARE: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    last = tl.load(LAST + b)      # position of the newest token in the ring (moshi: last_offset)
    end_a = tl.load(ENDA + b)     # end_offset after this write; slots >= end_a were never written
    pos_q = tl.load(POSQ + b)     # query position (moshi: state.offset)
    end_index = last % N
    start = pid_s * SPLIT
    stop = tl.minimum(start + SPLIT, N)
    if LEN_AWARE:
        stop = tl.minimum(stop, end_a)  # skip slots that hold nothing yet
    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * s_qb + h * s_qh + offs_d).to(tl.float32) * sm_scale
    m_i = tl.zeros([], dtype=tl.float32) - float("inf")  # loop-carried values must keep one type
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    kbase = b * s_kb + h * s_kh
    sbase = b * s_sb + h * s_sh
    for n0 in range(start, stop, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < stop
        k = tl.load(K + kbase + offs_n[:, None] * s_kn + offs_d[None, :], mask=nmask[:, None], other=0).to(tl.float32)
        if IS_INT8:
            k = k * tl.load(KS + sbase + offs_n, mask=nmask, other=0.0)[:, None]
        s = tl.sum(q[None, :] * k, axis=1)
        # moshi RingKVCache.complete + causal/context mask, recomputed per slot
        delta = offs_n - end_index
        pos = tl.where(delta <= 0, last + delta, last + delta - N)
        valid = nmask & (offs_n < end_a) & (pos >= 0) & (pos_q - pos >= 0) & (pos_q - pos < CONTEXT)
        s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(s - m_safe)
        alpha = tl.exp(m_i - m_safe)
        v = tl.load(V + kbase + offs_n[:, None] * s_kn + offs_d[None, :], mask=nmask[:, None], other=0).to(tl.float32)
        if IS_INT8:
            v = v * tl.load(VS + sbase + offs_n, mask=nmask, other=0.0)[:, None]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    tl.store(ACC + b * s_ab + h * s_ah + pid_s * s_as + offs_d, acc)
    tl.store(M + b * s_mb + h * s_mh + pid_s, m_i)
    tl.store(L + b * s_mb + h * s_mh + pid_s, l_i)


@triton.jit
def _merge_kernel(ACC, M, L, OUT, s_ab, s_ah, s_as, s_mb, s_mh, s_ob, s_oh, H,
                  D: tl.constexpr, S: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_s = tl.arange(0, S)
    offs_d = tl.arange(0, D)
    m = tl.load(M + b * s_mb + h * s_mh + offs_s)
    l = tl.load(L + b * s_mb + h * s_mh + offs_s)
    m_max = tl.max(m, axis=0)
    m_max = tl.where(m_max == -float("inf"), 0.0, m_max)
    w = tl.exp(m - m_max)  # splits with no valid slot have m = -inf -> weight 0
    acc = tl.load(ACC + b * s_ab + h * s_ah + offs_s[:, None] * s_as + offs_d[None, :])
    num = tl.sum(acc * w[:, None], axis=0)
    den = tl.sum(l * w, axis=0)
    out = num / tl.where(den == 0.0, 1.0, den)
    tl.store(OUT + b * s_ob + h * s_oh + offs_d, out.to(OUT.dtype.element_ty))


class FusedRingKV:
    """Drop-in for moshi's RingKVCache in the temporal transformer (respect_exec_mask=True), with
    optional int8 storage. Holds the ring, the end offsets and the split-K scratch buffers."""

    SPLITS = 16
    BLOCK_N = 64

    def __init__(self, batch_size, num_heads, dim_per_head, capacity, device, dtype, int8, len_aware):
        self.capacity, self.int8, self.len_aware = capacity, int8, len_aware
        B, H, N, D = batch_size, num_heads, capacity, dim_per_head
        store = torch.int8 if int8 else dtype
        self.k = torch.zeros(B, H, N, D, device=device, dtype=store)
        self.v = torch.zeros(B, H, N, D, device=device, dtype=store)
        if int8:
            self.ks = torch.zeros(B, H, N, device=device, dtype=torch.float32)
            self.vs = torch.zeros(B, H, N, device=device, dtype=torch.float32)
        else:
            self.ks = self.vs = torch.zeros(1, device=device, dtype=torch.float32)  # unused
        self.end_offset = torch.zeros(B, device=device, dtype=torch.long)
        self.last = torch.zeros(B, device=device, dtype=torch.long)
        S = self.SPLITS
        self.acc = torch.empty(B, H, S, D, device=device, dtype=torch.float32)
        self.m = torch.empty(B, H, S, device=device, dtype=torch.float32)
        self.l = torch.empty(B, H, S, device=device, dtype=torch.float32)

    def reset(self, reset_mask):
        self.end_offset[:] = torch.where(reset_mask, torch.zeros_like(self.end_offset), self.end_offset)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.ks, self.vs))

    @staticmethod
    def _quant(x):  # x [B, H, 1, D] -> int8 [B, H, 1, D], scale [B, H, 1]
        scale = x.float().abs().amax(-1).clamp(min=1e-8) / 127.0
        q = torch.round(x.float() / scale[..., None]).clamp(-127, 127).to(torch.int8)
        return q, scale

    def write(self, k, v, exec_mask):
        """k, v: [B, H, 1, D]. Same write/advance rule as RingKVCache.complete with respect_exec_mask."""
        B, H, T, D = k.shape
        assert T == 1, "fused path is for one-token decode steps"
        idx = (self.end_offset % self.capacity).view(B, 1, 1, 1).expand(B, H, 1, D)
        if self.int8:
            kq, ksc = self._quant(k)
            vq, vsc = self._quant(v)
            self.k.scatter_(2, idx, kq)
            self.v.scatter_(2, idx, vq)
            sidx = idx[..., 0]
            self.ks.scatter_(2, sidx, ksc)
            self.vs.scatter_(2, sidx, vsc)
        else:
            self.k.scatter_(2, idx, k.to(self.k.dtype))
            self.v.scatter_(2, idx, v.to(self.v.dtype))
        self.last.copy_(self.end_offset + T - 1)
        self.end_offset[:] = torch.where(exec_mask, self.end_offset + T, self.end_offset)

    def attend(self, q, pos_q, context):
        """q: [B, H, 1, D] (after RoPE); pos_q: [B] query positions. Returns [B, H, 1, D]."""
        B, H, _, D = q.shape
        q2 = q[:, :, 0, :]
        if not q2.is_contiguous():
            q2 = q2.contiguous()
        out = torch.empty_like(q2)
        S, N = self.SPLITS, self.capacity
        split = triton.cdiv(N, S)
        _split_kernel[(B * H, S)](
            q2, self.k, self.ks, self.v, self.vs, self.last, self.end_offset, pos_q,
            self.acc, self.m, self.l,
            q2.stride(0), q2.stride(1), self.k.stride(0), self.k.stride(1), self.k.stride(2),
            self.ks.stride(0) if self.int8 else 0, self.ks.stride(1) if self.int8 else 0,
            self.acc.stride(0), self.acc.stride(1), self.acc.stride(2), self.m.stride(0), self.m.stride(1),
            H, N, context if context is not None else N + 1, 1.0 / math.sqrt(D),
            D=D, BLOCK_N=self.BLOCK_N, SPLIT=split, IS_INT8=self.int8, LEN_AWARE=self.len_aware,
        )
        _merge_kernel[(B * H,)](
            self.acc, self.m, self.l, out,
            self.acc.stride(0), self.acc.stride(1), self.acc.stride(2), self.m.stride(0), self.m.stride(1),
            out.stride(0), out.stride(1), H, D=D, S=S,
        )
        return out[:, :, None, :]


def _patched_forward(self, query, key, value):
    """moshi StreamingMultiheadAttention.forward (self-attention, streaming, T=1) with steps 2-3 fused."""
    from einops import rearrange
    from moshi.modules.transformer import apply_weights_per_step
    state = self._streaming_state
    if state is None or not isinstance(state.kv_cache, FusedRingKV) or query.shape[1] != 1:
        return self._moshi_forward(query, key, value)
    offset, offset_cpu = state.offset, state.offset_cpu
    projected = apply_weights_per_step(self.in_projs, self.weights_per_step_schedule, query, offset_cpu)
    q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)
    if self.rope:
        q, k = self.rope(q, k, offset, time_before_heads=False)
    state.kv_cache.write(k, v, state.exec_mask)
    x = state.kv_cache.attend(q, offset, self.context)
    x = rearrange(x, "b h t d -> b t (h d)")
    x = apply_weights_per_step(self.out_projs, self.weights_per_step_schedule, x, offset_cpu)
    state.offset[:] = torch.where(state.exec_mask, state.offset + 1, state.offset)
    state.offset_cpu += 1
    return x


def patch_temporal_attention(lm, mode):
    """Swap the temporal transformer's self-attention onto the fused kernel. Call before streaming
    starts (LMGen.streaming_forever). Depth transformer and Mimi are untouched."""
    assert mode in MODES, mode
    if mode == "sdpa":
        return 0
    import types
    int8, len_aware = mode.startswith("int8"), mode.endswith("-len")
    n = 0
    for mod in lm.transformer.modules():
        if type(mod).__name__ != "StreamingMultiheadAttention" or mod.cross_attention:
            continue
        assert mod.kv_repeat == 1 and not mod.weights_per_step and mod.context is not None
        orig_init = mod._init_streaming_state

        def _init(self, batch_size, _orig=orig_init):
            st = _orig(batch_size)
            rk = st.kv_cache
            D = self.embed_dim // self.num_heads
            st.kv_cache = FusedRingKV(batch_size, self.num_heads, D, rk.capacity, rk.cache.device,
                                      rk.cache.dtype, int8, len_aware)
            del rk  # free moshi's bf16 ring
            return st

        mod._init_streaming_state = types.MethodType(_init, mod)
        mod._moshi_forward = mod.forward
        mod.forward = types.MethodType(_patched_forward, mod)
        n += 1
    return n
