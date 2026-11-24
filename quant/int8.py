"""Quantized loaders for the Moshi LM:

  int8wo        torchao Int8WeightOnly, temporal transformer only (depth transformer and Mimi stay bf16)
  int8wo-all    torchao Int8WeightOnly, temporal + depth transformers
  q8bnb         Kyutai's pre-quantized checkpoint (kyutai/moshiko-pytorch-q8, bitsandbytes QLinear).
                Not weight-only: QLinear casts activations to fp16 and runs LLM.int8-style bnb.matmul.
  int4wo*       torchao Int4WeightOnly (tinygemm, sm80+), temporal only; group 128 or 32, minmax or HQQ
  w8triton      kernels/w8a16.py, temporal only

torchao INT8 only fuses the dequantization into the GEMV under torch.compile; in eager mode it
materializes bf16 weights.
"""
import torch
from torch import nn

MODES = ("none", "int8wo", "int8wo-all", "q8bnb", "int4wo", "int4wo-hqq", "int4wo-g32", "int4wo-hqq-g32",
         "w8triton")
Q8_REPO = "kyutai/moshiko-pytorch-q8"
INT4_GROUP = 128


def _is_linear(mod, fqn):
    return isinstance(mod, nn.Linear)


def _is_int4_ok(mod, fqn):
    # tinygemm int4 packing needs K a multiple of 1024 and N a multiple of 8
    return isinstance(mod, nn.Linear) and mod.in_features % 1024 == 0 and mod.out_features % 8 == 0


def _int4_config(use_hqq, group=INT4_GROUP):
    """Int4 weight-only, tinygemm tensor-core tiled layout (aten _weight_int4pack_mm).
    torchao 0.14's default (version 2) packing needs fbgemm-gpu-genai, which we don't have."""
    from torchao.quantization import Int4WeightOnlyConfig
    try:
        return Int4WeightOnlyConfig(group_size=group, use_hqq=use_hqq, version=1)
    except TypeError:
        return Int4WeightOnlyConfig(group_size=group, use_hqq=use_hqq)


def load_quantized(mode, device="cuda"):
    """Returns (info, mimi, tok, lm) with the LM quantized per `mode`, keeping the GPU peak low:
    the bf16 LM is loaded on the CPU, then each linear is moved to the GPU and quantized there."""
    from moshi.models import loaders
    assert mode in MODES, mode
    if mode == "q8bnb":
        return _load_q8bnb(loaders, device)
    info = loaders.CheckpointInfo.from_hf_repo(loaders.DEFAULT_REPO)
    mimi = info.get_mimi(device=device)
    tok = info.get_text_tokenizer()
    if mode == "none":
        return info, mimi, tok, info.get_moshi(device=device, dtype=torch.bfloat16)

    lm = info.get_moshi(device="cpu", dtype=torch.bfloat16)
    if mode == "w8triton":
        # our Triton W8A16 kernel (kernels/w8a16.py), temporal transformer only
        from kernels.w8a16 import quantize_temporal_w8
        n = quantize_temporal_w8(lm, device)
        assert n == 32 * 4, n  # 32 layers x (in_proj, out_proj, ffn in, ffn out)
        lm.to(device)
        torch.cuda.empty_cache()
        return info, mimi, tok, lm
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    if mode.startswith("int4"):
        # temporal transformer only; depth transformer, embeddings and heads stay bf16
        group = 32 if mode.endswith("-g32") else INT4_GROUP
        quantize_(lm.transformer, _int4_config("hqq" in mode, group), filter_fn=_is_int4_ok, device=device)
        lm.to(device)
        torch.cuda.empty_cache()
        return info, mimi, tok, lm
    targets = [lm.transformer]
    if mode == "int8wo-all":
        targets += [m for n, m in lm.named_children() if n.startswith("depformer") or n == "linears"]
    for t in targets:
        quantize_(t, Int8WeightOnlyConfig(), filter_fn=_is_linear, device=device)
    lm.to(device)  # everything not quantized (embeddings, norms, heads) moves as bf16
    torch.cuda.empty_cache()
    return info, mimi, tok, lm


def _load_q8bnb(loaders, device):
    """Kyutai's bitsandbytes int8 checkpoint. moshi 0.2.13's get_moshi_lm casts every float tensor
    in the checkpoint to the model dtype, including the `*_scb` int8 scales that QLinear needs in
    float32, so QLinear.forward raises. Work around it without touching moshi: build the model
    without weights, then load the state dict ourselves, keeping `*_scb` in float32."""
    from safetensors.torch import load_file
    info = loaders.CheckpointInfo.from_hf_repo(Q8_REPO)
    mimi, tok = info.get_mimi(device=device), info.get_text_tokenizer()
    # build the (random-init) skeleton on the CPU so the GPU never holds a bf16 copy of the 7B;
    # fall back to building it on the GPU if bitsandbytes cannot quantize the skeleton on CPU
    try:
        lm = info.get_moshi(device="cpu", dtype=torch.bfloat16, load_weight=False)
    except Exception as e:
        print("q8 skeleton on CPU failed, building on GPU:", str(e)[:120])
        lm = info.get_moshi(device=device, dtype=torch.bfloat16, load_weight=False)
    state = load_file(str(info.moshi_weights), device=str(device))
    for k, v in state.items():
        if v.dtype.is_floating_point and not k.endswith("_scb"):
            keep_f32 = k.startswith("condition_provider.") or k.startswith("fuser.")
            state[k] = v.float() if keep_f32 else v.to(torch.bfloat16)
    lm.load_state_dict(state, assign=True)
    lm.to(device)  # anything not in the checkpoint (buffers) follows the weights
    for m in lm.modules():  # scales must stay float32 (QLinear.forward checks this)
        if hasattr(m, "weight_scb"):
            assert m.weight_scb.dtype == torch.float32, m.weight_scb.dtype
    lm.eval()
    torch.cuda.empty_cache()
    return info, mimi, tok, lm


def compile_temporal(lm):
    """torch.compile each Linear of the temporal transformer on its own, so the int8/int4 dequant
    fuses into the GEMV (gpt-fast pattern).

    Compiling lm.forward_text as a whole does not work: moshi's per-layer streaming state holds Python ints
    (offset_cpu) that are read and incremented inside the forward, so dynamo guards on their values
    and recompiles every step; the recompile on moshi's CUDAGraphed capture call touches the CUDA
    RNG and fails ("Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture").
    A single Linear has no streaming state and fixed shapes, so it compiles once during warm-up.
    The FFN never calls Linear.forward: moshi passes its weights into its own compiled
    gating_forward_kernel, which already fuses whatever weight type it is given."""
    n = 0
    for mod in lm.transformer.modules():
        if isinstance(mod, nn.Linear):
            mod.forward = torch.compile(mod.forward, dynamic=False, fullgraph=True)
            n += 1
    return n


def weight_bytes(lm):
    """Bytes actually stored for LM weights (int8 data + scales for quantized tensors)."""
    total = 0
    for p in lm.parameters():
        impl = getattr(p, "tensor_impl", None)
        if impl is None:
            impl = getattr(p, "layout_tensor", None)
        if impl is not None:  # torchao quantized tensor: count its stored pieces, not the logical bf16 size
            for name in ("int_data", "packed_weight", "scale", "zero_point", "scale_and_zero"):
                t = getattr(impl, name, None)
                if t is not None:
                    total += t.numel() * t.element_size()
        else:
            total += p.numel() * p.element_size()
    for b in lm.buffers():
        total += b.numel() * b.element_size()
    return total


if __name__ == "__main__":
    # self-check on a tiny model (CPU-safe logic check, GPU if present): int8wo halves linear bytes
    # and keeps outputs close
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = nn.Sequential(nn.Linear(256, 512, bias=False), nn.Linear(512, 256, bias=False)).to(dev, torch.bfloat16)
    x = torch.randn(4, 256, device=dev, dtype=torch.bfloat16)
    ref = m(x)
    b0 = weight_bytes(m)
    quantize_(m, Int8WeightOnlyConfig(), filter_fn=_is_linear)
    out = m(x)
    b1 = weight_bytes(m)
    rel = ((out - ref).float().norm() / ref.float().norm()).item()
    print(f"bytes {b0} -> {b1}, rel err {rel:.4f}")
    assert b1 < 0.6 * b0, (b0, b1)
    assert rel < 0.02, rel
    print("int8 self-check ok")
