"""Capture the rest of Mimi's streaming encode in a CUDA graph.

moshi graph-captures Mimi's SEANet encoder and encoder transformer, but the frame-rate
downsampling conv and the 8-codebook RVQ lookup (`quantizer.encode`) still run eagerly, as
dozens of small kernels launched from the CPU. On hosts with a slower or busier CPU those
launches stall the GPU, and the encode stage spikes from ~3-5 ms to 9-20 ms; in our runs every
frame over budget was such a spike. This patch wraps both steps in one moshi CUDAGraphed
callable from outside the model. Outputs are bit-identical to the stock path.
"""
import types

import torch


def _conv_forward_inplace(self, x):
    """moshi StreamingConv1d.forward with one change: the `first` flag is updated in place.
    The stock code reassigns `state.first`, so a CUDA graph keeps reading the tensor it captured
    while a stream reset writes into the new one, and the first-frame padding after a reset is lost."""
    B, C, T = x.shape
    assert T > 0 and T % self._stride == 0, "Steps must be multiple of stride"
    state = self._streaming_state
    if state is None:
        state = self._init_streaming_state(B)
    TP = state.previous.shape[-1]
    if TP and self.pad_mode == "replicate":
        assert T >= TP, "Not enough content to pad streaming."
        state.previous[:] = torch.where(state.first.view(-1, 1, 1) & state.exec_mask.view(-1, 1, 1),
                                        x[..., :1], state.previous)
    if TP:
        x = torch.cat([state.previous, x], dim=-1)
    y = self.conv(x)
    if TP:
        state.previous[:] = torch.where(state.exec_mask.view(-1, 1, 1), x[..., -TP:], state.previous)
        if self.pad_mode == "replicate":
            state.first[:] = torch.where(state.exec_mask, torch.zeros_like(state.first), state.first)
    return y


def patch_mimi_encode(mimi):
    from moshi.utils.compile import CUDAGraphed

    if getattr(mimi, "downsample", None) is not None:
        for m in mimi.downsample.modules():
            if type(m).__name__ == "StreamingConv1d":
                m.forward = types.MethodType(_conv_forward_inplace, m)

    def _post(emb):
        return mimi.quantizer.encode(mimi._to_framerate(emb))

    graphed_post = {}

    def encode(x):
        state = mimi._streaming_state
        if state is None or x.shape[-1] != mimi.frame_size:
            return type(mimi).encode(mimi, x)
        emb = state.graphed_encoder(x).clone()
        if mimi.encoder_transformer is not None:
            (emb,) = state.graphed_tr_enc(emb)
        g = graphed_post.get(id(state))
        if g is None:  # one graph per streaming state (batch size)
            g = graphed_post[id(state)] = CUDAGraphed(_post)
        return g(emb).clone()  # a graph replay overwrites its static output buffer

    mimi.encode = encode
    return mimi


@torch.no_grad()
def _self_test(n_frames=120):
    """Stock and patched Mimi must produce identical codes on the same audio, across a reset."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.common import load, write_result
    from bench.stream_bench import read_wav
    _, mimi, _, lm = load()
    del lm
    torch.cuda.empty_cache()
    x = torch.from_numpy(read_wav(sorted((Path(__file__).resolve().parents[1] / "audio").glob("clip*.wav"))[0]))
    x = x.cuda()[None, None]
    fn = mimi.frame_size

    def run(patched):
        mimi.streaming_forever(1)
        if patched:
            patch_mimi_encode(mimi)
        codes = []
        for f in range(n_frames):
            if f in (0, 60):
                mimi.reset_streaming()
            codes.append(mimi.encode(x[..., f * fn:(f + 1) * fn]).clone())
        torch.cuda.synchronize()
        out = torch.cat(codes, -1)
        if patched:
            del mimi.encode
        mimi._stop_streaming()
        return out

    ref, got = run(False), run(True)
    same = bool(torch.equal(ref, got))
    write_result("mimi_graph_check", {"what": "stock vs graph-captured Mimi encode, codes over 120 frames with resets",
                                      "frames": n_frames, "identical": same,
                                      "mismatching_codes": int((ref != got).sum())})
    assert same, int((ref != got).sum())
    print("mimi graph patch: identical codes over", n_frames, "frames")


if __name__ == "__main__":
    _self_test()
