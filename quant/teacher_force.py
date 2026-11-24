"""Teacher-forced quality of a quantized Moshi LM against bf16.

  --record          bf16: run LMGen (sampling, fixed seed) on the 5 clips through Mimi and save the exact
                    token stream the LM saw and produced under WORK.
  --quant none      replay the stream through bf16 and save the reference next-token distributions
                    (text + 8 audio codebooks). Run in its own process: LMGen holds the LM streaming state.
  --quant MODE      replay the same stream through MODE and compare with the reference:
                    KL(bf16 || quant) and top-1 agreement, per head. Writes results/quality_<MODE>.json.

Free-running samples drift apart from sampling noise alone, so they can't isolate quantization
error. Same inputs + same history means any difference in the distributions is the quantization.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import ROOT, make_lm_gen, seed_all, write_result
from bench.stream_bench import read_wav
from quant.int8 import MODES, load_quantized

WORK = Path("/root/tf")  # VM-local scratch, never pulled back
STEPS_PER_CLIP = 300  # 24 s of each clip
LONG_STEPS = 3200  # 256 s: one uninterrupted call that fills (and wraps) the 3000-slot ring


def _audio_inputs(long):
    clips = [read_wav(c) for c in sorted((ROOT / "audio").glob("clip*.wav"))]
    if long:  # the five clips twice over, as one 300 s conversation
        import numpy as np
        return [np.concatenate(clips + clips)], LONG_STEPS
    return clips, STEPS_PER_CLIP


@torch.no_grad()
def record(long=False):
    seed_all()
    info, mimi, tok, lm = load_quantized("none")
    stream = []  # per clip: list of (input [17] long, text_token long, audio_tokens [8] long)
    cur = {}
    lm_gen = make_lm_gen(info, lm,
                         on_text_hook=lambda t: cur.__setitem__("text", t.clone()),
                         on_audio_hook=lambda a: cur.__setitem__("audio", a.clone()))
    mimi.streaming_forever(1); lm_gen.streaming_forever(1)
    st = lm_gen._streaming_state
    main = st.graphed_main

    def spy(inp, *rest):
        cur["input"] = inp.clone()
        return main(inp, *rest)
    st.graphed_main = spy
    fn = int(mimi.sample_rate / mimi.frame_rate)
    audios, n_steps = _audio_inputs(long)
    for audio in audios:
        x = torch.from_numpy(audio).cuda()[None, None]
        mimi.reset_streaming(); lm_gen.reset_streaming()
        steps = []
        for f in range(n_steps):
            codes = mimi.encode(x[..., f * fn:(f + 1) * fn])
            if f == 0:
                mimi.reset_streaming()
            lm_gen.step(codes[:, :, :1])
            steps.append((cur["input"][0, :, 0].cpu(), cur["text"][0].cpu(), cur["audio"][0].cpu()))
        stream.append(steps)
    WORK.mkdir(exist_ok=True)
    torch.save(stream, WORK / ("stream_long.pt" if long else "stream.pt"))
    print("recorded", sum(len(s) for s in stream), "steps")


@torch.no_grad()
def replay(lm, stream):
    """Teacher-forced pass: returns per-step log-probs, text [N, Vt] and audio [N, 8, Va] (fp16, CPU)."""
    text_lp, audio_lp = [], []
    dev = lm.device
    for steps in stream:
        with lm.streaming(1):
            for inp, text_tok, audio_tok in steps:
                tr_out, text_logits = lm.forward_text(inp.to(dev)[None, :, None])
                text_lp.append(F.log_softmax(text_logits.float().view(-1), -1).half().cpu())
                per_cb = []
                with lm.depformer.streaming(1):
                    prev = text_tok.to(dev)
                    for cb in range(lm.dep_q):
                        logits = lm.forward_depformer(cb, prev.view(1, 1, 1), tr_out)
                        per_cb.append(F.log_softmax(logits.float().view(-1), -1).half().cpu())
                        prev = audio_tok[cb].to(dev)  # force the bf16 run's token, not our own
                audio_lp.append(torch.stack(per_cb))
    return {"text": torch.stack(text_lp), "audio": torch.stack(audio_lp)}


def compare(ref, q):
    def kl(p_log, q_log):  # KL(p || q) per row, in nats
        p_log, q_log = p_log.float(), q_log.float()
        return (p_log.exp() * (p_log - q_log)).sum(-1)
    out = {}
    t_kl = kl(ref["text"], q["text"])
    r_top, q_top = ref["text"].argmax(-1), q["text"].argmax(-1)
    words = ~((r_top == 0) | (r_top == 3))  # steps where bf16's top text token is a word, not padding
    out["text"] = {"kl_mean": float(t_kl.mean()), "kl_p99": float(t_kl.quantile(0.99)),
                   "top1_agree": float((r_top == q_top).float().mean()),
                   "word_steps": int(words.sum()),
                   "top1_agree_words": float((r_top == q_top)[words].float().mean()) if words.any() else None,
                   "kl_mean_words": float(t_kl[words].mean()) if words.any() else None}
    a_kl = kl(ref["audio"], q["audio"])  # [N, 8]
    a_top = (ref["audio"].argmax(-1) == q["audio"].argmax(-1)).float()
    out["audio"] = {"kl_mean": float(a_kl.mean()), "kl_p99": float(a_kl.flatten().quantile(0.99)),
                    "top1_agree": float(a_top.mean()),
                    "per_codebook_kl": [round(float(v), 5) for v in a_kl.mean(0)],
                    "per_codebook_top1": [round(float(v), 4) for v in a_top.mean(0)]}
    out["steps"] = int(ref["text"].shape[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--quant", choices=MODES)
    ap.add_argument("--kv", default="sdpa", choices=("sdpa", "bf16", "bf16-len", "int8", "int8-len"))
    ap.add_argument("--long", action="store_true", help="one 3200-step call that fills the 3000-slot ring")
    args = ap.parse_args()
    if args.record:
        record(args.long)
        return
    sfx = "_long" if args.long else ""
    stream = torch.load(WORK / f"stream{sfx}.pt")
    info, mimi, tok, lm = load_quantized(args.quant)
    del mimi
    if args.kv != "sdpa":
        from kernels.kv_attn import patch_temporal_attention
        assert patch_temporal_attention(lm, args.kv) == 32
    q = replay(lm, stream)
    if args.long:
        if args.quant == "none" and args.kv == "sdpa":
            torch.save(q, WORK / "ref_bf16_long.pt")
            print("saved long reference", q["text"].shape[0], "steps")
            return
        ref = torch.load(WORK / "ref_bf16_long.pt")
        res = compare(ref, q)
        n = ref["text"].shape[0]
        late = {k: v[2400:] for k, v in ref.items()}, {k: v[2400:] for k, v in q.items()}
        res["after_ring_fills"] = compare(*late)  # steps 2400-3199 (ring full from step 3000)
        res["what"] = "teacher-forced vs bf16 on one uninterrupted 3200-step call (fills the 3000-slot ring)"
        res["config"] = {"quant": args.quant, "kv": args.kv, "steps": n, "clips": "1-5 twice, concatenated"}
        write_result(f"quality_long_{args.quant}" + ("" if args.kv == "sdpa" else f"_kv-{args.kv}"), res)
        print(res)
        assert res["steps"] == LONG_STEPS
        return
    if args.quant == "none" and args.kv == "sdpa":
        torch.save(q, WORK / "ref_bf16.pt")
        q2 = replay(lm, stream[:1])  # determinism floor: same model, same stream, twice
        ref = {k: v[: q2[k].shape[0]] for k, v in q.items()}
        res = compare(ref, q2)
        res["what"] = "bf16 vs bf16 replay of clip 1 (noise floor of the measurement)"
        res["config"] = {"quant": "none", "steps_per_clip": STEPS_PER_CLIP, "clips": 1}
        write_result("quality_none", res)
        print(res)
        assert res["text"]["kl_mean"] < 1e-3 and res["audio"]["top1_agree"] > 0.99, res
        return
    ref = torch.load(WORK / "ref_bf16.pt")
    res = compare(ref, q)
    res["what"] = "teacher-forced next-token agreement vs bf16 on the 5 clips, same token history"
    res["config"] = {"quant": args.quant, "kv": args.kv, "steps_per_clip": STEPS_PER_CLIP, "clips": len(stream)}
    write_result(f"quality_{args.quant}" + ("" if args.kv == "sdpa" else f"_kv-{args.kv}"), res)
    print(res)
    # self-check: agreement numbers are probabilities, every step was compared
    assert 0 <= res["text"]["top1_agree"] <= 1 and 0 <= res["audio"]["top1_agree"] <= 1
    assert res["steps"] == STEPS_PER_CLIP * len(stream)


if __name__ == "__main__":
    main()
