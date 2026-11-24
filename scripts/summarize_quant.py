"""One row per quantization config: single-stream latency next to teacher-forced quality vs bf16.
Writes results/quant_table.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
# (label, latency file, quality file or None for the reference, weights touched)
ROWS = [
    ("bf16 (moshi stock)", "bf16_b1_r1", None, "none"),
    ("torchao int8 weight-only, eager", "int8wo_eager", "quality_int8wo", "temporal"),
    ("torchao int8 weight-only, temporal+depth", None, "quality_int8wo-all", "temporal+depth"),
    ("Kyutai q8 checkpoint (bnb, W8A8-style)", "q8bnb", "quality_q8bnb", "all LM linears"),
    ("Kyutai q8, 16 GB cap, CPU-built skeleton", "q8bnb_cap16", "quality_q8bnb", "all LM linears"),
    ("W8A16 Triton kernel (ours)", "w8triton", "quality_w8triton", "temporal"),
    ("int4 g128 minmax (tinygemm)", "int4wo_eager", "quality_int4wo", "temporal"),
    ("int4 g128 HQQ", None, "quality_int4wo-hqq", "temporal"),
    ("int4 g32 minmax", None, "quality_int4wo-g32", "temporal"),
    ("int4 g32 HQQ", "int4wo_hqq_g32", "quality_int4wo-hqq-g32", "temporal"),
]


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if name and p.exists() else None


def main():
    rows = []
    for label, lat_name, q_name, scope in ROWS:
        lat, q = load(lat_name), load(q_name)
        row = {"config": label, "quantized": scope, "latency_source": lat_name and f"results/{lat_name}.json",
               "quality_source": q_name and f"results/{q_name}.json"}
        if lat:
            s = lat["stages_ms"]
            row.update({"frame_p50_ms": s["wall"]["p50"], "frame_p99_ms": s["wall"]["p99"],
                        "frame_max_ms": s["wall"]["max"], "temporal_p50_ms": s["temporal"]["p50"],
                        "frames_over_80ms": lat["frames_over_80ms"], "frames_measured": lat["frames_measured"],
                        "peak_alloc_gb": lat["peak_mem_gb"]["allocated"],
                        "device_used_gb": lat["peak_mem_gb"].get("device_used_now"),
                        "lm_weight_gb": lat.get("lm_weight_gb")})
        if q:
            row.update({"audio_top1": round(q["audio"]["top1_agree"], 4), "audio_kl": round(q["audio"]["kl_mean"], 5),
                        "text_top1": round(q["text"]["top1_agree"], 4),
                        "text_top1_words": q["text"].get("top1_agree_words")})
        elif q_name is None:
            row.update({"audio_top1": 1.0, "audio_kl": 0.0, "text_top1": 1.0, "text_top1_words": 1.0})
        rows.append(row)
    out = {"what": "quantization configs on one L4 stream; quality = teacher-forced vs bf16, "
                   "1500 steps over 5 clips; noise floor (bf16 vs bf16) = exactly 0 (results/quality_none.json)",
           "rows": rows}
    (R / "quant_table.json").write_text(json.dumps(out, indent=2))
    hdr = f"{'config':44s} {'p50':>6s} {'p99':>6s} {'>80':>5s} {'peakGB':>7s} {'audio@1':>8s} {'audioKL':>8s}"
    print(hdr)
    for r in rows:
        print(f"{r['config']:44s} {r.get('frame_p50_ms', float('nan')):6.1f} {r.get('frame_p99_ms', float('nan')):6.1f} "
              f"{r.get('frames_over_80ms', -1):5d} {r.get('peak_alloc_gb', float('nan')):7.2f} "
              f"{r.get('audio_top1', float('nan')):8.4f} {r.get('audio_kl', float('nan')):8.5f}")
    return out


if __name__ == "__main__":
    o = main()
    by = {r["config"]: r for r in o["rows"]}
    ours, base = by["W8A16 Triton kernel (ours)"], by["bf16 (moshi stock)"]
    assert ours["frame_p50_ms"] < base["frame_p50_ms"] and ours["peak_alloc_gb"] < base["peak_alloc_gb"]
    assert all(0 <= r.get("audio_top1", 0) <= 1 for r in o["rows"])
