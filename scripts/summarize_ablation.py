"""CUDA-graph / torch.compile ablation summary -> results/ablation_graphs.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
RUNS = {"baseline": "baseline_l4_bf16", "no_graphs": "ablation_nograph",
        "no_compile": "ablation_nocompile", "no_graphs_no_compile": "ablation_nograph_nocompile"}
STAGES = ("encode", "temporal", "text", "depth", "decode", "host", "frame", "wall")


def main():
    runs = {k: json.loads((R / f"{v}.json").read_text()) for k, v in RUNS.items()}
    base = runs["baseline"]["stages_ms"]
    table = {}
    for k, r in runs.items():
        s = r["stages_ms"]
        table[k] = {
            "source": f"results/{RUNS[k]}.json",
            "cuda_graphs": r["config"].get("cuda_graphs", True),
            "torch_compile": r["config"].get("torch_compile", True),
            "p50_ms": {st: s[st]["p50"] for st in STAGES},
            "wall_p99_ms": s["wall"]["p99"],
            "frames_over_80ms_pct": r["frames_over_80ms_pct"],
            "slowdown_vs_baseline": {st: round(s[st]["p50"] / base[st]["p50"], 2) for st in STAGES},
        }
    prof = json.loads((ROOT / "profiles" / "profile_l4_bf16.json").read_text())
    gemv = max((t for t in prof["top_gpu_ops"] if "gemv" in t["name"] and "bfloat16" in t["name"]),
               key=lambda t: t["us_per_frame"])
    g = prof["weights_gb"]["lm_groups"]
    # the big-GEMV bucket (145 calls/frame) = 128 temporal linears + text_linear + 8 depformer_in
    # + 8 depth output heads ("linears"); count all of their bytes, not just the temporal ones
    bucket_gb = g["transformer"] + g["text_linear"] + g["depformer_in"] + g["linears"]
    probe = R / "bw_probe_l4.json"
    measured = json.loads(probe.read_text()) if probe.exists() else None
    # the torch.sum read probe (261 GB/s) is slower than moshi's own GEMVs, so it is not a ceiling;
    # the datasheet figure is used as the ceiling and the probe is reported alongside
    ceiling = prof["l4_peak_gbs"]
    temporal_stage_ms = base["temporal"]["p50"]
    out = {
        "what": "launch-overhead ablation: moshi's CUDA graphs and torch.compile turned off one at a time (L4, bf16, batch 1)",
        "runs": table,
        "temporal_bandwidth": {
            "source": "profiles/profile_l4_bf16.json" + (", results/bw_probe_l4.json" if measured else "")
                      + " (the profiler adds ~7 ms/frame)",
            "gemv_bucket_weight_gb": round(bucket_gb, 3),
            "gemv_bucket_us_per_frame": gemv["us_per_frame"],
            "gemv_calls_per_frame": gemv["per_frame"],
            "gemv_bucket_gb_per_s": round(bucket_gb / (gemv["us_per_frame"] / 1e6), 1),
            "temporal_stage_ms_p50": temporal_stage_ms,
            "temporal_stage_effective_gb_per_s": round(g["transformer"] / (temporal_stage_ms / 1e3), 1),
            "ceiling_gb_per_s": ceiling,
            "ceiling_kind": "datasheet",
            "probe_read_gb_per_s": measured["read_gbs"] if measured else None,
            "probe_copy_gb_per_s": measured["copy_gbs"] if measured else None,
        },
    }
    bw = out["temporal_bandwidth"]
    bw["gemv_fraction_of_ceiling"] = round(bw["gemv_bucket_gb_per_s"] / ceiling, 3)
    bw["stage_fraction_of_ceiling"] = round(bw["temporal_stage_effective_gb_per_s"] / ceiling, 3)
    out["diagnosis"] = (
        "The 7B temporal step is weight-bandwidth-bound: its large bf16 GEMVs stream {:.1f} GB at {:.0f} GB/s "
        "({:.0%} of the {} {:.0f} GB/s ceiling), and turning CUDA graphs off adds only {:.1f} ms to it. "
        "The depth transformer and Mimi are launch-bound and only real time because moshi CUDA-graphs them "
        "(depth {:.1f} -> {:.1f} ms without graphs)."
    ).format(bucket_gb, bw["gemv_bucket_gb_per_s"], bw["gemv_fraction_of_ceiling"], bw["ceiling_kind"], ceiling,
             table["no_graphs"]["p50_ms"]["temporal"] - table["baseline"]["p50_ms"]["temporal"],
             table["baseline"]["p50_ms"]["depth"], table["no_graphs"]["p50_ms"]["depth"])
    (R / "ablation_graphs.json").write_text(json.dumps(out, indent=2))
    print(out["diagnosis"])
    return out


if __name__ == "__main__":
    o = main()
    b, g = o["runs"]["baseline"], o["runs"]["no_graphs"]
    assert b["cuda_graphs"] and not g["cuda_graphs"] and not o["runs"]["no_compile"]["torch_compile"]
    assert g["p50_ms"]["frame"] > b["p50_ms"]["frame"]
    bw = o["temporal_bandwidth"]
    assert 0 < bw["stage_fraction_of_ceiling"] < bw["gemv_fraction_of_ceiling"] <= 1.05
