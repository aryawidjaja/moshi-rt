"""Concurrent-stream capacity from all repeated runs -> results/capacity.json.

Runs are grouped by their recorded config (quantization, KV mode, Mimi graph, batch), not by file
name. For each group: number of runs, late frames, worst p99 / max, and mean +- sd of p50 and p99.
Capacity (largest batch that passes) is reported under several criteria, from strict (no late
frame in any run) to loose (worst p99 < 80 ms only). Colab and memory-capped runs are excluded."""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
BUDGET = 80.0
FAMILIES = {("none", "sdpa", False): "bf16 (stock)", ("w8triton", "sdpa", False): "W8A16",
            ("w8triton", "int8", False): "W8A16 + INT8 KV", ("int4wo-hqq-g32", "int8", False): "INT4 g32 + INT8 KV",
            ("none", "sdpa", True): "bf16 + Mimi graph", ("w8triton", "sdpa", True): "W8A16 + Mimi graph",
            ("w8triton", "int8", True): "W8A16 + INT8 KV + Mimi graph"}
SKIP = ("baseline_l4_bf16", "ablation_", "colab_", "quality", "kernel_", "bw_probe", "quant_table", "capacity",
        "mimi_graph_check")
CRITERIA = {"strict (no late frame)": lambda g: g["late"] == 0,
            "late <= 0.01%": lambda g: g["late_pct"] <= 0.01 and g["worst_p99"] < BUDGET,
            "late <= 0.1%": lambda g: g["late_pct"] <= 0.1 and g["worst_p99"] < BUDGET,
            "late <= 1%": lambda g: g["late_pct"] <= 1.0 and g["worst_p99"] < BUDGET,
            "worst p99 < 80 ms": lambda g: g["worst_p99"] < BUDGET}


def main():
    groups = defaultdict(list)
    for p in sorted(R.glob("*.json")):
        if p.stem.startswith(SKIP):
            continue
        d = json.loads(p.read_text())
        c = d.get("config", {})
        if "stages_ms" not in d or c.get("mem_cap_gb") or d.get("oom"):
            continue
        key = (c.get("quant", "none"), c.get("kv", "sdpa"), bool(c.get("graph_mimi", False)))
        if key in FAMILIES and not c.get("compile_temporal"):
            groups[(FAMILIES[key], c.get("batch", 1))].append((p.stem, d))
    rows = []
    for (fam, b), runs in sorted(groups.items(), key=lambda kv: (list(FAMILIES.values()).index(kv[0][0]), kv[0][1])):
        w = [d["stages_ms"]["wall"] for _, d in runs]
        late = sum(d["frames_over_80ms"] for _, d in runs)
        frames = sum(d["frames_measured"] for _, d in runs)
        sd = lambda xs: round(st.stdev(xs), 2) if len(xs) > 1 else None
        rows.append({
            "config": fam, "batch": b, "runs": len(runs),
            "paced_runs": sum(bool(d["config"].get("paced")) for _, d in runs),
            "files": [n for n, _ in runs],
            "late": late, "frames": frames, "late_pct": round(100 * late / frames, 4),
            "worst_p99": round(max(x["p99"] for x in w), 2), "worst_max": round(max(x["max"] for x in w), 2),
            "p50_mean": round(st.mean(x["p50"] for x in w), 2), "p50_sd": sd([x["p50"] for x in w]),
            "p99_mean": round(st.mean(x["p99"] for x in w), 2), "p99_sd": sd([x["p99"] for x in w]),
            "late_frame_encode_ms": sorted(lf["encode"] for _, d in runs for lf in d.get("late_frames", [])),
        })
    capacity = {}
    for fam in dict.fromkeys(r["config"] for r in rows):
        capacity[fam] = {}
        for crit, ok in CRITERIA.items():
            passing = [r["batch"] for r in rows if r["config"] == fam and ok(r)]
            capacity[fam][crit] = max(passing) if passing else 0
    out = {"what": "concurrent-stream capacity on one L4 from repeated runs (Modal; Colab and capped runs excluded)",
           "criteria": list(CRITERIA), "capacity": capacity, "groups": rows}
    (R / "capacity.json").write_text(json.dumps(out, indent=2))
    for fam, caps in capacity.items():
        print(f"{fam:30s}", "  ".join(f"{k.split(' (')[0]}: {v}" for k, v in caps.items()))
    for r in rows:
        print(f"{r['config']:30s} B={r['batch']:2d} runs={r['runs']} paced={r['paced_runs']} late={r['late']}/{r['frames']}"
              f" worst_p99={r['worst_p99']} max={r['worst_max']} p50={r['p50_mean']}+-{r['p50_sd']}")
    return out


if __name__ == "__main__":
    o = main()
    assert all(r["frames"] >= 1821 * r["runs"] - 10 for r in o["groups"])
    assert o["capacity"]["bf16 (stock)"]["worst p99 < 80 ms"] <= 1
