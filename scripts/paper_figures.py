"""Vector figures for the paper (paper/figures/*.pdf), single-column width, from results/ only."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "paper" / "figures"
COL = 3.33  # single column width, inches
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#111111", "#555555", "#dddddd"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


PNG = ROOT / "figures"


def load(n):
    return json.loads((R / f"{n}.json").read_text())


def save(fig, name):
    """Vector PDF for the paper, PNG for the README."""
    fig.savefig(OUT / f"{name}.pdf")
    PNG.mkdir(exist_ok=True)
    fig.savefig(PNG / f"{name}.png", dpi=220, facecolor="white")
    plt.close(fig)


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_budget():
    rows = [("bf16 (stock)", "bf16_b1_r1"), ("W8A16", "w8triton"), ("W8A16 + INT8 KV", "streams_w8kv8_b1"),
            ("Kyutai q8", "q8bnb"), ("INT4 g32", "int4wo_hqq_g32")]
    stages = [("Mimi enc.", ["encode"]), ("temporal", ["temporal"]), ("depth", ["depth"]),
              ("Mimi dec.", ["decode"]), ("other", ["text", "host"])]
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    clean(ax, "x")
    for i, (label, name) in enumerate(rows):
        s, left = load(name)["stages_ms"], 0.0
        for (st, keys), col in zip(stages, C):
            w = sum(s[k]["p50"] for k in keys)
            ax.barh(i, w, left=left, height=0.64, color=col, edgecolor="white", linewidth=0.6,
                    label=st if i == 0 else None)
            if w >= 7:
                ax.text(left + w / 2, i, f"{w:.1f}", ha="center", va="center", fontsize=6.3, color=INK)
            left += w
    ax.axvline(80, color=INK, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_yticks(range(len(rows)), [r for r, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 82)
    ax.set_xlabel("p50 time per 80 ms frame (ms), one stream")
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, handlelength=0.9,
              columnspacing=0.8, handletextpad=0.35)
    save(fig, "budget")


def _sweep(prefix, extra=()):
    pts = {}
    files = [p for p in R.glob(f"{prefix}_b*.json") if re.fullmatch(rf"{prefix}_b(\d+)(_paced)?(_r\d+)?", p.stem)]
    for p in files + [R / f"{n}.json" for n in extra]:
        d = json.loads(p.read_text())
        if d.get("oom"):
            continue
        b = d["config"]["batch"]
        pts[b] = max(pts.get(b, 0.0), d["stages_ms"]["wall"]["p99"])
    return dict(sorted(pts.items()))


def fig_streams():
    bf16_b1 = [p.stem for p in sorted(R.glob("bf16_b1_*.json"))]
    series = [("bf16 (stock)", _sweep("streams_bf16", bf16_b1), "o"),
              ("W8A16", _sweep("streams_w8", ["w8triton"]), "s"),
              ("W8A16 + INT8 KV", _sweep("streams_w8kv8", ["streams_w8kv8_b1"]), "D"),
              ("W8A16 + INT8 KV + Mimi graph", _sweep("gm_w8kv8"), "P"),
              ("INT4 g32 HQQ + INT8 KV", _sweep("streams_i4kv8"), "^")]
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    clean(ax, "both")
    for (label, pts, m), col in zip(series, C):
        ax.plot(list(pts), list(pts.values()), color=col, marker=m, markersize=3.2, linewidth=1.1, label=label)
    ax.axhline(80, color=INK, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(16, 80.8, "80 ms", ha="right", fontsize=6.5, color=INK)
    ax.set_xlabel("concurrent streams (batch size)")
    ax.set_ylabel("worst p99 frame time (ms)")
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.0,
              handlelength=1.6)
    save(fig, "streams")


def fig_tail():
    """Per-run maximum Mimi-encode time and maximum frame time at 6 and 8 streams (W8A16 + INT8 KV),
    with stock Mimi vs. graph-captured Mimi encode."""
    def runs(prefix):
        out = []
        for p in sorted(R.glob(f"{prefix}_b*.json")):
            m = re.fullmatch(rf"{prefix}_b(6|8)(_paced)?(_r\d+)?", p.stem)
            if m:
                d = json.loads(p.read_text())
                out.append((d["stages_ms"]["encode"]["max"], d["stages_ms"]["wall"]["max"], int(m.group(1))))
        return out
    groups = [("stock Mimi", runs("streams_w8kv8"), C[0]), ("graph-captured\nMimi encode", runs("gm_w8kv8"), C[2])]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=False)
    for ax, idx, title in ((axes[0], 0, "max encode per run (ms)"), (axes[1], 1, "max frame per run (ms)")):
        clean(ax, "y")
        for gi, (label, pts, col) in enumerate(groups):
            for k, (enc, frm, b) in enumerate(pts):
                x = gi + (k % 7 - 3) * 0.045
                ax.scatter([x], [enc if idx == 0 else frm], s=10, color=col, marker="o" if b == 8 else "s",
                           edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_xticks([0, 1], [g[0] for g in groups])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title, fontsize=7.2, color=INK)
        if idx == 1:
            ax.axhline(80, color=INK, linestyle=(0, (3, 2)), linewidth=0.8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="s", linestyle="", color=INK2, markersize=3.5, label="6 streams"),
               Line2D([], [], marker="o", linestyle="", color=INK2, markersize=3.5, label="8 streams")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.07),
               fontsize=6.5, handletextpad=0.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "tail")


def fig_quality():
    pts = [("bf16 (stock)", "bf16_b1_r1", None), ("torchao INT8 (eager)", "int8wo_eager", "quality_int8wo"),
           ("Kyutai q8", "q8bnb", "quality_q8bnb"), ("W8A16", "w8triton", "quality_w8triton"),
           ("W8A16 + INT8 KV", "streams_w8kv8_b1", "quality_w8triton_kv-int8"),
           ("INT4 g128", "int4wo_eager", "quality_int4wo"), ("INT4 g32 HQQ", "int4wo_hqq_g32", "quality_int4wo-hqq-g32")]
    floor = load("quality_none_kv-bf16")["audio"]["kl_mean"]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    clean(ax, "both")
    ax.axhspan(1e-5, floor, color="#eeeeee", zorder=0)
    ax.text(83.5, floor * 0.62, "kernel reordering floor", fontsize=6.2, color=INK2, ha="right")
    offs = {"bf16 (stock)": (-4, 5), "torchao INT8 (eager)": (-4, 5), "Kyutai q8": (4, 3), "W8A16": (5, -9),
            "W8A16 + INT8 KV": (-5, 5), "INT4 g128": (4, 2), "INT4 g32 HQQ": (4, -8)}
    for label, lat, q in pts:
        x = load(lat)["stages_ms"]["wall"]["p50"]
        y = load(q)["audio"]["kl_mean"] if q else 1.2e-5
        ours = label.startswith("W8A16")
        ax.scatter([x], [y], s=16 if ours else 11, color=C[2] if ours else C[0], marker="D" if ours else "o",
                   zorder=3, edgecolors="white", linewidths=0.4)
        dx, dy = offs[label]
        text = label + (" (KL = 0)" if q is None else "")
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=6.2, color=INK,
                    ha="right" if dx < 0 else "left")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.1)
    ax.set_xlim(35, 85)
    ax.set_xlabel("p50 frame time, one stream (ms)")
    ax.set_ylabel("audio KL vs bf16 (nats)")
    save(fig, "quality")


def fig_attention():
    rows = load("kernel_kv_attn")["bench"]
    modes = [("sdpa", "SDPA (moshi)"), ("bf16", "fused, bf16 KV"), ("int8", "fused, INT8 KV")]
    bs = [1, 4, 8]
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    clean(ax, "y")
    w = 0.26
    for j, ((m, label), col) in enumerate(zip(modes, [C[0], C[1], C[2]])):
        ys = [next(r["ms_per_frame_32_layers"] for r in rows if r["batch"] == b and r["used_slots"] == 3000
                   and r["mode"] == m) for b in bs]
        xs = [i + (j - 1) * w for i in range(len(bs))]
        ax.bar(xs, ys, width=w, color=col, edgecolor="white", linewidth=0.5, label=label)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.8, f"{y:.0f}", ha="center", fontsize=5.8, color=INK2)
    ax.set_xticks(range(len(bs)), [f"{b} stream{'s' if b > 1 else ''}" for b in bs])
    ax.set_ylabel("attention ms / frame\n(32 layers, full ring)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "attention")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_budget, fig_streams, fig_tail, fig_quality, fig_attention):
        f()
    made = sorted(p.name for p in OUT.glob("*.pdf"))
    print(made)
    assert made == ["attention.pdf", "budget.pdf", "quality.pdf", "streams.pdf", "tail.pdf"]
    assert all((PNG / f"{n[:-4]}.png").exists() for n in made)
