"""
visualize_segmentation.py
Temporal segmentation analysis for TEMPO paper figures.

Outputs (saved to ./figures/):
  seg_breakpoint_dist.pdf  — ① histogram of breakpoint positions
  seg_early_late_ratio.pdf — ② early/late frame ratio bar
  seg_gt_prob.pdf          — ③ early vs late gt_prob (abnormal vs normal)
  seg_type_breakdown.pdf   — ④ per-scam-type breakpoint + gt_prob
  seg_combined.pdf         — all four panels in one figure (for paper)
"""
import csv, json, re, statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["보이스피싱", "투자사기", "로맨스스캠", "부업사기"]
CLASS_EN = ["Voice\nPhishing", "Investment\nFraud", "Romance\nScam", "Part-time\nScam"]
COLORS   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── helpers ───────────────────────────────────────────────────────────────────
def extract_vid_id(link: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", link)
    return m.group(1) if m else ""

def mean(lst):
    return statistics.mean(lst) if lst else 0.0

# ── load data ─────────────────────────────────────────────────────────────────
seg_data = json.load(open(DATA_DIR / "segmentation.json", encoding="utf-8"))

type_map: dict[str, str] = {}
with open(DATA_DIR / "LABELS.csv", encoding="utf-8") as f:
    for row in csv.reader(f):
        if not row or row[0] == "라벨":
            continue
        link = row[2].strip() if len(row) > 2 else ""
        vid_id = extract_vid_id(link)
        if vid_id and len(row) > 1:
            type_map[vid_id] = row[1].strip()

# ── aggregate ─────────────────────────────────────────────────────────────────
all_bkp_pct   = []
bkp_abn, bkp_nor = [], []
early_gt_abn, late_gt_abn = [], []
early_gt_nor, late_gt_nor = [], []
early_ratios, late_ratios = [], []

cat_bkp:      dict[str, list] = defaultdict(list)
cat_early_gt: dict[str, list] = defaultdict(list)
cat_late_gt:  dict[str, list] = defaultdict(list)

for v in seg_data:
    n   = v["n_frames"]
    bkp = v["breakpoint"]
    pct = bkp / n * 100
    all_bkp_pct.append(pct)

    n_early = sum(1 for f in v["frame_segments"] if f["segment"] == "early")
    early_ratios.append(n_early / n * 100)
    late_ratios.append((n - n_early) / n * 100)

    s  = v.get("segment_summary", {})
    ep = s.get("early", {}).get("mean_gt_prob")
    lp = s.get("late",  {}).get("mean_gt_prob")

    if v["label"] == "abnormal":
        bkp_abn.append(pct)
        if ep is not None: early_gt_abn.append(ep)
        if lp is not None: late_gt_abn.append(lp)
    else:
        bkp_nor.append(pct)
        if ep is not None: early_gt_nor.append(ep)
        if lp is not None: late_gt_nor.append(lp)

    yt_id = v["video_name"].split("_", 1)[1] if "_" in v["video_name"] else v["video_name"]
    cat   = type_map.get(yt_id)
    if cat in CLASSES:
        cat_bkp[cat].append(pct)
        if ep is not None: cat_early_gt[cat].append(ep)
        if lp is not None: cat_late_gt[cat].append(lp)

# ── figure: combined 4-panel ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Temporal Segmentation Analysis", fontsize=14, fontweight="bold", y=1.01)

# ── Panel ①: Breakpoint position histogram ───────────────────────────────────
ax = axes[0, 0]
bins = np.arange(0, 105, 5)
ax.hist(bkp_abn, bins=bins, alpha=0.65, color="#C44E52", label=f"Scam (n={len(bkp_abn)})",    density=True)
ax.hist(bkp_nor, bins=bins, alpha=0.65, color="#4C72B0", label=f"Normal (n={len(bkp_nor)})", density=True)
ax.axvline(mean(bkp_abn), color="#C44E52", linestyle="--", linewidth=1.5, label=f"Scam mean {mean(bkp_abn):.1f}%")
ax.axvline(mean(bkp_nor), color="#4C72B0", linestyle="--", linewidth=1.5, label=f"Normal mean {mean(bkp_nor):.1f}%")
ax.set_xlabel("Change-point Position (% of video)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("① Change-point Position Distribution", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 100)

# ── Panel ②: Early/Late frame ratio ──────────────────────────────────────────
ax = axes[0, 1]
labels_bar = ["Early\nFrames", "Late\nFrames"]
vals_bar   = [mean(early_ratios), mean(late_ratios)]
bars = ax.bar(labels_bar, vals_bar, color=["#4C72B0", "#DD8452"], width=0.4, edgecolor="white")
ax.set_ylim(0, 75)
ax.set_ylabel("Mean Frame Ratio (%)", fontsize=11)
ax.set_title("② Early / Late Frame Ratio", fontsize=11, fontweight="bold")
for bar, val in zip(bars, vals_bar):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

# ── Panel ③: Early vs Late gt_prob ───────────────────────────────────────────
ax = axes[1, 0]
x      = np.array([0, 1])
width  = 0.3
e_vals = [mean(early_gt_abn), mean(early_gt_nor)]
l_vals = [mean(late_gt_abn),  mean(late_gt_nor)]
b1 = ax.bar(x - width/2, e_vals, width, label="Early", color="#4C72B0", edgecolor="white")
b2 = ax.bar(x + width/2, l_vals, width, label="Late",  color="#DD8452", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(["Scam Videos", "Normal Videos"], fontsize=10)
ax.set_ylabel("Mean GT Prob", fontsize=11)
ax.set_title("③ Early vs Late Detection Signal (GT Prob)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 0.7)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
            ha="center", va="bottom", fontsize=9)

# ── Panel ④: Per-type breakpoint + Early-Late diff ───────────────────────────
ax = axes[1, 1]
ax2 = ax.twinx()

bkp_means  = [mean(cat_bkp[c])      for c in CLASSES]
diff_means  = [mean(cat_early_gt[c]) - mean(cat_late_gt[c]) for c in CLASSES]

x = np.arange(len(CLASSES))
bars = ax.bar(x, bkp_means, color=COLORS, alpha=0.75, width=0.5, edgecolor="white",
              label="Change-point (%)")
ax2.plot(x, diff_means, "D--", color="black", markersize=7, linewidth=1.5,
         label="Early−Late GT diff")
ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")

ax.set_xticks(x)
ax.set_xticklabels(CLASS_EN, fontsize=9)
ax.set_ylabel("Change-point Position (%)", fontsize=10)
ax2.set_ylabel("Early − Late GT Prob", fontsize=10)
ax.set_title("④ Per-type Change-point & Signal Diff", fontsize=11, fontweight="bold")
ax.set_ylim(0, 70)
ax2.set_ylim(0, 0.08)

for bar, val in zip(bars, bkp_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

fig.tight_layout()
out_path = OUT_DIR / "seg_combined.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"Saved: {out_path}")

# ── also save PNG for quick preview ──────────────────────────────────────────
out_png = OUT_DIR / "seg_combined.png"
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved: {out_png}")

plt.close(fig)

# ── print summary table ───────────────────────────────────────────────────────
print("\n=== Summary Table ===")
print(f"{'Type':<12} {'N':>4} {'BKP%':>7} {'Early GT':>9} {'Late GT':>9} {'Diff':>7}")
print("-" * 54)
for cat in CLASSES:
    n  = len(cat_bkp[cat])
    bp = mean(cat_bkp[cat])
    eg = mean(cat_early_gt[cat])
    lg = mean(cat_late_gt[cat])
    print(f"{cat:<12} {n:>4} {bp:>7.1f}% {eg:>9.3f} {lg:>9.3f} {eg-lg:>+7.3f}")

print()
print(f"Overall breakpoint: mean={mean(all_bkp_pct):.1f}%  "
      f"median={statistics.median(all_bkp_pct):.1f}%  "
      f"std={statistics.stdev(all_bkp_pct):.1f}%")
print(f"Early frames: {mean(early_ratios):.1f}%  Late frames: {mean(late_ratios):.1f}%")
print(f"Abnormal — Early GT: {mean(early_gt_abn):.3f}  Late GT: {mean(late_gt_abn):.3f}")
print(f"Normal   — Early GT: {mean(early_gt_nor):.3f}  Late GT: {mean(late_gt_nor):.3f}")
