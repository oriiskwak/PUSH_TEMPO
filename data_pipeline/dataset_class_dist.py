"""
dataset_class_dist.py — CyberScam-Ko 데이터셋 유형별 영상 개수 막대그래프.
표본이 적은 클래스(보이스피싱, 로맨스스캠)를 경고색으로 강조.

Usage:
  python data_pipeline/dataset_class_dist.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent

COUNTS = {
    "Voice Phishing": 12,
    "Investment Fraud": 87,
    "Romance Scam": 33,
    "Job Scam": 94,
    "Normal": 375,
}

SMALL_SAMPLE_THRESHOLD = 50  # 이 미만이면 경고색으로 강조
COLOR_NORMAL_CLASS = "#2a78d6"   # 표본 충분한 클래스 (categorical, 고정)
COLOR_SMALL_SAMPLE = "#c0392b"   # 표본 적은 클래스 (경고색)

if __name__ == "__main__":
    labels = list(COUNTS.keys())
    values = list(COUNTS.values())
    colors = [COLOR_SMALL_SAMPLE if v < SMALL_SAMPLE_THRESHOLD else COLOR_NORMAL_CLASS for v in values]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 6, str(v),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(SMALL_SAMPLE_THRESHOLD, color=COLOR_SMALL_SAMPLE, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(len(labels) - 0.4, SMALL_SAMPLE_THRESHOLD + 6, f"n<{SMALL_SAMPLE_THRESHOLD}",
             ha="right", va="bottom", fontsize=9, color=COLOR_SMALL_SAMPLE, style="italic")

    ax.set_ylabel("Number of videos", fontsize=11)
    ax.set_ylim(0, max(values) * 1.12)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_png = ROOT / "results" / "dataset_class_distribution.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"저장: {out_png}")
