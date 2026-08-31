"""
ratio_sweep_group_rank.py — ratio_sweep_result.csv 기준, scam(abnormal) 영상 그룹과
normal 영상 그룹 각각의 조건별 평균 순위(|예측값-정답| 오차 기준, 1=최고)를 그룹화
막대그래프로 그림. GPU 불필요.

Usage:
  python data_pipeline/ratio_sweep_group_rank.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HIGHLIGHT = "--no-highlight" not in sys.argv

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "results" / "ratio_sweep_result.csv"
CONDITIONS = ["동적(GT)", "5:5", "7:3", "8:2", "ensemble"]
COND_EN = {"동적(GT)": "Dynamic(GT)", "5:5": "5:5", "7:3": "7:3", "8:2": "8:2", "ensemble": "Ensemble"}

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH).dropna(subset=CONDITIONS)

    # 정답과의 거리(|예측값-정답|, 낮을수록 좋음) 기준 순위 — ratio_sweep_stats.py와 동일 로직.
    # raw 점수 그대로 랭킹하면 "그냥 전반적으로 점수 높은 조건"이 유리해지는 왜곡이 생겨서,
    # abnormal=1/normal=0과의 오차로 방향을 통일해야 scam/normal 두 그룹이 공정하게 비교됨.
    y_true = (df["label"] == "abnormal").astype(int)
    err_df = df.copy()
    for c in CONDITIONS:
        err_df[c] = (df[c] - y_true).abs()
    ranks = err_df[CONDITIONS].rank(axis=1, ascending=True)
    ranks["label"] = df["label"].values

    scam_mean = ranks[ranks["label"] == "abnormal"][CONDITIONS].mean()
    normal_mean = ranks[ranks["label"] == "normal"][CONDITIONS].mean()

    print("=== 조건별 평균 순위 (그룹별, 낮을수록 좋음) ===")
    print(f"{'조건':<12} {'Scam(n=' + str((ranks['label']=='abnormal').sum()) + ')':<14} "
          f"{'Normal(n=' + str((ranks['label']=='normal').sum()) + ')':<14}")
    for c in CONDITIONS:
        print(f"{c:<12} {scam_mean[c]:<14.3f} {normal_mean[c]:<14.3f}")

    # ── plot ──
    x = np.arange(len(CONDITIONS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))

    COLOR_SCAM, COLOR_NORMAL = "#c0392b", "#2a78d6"  # 범죄=red계열, 정상=blue계열 (categorical, 고정 순서)
    HIGHLIGHT_COND = "5:5"  # 채택된 방법 — scam 막대만 강조
    HIGHLIGHT_EDGE = "#1a7a3c"

    scam_bars = ax.bar(x - width / 2, [scam_mean[c] for c in CONDITIONS], width,
                        label="Scam (abnormal)", color=COLOR_SCAM)
    ax.bar(x + width / 2, [normal_mean[c] for c in CONDITIONS], width,
           label="Normal", color=COLOR_NORMAL)

    if HIGHLIGHT:
        hi_idx = CONDITIONS.index(HIGHLIGHT_COND)
        scam_bars[hi_idx].set_edgecolor(HIGHLIGHT_EDGE)
        scam_bars[hi_idx].set_linewidth(3.5)
        scam_bars[hi_idx].set_zorder(5)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_EN[c] for c in CONDITIONS], fontsize=11)
    ax.set_ylabel("Mean rank (1=best, 5=worst)", fontsize=11)
    title = "Per-group mean rank by segmentation condition\n(rank based on |pred - true label|)"
    if HIGHLIGHT:
        title += "  —  ★ = adopted method"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.7, 5.3)
    ax.invert_yaxis()  # 1이 위로 가게

    if HIGHLIGHT:
        ax.text(hi_idx, -0.45, "★ Adopted", ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold", color=HIGHLIGHT_EDGE)
    ax.axhline(3.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)  # 중간(3)선 참조용
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    for xi, c in zip(x - width / 2, CONDITIONS):
        ax.text(xi, scam_mean[c] - 0.08, f"{scam_mean[c]:.2f}", ha="center", va="bottom", fontsize=8.5)
    for xi, c in zip(x + width / 2, CONDITIONS):
        ax.text(xi, normal_mean[c] - 0.08, f"{normal_mean[c]:.2f}", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out_name = "ratio_sweep_group_rank.png" if HIGHLIGHT else "ratio_sweep_group_rank_plain.png"
    out_png = ROOT / "results" / out_name
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n저장: {out_png}")
