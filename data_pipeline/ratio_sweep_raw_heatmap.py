"""
ratio_sweep_raw_heatmap.py — ratio_sweep_result.csv의 0-1 raw 예측값을 그대로
히트맵으로 그림 (순위/스케일링 없음). 행 정렬은 ratio_sweep_stats.py와 동일하게
사기유형 block 유지 + block 내 hierarchical clustering(오차 기준)을 재사용해
두 히트맵을 나란히 비교할 수 있게 함.

GPU 불필요. Usage:
  python data_pipeline/ratio_sweep_raw_heatmap.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "results" / "ratio_sweep_result.csv"
CONDITIONS = ["동적(GT)", "5:5", "7:3", "8:2", "ensemble"]
SCAM_ORDER = ["보이스피싱", "투자사기", "로맨스스캠", "부업사기", "정상", "미상"]

COND_EN = {"동적(GT)": "Dynamic(GT)", "5:5": "5:5", "7:3": "7:3", "8:2": "8:2", "ensemble": "Ensemble"}
SCAM_EN = {"보이스피싱": "Voice Phishing", "투자사기": "Investment Fraud", "로맨스스캠": "Romance Scam",
           "부업사기": "Job Scam", "정상": "Normal", "미상": "Unknown"}

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH).dropna(subset=CONDITIONS)
    print(f"영상 수: {len(df)}")

    # 정렬 기준은 rank heatmap과 동일(오차 기준 clustering)해서 두 그림을 나란히
    # 비교했을 때 같은 영상이 같은 행에 오도록 맞춤.
    y_true = (df["label"] == "abnormal").astype(int)
    err_df = df.copy()
    for c in CONDITIONS:
        err_df[c] = (df[c] - y_true).abs()
    ranks = err_df[CONDITIONS].rank(axis=1, ascending=True)

    ordered_idx = []
    block_boundaries = []
    for scam in SCAM_ORDER:
        block_idx = df.index[df["scam_type"] == scam].tolist()
        if not block_idx:
            continue
        if len(block_idx) > 2:
            Z = linkage(ranks.loc[block_idx].values, method="average", metric="euclidean")
            order = leaves_list(Z)
            block_idx = [block_idx[i] for i in order]
        ordered_idx.extend(block_idx)
        block_boundaries.append(len(ordered_idx))

    raw_matrix = df.loc[ordered_idx, CONDITIONS].values.astype(float)

    fig, ax = plt.subplots(figsize=(6, max(8, len(ordered_idx) * 0.09)))
    im = ax.imshow(raw_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_EN[c] for c in CONDITIONS], fontsize=10)
    ax.set_yticks([])
    ax.set_ylabel(f"Videos (n={len(ordered_idx)}, grouped by scam type)", fontsize=10)
    ax.set_title("Per-video raw prediction score (0-1, no scaling)", fontsize=12, fontweight="bold")

    for b in block_boundaries[:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=1.2)
    prev = 0
    for b, scam in zip(block_boundaries, [s for s in SCAM_ORDER if s in df["scam_type"].values]):
        mid = (prev + b) / 2 - 0.5
        ax.text(-0.7, mid, SCAM_EN.get(scam, scam), ha="right", va="center", fontsize=9)
        prev = b

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Raw predicted score (0=normal-like, 1=scam-like)")
    plt.tight_layout()
    out_png = ROOT / "results" / "ratio_sweep_raw_heatmap.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"저장: {out_png}")
