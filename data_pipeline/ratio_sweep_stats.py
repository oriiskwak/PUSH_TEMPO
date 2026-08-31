"""
ratio_sweep_stats.py — ratio_sweep_result.csv에 대해:
  1. Friedman test (5개 조건 간 유의차) + 사후검정(Nemenyi)
  2. 영상별 조건 순위(rank) 히트맵 — 사기유형 block 유지, block 내에서만 hierarchical clustering

GPU 불필요. Usage:
  python data_pipeline/ratio_sweep_stats.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from scipy.cluster.hierarchy import linkage, leaves_list
import scikit_posthocs as sp

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "results" / "ratio_sweep_result.csv"
CONDITIONS = ["동적(GT)", "5:5", "7:3", "8:2", "ensemble"]
SCAM_ORDER = ["보이스피싱", "투자사기", "로맨스스캠", "부업사기", "정상", "미상"]

# 시스템에 한글 폰트가 없어서 플롯 라벨만 영문 매핑 (데이터/CSV는 한글 그대로 유지)
COND_EN = {"동적(GT)": "Dynamic(GT)", "5:5": "5:5", "7:3": "7:3", "8:2": "8:2", "ensemble": "Ensemble"}
SCAM_EN = {"보이스피싱": "Voice Phishing", "투자사기": "Investment Fraud", "로맨스스캠": "Romance Scam",
           "부업사기": "Job Scam", "정상": "Normal", "미상": "Unknown"}

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=CONDITIONS)
    print(f"영상 수: {len(df)}")

    # 정답과의 거리(|예측값-정답|, 낮을수록 좋음)로 변환 — raw 점수를 그대로 쓰면 abnormal은
    # 높을수록/normal은 낮을수록 좋다는 방향성이 무시되어, "그냥 전반적으로 점수를 높게 주는
    # 조건"이 판별력과 무관하게 유리해지는 왜곡이 생김(실제로 이 왜곡 때문에 최초 버전에서
    # Dynamic(GT)가 AUC 최하위인데도 평균 순위 2위로 나오는 모순이 있었음).
    y_true = (df["label"] == "abnormal").astype(int)
    err_df = df.copy()
    for c in CONDITIONS:
        err_df[c] = (df[c] - y_true).abs()

    # ── 1. Friedman test (오차 기준 — 낮을수록 좋음) ──
    stat, p = friedmanchisquare(*[err_df[c].values for c in CONDITIONS])
    print(f"\n=== Friedman test (|예측값-정답| 오차 기준) ===")
    print(f"chi2={stat:.4f}  p={p:.6f}  {'(유의함, p<0.05)' if p < 0.05 else '(유의하지 않음)'}")

    # 사후검정: Nemenyi (5x5 pairwise p-value matrix)
    posthoc = sp.posthoc_nemenyi_friedman(err_df[CONDITIONS].values)
    posthoc.columns = CONDITIONS
    posthoc.index = CONDITIONS
    print(f"\n=== 사후검정 (Nemenyi, pairwise p-value) ===")
    print(posthoc.round(4).to_string())

    # ── 2. 영상별 조건 순위 (오차 기준, 1=오차 최소=최고) ──
    ranks = err_df[CONDITIONS].rank(axis=1, ascending=True)

    # scam_type block 유지, block 내에서만 hierarchical clustering으로 행 재정렬
    df["_scam_order"] = df["scam_type"].apply(
        lambda s: SCAM_ORDER.index(s) if s in SCAM_ORDER else len(SCAM_ORDER)
    )
    ordered_idx = []
    block_boundaries = []
    for scam in SCAM_ORDER:
        block_mask = df["scam_type"] == scam
        block_idx = df.index[block_mask].tolist()
        if not block_idx:
            continue
        if len(block_idx) > 2:
            sub_ranks = ranks.loc[block_idx].values
            Z = linkage(sub_ranks, method="average", metric="euclidean")
            order = leaves_list(Z)
            block_idx = [block_idx[i] for i in order]
        ordered_idx.extend(block_idx)
        block_boundaries.append(len(ordered_idx))

    rank_matrix = ranks.loc[ordered_idx].values
    scam_labels = df.loc[ordered_idx, "scam_type"].values

    # ── plot ──
    fig, ax = plt.subplots(figsize=(6, max(8, len(ordered_idx) * 0.09)))
    im = ax.imshow(rank_matrix, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=5)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_EN[c] for c in CONDITIONS], fontsize=10)
    ax.set_yticks([])
    ax.set_ylabel(f"Videos (n={len(ordered_idx)}, grouped by scam type)", fontsize=10)
    ax.set_title(
        f"Per-video condition rank heatmap (1=best, 5=worst)\nFriedman chi2={stat:.2f}, p={p:.4f}",
        fontsize=12, fontweight="bold",
    )

    # block 구분선 + 라벨
    prev = 0
    for b in block_boundaries[:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=1.2)
    prev = 0
    for b, scam in zip(block_boundaries, [s for s in SCAM_ORDER if s in df["scam_type"].values]):
        mid = (prev + b) / 2 - 0.5
        ax.text(-0.7, mid, SCAM_EN.get(scam, scam), ha="right", va="center", fontsize=9)
        prev = b

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Rank (1=best)")
    plt.tight_layout()
    out_png = ROOT / "results" / "ratio_sweep_rank_heatmap.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n히트맵 저장: {out_png}")

    # ── 결과 저장 ──
    out = {
        "n_videos": len(df),
        "friedman": {"chi2": float(stat), "p_value": float(p)},
        "posthoc_nemenyi_pvalues": posthoc.round(6).to_dict(),
        "mean_rank_per_condition": {c: float(ranks[c].mean()) for c in CONDITIONS},
    }
    out_path = ROOT / "results" / "ratio_sweep_stats_result.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"통계 결과 저장: {out_path}")

    print(f"\n=== 조건별 평균 순위 (낮을수록 좋음) ===")
    for c in CONDITIONS:
        print(f"  {c}: {ranks[c].mean():.3f}")
