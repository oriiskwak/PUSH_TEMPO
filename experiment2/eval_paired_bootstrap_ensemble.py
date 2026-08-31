"""
eval_paired_bootstrap_ensemble.py — CYREN(InternVL, top-3 ensemble) vs UniAPO의
paired bootstrap 95% CI. eval_paired_bootstrap.py의 InternVL 전용 앙상블 버전
(CYREN 쪽 점수만 cyren_test_scores_ensemble.json으로 교체, 나머지 로직은 동일).

GPU 불필요. Usage:
  python experiment2/eval_paired_bootstrap_ensemble.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).parent.parent
CYREN_PATH  = ROOT / "results_intern" / "cyren_test_scores_ensemble.json"
UNIAPO_PATH = ROOT / "results_intern" / "uniapo_test_scores.json"
OUT_PATH    = ROOT / "results_intern" / "paired_bootstrap_ensemble_result.json"


def _metric(y_true, y_score, metric):
    if len(set(y_true.tolist())) < 2:
        return None
    if metric == "auc":
        return float(roc_auc_score(y_true, y_score))
    return float(average_precision_score(y_true, y_score))


def paired_bootstrap(y_true, score_a, score_b, metric, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    tries = 0
    while len(diffs) < n_boot and tries < n_boot * 20:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ma = _metric(yt, score_a[idx], metric)
        mb = _metric(yt, score_b[idx], metric)
        if ma is None or mb is None:
            continue
        diffs.append(ma - mb)
    return np.array(diffs)


if __name__ == "__main__":
    cyren = json.load(open(CYREN_PATH, encoding="utf-8"))
    uniapo = json.load(open(UNIAPO_PATH, encoding="utf-8"))

    vids_c = set(cyren["video_scores"].keys())
    vids_u = set(uniapo["video_scores"].keys())
    common = sorted(vids_c & vids_u)
    if len(common) != len(vids_c) or len(common) != len(vids_u):
        print(f"[Warning] 영상 집합이 다릅니다 (CYREN={len(vids_c)}, UniAPO={len(vids_u)}, "
              f"공통={len(common)}) — 공통 영상만 사용합니다.")

    y_true = np.array([1 if cyren["video_labels"][v] == "abnormal" else 0 for v in common])
    score_c = np.array([cyren["video_scores"][v] for v in common])
    score_u = np.array([uniapo["video_scores"][v] for v in common])

    print(f"=== backend=intern (CYREN ensemble)  n_videos={len(common)}  n_boot=10000  seed=42 ===\n")

    out = {}
    for metric in ("auc", "map"):
        point_c = _metric(y_true, score_c, metric)
        point_u = _metric(y_true, score_u, metric)
        point_diff = point_c - point_u

        diffs = paired_bootstrap(y_true, score_c, score_u, metric)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        contains_zero = lo <= 0 <= hi

        print(f"[{metric.upper()}]")
        print(f"  CYREN(ensemble)={point_c:.4f}  UniAPO={point_u:.4f}  diff={point_diff:+.4f}")
        print(f"  95% CI (paired bootstrap, n={len(diffs)}): [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {'CI가 0을 포함 → 유의한 차이 아님' if contains_zero else 'CI가 0을 미포함 → 유의한 차이'}")
        print()

        out[metric] = {
            "cyren_ensemble": point_c, "uniapo": point_u, "diff": point_diff,
            "ci_95": [float(lo), float(hi)], "contains_zero": bool(contains_zero),
            "n_boot_valid": len(diffs),
        }

    json.dump({"backend": "intern", "cyren_variant": "ensemble_top3",
               "n_videos": len(common), **out},
              open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"결과 저장: {OUT_PATH}")
