"""
eval_paired_bootstrap.py — CYREN vs UniAPO의 test set 예측 점수(eval_cyren_rescore.py /
eval_uniapo_rescore.py로 만든 영상별 raw 점수)로 paired bootstrap 95% CI 계산.

같은 test set(같은 영상, 같은 seed=42 split)에서 나온 점수라 영상 단위로 paired.
매 리샘플에서 같은 인덱스를 CYREN/UniAPO 양쪽에 동일하게 적용(paired resampling).
AUC/mAP 차이(diff = CYREN - UniAPO)의 2.5/97.5 퍼센타일을 95% CI로 보고,
CI가 0을 포함하는지로 유의성 판단.

GPU 불필요. Usage:
  python experiment2/eval_paired_bootstrap.py --backend qwen
  python experiment2/eval_paired_bootstrap.py --backend intern
"""
import argparse, json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).parent.parent

CYREN_PATH = {
    "qwen":   ROOT / "results" / "qwen" / "cyren_test_scores.json",
    "intern": ROOT / "results_intern" / "cyren_test_scores.json",
}
UNIAPO_PATH = {
    "qwen":   ROOT / "results" / "qwen" / "uniapo_test_scores.json",
    "intern": ROOT / "results_intern" / "uniapo_test_scores.json",
}


def _metric(y_true, y_score, metric):
    if len(set(y_true.tolist())) < 2:
        return None
    if metric == "auc":
        return float(roc_auc_score(y_true, y_score))
    return float(average_precision_score(y_true, y_score))


def paired_bootstrap(y_true, score_a, score_b, metric, n_boot=1000, seed=42):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qwen", "intern"], required=True)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cyren = json.load(open(CYREN_PATH[args.backend], encoding="utf-8"))
    uniapo = json.load(open(UNIAPO_PATH[args.backend], encoding="utf-8"))

    vids_c = set(cyren["video_scores"].keys())
    vids_u = set(uniapo["video_scores"].keys())
    common = sorted(vids_c & vids_u)
    if len(common) != len(vids_c) or len(common) != len(vids_u):
        print(f"[Warning] 영상 집합이 다릅니다 (CYREN={len(vids_c)}, UniAPO={len(vids_u)}, "
              f"공통={len(common)}) — 공통 영상만 사용합니다.")

    y_true = np.array([1 if cyren["video_labels"][v] == "abnormal" else 0 for v in common])
    score_c = np.array([cyren["video_scores"][v] for v in common])
    score_u = np.array([uniapo["video_scores"][v] for v in common])

    print(f"=== backend={args.backend}  n_videos={len(common)}  n_boot={args.n_boot} ===\n")

    for metric in ("auc", "map"):
        point_c = _metric(y_true, score_c, metric)
        point_u = _metric(y_true, score_u, metric)
        point_diff = point_c - point_u

        diffs = paired_bootstrap(y_true, score_c, score_u, metric,
                                 n_boot=args.n_boot, seed=args.seed)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        contains_zero = lo <= 0 <= hi

        print(f"[{metric.upper()}]")
        print(f"  CYREN={point_c:.4f}  UniAPO={point_u:.4f}  diff(CYREN-UniAPO)={point_diff:+.4f}")
        print(f"  95% CI (paired bootstrap, n={len(diffs)}): [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {'CI가 0을 포함 → 유의한 차이 아님' if contains_zero else 'CI가 0을 미포함 → 유의한 차이'}")
        print()

    out = {}
    for metric in ("auc", "map"):
        point_c = _metric(y_true, score_c, metric)
        point_u = _metric(y_true, score_u, metric)
        diffs = paired_bootstrap(y_true, score_c, score_u, metric,
                                 n_boot=args.n_boot, seed=args.seed)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[metric] = {
            "cyren": point_c, "uniapo": point_u, "diff": point_c - point_u,
            "ci_95": [float(lo), float(hi)], "contains_zero": bool(lo <= 0 <= hi),
            "n_boot_valid": len(diffs),
        }

    out_path = ROOT / ("results/qwen" if args.backend == "qwen" else "results_intern") / "paired_bootstrap_result.json"
    json.dump({"backend": args.backend, "n_videos": len(common), **out},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")
