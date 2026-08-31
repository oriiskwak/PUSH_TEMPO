"""
eval_ratio_devselect.py — leakage 없는 분할 비율 선택.

기존 eval_ratio_sweep.py는 test set(121개)에 직접 비교해서 승자를 골랐는데,
그러면 "test 성능 보고 하이퍼파라미터(분할 비율) 골랐다"는 leakage가 된다.
이 스크립트는 그 문제를 고친 버전:

  1. devset(120개, train/dev/test split에서 원래 있던 validation)에서
     동적(GT)/5:5/7:3/8:2/ensemble 비교 → 승자 선택 (test 안 봄)
  2. 승자 조건 "하나만" testset에 적용해서 최종 AUC/mAP 보고 (다른 조건과
     test에서 비교하지 않음 — 이래야 진짜 leakage-free)

재학습 없음, 저장된 best prompt로 재채점만. tool은 끔(fb_mem=None). Qwen 백본.

Usage:
  TP_SIZE=1 python experiment2/eval_ratio_devselect.py --result results/qwen/tempo_cov_total_covfix_result.json
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CONDITIONS = [
    ("동적(GT)", True, None),
    ("5:5", False, 0.5),
    ("7:3", False, 0.7),
    ("8:2", False, 0.8),
]


def _video_level_scores(frame_results):
    acc = defaultdict(list)
    label_of = {}
    for r in frame_results:
        acc[r["video"]].append(r["pred"])
        label_of[r["video"]] = r["label"]
    return {v: sum(ps) / len(ps) for v, ps in acc.items()}, label_of


def _video_auc_map(scores: dict, label_of: dict):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    vids = list(scores.keys())
    y_true = np.array([1 if label_of[v] == "abnormal" else 0 for v in vids])
    y_score = np.array([scores[v] for v in vids])
    if len(set(y_true.tolist())) < 2:
        return None, None
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def _fmt(v):
    return f"{v:.4f}" if v is not None else "None"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="results/qwen/tempo_cov_total_covfix_result.json",
                        help="best_early_prompt/best_late_prompt를 가져올 저장된 결과 JSON")
    parser.add_argument("--select_metric", default="map", choices=["map", "auc"],
                        help="dev set에서 승자를 고를 기준 지표")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    result_path = root / args.result if not Path(args.result).is_absolute() else Path(args.result)
    saved = json.load(open(result_path, encoding="utf-8"))
    best_early = saved["best_early_prompt"]
    best_late = saved["best_late_prompt"]
    print(f"=== {args.result} 에서 prompt 로드 (method: {saved.get('method')}) ===")

    import experiment2.tempo_cov as tc
    from utils import split_data, load_gt

    print("\n=== 데이터셋 로드 (동일 SEED=42 split) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, tc.TRAIN_RATIO, tc.DEV_RATIO, tc.SEED)
    print(f"Dev={len(devset)}  Test={len(testset)}")

    gt_seg_map = dict(tc._SEG_MAP)

    def _run_condition(dataset, is_dynamic, ratio):
        if is_dynamic:
            tc._SEG_MAP, tc._FIXED_RATIO = gt_seg_map, None
        else:
            tc._SEG_MAP, tc._FIXED_RATIO = {}, ratio
        return tc.score_dataset(dataset, best_early, best_late, None, None)

    # ── 1단계: dev set에서 승자 선택 (test 안 봄) ──
    print("\n" + "=" * 60)
    print("[1단계] Dev set(validation)에서 조건 비교 — test는 아직 안 봄")
    print("=" * 60)

    dev_condition_scores = {}
    dev_label_of = {}
    dev_summary = {}
    for cond_name, is_dynamic, ratio in CONDITIONS:
        print(f"\n▶ [DEV] 조건: {cond_name}")
        res = _run_condition(devset, is_dynamic, ratio)
        scores, label_of = _video_level_scores(res)
        dev_condition_scores[cond_name] = scores
        dev_label_of.update(label_of)
        auc, mAP = _video_auc_map(scores, label_of)
        dev_summary[cond_name] = {"auc": auc, "map": mAP}
        print(f"  DEV 영상 단위 AUC={_fmt(auc)}  mAP={_fmt(mAP)}")

    # ensemble = 4개 조건 dev 예측 평균
    ensemble_dev = {
        v: sum(dev_condition_scores[c][v] for c, *_ in CONDITIONS) / len(CONDITIONS)
        for v in dev_label_of
    }
    dev_condition_scores["ensemble"] = ensemble_dev
    ens_auc, ens_map = _video_auc_map(ensemble_dev, dev_label_of)
    dev_summary["ensemble"] = {"auc": ens_auc, "map": ens_map}
    print(f"\n▶ [DEV] 조건: ensemble")
    print(f"  DEV 영상 단위 AUC={_fmt(ens_auc)}  mAP={_fmt(ens_map)}")

    winner = max(dev_summary, key=lambda c: dev_summary[c][args.select_metric] or -1)
    print(f"\n{'=' * 60}")
    print(f"[승자] Dev set 기준({args.select_metric}) 최고 조건: {winner}  "
          f"(dev {args.select_metric}={dev_summary[winner][args.select_metric]:.4f})")
    print(f"{'=' * 60}")

    # ── 2단계: 승자 조건 "하나만" test set에 적용 (leakage-free 최종 성능) ──
    print(f"\n[2단계] 승자 조건({winner})만 test set에 적용 — 최종 보고용 성능")
    winner_cfg = next(((n, d, r) for n, d, r in CONDITIONS if n == winner), None)

    tc._SEG_MAP, tc._FIXED_RATIO = gt_seg_map, None  # 원복
    if winner == "ensemble":
        test_scores = {}
        test_label_of = {}
        for cond_name, is_dynamic, ratio in CONDITIONS:
            res = _run_condition(testset, is_dynamic, ratio)
            scores, label_of = _video_level_scores(res)
            test_label_of.update(label_of)
            for v, s in scores.items():
                test_scores.setdefault(v, []).append(s)
        test_scores = {v: sum(ps) / len(ps) for v, ps in test_scores.items()}
    else:
        _, is_dynamic, ratio = winner_cfg
        res = _run_condition(testset, is_dynamic, ratio)
        test_scores, test_label_of = _video_level_scores(res)

    tc._SEG_MAP, tc._FIXED_RATIO = gt_seg_map, None  # 원복

    test_auc, test_map = _video_auc_map(test_scores, test_label_of)
    print(f"\n▶ [TEST, leakage-free 최종] 조건={winner}")
    print(f"  AUC={_fmt(test_auc)}  mAP={_fmt(test_map)}")

    out = {
        "source_result": args.result,
        "select_metric": args.select_metric,
        "note": "1단계: dev set에서만 조건 비교해 승자 선택(test 미사용). "
                "2단계: 승자 조건 하나만 test set에 적용해 최종 성능 산출 — leakage-free.",
        "dev_summary": dev_summary,
        "winner": winner,
        "test_final": {"auc": test_auc, "map": test_map, "n": len(test_scores)},
    }
    out_path = Path(args.output) if args.output else result_path.parent / "ratio_devselect_result.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
