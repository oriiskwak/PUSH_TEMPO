"""
eval_ratio_devset_csv.py — early:late 비율 스윕을 DEV set(120개)에 대해 실행하고
영상별 raw 예측 점수를 CSV로 저장 (히트맵용). eval_ratio_devselect.py의 1단계
(dev 비교)와 동일한 조건/로직이지만, 집계 요약뿐 아니라 영상별 원본 점수까지 남긴다.
eval_ratio_sweep.py / eval_ratio_devselect.py는 수정하지 않고 독립 스크립트로 분리.

Usage:
  TP_SIZE=1 python experiment2/eval_ratio_devset_csv.py \
      --result results/qwen/tempo_cov_total_covfix_result.json
"""
import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CLASSES = ["보이스피싱", "투자사기", "로맨스스캠", "부업사기", "정상"]

CONDITIONS = [
    ("동적(GT)", True, None),
    ("5:5", False, 0.5),
    ("7:3", False, 0.7),
    ("8:2", False, 0.8),
]


def _extract_vid_id(link: str):
    m = re.search(r'(?:v=|/shorts/|youtu\.be/)([a-zA-Z0-9_-]{11})', link)
    if m:
        return m.group(1)
    m = re.search(r'tiktok\.com/.*/video/(\d+)', link)
    if m:
        return m.group(1)
    return None


def _load_type_map(data_dir: Path) -> dict:
    type_map = {}
    with open(data_dir / "LABELS.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row[0] == "라벨":
                continue
            link = row[2] if len(row) > 2 else ""
            vid_id = _extract_vid_id(link)
            if vid_id:
                type_map[vid_id] = row[1]
    return type_map


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="results/qwen/tempo_cov_total_covfix_result.json",
                        help="best_early_prompt/best_late_prompt를 가져올 저장된 결과 JSON")
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
    print(f"Dev={len(devset)}")

    gt_seg_map = dict(tc._SEG_MAP)
    type_map = _load_type_map(root / "data")

    def scam_type_of(vname, label):
        if label == "normal":
            return "정상"
        yt_id = vname.split("_", 1)[1] if "_" in vname else vname
        t = type_map.get(yt_id)
        return t if t in CLASSES else (t or "미상")

    condition_scores = {}
    label_of_global = {}

    for cond_name, is_dynamic, ratio in CONDITIONS:
        if is_dynamic:
            tc._SEG_MAP, tc._FIXED_RATIO = gt_seg_map, None
        else:
            tc._SEG_MAP, tc._FIXED_RATIO = {}, ratio

        print(f"\n▶ [DEV] 조건: {cond_name}")
        res = tc.score_dataset(devset, best_early, best_late, None, None)
        scores, label_of = _video_level_scores(res)
        condition_scores[cond_name] = scores
        label_of_global.update(label_of)

        auc, mAP = _video_auc_map(scores, label_of)
        if auc is not None:
            print(f"  DEV 영상 단위 AUC={auc:.4f}  mAP={mAP:.4f}")

    tc._SEG_MAP, tc._FIXED_RATIO = gt_seg_map, None  # 원복

    ensemble_scores = {
        v: sum(condition_scores[c][v] for c, *_ in CONDITIONS) / len(CONDITIONS)
        for v in label_of_global
    }
    condition_scores["ensemble"] = ensemble_scores
    ens_auc, ens_map = _video_auc_map(ensemble_scores, label_of_global)
    print(f"\n▶ [DEV] 앙상블(4조건 평균)  AUC={ens_auc:.4f}  mAP={ens_map:.4f}")

    all_cond_names = [c for c, *_ in CONDITIONS] + ["ensemble"]

    table_rows = []
    for i, vname in enumerate(sorted(label_of_global.keys()), 1):
        row = {
            "video_id": i, "video_name": vname,
            "label": label_of_global[vname],
            "scam_type": scam_type_of(vname, label_of_global[vname]),
        }
        for c in all_cond_names:
            row[c] = round(condition_scores[c].get(vname, float("nan")), 4)
        table_rows.append(row)

    out_path = Path(args.output) if args.output else result_path.parent / "ratio_sweep_devset_result.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "video_name", "label", "scam_type"] + all_cond_names)
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"\nCSV 저장: {out_path}")
