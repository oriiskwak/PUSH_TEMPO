"""
majority_baseline_mc.py — 5-class 다수결(majority-class) baseline.

VLM 호출 전혀 없이, train set에서 가장 흔한 클래스를 test set 전체에 그대로 찍었을 때
나오는 ACC/macro-F1을 계산한다. accuracy가 클래스 불균형 때문에 얼마나 쉽게 부풀려질
수 있는지 보여주는 근거용 — macro-F1을 주 지표로 쓰는 이유를 뒷받침한다.

Usage:
  python experiment_4class/majority_baseline_mc.py
"""
import json, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils_mc import OUT_DIR, CLASSES, load_gt_mc, split_data_mc, compute_metrics_mc

TRAIN_RATIO = 0.6
DEV_RATIO   = 0.2
SEED        = 42
OUTPUT      = "majority_mc_result.json"


def _fmt(v):
    return f"{v:.3f}" if v is not None else "None"


def main():
    gt_list = load_gt_mc()
    trainset, devset, testset = split_data_mc(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    train_counts = Counter(v["label"] for v in trainset)
    majority_class = train_counts.most_common(1)[0][0]
    print(f"\nTrain 클래스 분포: {dict(train_counts)}")
    print(f"Majority class (train 기준): {majority_class}")

    # VLM 없이 모든 test 영상에 majority_class를 그대로 찍은 frame_results 구성
    fake_frame_results = [
        {"video": v["video_name"], "label": v["label"], "pred_label": majority_class}
        for v in testset
    ]

    m = compute_metrics_mc(fake_frame_results)
    print(f"\n=== Majority Baseline (always predict '{majority_class}') ===")
    print(f"  ACC={_fmt(m['acc'])}  macro_F1={_fmt(m['macro_f1'])}")
    for c in CLASSES:
        print(f"    {c}: F1={_fmt(m['per_class_f1'][c])}")

    result = {
        "method":         "Majority-class baseline",
        "majority_class": majority_class,
        "train_class_counts": dict(train_counts),
        "split": {"train": len(trainset), "dev": len(devset), "test": len(testset), "seed": SEED},
        "best_test": m,
    }
    out_path = OUT_DIR / OUTPUT
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
