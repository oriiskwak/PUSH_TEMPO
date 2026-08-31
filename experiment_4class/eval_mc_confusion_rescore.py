"""
eval_mc_confusion_rescore.py — TEMPO(early/late)와 UniAPO-MC-v2(단일 프롬프트)의
저장된 best prompt로 5-class test set(124개)을 재채점해서 영상별
(video_name, true_label, pred_label)을 CSV로 저장. 보이스피싱 등 개별 클래스
confusion matrix를 나중에 자유롭게 재구성할 수 있게 하기 위함.
재학습 없음. tempo_qwen.py/uniapo_mc_v2.py는 수정하지 않고 그대로 import.

Usage:
  TP_SIZE=1 python experiment_4class/eval_mc_confusion_rescore.py --method tempo
  TP_SIZE=1 python experiment_4class/eval_mc_confusion_rescore.py --method uniapo
"""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULT_PATH = {
    "tempo":  ROOT / "results_4class" / "tempo_qwen_result.json",
    "uniapo": ROOT / "results_4class" / "uniapo_mc_v2_result.json",
}
OUT_PATH = {
    "tempo":  ROOT / "results_4class" / "tempo_qwen_test_predictions.csv",
    "uniapo": ROOT / "results_4class" / "uniapo_mc_v2_test_predictions.csv",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tempo", "uniapo"], required=True)
    args = parser.parse_args()

    saved = json.load(open(RESULT_PATH[args.method], encoding="utf-8"))
    print(f"=== {RESULT_PATH[args.method]} 로드 (method: {saved.get('method')}) ===")

    from utils_mc import load_gt_mc, split_data_mc, aggregate_video_pred, CLASSES

    if args.method == "tempo":
        import experiment_4class.tempo_qwen as mod
        best_early = saved["best_early_prompt"]
        best_late  = saved["best_late_prompt"]
        score_fn = lambda testset: mod.score_dataset(testset, best_early, best_late)
    else:
        import experiment_4class.uniapo_mc_v2 as mod
        best_prompt = saved["best_prompt"]
        score_fn = lambda testset: mod.score_dataset(testset, best_prompt)

    print("\n=== 데이터셋 로드 (동일 SEED=42 split) ===")
    gt_list = load_gt_mc()
    trainset, devset, testset = split_data_mc(gt_list, mod.TRAIN_RATIO, mod.DEV_RATIO, mod.SEED)
    print(f"Test={len(testset)}")

    print("\n=== Test set 재채점 ===")
    frame_results = score_fn(testset)

    from collections import defaultdict
    by_video = defaultdict(list)
    label_of = {}
    for r in frame_results:
        by_video[r["video"]].append(r["pred_label"])
        label_of[r["video"]] = r["label"]

    rows = []
    for vname, preds in by_video.items():
        pred_label = aggregate_video_pred(preds)
        rows.append({"video_name": vname, "true_label": label_of[vname], "pred_label": pred_label})

    out_path = OUT_PATH[args.method]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_name", "true_label", "pred_label"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n영상 {len(rows)}개 저장: {out_path}")

    # 참고용 5x5 confusion matrix 콘솔 출력
    confusion = {c: {p: 0 for p in CLASSES} for c in CLASSES}
    for r in rows:
        if r["true_label"] in confusion and r["pred_label"] in confusion[r["true_label"]]:
            confusion[r["true_label"]][r["pred_label"]] += 1
    print("\nConfusion (행=true, 열=pred):")
    header = "         " + "  ".join(f"{c[:4]:>4}" for c in CLASSES)
    print(f"  {header}")
    for tc in CLASSES:
        row = "  ".join(f"{confusion[tc][pc]:>4}" for pc in CLASSES)
        print(f"  {tc[:4]:>4}  {row}")
