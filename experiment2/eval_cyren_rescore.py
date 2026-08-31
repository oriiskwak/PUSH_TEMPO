"""
eval_cyren_rescore.py — CYREN(TEMPO-COV-Total)의 저장된 best_early/late 프롬프트로
test set을 재채점해서 영상별 raw 예측 점수를 저장 (CYREN vs UniAPO paired bootstrap 준비용).
tempo_cov.py(qwen)/experiment_intern.tempo_cov.py(intern)는 수정하지 않고 그대로 import.
Segmentation은 원래 학습·평가 때와 동일하게 동적(GT) 그대로 둔다 (ratio override 없음).

Usage:
  TP_SIZE=1                python experiment2/eval_cyren_rescore.py --backend qwen
  TP_SIZE=1 PP_SIZE=1      python experiment2/eval_cyren_rescore.py --backend intern
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
ROOT = Path(__file__).parent.parent
FRAMES_CACHE_DIR = ROOT / "data" / "frames_cache_2s"


def _make_cached_loader(frame_step: int, max_sec: int):
    """data/frames_cache_2s의 사전 추출된 448x448 JPG를 읽어 원본 mp4 디코딩을
    건너뛴다 (tempo_cov.py의 _load_video 대체용, 원본 파일은 안 건드림)."""
    from PIL import Image

    def _cached_load_video(vid):
        vname = vid["video_name"]
        label = vid["label"]
        gt_map = {fp["second"]: fp["probability"] for fp in vid["frame_probabilities"]}
        frames = []
        for sec in range(0, max_sec, frame_step):
            p = FRAMES_CACHE_DIR / f"{vname}_s{sec:04d}.jpg"
            if not p.exists():
                continue
            img = Image.open(p).convert("RGB")
            gt = gt_map.get(sec, float(label == "abnormal"))
            frames.append((sec, gt, img))
        return vname, label, frames

    return _cached_load_video


DEFAULT_RESULT = {
    "qwen":   "results/qwen/tempo_cov_total_covfix_result.json",
    "intern": "results_intern/tempo_cov_intern_evolve_result.json",
}
DEFAULT_OUTPUT = {
    "qwen":   "results/qwen/cyren_test_scores.json",
    "intern": "results_intern/cyren_test_scores.json",
}


def _video_level(frame_results):
    acc = defaultdict(list)
    label_of = {}
    for r in frame_results:
        acc[r["video"]].append(r["pred"])
        label_of[r["video"]] = r["label"]
    return {v: sum(ps) / len(ps) for v, ps in acc.items()}, label_of


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qwen", "intern"], required=True)
    parser.add_argument("--result", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    result_path = root / (args.result or DEFAULT_RESULT[args.backend])
    saved = json.load(open(result_path, encoding="utf-8"))
    best_early = saved["best_early_prompt"]
    best_late = saved["best_late_prompt"]
    print(f"=== {result_path} 에서 프롬프트 로드 (method: {saved.get('method')}) ===")

    if args.backend == "qwen":
        import experiment2.tempo_cov as tc
        from utils import split_data, load_gt
    else:
        import experiment_intern.tempo_cov as tc
        from utils_intern import split_data, load_gt

    # 원본 mp4 디코딩 대신 사전 추출된 프레임 캐시(data/frames_cache_2s)를 읽도록 교체
    tc._load_video = _make_cached_loader(tc.FRAME_STEP, tc.MAX_SEC)

    print("\n=== 데이터셋 로드 (동일 SEED=42 split) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, tc.TRAIN_RATIO, tc.DEV_RATIO, tc.SEED)
    print(f"Test={len(testset)}")

    res = tc.score_dataset(testset, best_early, best_late, None, None)
    scores, label_of = _video_level(res)

    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    vids = sorted(scores.keys())
    y_true = np.array([1 if label_of[v] == "abnormal" else 0 for v in vids])
    y_score = np.array([scores[v] for v in vids])
    auc = float(roc_auc_score(y_true, y_score))
    mAP = float(average_precision_score(y_true, y_score))
    print(f"\n재현 검증: AUC={auc:.4f}  mAP={mAP:.4f}  "
          f"(원본 best_test 값과 비슷하면 재채점이 제대로 된 것)")

    out = {
        "method": "CYREN", "backend": args.backend, "source_result": str(result_path),
        "verify_auc": auc, "verify_map": mAP, "n": len(vids),
        "video_scores": {v: scores[v] for v in vids},
        "video_labels": {v: label_of[v] for v in vids},
    }
    out_path = root / (args.output or DEFAULT_OUTPUT[args.backend])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")
