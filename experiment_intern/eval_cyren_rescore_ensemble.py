"""
eval_cyren_rescore_ensemble.py — CYREN(InternVL)의 top-3 prompt ensemble(best_test와 동일
방식: early_pm/late_pm 상위 3쌍을 (early_score+late_score)/2 가중 평균)로 test set을
재채점해서 영상별 raw 예측 점수를 저장 (paired bootstrap 준비용).
eval_cyren_rescore.py(단일쌍, tool OFF)의 앙상블 버전 — tool ON, top-3 앙상블.
experiment_intern/tempo_cov.py는 수정하지 않고 그대로 import.

Usage:
  CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 PP_SIZE=1 python experiment_intern/eval_cyren_rescore_ensemble.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
ROOT = Path(__file__).parent.parent

RESULT_PATH     = ROOT / "results_intern" / "tempo_cov_intern_evolve_result.json"
CHECKPOINT_PATH = ROOT / "results_intern" / "tempo_cov_intern_evolve_checkpoint.json"
OUTPUT_PATH     = ROOT / "results_intern" / "cyren_test_scores_ensemble.json"
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


def _video_level(frame_results):
    acc = defaultdict(list)
    label_of = {}
    for r in frame_results:
        acc[r["video"]].append(r["pred"])
        label_of[r["video"]] = r["label"]
    return {v: sum(ps) / len(ps) for v, ps in acc.items()}, label_of


if __name__ == "__main__":
    saved = json.load(open(RESULT_PATH, encoding="utf-8"))
    print(f"=== {RESULT_PATH} (method: {saved.get('method')}) ===")

    import experiment_intern.tempo_cov as tc
    from utils_intern import split_data, load_gt
    from em_core_tool2 import FeedbackMemory, PromptMemory

    # 원본 mp4 디코딩 대신 사전 추출된 프레임 캐시(data/frames_cache_2s)를 읽도록 교체
    tc._load_video = _make_cached_loader(tc.FRAME_STEP, tc.MAX_SEC)

    ckpt = json.load(open(CHECKPOINT_PATH, encoding="utf-8"))
    early_fb_mem = FeedbackMemory(top_k=3)
    early_fb_mem.memory = ckpt["early_fb_mem"]
    late_fb_mem = FeedbackMemory(top_k=3)
    late_fb_mem.memory = ckpt["late_fb_mem"]
    early_pm = PromptMemory(beam_size=tc.BEAM_SIZE, top_k=2)
    early_pm.memory = ckpt["early_pm"]
    late_pm = PromptMemory(beam_size=tc.BEAM_SIZE, top_k=2)
    late_pm.memory = ckpt["late_pm"]
    print(f"early_fb_mem entries: {len(early_fb_mem.memory)}  "
          f"early_pm entries: {len(early_pm.memory)}  late_pm entries: {len(late_pm.memory)}")

    # tempo_cov.py 메인 블록과 동일한 top-3 앙상블 구성 로직
    _top_e = sorted([(p, s) for p, s in early_pm.memory], key=lambda x: x[1], reverse=True)[:3]
    _top_l = sorted([(p, s) for p, s in late_pm.memory],  key=lambda x: x[1], reverse=True)[:3]
    ensemble_pairs = [(ep, lp, max((es + ls) / 2, 1e-6))
                      for (ep, es), (lp, ls) in zip(_top_e, _top_l)]
    if not ensemble_pairs:
        ensemble_pairs = [(saved["best_early_prompt"], saved["best_late_prompt"], 1.0)]
    print(f"앙상블 프롬프트 수: {len(ensemble_pairs)}  (원본 n_ensemble_pairs={saved.get('n_ensemble_pairs')})")

    # score_dataset이 tool을 쓰려면 TOOLS_ENABLED=True 필요 (학습 루프 밖 직접 호출 시
    # 모듈 기본값 False라서 명시적으로 켜야 함)
    tc.TOOLS_ENABLED = True

    print("\n=== 데이터셋 로드 (동일 SEED=42 split) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, tc.TRAIN_RATIO, tc.DEV_RATIO, tc.SEED)
    print(f"Test={len(testset)}")

    def _weighted_ensemble_score(dataset):
        acc: dict = defaultdict(list)
        lookup: dict = {}
        for ep, lp, w in ensemble_pairs:
            _res = tc.score_dataset(dataset, ep, lp, early_fb_mem, late_fb_mem)
            for r in _res:
                key = (r["video"], r["second"])
                acc[key].append((r["pred"], w))
                if key not in lookup:
                    lookup[key] = r
        out = []
        for key, pw_list in acc.items():
            r = dict(lookup[key])
            wsum = sum(w for _, w in pw_list)
            r["pred"] = sum(p * w for p, w in pw_list) / wsum
            out.append(r)
        return out

    res = _weighted_ensemble_score(testset)
    scores, label_of = _video_level(res)

    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    vids = sorted(scores.keys())
    y_true = np.array([1 if label_of[v] == "abnormal" else 0 for v in vids])
    y_score = np.array([scores[v] for v in vids])
    auc = float(roc_auc_score(y_true, y_score))
    mAP = float(average_precision_score(y_true, y_score))
    print(f"\n재현 검증: AUC={auc:.4f}  mAP={mAP:.4f}  "
          f"(원본 best_test AUC={saved['best_test']['auc']:.4f} MAP={saved['best_test']['map']:.4f}과 "
          f"비슷하면 재채점이 제대로 된 것)")

    out = {
        "method": "CYREN-ensemble", "backend": "intern", "source_result": str(RESULT_PATH),
        "n_ensemble_pairs": len(ensemble_pairs),
        "verify_auc": auc, "verify_map": mAP, "n": len(vids),
        "video_scores": {v: scores[v] for v in vids},
        "video_labels": {v: label_of[v] for v in vids},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTPUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"저장: {OUTPUT_PATH}")
