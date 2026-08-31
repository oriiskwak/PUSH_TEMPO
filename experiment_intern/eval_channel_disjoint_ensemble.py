"""
eval_channel_disjoint_ensemble.py — CYREN(InternVL) top-3 앙상블(eval_cyren_rescore_ensemble.py와
동일 방식: early_pm/late_pm 상위 3쌍을 (early_score+late_score)/2 가중 평균)로, 전체 test(121)와
channel-clean 서브셋을 각각 재채점해서 비교. eval_channel_disjoint.py의 앙상블 버전
(기존 스크립트는 단일쌍만 지원 — Table 1이 앙상블 기준(AUC 0.6826/mAP 0.5514)으로
업데이트됐으므로, channel-clean 비교도 동일 앙상블 기준으로 맞춤).
재학습 없음. 반올림 없이 저장(정밀도 손실로 인한 tie 버그 방지).

Usage:
  CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 PP_SIZE=1 python experiment_intern/eval_channel_disjoint_ensemble.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
ROOT = Path(__file__).parent.parent

RESULT_PATH     = ROOT / "results_intern" / "tempo_cov_intern_evolve_result.json"
CHECKPOINT_PATH = ROOT / "results_intern" / "tempo_cov_intern_evolve_checkpoint.json"
OUT_PATH        = ROOT / "results_intern" / "channel_disjoint_ensemble_result.json"
FRAMES_CACHE_DIR = ROOT / "data" / "frames_cache_2s"


def _fmt(v):
    return f"{v:.4f}" if v is not None else "None"


def _make_cached_loader(frame_step: int, max_sec: int):
    """data/frames_cache_2s의 사전 추출된 448x448 JPG를 읽어 원본 mp4 디코딩을 건너뛴다."""
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


if __name__ == "__main__":
    print("=== channel_disjoint_analysis_intern.json 로드 ===")
    cd = json.load(open(ROOT / "data" / "channel_disjoint_analysis_intern.json", encoding="utf-8"))
    clean_names = set(cd["clean_test_video_names"])
    print(f"channel-clean 대상: {len(clean_names)}개")

    saved = json.load(open(RESULT_PATH, encoding="utf-8"))
    print(f"=== {RESULT_PATH} (method: {saved.get('method')}) ===")

    import experiment_intern.tempo_cov as tc
    from utils_intern import split_data, load_gt
    from metrics_core import compute_metrics_from_frames
    from em_core_tool2 import FeedbackMemory, PromptMemory

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
    print(f"early_fb_mem entries: {len(early_fb_mem.memory)}")

    # eval_cyren_rescore_ensemble.py와 동일한 top-3 앙상블 구성 로직
    _top_e = sorted([(p, s) for p, s in early_pm.memory], key=lambda x: x[1], reverse=True)[:3]
    _top_l = sorted([(p, s) for p, s in late_pm.memory],  key=lambda x: x[1], reverse=True)[:3]
    ensemble_pairs = [(ep, lp, max((es + ls) / 2, 1e-6))
                      for (ep, es), (lp, ls) in zip(_top_e, _top_l)]
    if not ensemble_pairs:
        ensemble_pairs = [(saved["best_early_prompt"], saved["best_late_prompt"], 1.0)]
    print(f"앙상블 프롬프트 수: {len(ensemble_pairs)}")

    tc.TOOLS_ENABLED = True

    print("\n=== 데이터셋 로드 (동일 SEED=42 split) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, tc.TRAIN_RATIO, tc.DEV_RATIO, tc.SEED)
    print(f"Test 전체={len(testset)}")

    clean_testset = [v for v in testset if v["video_name"] in clean_names]
    print(f"Channel-clean 서브셋={len(clean_testset)}  (원본 test 중 {len(testset) - len(clean_testset)}개 제외)")

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

    def _run(label, dataset):
        res = _weighted_ensemble_score(dataset)
        m = compute_metrics_from_frames(res)
        print(f"\n▶ {label}  (n={len(dataset)} videos, {len(res)} frames)")
        print(f"  AUC={_fmt(m['auc'])}  MAP={_fmt(m['map'])}  F1={_fmt(m['f1'])}  ACC={_fmt(m['acc'])}")
        return m, res

    print("\n" + "=" * 60)
    full_m,  full_res  = _run("전체 test set (121개, 원본 — channel 오염 포함)", testset)
    clean_m, clean_res = _run("Channel-clean 서브셋", clean_testset)

    print("\n" + "=" * 60)
    print("=== 비교 요약 ===")
    for k in ("auc", "map", "f1", "acc"):
        f, c = full_m.get(k), clean_m.get(k)
        d = (c - f) if (f is not None and c is not None) else None
        print(f"  {k:>4}: 전체={_fmt(f)}  clean={_fmt(c)}  diff(clean-전체)={_fmt(d)}")

    out = {
        "backbone": "InternVL2.5-78B-AWQ",
        "method": "CYREN-ensemble (top-3)",
        "source_result": str(RESULT_PATH),
        "n_ensemble_pairs": len(ensemble_pairs),
        "note": "Table 1 앙상블 기준(재채점 AUC=0.6826/mAP=0.5514)과 동일한 top-3 앙상블 방식으로 "
                "전체 test와 channel-clean 서브셋을 각각 재채점 (재학습 없음, 동일 학습된 fb_mem 사용).",
        "full_test":    {"n": len(testset), **full_m},
        "channel_clean": {"n": len(clean_testset), **clean_m},
        "n_contaminated_excluded": len(testset) - len(clean_testset),
    }
    json.dump(out, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {OUT_PATH}")
