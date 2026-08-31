"""
eval_channel_disjoint.py — InternVL binary CYREN 체크포인트의 실제 학습된 fb_mem +
best_early_prompt/best_late_prompt로, (1) 전체 test 121개와 (2) channel-clean
서브셋(train/dev와 겹치는 채널의 영상 제외, 91개)을 Tool+Segment 전부 켠 상태로
재채점해서 AUC/mAP/F1/ACC를 비교한다. 재학습 없음.

Usage:
  CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 PP_SIZE=1 python experiment_intern/eval_channel_disjoint.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
ROOT = Path(__file__).parent.parent


def _fmt(v):
    return f"{v:.4f}" if v is not None else "None"


if __name__ == "__main__":
    print("=== channel_disjoint_analysis_intern.json 로드 ===")
    cd = json.load(open(ROOT / "data" / "channel_disjoint_analysis_intern.json", encoding="utf-8"))
    clean_names = set(cd["clean_test_video_names"])
    contaminated_names = set(cd["contaminated_video_names"])
    print(f"channel-clean 대상: {len(clean_names)}개  /  제외 대상: {len(contaminated_names)}개")

    print("\n=== best prompt + fb_mem 로드 ===")
    saved = json.load(open(ROOT / "results_intern" / "tempo_cov_intern_evolve_result.json", encoding="utf-8"))
    best_early = saved["best_early_prompt"]
    best_late  = saved["best_late_prompt"]
    print(f"method: {saved.get('method')}  (Table 1 인터널 numbers와 매칭되는 실행)")

    import experiment_intern.tempo_cov as tc
    from utils_intern import split_data, load_gt
    from metrics_core import compute_metrics_from_frames
    from em_core_tool2 import FeedbackMemory

    print("\n=== 데이터셋 로드 (동일 SEED=42 split) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, tc.TRAIN_RATIO, tc.DEV_RATIO, tc.SEED)
    print(f"Test 전체={len(testset)}")

    clean_testset = [v for v in testset if v["video_name"] in clean_names]
    print(f"Channel-clean 서브셋={len(clean_testset)}  (원본 test 중 {len(testset) - len(clean_testset)}개 제외)")

    ckpt = json.load(open(ROOT / "results_intern" / "tempo_cov_intern_evolve_checkpoint.json", encoding="utf-8"))
    early_fb_mem = FeedbackMemory(top_k=3)
    early_fb_mem.memory = ckpt["early_fb_mem"]
    late_fb_mem = FeedbackMemory(top_k=3)
    late_fb_mem.memory = ckpt["late_fb_mem"]
    print(f"early_fb_mem entries: {len(early_fb_mem.memory)}")

    # score_dataset이 tool을 쓰려면 TOOLS_ENABLED=True 필요 (학습 루프 밖에서 직접 호출 시
    # 모듈 기본값 False라서 명시적으로 켜야 함)
    tc.TOOLS_ENABLED = True

    def _run(label, dataset):
        res = tc.score_dataset(dataset, best_early, best_late, early_fb_mem, late_fb_mem)
        m = compute_metrics_from_frames(res)
        print(f"\n▶ {label}  (n={len(dataset)} videos, {len(res)} frames)")
        print(f"  AUC={_fmt(m['auc'])}  MAP={_fmt(m['map'])}  F1={_fmt(m['f1'])}  ACC={_fmt(m['acc'])}")
        return m

    print("\n" + "=" * 60)
    full_m  = _run("전체 test set (121개, 원본 — channel 오염 포함)", testset)
    clean_m = _run("Channel-clean 서브셋 (91개)", clean_testset)

    print("\n" + "=" * 60)
    print("=== 비교 요약 ===")
    for k in ("auc", "map", "f1", "acc"):
        f, c = full_m.get(k), clean_m.get(k)
        d = (c - f) if (f is not None and c is not None) else None
        print(f"  {k:>4}: 전체={_fmt(f)}  clean={_fmt(c)}  diff(clean-전체)={_fmt(d)}")

    out = {
        "backbone": "InternVL2.5-78B-AWQ",
        "source_result": "results_intern/tempo_cov_intern_evolve_result.json",
        "note": "Tool+Segment 전부 적용된 완전체 CYREN(InternVL)으로, 전체 test와 channel-clean "
                "서브셋을 각각 재채점 (재학습 없음, 동일 best prompt + 동일 학습된 fb_mem 사용).",
        "full_test":       {"n": len(testset), **full_m},
        "channel_clean":    {"n": len(clean_testset), **clean_m},
        "n_contaminated_excluded": len(testset) - len(clean_testset),
        "channel_info_coverage_caveat": (
            "test 121개 중 채널 정보가 있는 영상은 42개(34.7%)뿐이며, 나머지 79개(65.3%)는 "
            "channel-disjoint 여부를 확인할 수 없음 — 이 79개는 clean/contaminated 판정 없이 "
            "channel-clean 서브셋에 그대로 포함되어 있음."
        ),
    }
    out_path = ROOT / "results_intern" / "channel_disjoint_result.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
