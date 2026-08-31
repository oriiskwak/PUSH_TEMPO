"""
eval_segmentation_ablation.py — GT(GPT-5.4) 기반 change-point 세그멘테이션 vs
고정 50:50 중간점 분할 비교 (재학습 없음, 저장된 best prompt로 test만 재채점).

논문 리뷰에서 나온 "test 세그멘테이션이 GT(오라클) 정보를 쓰고 있다"는 지적에 대한
정량적 근거 확보용. tempo_cov.py를 모듈로 import해서 score_dataset을 그대로 재사용하고,
tempo_cov._SEG_MAP만 비워서(={}) 모든 영상이 고정 중간점 폴백(`"early" if i < mid else
"late"`)을 타도록 만든 뒤 GT 버전과 동일 test set에서 비교한다.

Usage:
  TP_SIZE=1 python experiment2/eval_segmentation_ablation.py \
      --result results/tempo_cov_total_result.json
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="results/tempo_cov_total_result.json",
                        help="best_early_prompt/best_late_prompt를 가져올 저장된 결과 JSON")
    parser.add_argument("--output", default=None,
                        help="비교 결과 저장 경로 (기본: <result>_segablation.json)")
    args = parser.parse_args()

    root       = Path(__file__).parent.parent
    result_path = root / args.result if not Path(args.result).is_absolute() else Path(args.result)
    saved = json.load(open(result_path, encoding="utf-8"))
    best_early = saved["best_early_prompt"]
    best_late  = saved["best_late_prompt"]
    print(f"=== {args.result} 에서 prompt 로드 ===")
    print(f"method: {saved.get('method')}")
    print(f"best_early_prompt: {best_early[:80]}...")
    print(f"best_late_prompt : {best_late[:80]}...")

    # tempo_cov.py를 모듈로 import — 이 시점에 utils.py의 vLLM 모델이 로드됨
    import experiment2.tempo_cov as tc
    from utils import split_data, load_gt
    from metrics_core import compute_metrics_from_frames

    print("\n=== 데이터셋 로드 (동일 SEED=42 split) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, tc.TRAIN_RATIO, tc.DEV_RATIO, tc.SEED)
    print(f"Test={len(testset)}")

    gt_seg_map = dict(tc._SEG_MAP)  # 원본 GT 기반 세그멘테이션 백업

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "None"

    def _run(label: str, seg_map: dict) -> dict:
        tc._SEG_MAP = seg_map
        # fb_mem=None → 도구 주입 없이 순수 세그멘테이션 효과만 격리해서 비교
        res = tc.score_dataset(testset, best_early, best_late, None, None)
        m   = compute_metrics_from_frames(res)
        print(f"\n▶ {label}")
        print(f"  AUC={_fmt(m['auc'])}  MAP={_fmt(m['map'])}  "
              f"F1={_fmt(m['f1'])}  ACC={_fmt(m['acc'])}")
        return m

    gt_based   = _run("GT(GPT-5.4) 기반 change-point 세그멘테이션 (기존 방식)", gt_seg_map)
    fixed_5050 = _run("고정 50:50 중간점 분할 (세그멘테이션 없이)", {})

    tc._SEG_MAP = gt_seg_map  # 원복

    print("\n=== 비교 요약 ===")
    for k in ("auc", "map", "f1", "acc"):
        g, f = gt_based.get(k), fixed_5050.get(k)
        diff = (g - f) if (g is not None and f is not None) else None
        print(f"  {k:>4}: GT={_fmt(g)}  fixed50:50={_fmt(f)}  diff={_fmt(diff)}")

    out = {
        "source_result":       args.result,
        "method":               saved.get("method"),
        "best_early_prompt":    best_early,
        "best_late_prompt":     best_late,
        "gt_segmentation":      gt_based,
        "fixed_5050_segmentation": fixed_5050,
    }
    out_path = Path(args.output) if args.output else result_path.parent / (result_path.stem + "_segablation.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
