"""
eval_prompt_swap_random.py — Early/Late 이중 프롬프트의 "역할 특화"와 "시간 구조"가
진짜 의미 있는지 검증. 재학습 없음, 저장된 best prompt로 test set만 재채점. tool은 끔.
tempo_cov.py는 수정하지 않고, score_dataset을 이 파일 안에 독립적으로 복제해서 사용
(다른 실험들이 의존하는 공용 파일을 건드리지 않기 위함).

3개 조건 (전부 5:5 고정 비율 기준):
  1. normal            — best_early_prompt를 early 프레임에, best_late_prompt를 late 프레임에
                          (정상 배치, ratio_sweep의 "5:5"와 동일 조건)
  2. swapped            — 프롬프트 위치를 바꿔서: best_late_prompt를 early 프레임에,
                          best_early_prompt를 late 프레임에. 크게 나빠지면 "역할 특화"가
                          진짜라는 근거, 별 차이 없으면 "그냥 프롬프트 2개 쓰는 앙상블 효과"
  3. random_partition  — 시간순 대신 영상별로 프레임을 무작위로 섞어서 절반씩 early/late로
                          배정(영상마다 재현 가능한 시드). normal과 비슷하면 "시간 구조"
                          자체는 안 중요하다는 근거.

Usage:
  TP_SIZE=1 python experiment2/eval_prompt_swap_random.py --result results/tempo_cov_total_result.json
"""
import argparse, json, random, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


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


def score_dataset_local(tc, dataset, early_prompt, late_prompt, random_seed=None):
    """tempo_cov.score_dataset의 독립 복제본. tool은 항상 끔(fb_mem 없음).
    random_seed가 주어지면 시간순 대신 영상별로 프레임을 무작위로 섞어 5:5 분할."""
    loaded = []
    with ThreadPoolExecutor(max_workers=tc.LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(tc._load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    all_items, all_meta = [], []
    for vname, label, frames in loaded:
        mid = max(1, min(len(frames) - 1, len(frames) // 2)) if len(frames) > 1 else 1

        if random_seed is not None:
            order = list(range(len(frames)))
            random.Random(f"{random_seed}_{vname}").shuffle(order)
            seg_of_idx = {idx: ("early" if pos < mid else "late") for pos, idx in enumerate(order)}
        else:
            seg_of_idx = None

        for i, (sec, gt, img) in enumerate(frames):
            seg = seg_of_idx[i] if seg_of_idx is not None else ("early" if i < mid else "late")
            prompt = early_prompt if seg == "early" else late_prompt
            all_items.append((img, prompt))
            all_meta.append((vname, label, gt, sec, seg))

    all_preds = tc._predict_batch(all_items)

    frame_results = []
    for (vname, label, gt, sec, seg), pred in zip(all_meta, all_preds):
        if pred is None:
            continue
        frame_results.append({
            "video": vname, "label": label, "pred": pred, "gt": gt,
            "segment": seg, "second": sec,
        })
    return frame_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="results/tempo_cov_total_result.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--random_seed", type=int, default=42)
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
    print(f"Test={len(testset)}")

    CONDITIONS = [
        ("normal",           best_early, best_late, None),
        ("swapped",          best_late,  best_early, None),
        ("random_partition", best_early, best_late,  args.random_seed),
    ]

    results = {}
    for cond_name, e_prompt, l_prompt, rseed in CONDITIONS:
        print(f"\n▶ 조건: {cond_name}")
        res = score_dataset_local(tc, testset, e_prompt, l_prompt, random_seed=rseed)
        scores, label_of = _video_level_scores(res)
        auc, mAP = _video_auc_map(scores, label_of)
        results[cond_name] = {"auc": auc, "map": mAP, "n": len(scores)}
        if auc is not None:
            print(f"  영상 단위 AUC={auc:.4f}  mAP={mAP:.4f}")

    print("\n=== 비교 요약 ===")
    for cond_name, *_ in CONDITIONS:
        r = results[cond_name]
        print(f"  {cond_name:<18} AUC={_fmt(r['auc'])}  mAP={_fmt(r['map'])}")

    normal_auc = results["normal"]["auc"]
    swap_auc = results["swapped"]["auc"]
    rand_auc = results["random_partition"]["auc"]
    if normal_auc and swap_auc:
        print(f"\n  swapped - normal AUC diff: {swap_auc - normal_auc:+.4f}  "
              f"({'역할 특화 근거 (많이 나빠짐)' if swap_auc < normal_auc - 0.02 else '역할 특화 근거 약함 (차이 작음)'})")
    if normal_auc and rand_auc:
        print(f"  random_partition - normal AUC diff: {rand_auc - normal_auc:+.4f}  "
              f"({'시간 구조 근거 (많이 나빠짐)' if rand_auc < normal_auc - 0.02 else '시간 구조 근거 약함 (차이 작음)'})")

    out = {
        "source_result": args.result,
        "note": "재학습 없음, 저장된 best prompt로 재채점만. 전 조건 5:5 고정 분할, tool 없음. "
                "tempo_cov.py는 수정하지 않고 score_dataset을 이 파일 안에서 독립 복제해서 사용.",
        "results": results,
    }
    out_path = Path(args.output) if args.output else result_path.parent / "prompt_swap_random_result.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
