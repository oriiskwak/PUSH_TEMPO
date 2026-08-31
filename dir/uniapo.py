"""
run_uniapo.py — UniAPO EM: 8프레임 균일 샘플링 + EM 최적화

utils.py + em_core_tool.py 활용:
  · 8프레임 균일 샘플링 → 영상 전체를 한 번에 VLM 입력
  · EM E-step: 오류 영상 클러스터링 → 피드백 생성 (em_core_tool)
  · EM M-step: 후보 생성 + Crossover → PromptMemory 빔서치 (em_core_tool)

※ MM_LIMIT=8 필수: utils.py LLM이 8장 동시 입력으로 초기화됨

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=2 MM_LIMIT=8 python run_uniapo.py
"""

import json, re, os, random, argparse, cv2, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from vllm import SamplingParams

# 8프레임 동시 입력 → MM_LIMIT=8 필수
if int(os.environ.get("MM_LIMIT", "1")) < 8:
    print("[ERROR] MM_LIMIT=8 이상으로 설정해야 합니다.")
    print("  실행: CUDA_VISIBLE_DEVICES=0,1 MM_LIMIT=8 python run_uniapo.py")
    sys.exit(1)

from utils import (
    VIDEO_DIR, OUT_DIR, VLLM_BATCH,
    _llm, _processor, _sampling,
    compute_metrics, find_best_threshold,
    split_data, load_gt, bernoulli_kl,
)
from em_core_tool import (
    FeedbackMemory, PromptMemory,
    cluster_errors, generate_feedback, evolve_feedback,
    generate_prompt_candidates, crossover_prompts,
    init_logger, get_logger,
)

RANDOM_SEED  = 42
N_FRAMES     = 8       # 균일 샘플링 프레임 수
FRAME_SIZE   = 448
LOAD_WORKERS = 8
UNIAPO_BATCH = 4       # 8장×4 = 32 이미지/call (OOM 방지)

SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."

INITIAL_PROMPT = (
"Predict the probability (0.0-1.0) that this video contains a cyber scam."
)

_v_sampling = SamplingParams(temperature=0.0, max_tokens=10)


# ── 8프레임 균일 샘플링 ────────────────────────────────────────────────────────
def _sample_8frames(video_path: Path) -> list[Image.Image]:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    poses = [int(total * (2 * i + 1) / (2 * N_FRAMES)) for i in range(N_FRAMES)]
    frames = []
    for pos in poses:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img.resize((FRAME_SIZE, FRAME_SIZE), Image.LANCZOS))
    cap.release()
    return frames


def _load_video(vid: dict) -> tuple[str, str, int, list[Image.Image]]:
    """(vname, label, representative_second, 8 frames)."""
    vname  = vid["video_name"]
    label  = vid["label"]
    vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
    if not vfiles:
        return vname, label, 0, []
    fps_list = vid.get("frame_probabilities", [])
    rep_sec  = fps_list[len(fps_list) // 2]["second"] if fps_list else 0
    return vname, label, rep_sec, _sample_8frames(vfiles[0])


# ── 8프레임 동시 배치 추론 ─────────────────────────────────────────────────────
def _predict_8frame_batch(items: list[tuple[list[Image.Image], str]]) -> list[float | None]:
    """[(8 PIL frames, prompt)] → [score | None]."""
    results: list[float | None] = []
    for start in range(0, len(items), UNIAPO_BATCH):
        chunk  = items[start:start + UNIAPO_BATCH]
        inputs, valid = [], []
        for i, (frames, prompt) in enumerate(chunk):
            if not frames:
                continue
            content = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": prompt + SCORE_SUFFIX})
            text = _processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs.append({"prompt": text, "multi_modal_data": {"image": frames}})
            valid.append(i)

        chunk_res = [None] * len(chunk)
        if inputs:
            outs = _llm.generate(inputs, _v_sampling, use_tqdm=False)
            for idx, out in zip(valid, outs):
                raw = out.outputs[0].text.strip()
                m   = re.search(r"[0-9]*\.?[0-9]+", raw)
                chunk_res[idx] = max(0.0, min(1.0, float(m.group()))) if m else None
        results.extend(chunk_res)
    return results


# ── 데이터셋 채점 (영상 레벨) ──────────────────────────────────────────────────
def score_dataset(dataset: list[dict], prompt: str) -> list[dict]:
    """영상별 8프레임 → 영상 레벨 스코어 dict 리스트."""
    loaded: list[tuple[str, str, int, list]] = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, rep_sec, frames = fut.result()
            if frames:
                loaded.append((vname, label, rep_sec, frames))

    preds = _predict_8frame_batch([(frames, prompt) for _, _, _, frames in loaded])

    results = []
    for (vname, label, rep_sec, _), pred in zip(loaded, preds):
        if pred is None:
            continue
        gt = 1.0 if label == "abnormal" else 0.0
        results.append({
            "video":  vname,
            "label":  label,
            "second": rep_sec,       # em_core_tool 피드백용 대표 프레임
            "pred":   pred,
            "gt":     gt,
            "kl":     bernoulli_kl(pred, gt),
        })
    return results


def _video_f1(results: list[dict], thr: float = 0.5) -> float:
    if not results:
        return 0.0
    true_l = [1 if r["label"] == "abnormal" else 0 for r in results]
    pred_l = [1 if r["pred"] >= thr else 0 for r in results]
    if len(set(true_l)) < 2:
        return 0.0
    from sklearn.metrics import f1_score
    return float(f1_score(true_l, pred_l, zero_division=0))


# ── E-step ────────────────────────────────────────────────────────────────────
def e_step(
    train_results: list[dict],
    current_prompt: str,
    fb_mem: FeedbackMemory,
    iteration: int,
) -> tuple[str, FeedbackMemory]:
    """오류 영상 클러스터링 → 피드백 생성 → FeedbackMemory 업데이트."""
    errors = [r for r in train_results
              if (r["pred"] >= 0.5) != (r["label"] == "abnormal")]

    if not errors:
        return "", fb_mem

    clusters   = cluster_errors(errors, em_name="uniapo", iteration=iteration)
    past       = fb_mem.retrieve(query="video scam detection error")
    chunk_fbs  = []

    for cid, chunk in enumerate(clusters):
        n_fn = sum(1 for r in chunk if r["gt"] > 0.5 and r["pred"] <= 0.5)
        n_fp = sum(1 for r in chunk if r["gt"] <= 0.5 and r["pred"] > 0.5)
        err_type = "fn" if n_fn >= n_fp else "fp"

        fb, tools = generate_feedback(
            current_prompt, chunk, past,
            em_name="uniapo", iteration=iteration,
            cluster_id=cid, error_type=err_type,
        )
        fb_mem.add(fb, tools)
        chunk_fbs.append(fb)

    f_short  = "\n".join(chunk_fbs)
    f_long   = fb_mem.retrieve(query=f_short)
    feedback = evolve_feedback(f_short, f_long)
    return feedback, fb_mem


# ── M-step ────────────────────────────────────────────────────────────────────
def m_step(
    current_prompt: str,
    feedback: str,
    pm: PromptMemory,
    pos_examples: list[dict],
    devset: list[dict],
    fb_mem: FeedbackMemory,
    dev_subsample: int = 50,
    n_candidates: int = 3,
) -> tuple[str, PromptMemory, float, dict]:
    """모든 후보를 각각 dev 평가 → 가장 높은 MAP 후보 선택 (빔서치)."""
    if not feedback:
        return current_prompt, pm, pm.best()[1], {}

    candidates  = generate_prompt_candidates(current_prompt, feedback, pos_examples, n=n_candidates)
    long_top    = pm.get_top_k()
    merged      = crossover_prompts(candidates, long_top)
    all_cands   = candidates + [merged]

    dev_sub = random.sample(devset, min(dev_subsample, len(devset)))

    best_cand    = current_prompt
    best_map     = pm.best()[1] or 0.0
    best_metrics = {}

    for cand in all_cands:
        dev_res     = score_dataset(dev_sub, cand)
        dev_metrics = compute_metrics(dev_res)
        dev_map     = dev_metrics.get("map") or 0.0
        pm.add(cand, dev_map)
        print(f"    후보 MAP={dev_map:.3f}")
        if dev_map > best_map:
            best_map     = dev_map
            best_cand    = cand
            best_metrics = dev_metrics

    return pm.get_beam()[0], pm, best_map, best_metrics


# ── UniAPO EM 메인 루프 ────────────────────────────────────────────────────────
def run_uniapo_em(
    trainset: list[dict],
    devset: list[dict],
    max_iter: int = 12,
    beam_size: int = 3,
    minibatch: int = 32,
    dev_subsample: int = 50,
    patience: int = 3,
) -> tuple[str, list[dict]]:

    init_logger("uniapo_em")
    lg = get_logger()

    prompt  = INITIAL_PROMPT
    fb_mem  = FeedbackMemory(top_k=3)
    pm      = PromptMemory(beam_size=beam_size, top_k=2)

    # 초기 dev 평가
    dev_sub      = random.sample(devset, min(dev_subsample, len(devset)))
    init_res     = score_dataset(dev_sub, prompt)
    init_metrics = compute_metrics(init_res)
    init_map     = init_metrics.get("map") or 0.0
    pm.add(prompt, init_map)
    print(f"  [Init] dev_map={init_map:.3f}  auc={init_metrics.get('auc') or 0:.3f}  "
          f"f1={init_metrics.get('f1') or 0:.3f}  acc={init_metrics.get('acc') or 0:.3f}")

    best_map   = init_map
    no_improve = 0
    log        = []

    for t in range(max_iter):
        print(f"\n── Iteration {t + 1}/{max_iter} ──")

        # 훈련 서브샘플 채점
        train_sub     = random.sample(trainset, min(minibatch, len(trainset)))
        train_results = score_dataset(train_sub, prompt)
        train_metrics = compute_metrics(train_results)
        train_map     = train_metrics.get("map") or 0.0

        n_errors = sum(1 for r in train_results
                       if (r["pred"] >= 0.5) != (r["label"] == "abnormal"))
        print(f"  [E-step] train_map={train_map:.3f}  train_f1={train_metrics.get('f1') or 0:.3f}  errors={n_errors}")

        # E-step
        feedback, fb_mem = e_step(train_results, prompt, fb_mem, iteration=t + 1)

        # Positive examples
        pos_examples = [r for r in train_results
                        if (r["pred"] >= 0.5) == (r["label"] == "abnormal")][:5]

        # M-step (MAP 기준 빔서치)
        prompt, pm, dev_map, dev_metrics = m_step(
            prompt, feedback, pm, pos_examples, devset, fb_mem,
            dev_subsample=dev_subsample,
        )
        print(f"  [M-step] dev_map={dev_map:.3f}  auc={dev_metrics.get('auc') or 0:.3f}  "
              f"f1={dev_metrics.get('f1') or 0:.3f}  acc={dev_metrics.get('acc') or 0:.3f}  "
              f"prompt={prompt[:60]}...")

        if lg:
            lg.log_iteration("uniapo_em", t + 1, train_map, dev_map,
                             n_errors, len(cluster_errors(
                                 [r for r in train_results if (r["pred"] >= 0.5) != (r["label"] == "abnormal")]
                             )), prompt)

        log.append({
            "iter":      t + 1,
            "train_map": train_map,
            "train_f1":  train_metrics.get("f1"),
            "dev_map":   dev_map,
            "dev_auc":   dev_metrics.get("auc"),
            "dev_f1":    dev_metrics.get("f1"),
            "dev_acc":   dev_metrics.get("acc"),
            "dev_kl":    dev_metrics.get("kl"),
            "prompt":    prompt,
            "feedback":  feedback,
        })

        if dev_map > best_map:
            best_map   = dev_map
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping (patience={patience} 달성)")
                break

    best_prompt, _ = pm.best()
    if lg:
        lg.close()

    return best_prompt, log


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter",      type=int,   default=12)
    parser.add_argument("--beam_size",     type=int,   default=3)
    parser.add_argument("--minibatch",     type=int,   default=32)
    parser.add_argument("--dev_subsample", type=int,   default=50)
    parser.add_argument("--patience",      type=int,   default=3)
    parser.add_argument("--train_ratio",   type=float, default=0.6)
    parser.add_argument("--dev_ratio",     type=float, default=0.2)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--output",        default="uniapo_em_result.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("\n=== 데이터셋 로드 ===")
    gt_list  = load_gt()
    trainset, devset, testset = split_data(
        gt_list, args.train_ratio, args.dev_ratio, args.seed
    )
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")
    print(f"설정: N_FRAMES={N_FRAMES}  beam_size={args.beam_size}  max_iter={args.max_iter}")

    print("\n=== UniAPO EM 최적화 시작 ===")
    best_prompt, log = run_uniapo_em(
        trainset, devset,
        max_iter=args.max_iter,
        beam_size=args.beam_size,
        minibatch=args.minibatch,
        dev_subsample=args.dev_subsample,
        patience=args.patience,
    )
    print(f"\n최적 프롬프트:\n  {best_prompt}")

    print("\n=== 테스트 평가 ===")
    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    print("▶ Baseline (초기 프롬프트)")
    base_res = score_dataset(testset, INITIAL_PROMPT)
    base_thr = find_best_threshold(score_dataset(devset, INITIAL_PROMPT))
    base_m   = compute_metrics(base_res)
    base_m_o = compute_metrics(base_res, base_thr)
    print(f"  KL={_fmt(base_m['kl'])}  AUC={_fmt(base_m['auc'])}  MAP={_fmt(base_m['map'])}"
          f"  F1={_fmt(base_m['f1'])}  ACC={_fmt(base_m['acc'])}  F1_opt={_fmt(base_m_o['f1'])}")

    print("▶ UniAPO EM (최적화 후)")
    best_dev_res = score_dataset(devset, best_prompt)
    best_thr     = find_best_threshold(best_dev_res)
    best_res     = score_dataset(testset, best_prompt)
    best_m       = compute_metrics(best_res)
    best_m_o     = compute_metrics(best_res, best_thr)
    print(f"  KL={_fmt(best_m['kl'])}  AUC={_fmt(best_m['auc'])}  MAP={_fmt(best_m['map'])}"
          f"  F1={_fmt(best_m['f1'])}  ACC={_fmt(best_m['acc'])}  F1_opt={_fmt(best_m_o['f1'])}")

    out_path = OUT_DIR / args.output
    json.dump({
        "method":           "UniAPO-EM",
        "best_prompt":      best_prompt,
        "baseline_prompt":  INITIAL_PROMPT,
        "baseline_test":    base_m,
        "best_test":        best_m,
        "best_test_f1_opt": best_m_o["f1"],
        "best_test_acc_opt": best_m_o["acc"],
        "best_threshold":   best_thr,
        "log":              log,
        "config": {
            "n_frames":      N_FRAMES,
            "frame_size":    FRAME_SIZE,
            "max_iter":      args.max_iter,
            "beam_size":     args.beam_size,
            "minibatch":     args.minibatch,
            "dev_subsample": args.dev_subsample,
        },
        "test_results": best_res,
    }, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path.name}")
