"""
uniapo_total.py — Baseline EM (frame-by-frame, no tool, no temporal, no COV)

추론: 프레임 1장씩 개별 추론 → mean score → 영상 레벨 집계
최적화: EM (tool 없음, temporal 없음, COV 없음)

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MM_LIMIT=1 python experiment/uniapo_total.py
"""

import json, re, os, random, cv2, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from vllm import SamplingParams

from utils import (
    VIDEO_DIR, OUT_DIR,
    _llm, _processor,
    compute_metrics, find_best_threshold,
    split_data, load_gt, bernoulli_kl,
)
from em_core_tool import (
    FeedbackMemory, PromptMemory,
    cluster_errors, generate_feedback, evolve_feedback,
    generate_prompt_candidates, crossover_prompts,
    init_logger, get_logger,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────
MAX_ROUNDS    = 12
BEAM_SIZE     = 3
MINIBATCH     = 32
DEV_SUBSAMPLE = 50
PATIENCE      = 2
PATIENCE_EPS  = 0.01
TRAIN_RATIO   = 0.6
DEV_RATIO     = 0.2
SEED          = 42
OUTPUT        = "baseline_em_result.json"

FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8
VLLM_BATCH   = 32   # 프레임 1장씩 → 배치로 묶어서 throughput 확보

SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."

INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)

CHECKPOINT_PATH = OUT_DIR / "baseline_em_checkpoint.json"


def _save_checkpoint(rnd: int, prompt: str, fb_mem: "FeedbackMemory",
                     pm: "PromptMemory", best_map: float, best_prompt: str,
                     no_improve: int, log: list):
    state = {
        "round":       rnd,
        "prompt":      prompt,
        "fb_mem":      fb_mem.memory,
        "pm_mem":      [[p, s] for p, s in pm.memory],
        "best_map":    best_map,
        "best_prompt": best_prompt,
        "no_improve":  no_improve,
        "log":         log,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_checkpoint(fb_mem: "FeedbackMemory", pm: "PromptMemory"):
    if not CHECKPOINT_PATH.exists():
        return None, fb_mem, pm, 0.0, "", 0, []
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        state = json.load(f)
    fb_mem.memory = state["fb_mem"]
    pm.memory     = [[p, s] for p, s in state["pm_mem"]]
    best_map    = state.get("best_map") or state.get("best_auc") or 0.0
    best_prompt = state.get("best_prompt") or state["prompt"]
    print(f"  [Resume] Round {state['round']} 이후부터 재시작  best_map={best_map:.3f}")
    return (state["prompt"], fb_mem, pm,
            best_map, best_prompt, state["no_improve"], state["log"])


# ── 프레임별 배치 추론 (1장씩) ────────────────────────────────────────────────
def _predict_batch(items: list[tuple[Image.Image, str]]) -> list[float | None]:
    """(PIL Image, prompt) 리스트 → 프레임별 float 스코어 리스트 (1장씩 추론)."""
    results: list[float | None] = []
    sp = SamplingParams(temperature=0.0, max_tokens=10)

    for start in range(0, len(items), VLLM_BATCH):
        chunk  = items[start:start + VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            img_np = np.array(img)
            content = [
                {"type": "image", "image": img_np},
                {"type": "text",  "text":  prompt + SCORE_SUFFIX},
            ]
            text = _processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs.append({"prompt": text, "multi_modal_data": {"image": img_np}})

        try:
            outs = _llm.generate(inputs, sp, use_tqdm=False)
            for out in outs:
                raw = out.outputs[0].text.strip()
                m   = re.search(r"[0-9]*\.?[0-9]+", raw)
                results.append(max(0.0, min(1.0, float(m.group()))) if m else None)
        except Exception as e:
            print(f"  [Warning] vLLM Inference Error: {e}")
            results.extend([None] * len(chunk))

    return results


# ── 프레임 추출 ───────────────────────────────────────────────────────────────
def _extract_frames(video_path: Path) -> list[tuple[int, Image.Image]]:
    cap   = cv2.VideoCapture(str(video_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sec = min(int(total / fps_v), MAX_SEC)
    out   = []
    for sec in range(0, n_sec, FRAME_STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_v))
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out.append((sec, img.resize((448, 448), Image.LANCZOS)))
    cap.release()
    return out


def _load_video(vid: dict) -> tuple[str, str, list[tuple[int, float, Image.Image]]]:
    vname  = vid["video_name"]
    label  = vid["label"]
    vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
    if not vfiles:
        return vname, label, []
    gt_map = {fp["second"]: fp["probability"] for fp in vid["frame_probabilities"]}
    raw    = _extract_frames(vfiles[0])
    return vname, label, [
        (sec, gt_map.get(sec, float(label == "abnormal")), img) for sec, img in raw
    ]


# ── 데이터셋 채점 (프레임별 추론 → 영상 레벨 집계) ───────────────────────────
def score_dataset(dataset: list[dict], prompt: str) -> list[dict]:
    """프레임 1장씩 추론 → 영상별 mean score."""
    loaded: list[tuple[str, str, list]] = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    # 전체 프레임 플랫하게 모아서 한번에 배치 추론
    all_items: list[tuple[Image.Image, str]] = []
    frame_counts: list[int] = []
    for vname, label, frames in loaded:
        all_items.extend([(img, prompt) for _, _, img in frames])
        frame_counts.append(len(frames))

    all_preds = _predict_batch(all_items)

    frame_results: list[dict] = []
    idx = 0
    for (vname, label, frames), count in zip(loaded, frame_counts):
        preds = all_preds[idx:idx + count]
        idx  += count
        for (sec, gt, _), pred in zip(frames, preds):
            if pred is None:
                continue
            frame_results.append({
                "video": vname, "label": label,
                "pred":  pred,  "gt":    gt,
                "kl":    bernoulli_kl(pred, gt),
                "second": sec,
            })

    return frame_results


# ── E-step ────────────────────────────────────────────────────────────────────
def _e_step(
    frame_results: list[dict],
    current_prompt: str,
    fb_mem: FeedbackMemory,
    iteration: int,
) -> tuple[str, FeedbackMemory]:
    errors = [r for r in frame_results
              if (r["pred"] >= 0.5) != (r["label"] == "abnormal")]
    if not errors:
        return "", fb_mem

    clusters  = cluster_errors(errors, em_name="baseline_em", iteration=iteration)
    past      = fb_mem.retrieve(query="scam frame detection error")
    chunk_fbs = []

    for cid, chunk in enumerate(clusters):
        n_fn     = sum(1 for r in chunk if r["gt"] > 0.5 and r["pred"] <= 0.5)
        n_fp     = sum(1 for r in chunk if r["gt"] <= 0.5 and r["pred"] > 0.5)
        err_type = "fn" if n_fn >= n_fp else "fp"
        fb, tools = generate_feedback(
            current_prompt, chunk, past,
            em_name="baseline_em", iteration=iteration,
            cluster_id=cid, error_type=err_type,
        )
        fb_mem.add(fb, tools)
        chunk_fbs.append(fb)

    f_short  = "\n".join(chunk_fbs)
    f_long   = fb_mem.retrieve(query=f_short)
    feedback = evolve_feedback(f_short, f_long)
    return feedback, fb_mem


# ── M-step ────────────────────────────────────────────────────────────────────
def _m_step(
    current_prompt: str,
    feedback: str,
    pm: PromptMemory,
    pos_examples: list[dict],
    devset: list[dict],
    n_candidates: int = 3,
) -> tuple[str, PromptMemory, float, dict]:
    if not feedback:
        return current_prompt, pm, 0.0, {}

    candidates = generate_prompt_candidates(current_prompt, feedback, pos_examples, n=n_candidates)
    long_top   = pm.get_top_k()
    new_prompt = crossover_prompts(candidates, long_top)

    dev_sub     = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
    dev_res     = score_dataset(dev_sub, new_prompt)
    dev_metrics = compute_metrics(dev_res)
    dev_map     = dev_metrics.get("map") or 0.0

    print(f"    MAP={dev_map:.3f}  AUC={dev_metrics.get('auc') or 0:.3f}")
    return new_prompt, pm, dev_map, dev_metrics


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run_baseline_em(trainset: list[dict], devset: list[dict]):
    init_logger("baseline_em")
    lg = get_logger()

    fb_mem = FeedbackMemory(top_k=3)
    pm     = PromptMemory(beam_size=BEAM_SIZE, top_k=2)

    resume_prompt, fb_mem, pm, best_map, best_prompt, no_improve, log = _load_checkpoint(fb_mem, pm)

    if resume_prompt is not None:
        prompt    = resume_prompt
        start_rnd = log[-1]["round"] if log else 0
    else:
        prompt = INITIAL_PROMPT
        dev_sub      = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
        init_res     = score_dataset(dev_sub, prompt)
        init_metrics = compute_metrics(init_res)
        init_map     = init_metrics.get("map") or 0.0
        pm.add(prompt, init_map)
        print(f"  [Init] dev_map={init_map:.3f}  dev_auc={init_metrics.get('auc') or 0:.3f}  "
              f"f1={init_metrics.get('f1') or 0:.3f}  acc={init_metrics.get('acc') or 0:.3f}")
        best_map    = init_map
        best_prompt = prompt
        start_rnd   = 0

    for rnd in range(start_rnd, MAX_ROUNDS):
        print(f"\n{'═'*55}")
        print(f"  Round {rnd + 1}/{MAX_ROUNDS}")
        print(f"{'═'*55}")

        train_sub     = random.sample(trainset, min(MINIBATCH, len(trainset)))
        train_results = score_dataset(train_sub, prompt)
        train_metrics = compute_metrics(train_results)
        train_map     = train_metrics.get("map") or 0.0
        n_errors      = sum(1 for r in train_results
                            if (r["pred"] >= 0.5) != (r["label"] == "abnormal"))
        print(f"  [E-step] train_map={train_map:.3f}  errors={n_errors}")

        feedback, fb_mem = _e_step(train_results, prompt, fb_mem, iteration=rnd + 1)

        pos = [r for r in train_results
               if (r["pred"] >= 0.5) == (r["label"] == "abnormal")][:5]

        new_prompt, pm, dev_map, dev_metrics = _m_step(
            prompt, feedback, pm, pos, devset,
        )
        dev_auc = dev_metrics.get("auc") or 0.0
        print(f"  [M-step] map={dev_map:.3f}  auc={dev_auc:.3f}  "
              f"f1={dev_metrics.get('f1') or 0:.3f}  acc={dev_metrics.get('acc') or 0:.3f}  "
              f"prompt={new_prompt[:55]}...")

        prompt = new_prompt

        if lg:
            lg.log_iteration("baseline_em", rnd + 1,
                             train_map, dev_map, n_errors, 0, new_prompt)

        # 라운드 끝 전체 devset 평가 (early stopping 기준 — temporal/tempo_cov와 동일)
        full_dev      = score_dataset(devset, prompt)
        round_metrics = compute_metrics(full_dev)
        round_map     = round_metrics.get("map") or 0.0
        round_auc     = round_metrics.get("auc") or 0.0
        print(f"\n  [Round {rnd+1} 완료] dev_map={round_map:.3f}  dev_auc={round_auc:.3f}  "
              f"f1={round_metrics.get('f1') or 0:.3f}  acc={round_metrics.get('acc') or 0:.3f}")

        pm.add(prompt, round_map)

        log.append({
            "round":     rnd + 1,
            "train_map": train_map,
            "dev_map":   dev_map,
            "dev_auc":   dev_auc,
            "round_map": round_map,
            "round_auc": round_auc,
            "dev_f1":    round_metrics.get("f1"),
            "dev_acc":   round_metrics.get("acc"),
            "prompt":    new_prompt,
            "feedback":  feedback,
        })

        if round_map > best_map:
            best_map    = round_map
            best_prompt = prompt
            no_improve  = 0
        elif round_map >= best_map - PATIENCE_EPS:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})")
                _save_checkpoint(rnd + 1, prompt, fb_mem, pm, best_map, best_prompt, no_improve, log)
                break

        prompt = pm.get_beam()[0]
        _save_checkpoint(rnd + 1, prompt, fb_mem, pm, best_map, best_prompt, no_improve, log)

    if lg:
        lg.close()
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    return best_prompt, log


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 데이터셋 로드 ===")
    gt_list  = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    print("\n=== Baseline EM 시작 ===")
    best_prompt, log = run_baseline_em(trainset, devset)
    print(f"\n최적 프롬프트:\n  {best_prompt}")

    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    print("\n=== 테스트 평가 ===")
    print("▶ Shared Baseline (baseline_eval.py 결과)")
    _bl_path = OUT_DIR / "baseline_result.json"
    if _bl_path.exists():
        _bl      = json.load(open(_bl_path))
        base_m   = _bl["test"]
        base_m_o = {"f1": _bl["test_f1_opt"], "acc": _bl["test_acc_opt"]}
        base_thr = _bl["best_threshold"]
    else:
        print("  [Warning] baseline_result.json 없음 — baseline_eval.py를 먼저 실행하세요.")
        base_m   = {"auc": None, "map": None, "f1": None, "acc": None}
        base_m_o = {"f1": None, "acc": None}
        base_thr = 0.5
    print(f"  AUC={_fmt(base_m['auc'])}  MAP={_fmt(base_m['map'])}  "
          f"F1={_fmt(base_m['f1'])}  ACC={_fmt(base_m['acc'])}  F1_opt={_fmt(base_m_o['f1'])}")

    print("▶ Baseline EM (최적화 후)")
    best_dev_res = score_dataset(devset, best_prompt)
    best_thr     = find_best_threshold(best_dev_res)
    best_res     = score_dataset(testset, best_prompt)
    best_m       = compute_metrics(best_res)
    best_m_o     = compute_metrics(best_res, best_thr)
    print(f"  AUC={_fmt(best_m['auc'])}  MAP={_fmt(best_m['map'])}  "
          f"F1={_fmt(best_m['f1'])}  ACC={_fmt(best_m['acc'])}  F1_opt={_fmt(best_m_o['f1'])}")

    result = {
        "method":           "Baseline-EM",
        "best_prompt":      best_prompt,
        "baseline_test":    base_m,
        "best_test":        best_m,
        "best_test_f1_opt": best_m_o["f1"],
        "best_test_acc_opt":best_m_o["acc"],
        "log":              log,
    }
    out_path = OUT_DIR / OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
