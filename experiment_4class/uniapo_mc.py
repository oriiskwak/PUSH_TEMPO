"""
uniapo_mc.py — 4-class Baseline EM (단일 프롬프트, temporal X, COV X)

추론: 프레임 1장씩 개별 추론 → majority vote → 영상 레벨 클래스
최적화: EM (4-class confusion pair 기반 피드백, COV 없음)

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MM_LIMIT=1 python experiment_4class/uniapo_mc.py
"""

import json, re, os, random, cv2, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from vllm import SamplingParams

from utils import _llm, _processor, VIDEO_DIR
from utils_mc import (
    OUT_DIR, CLASSES, TYPE_MAP, TYPE_SUFFIX,
    load_gt_mc, split_data_mc, compute_metrics_mc,
)
from em_core_mc import (
    cluster_errors_mc, generate_feedback_mc, generate_prompt_candidates_mc,
    crossover_prompts_mc,
)
from em_core_tool import (
    FeedbackMemory, PromptMemory,
    evolve_feedback, init_logger, get_logger,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────
MAX_ROUNDS    = 8
BEAM_SIZE     = 3
MINIBATCH     = 32
DEV_SUBSAMPLE = 50
PATIENCE      = 3
TRAIN_RATIO   = 0.6
DEV_RATIO     = 0.2
SEED          = 42
OUTPUT        = "uniapo_mc_result.json"
CHECKPOINT_PATH = OUT_DIR / "uniapo_mc_checkpoint.json"

FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8
VLLM_BATCH   = 32

CLASS_SUFFIX = TYPE_SUFFIX

INITIAL_PROMPT = (
    "Classify this video frame into one of five fraud categories.\n"
    "- Type-A\n"
    "- Type-B\n"
    "- Type-C\n"
    "- Type-D\n"
    "- Type-E"
)


# ── 체크포인트 ────────────────────────────────────────────────────────────────
def _save_checkpoint(rnd: int, prompt: str, fb_mem: "FeedbackMemory",
                     pm: "PromptMemory", best_f1: float,
                     no_improve: int, log: list):
    state = {
        "round":      rnd,
        "prompt":     prompt,
        "fb_mem":     fb_mem.memory,
        "pm_mem":     [[p, s] for p, s in pm.memory],
        "best_f1":    best_f1,
        "no_improve": no_improve,
        "log":        log,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_checkpoint(fb_mem: "FeedbackMemory", pm: "PromptMemory"):
    if not CHECKPOINT_PATH.exists():
        return None, fb_mem, pm, 0.0, 0, []
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        state = json.load(f)
    fb_mem.memory = state["fb_mem"]
    pm.memory     = [[p, s] for p, s in state["pm_mem"]]
    print(f"  [Resume] Round {state['round']} 이후부터 재시작  best_f1={state['best_f1']:.3f}")
    return (state["prompt"], fb_mem, pm,
            state["best_f1"], state["no_improve"], state["log"])


# ── 프레임별 배치 추론 ────────────────────────────────────────────────────────
def _predict_batch(items: list[tuple[Image.Image, str]]) -> list[str | None]:
    results: list[str | None] = []
    sp = SamplingParams(temperature=0.0, max_tokens=20)

    for start in range(0, len(items), VLLM_BATCH):
        chunk  = items[start:start + VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            img_np  = np.array(img)
            content = [
                {"type": "image", "image": img_np},
                {"type": "text",  "text":  prompt + CLASS_SUFFIX},
            ]
            text = _processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs.append({"prompt": text, "multi_modal_data": {"image": img_np}})

        try:
            outs = _llm.generate(inputs, sp, use_tqdm=False)
            for out in outs:
                raw     = out.outputs[0].text.strip()
                matched = next((c for c in CLASSES if c in raw), None)
                if matched is None:
                    matched = next((cls for key, cls in TYPE_MAP.items() if key in raw), None)
                results.append(matched)
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


def _load_video(vid: dict) -> tuple[str, str, list]:
    vname  = vid["video_name"]
    label  = vid["label"]
    vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
    if not vfiles:
        return vname, label, []
    raw = _extract_frames(vfiles[0])
    return vname, label, [(sec, img) for sec, img in raw]


# ── 데이터셋 채점 ─────────────────────────────────────────────────────────────
def score_dataset(dataset: list[dict], prompt: str) -> list[dict]:
    """프레임 1장씩 추론 → frame_results (pred_label 포함)."""
    loaded = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    all_items: list[tuple[Image.Image, str]] = []
    frame_counts: list[int] = []
    for _, _, frames in loaded:
        all_items.extend([(img, prompt) for _, img in frames])
        frame_counts.append(len(frames))

    all_preds = _predict_batch(all_items)

    frame_results = []
    idx = 0
    for (vname, label, frames), count in zip(loaded, frame_counts):
        preds = all_preds[idx:idx + count]
        idx  += count
        for (sec, _), pred in zip(frames, preds):
            frame_results.append({
                "video":      vname,
                "label":      label,
                "pred_label": pred,
                "second":     sec,
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
              if r["pred_label"] is not None and r["pred_label"] != r["label"]]
    if not errors:
        return "", fb_mem

    # confusion pair 기반 클러스터링
    confusion_clusters = cluster_errors_mc(errors)
    past      = fb_mem.retrieve(query="scam type classification error")
    chunk_fbs = []

    for cid, (conf_key, chunk) in enumerate(confusion_clusters.items()):
        true_class, pred_class = conf_key.split("→", 1)
        fb, tools = generate_feedback_mc(
            current_prompt, chunk, past,
            true_class=true_class, pred_class=pred_class,
            em_name="uniapo_mc", iteration=iteration, cluster_id=cid,
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
        best_p, best_f1 = pm.best()
        return best_p if best_p else current_prompt, pm, best_f1 or 0.0, {}

    candidates = generate_prompt_candidates_mc(current_prompt, feedback, pos_examples, n=n_candidates)
    long_top   = pm.get_top_k()
    new_prompt = crossover_prompts_mc(candidates, long_top)

    dev_sub     = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
    dev_res     = score_dataset(dev_sub, new_prompt)
    dev_metrics = compute_metrics_mc(dev_res)
    dev_f1      = dev_metrics.get("macro_f1") or 0.0

    pm.add(new_prompt, dev_f1)
    print(f"    macro_F1={dev_f1:.3f}  ACC={dev_metrics.get('acc') or 0:.3f}")

    return new_prompt, pm, dev_f1, dev_metrics


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run_uniapo_mc(trainset: list[dict], devset: list[dict]):
    init_logger("uniapo_mc")
    lg = get_logger()

    fb_mem = FeedbackMemory(top_k=3)
    pm     = PromptMemory(beam_size=BEAM_SIZE, top_k=2)

    resume_prompt, fb_mem, pm, best_f1, no_improve, log = _load_checkpoint(fb_mem, pm)

    if resume_prompt is not None:
        prompt    = resume_prompt
        start_rnd = log[-1]["round"] if log else 0
    else:
        prompt = INITIAL_PROMPT
        dev_sub      = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
        init_res     = score_dataset(dev_sub, prompt)
        init_metrics = compute_metrics_mc(init_res)
        init_f1      = init_metrics.get("macro_f1") or 0.0
        pm.add(prompt, init_f1)
        print(f"  [Init] macro_F1={init_f1:.3f}  ACC={init_metrics.get('acc') or 0:.3f}")
        best_f1    = init_f1
        no_improve = 0
        start_rnd  = 0

    for rnd in range(start_rnd, MAX_ROUNDS):
        print(f"\n{'═'*55}")
        print(f"  Round {rnd + 1}/{MAX_ROUNDS}")
        print(f"{'═'*55}")

        train_sub     = random.sample(trainset, min(MINIBATCH, len(trainset)))
        train_results = score_dataset(train_sub, prompt)
        train_metrics = compute_metrics_mc(train_results)
        train_f1      = train_metrics.get("macro_f1") or 0.0
        n_errors      = sum(1 for r in train_results
                            if r["pred_label"] is not None and r["pred_label"] != r["label"])
        print(f"  [E-step] train_F1={train_f1:.3f}  errors={n_errors}")

        feedback, fb_mem = _e_step(train_results, prompt, fb_mem, iteration=rnd + 1)

        pos = [r for r in train_results
               if r["pred_label"] == r["label"]][:5]

        new_prompt, pm, dev_f1, dev_metrics = _m_step(
            prompt, feedback, pm, pos, devset,
        )
        print(f"  [M-step] macro_F1={dev_f1:.3f}  ACC={dev_metrics.get('acc') or 0:.3f}  "
              f"prompt={new_prompt[:55]}...")

        prompt = new_prompt

        if lg:
            lg.log_iteration("uniapo_mc", rnd + 1,
                             train_f1, dev_f1, n_errors, 0, new_prompt)

        log.append({
            "round":         rnd + 1,
            "train_f1":      train_f1,
            "dev_f1":        dev_f1,
            "dev_acc":       dev_metrics.get("acc"),
            "dev_per_class": dev_metrics.get("per_class_f1"),
            "prompt":        new_prompt,
            "feedback":      feedback,
        })

        if dev_f1 > best_f1:
            best_f1    = dev_f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})")
                _save_checkpoint(rnd + 1, prompt, fb_mem, pm, best_f1, no_improve, log)
                break

        _save_checkpoint(rnd + 1, prompt, fb_mem, pm, best_f1, no_improve, log)

    best_p, _ = pm.best()
    if lg:
        lg.close()
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    return best_p or prompt, log


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 4-class 데이터셋 로드 ===")
    gt_list  = load_gt_mc()
    trainset, devset, testset = split_data_mc(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    print("\n=== UniAPO-MC (4-class EM) 시작 ===")
    best_prompt, log = run_uniapo_mc(trainset, devset)
    print(f"\n최적 프롬프트:\n  {best_prompt}")

    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    print("\n=== 테스트 평가 ===")
    print("▶ Shared Baseline (baseline_eval_mc.py 결과)")
    _bl_path = OUT_DIR / "baseline_mc_result.json"
    if _bl_path.exists():
        _bl    = json.load(open(_bl_path))
        base_m = _bl["test"]
    else:
        print("  [Warning] baseline_mc_result.json 없음 — baseline_eval_mc.py를 먼저 실행하세요.")
        base_m = {"acc": None, "macro_f1": None}
    print(f"  ACC={_fmt(base_m['acc'])}  macro_F1={_fmt(base_m['macro_f1'])}")

    print("▶ UniAPO-MC (최적화 후)")
    best_res = score_dataset(testset, best_prompt)
    best_m   = compute_metrics_mc(best_res)
    print(f"  ACC={_fmt(best_m['acc'])}  macro_F1={_fmt(best_m['macro_f1'])}")
    for c in CLASSES:
        print(f"    {c}: F1={_fmt(best_m['per_class_f1'][c])}")
    print("  Confusion (행=true, 열=pred):")
    header = "         " + "  ".join(f"{c[:4]:>4}" for c in CLASSES)
    print(f"  {header}")
    for tc in CLASSES:
        row = "  ".join(f"{best_m['confusion'][tc][pc]:>4}" for pc in CLASSES)
        print(f"  {tc[:4]:>4}  {row}")

    result = {
        "method":       "UniAPO-MC",
        "best_prompt":  best_prompt,
        "baseline_test": base_m,
        "best_test":    best_m,
        "log":          log,
    }
    out_path = OUT_DIR / OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
