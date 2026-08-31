"""
tool_gpt.py — tool.py(uniapo_total_tool5b, Tool EM v5b)의 GPT 백엔드 버전.

tool.py와 동일한 EM 알고리즘(tool routing: title/channel/description/rag/web_search,
MIN_FB_ENTRIES 안정 라우팅 게이트)을 쓰지만, Qwen2.5-VL(_llm/_processor, vLLM 로컬 추론)이
맡던 predictor + optimizer 역할을 모두 GPT(gpt_backend.py, 기본 gpt-4o-mini)로 교체했다.

  - No text metadata injected by default (pure vision base)
  - Retrieve tools: {title, channel, description, rag, web_search} via pattern-tag embeddings
  - Retrieval query = video title (top_k=3, sim_threshold=0.4)
  - FeedbackMemory stores (feedback, tools, pattern) per error cluster

Usage:
  OPENAI_API_KEY=... GPT_MODEL=gpt-4o-mini python gpt_experiment/tool_gpt.py
"""

import json, os, random, cv2, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from utils_gpt import (
    VIDEO_DIR, OUT_DIR,
    compute_metrics, find_best_threshold,
    split_data, load_gt, bernoulli_kl,
)
from gpt_backend import predict_gpt_batch
from em_core_tool2_gpt import (
    FeedbackMemory, PromptMemory,
    cluster_errors, generate_feedback, evolve_feedback,
    get_video_query,
    generate_prompt_candidates, crossover_prompts,
    init_logger, get_logger,
)
from tools import (
    inject_title, inject_channel,
    inject_rag, inject_web_search, inject_description, batch_web_search,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────
MAX_ROUNDS    = 12   # tool.py(Qwen/Intern)와 동일 옵션으로 통일
BEAM_SIZE     = 3
MINIBATCH     = 32
DEV_SUBSAMPLE = 50
PATIENCE      = 2    # tool.py(Qwen/Intern)와 동일 옵션으로 통일
PATIENCE_EPS  = 0.01 # tool.py(Qwen/Intern)와 동일 옵션으로 통일
TRAIN_RATIO   = 0.6
DEV_RATIO     = 0.2
SEED          = 42
OUTPUT        = "tool_gpt4omini_result.json"
MIN_FB_ENTRIES = 1   # fb_mem 엔트리 이 수 이상일 때만 툴 라우팅 활성화 (안정 라우팅)

FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8

SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."

INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)

CHECKPOINT_PATH = OUT_DIR / "tool_gpt4omini_checkpoint.json"


def _save_checkpoint(rnd: int, prompt: str, fb_mem: "FeedbackMemory",
                     pm: "PromptMemory", best_map: float, best_prompt: str,
                     no_improve: int, log: list, phase: int):
    state = {
        "round":       rnd,
        "prompt":      prompt,
        "fb_mem":      fb_mem.memory,
        "pm_mem":      [[p, s] for p, s in pm.memory],
        "best_map":    best_map,
        "best_prompt": best_prompt,
        "no_improve":  no_improve,
        "log":         log,
        "phase":       phase,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_checkpoint(fb_mem: "FeedbackMemory", pm: "PromptMemory"):
    if not CHECKPOINT_PATH.exists():
        return None, fb_mem, pm, 0.0, None, 0, [], 1
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        state = json.load(f)
    fb_mem.memory = state["fb_mem"]
    pm.memory     = [[p, sc] for p, sc in state["pm_mem"]]
    print(f"  [Resume] Round {state['round']} 이후부터 재시작  best_map={state['best_map']:.3f}")
    return (state["prompt"], fb_mem, pm,
            state["best_map"], state.get("best_prompt"), state["no_improve"],
            state["log"], state.get("phase", 1))


# ── 도구 컨텍스트 주입 ────────────────────────────────────────────────────────
def _build_tool_prompt(vname: str, base_prompt: str, tools: list[str],
                       web_res: str = "") -> str:
    """title 쿼리로 선택된 툴만 주입. 기본 입력은 프레임 이미지뿐 (순수 비전)."""
    prompt = base_prompt
    tool_set = set(tools)
    if "title"       in tool_set: prompt = inject_title(vname, prompt)
    if "channel"     in tool_set: prompt = inject_channel(vname, prompt)
    if "description" in tool_set: prompt = inject_description(vname, prompt)
    if "rag"         in tool_set: prompt = inject_rag(vname, prompt)
    if "web_search"  in tool_set: prompt = inject_web_search(vname, prompt, web_res)
    return prompt


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


# ── 데이터셋 채점 ──────────────────────────────────────────────────────────────
def score_dataset(
    dataset: list[dict],
    prompt: str,
    fb_mem: FeedbackMemory | None = None,
    enable_tools: bool = True,
) -> list[dict]:
    loaded: list[tuple[str, str, list]] = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    _use_tools = enable_tools and fb_mem is not None and len(fb_mem.memory) >= MIN_FB_ENTRIES

    web_search_videos: list[str] = []
    if _use_tools:
        for vname, _, _ in loaded:
            if "web_search" in fb_mem.retrieve_tools(get_video_query(vname)):
                web_search_videos.append(vname)
    web_data: dict[str, str] = batch_web_search(web_search_videos) if web_search_videos else {}

    all_items: list[tuple[Image.Image, str]] = []
    frame_meta: list[tuple[str, str, float, int]] = []

    for vname, label, frames in loaded:
        t_list  = fb_mem.retrieve_tools(get_video_query(vname)) if _use_tools else []
        web_res = web_data.get(vname, "")
        p       = _build_tool_prompt(vname, prompt, t_list, web_res)

        for sec, gt, img in frames:
            all_items.append((img, p))
            frame_meta.append((vname, label, gt, sec))

    all_preds = predict_gpt_batch(all_items, score_suffix=SCORE_SUFFIX)

    frame_results: list[dict] = []
    for (vname, label, gt, sec), pred in zip(frame_meta, all_preds):
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

    clusters  = cluster_errors(errors, em_name="tool_gpt", iteration=iteration)
    past      = fb_mem.retrieve(query="scam frame detection error")
    chunk_fbs = []

    for cid, chunk in enumerate(clusters):
        n_fn     = sum(1 for r in chunk if r["gt"] > 0.5 and r["pred"] <= 0.5)
        n_fp     = sum(1 for r in chunk if r["gt"] <= 0.5 and r["pred"] > 0.5)
        err_type = "fn" if n_fn >= n_fp else "fp"
        fb, tools, pattern = generate_feedback(
            current_prompt, chunk, past,
            em_name="tool_gpt", iteration=iteration,
            cluster_id=cid, error_type=err_type,
        )
        fb_mem.add(fb, tools, pattern)
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
    fb_mem: FeedbackMemory,
    n_candidates: int = 3,
    enable_tools: bool = False,
) -> tuple[str, PromptMemory, float, dict]:
    if not feedback:
        return current_prompt, pm, 0.0, {}

    candidates = generate_prompt_candidates(current_prompt, feedback, pos_examples, n=n_candidates)
    long_top   = pm.get_top_k()
    new_prompt = crossover_prompts(candidates, long_top)

    dev_sub     = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
    dev_res     = score_dataset(dev_sub, new_prompt, fb_mem, enable_tools=enable_tools)
    dev_metrics = compute_metrics(dev_res)
    dev_map     = dev_metrics.get("map") or 0.0

    print(f"    MAP={dev_map:.3f}  AUC={dev_metrics.get('auc') or 0:.3f}")
    return new_prompt, pm, dev_map, dev_metrics


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run_tool_em(trainset: list[dict], devset: list[dict]):
    init_logger("tool_gpt")
    lg = get_logger()

    fb_mem = FeedbackMemory(top_k=3)
    pm     = PromptMemory(beam_size=BEAM_SIZE, top_k=2)

    (resume_prompt, fb_mem, pm,
     best_map, best_prompt, no_improve, log, phase) = _load_checkpoint(fb_mem, pm)

    if resume_prompt is not None:
        prompt    = resume_prompt
        start_rnd = log[-1]["round"] if log else 0
    else:
        prompt    = INITIAL_PROMPT
        phase     = 1
        dev_sub      = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
        init_res     = score_dataset(dev_sub, prompt, fb_mem, enable_tools=False)
        init_metrics = compute_metrics(init_res)
        init_map     = init_metrics.get("map") or 0.0
        pm.add(prompt, init_map)
        print(f"  [Init] dev_map={init_map:.3f}  auc={init_metrics.get('auc') or 0:.3f}  "
              f"f1={init_metrics.get('f1') or 0:.3f}  acc={init_metrics.get('acc') or 0:.3f}")
        best_map    = init_map
        best_prompt = prompt
        no_improve  = 0
        log         = []
        start_rnd   = 0

    for rnd in range(start_rnd, MAX_ROUNDS):
        print(f"\n{'═'*55}")
        print(f"  Round {rnd + 1}/{MAX_ROUNDS}  [Phase {phase}]")
        print(f"{'═'*55}")

        enable        = (phase == 2)
        train_sub     = random.sample(trainset, min(MINIBATCH, len(trainset)))
        train_results = score_dataset(train_sub, prompt, fb_mem, enable_tools=enable)
        train_metrics = compute_metrics(train_results)
        train_map     = train_metrics.get("map") or 0.0
        n_errors      = sum(1 for r in train_results
                            if (r["pred"] >= 0.5) != (r["label"] == "abnormal"))
        print(f"  [E-step] train_map={train_map:.3f}  errors={n_errors}")

        feedback, fb_mem = _e_step(train_results, prompt, fb_mem, iteration=rnd + 1)

        pos = [r for r in train_results
               if (r["pred"] >= 0.5) == (r["label"] == "abnormal")][:5]

        if phase == 1:
            new_prompt, pm, dev_map, dev_metrics = _m_step(
                prompt, feedback, pm, pos, devset, fb_mem, enable_tools=False,
            )
            prompt = new_prompt
        else:
            new_prompt  = best_prompt
            dev_sub     = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
            dev_res     = score_dataset(dev_sub, new_prompt, fb_mem, enable_tools=True)
            dev_metrics = compute_metrics(dev_res)
            dev_map     = dev_metrics.get("map") or 0.0
            prompt      = new_prompt

        dev_auc = dev_metrics.get("auc") or 0.0
        print(f"  [{'M-step' if phase == 1 else 'Phase2'}] "
              f"map={dev_map:.3f}  auc={dev_auc:.3f}  prompt={prompt[:55]}...")

        if lg:
            lg.log_iteration("tool_gpt", rnd + 1, train_map, dev_map, n_errors, 0, prompt)

        full_dev      = score_dataset(devset, prompt, fb_mem, enable_tools=enable)
        round_metrics = compute_metrics(full_dev)
        round_map     = round_metrics.get("map") or 0.0
        round_auc     = round_metrics.get("auc") or 0.0
        print(f"\n  [Round {rnd+1} 완료] dev_map={round_map:.3f}  dev_auc={round_auc:.3f}  "
              f"f1={round_metrics.get('f1') or 0:.3f}  acc={round_metrics.get('acc') or 0:.3f}")

        pm.add(prompt, round_map)
        log.append({
            "round":     rnd + 1,
            "phase":     phase,
            "train_map": train_map,
            "dev_map":   dev_map,
            "dev_auc":   dev_auc,
            "round_map": round_map,
            "round_auc": round_auc,
            "dev_f1":    round_metrics.get("f1"),
            "dev_acc":   round_metrics.get("acc"),
            "prompt":    prompt,
            "feedback":  feedback,
        })

        if round_map > best_map:
            best_map    = round_map
            no_improve  = 0
            best_prompt = prompt
        elif round_map >= best_map - PATIENCE_EPS:
            no_improve  = 0
        else:
            no_improve += 1

        if phase == 1 and no_improve >= PATIENCE:
            print(f"  [Phase 1→2] Prompt 수렴 — 툴 라우팅 단계 시작")
            phase      = 2
            no_improve = 0
            prompt     = best_prompt
        elif phase == 2 and no_improve >= PATIENCE:
            print(f"  Early stopping (patience={PATIENCE})")
            _save_checkpoint(rnd + 1, prompt, fb_mem, pm, best_map, best_prompt, no_improve, log, phase)
            break
        elif phase == 1:
            prompt = pm.get_beam()[0]

        _save_checkpoint(rnd + 1, prompt, fb_mem, pm, best_map, best_prompt, no_improve, log, phase)

    if lg:
        lg.close()
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    return best_prompt, fb_mem, log


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 데이터셋 로드 ===")
    gt_list  = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    print("\n=== Tool EM (GPT) 시작 ===")
    best_prompt, fb_mem, log = run_tool_em(trainset, devset)
    print(f"\n최적 프롬프트:\n  {best_prompt}")

    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    print("\n=== 테스트 평가 ===")
    print("▶ Shared Baseline (baseline_eval.py 결과)")
    _bl_path = OUT_DIR / "baseline_result.json"
    if _bl_path.exists():
        _bl      = json.load(open(_bl_path))
        base_m   = _bl["test"]
        base_m_o = {"f1": _bl["test_f1_opt"], "acc": _bl["test_acc_opt"]}
    else:
        print("  [Warning] baseline_result.json 없음 — baseline_eval.py를 먼저 실행하세요.")
        base_m   = {"auc": None, "map": None, "f1": None, "acc": None}
        base_m_o = {"f1": None, "acc": None}
    print(f"  AUC={_fmt(base_m['auc'])}  MAP={_fmt(base_m['map'])}  "
          f"F1={_fmt(base_m['f1'])}  ACC={_fmt(base_m['acc'])}  F1_opt={_fmt(base_m_o['f1'])}")

    print("▶ Tool EM - GPT (최적화 후)")
    best_dev_res = score_dataset(devset,    best_prompt, fb_mem=fb_mem, enable_tools=True)
    best_thr     = find_best_threshold(best_dev_res)
    best_res     = score_dataset(testset,   best_prompt, fb_mem=fb_mem, enable_tools=True)
    best_m       = compute_metrics(best_res)
    best_m_o     = compute_metrics(best_res, best_thr)
    print(f"  AUC={_fmt(best_m['auc'])}  MAP={_fmt(best_m['map'])}  "
          f"F1={_fmt(best_m['f1'])}  ACC={_fmt(best_m['acc'])}  F1_opt={_fmt(best_m_o['f1'])}")

    result = {
        "method":           "Tool-EM-GPT",
        "gpt_model":        os.environ.get("GPT_MODEL", "gpt-4o-mini"),
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
