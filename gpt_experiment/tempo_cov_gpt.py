"""
tempo_cov_gpt.py — tempo_cov.py(TEMPO-COV-Total: Temporal+Tool+COV)의 GPT 백엔드 버전.

추론: 프레임 1장씩 개별 추론
      Early 프레임 → early_prompt + tool, Late 프레임 → late_prompt + tool (OpenAI vision API)
최적화: Early/Late Coordinate Descent + Coverage Diversity — 텍스트/비전 생성도 GPT

Qwen2.5-VL(_llm/_processor, vLLM 로컬 추론)이 맡던 predictor + optimizer 역할을
모두 GPT(gpt_backend.py, 기본 gpt-4o-mini)로 교체했다.

Usage:
  OPENAI_API_KEY=... GPT_MODEL=gpt-4o-mini python gpt_experiment/tempo_cov_gpt.py
"""

import json, os, random, cv2, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

# segmentation.json 로드: {video_name: {second: "early"/"late"}}
_SEG_PATH = Path(__file__).parent.parent / "data" / "segmentation.json"
_SEG_MAP: dict[str, dict[int, str]] = {}
if _SEG_PATH.exists():
    for _e in json.load(open(_SEG_PATH, encoding="utf-8")):
        _SEG_MAP[_e["video_name"]] = {fs["second"]: fs["segment"] for fs in _e["frame_segments"]}
    print(f"segmentation.json 로드: {len(_SEG_MAP)}개 영상")

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
    inject_title, inject_asr, inject_channel, inject_description,
    inject_rag, inject_web_search, batch_web_search,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────
MAX_ROUNDS      = 12  # tempo_cov.py(Qwen)와 동일 옵션으로 통일
BEAM_SIZE       = 3
MINIBATCH       = 32
DEV_SUBSAMPLE   = 50
PATIENCE        = 2   # tempo_cov.py(Qwen)와 동일 옵션으로 통일
PATIENCE_EPS    = 0.01   # tempo_cov.py(Qwen)와 동일 옵션으로 통일
TRAIN_RATIO     = 0.6
DEV_RATIO       = 0.2
SEED            = 42
OUTPUT          = "tempo_cov_gpt4omini_result.json"
MIN_FB_ENTRIES  = 1   # fb_mem 엔트리 이 수 이상일 때만 툴 라우팅 활성화 (Round 2부터)

PHASE1_MAX_ROUNDS = 6   # Phase 1(temporal만) 라운드 상한 (= MAX_ROUNDS의 절반, tempo_cov.py와 동일) — patience로 더 일찍 끝날 수 있음
TOOLS_ENABLED     = False   # Phase 1=False(tool 비활성), Phase 2=True. run_tempo_cov_gpt가 갱신.

FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8

SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."

INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)


CHECKPOINT_PATH = OUT_DIR / "tempo_cov_gpt4omini_checkpoint.json"


def _save_checkpoint_cov(
    rnd: int, global_it: int,
    early_prompt: str, late_prompt: str,
    early_fb_mem: "FeedbackMemory", late_fb_mem: "FeedbackMemory",
    early_pm: "PromptMemory", late_pm: "PromptMemory",
    best_map: float, no_improve: int, log: list,
    best_pair_early: str, best_pair_late: str,
    phase: int,
):
    state = {
        "round":        rnd,
        "global_it":    global_it,
        "early_prompt": early_prompt,
        "late_prompt":  late_prompt,
        "early_fb_mem": early_fb_mem.memory,
        "late_fb_mem":  late_fb_mem.memory,
        "early_pm":     [[p, s] for p, s in early_pm.memory],
        "late_pm":      [[p, s] for p, s in late_pm.memory],
        "best_map":     best_map,
        "no_improve":   no_improve,
        "log":          log,
        "best_pair_early": best_pair_early,
        "best_pair_late":  best_pair_late,
        "phase":        phase,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_checkpoint_cov(
    early_fb_mem: "FeedbackMemory", late_fb_mem: "FeedbackMemory",
    early_pm: "PromptMemory", late_pm: "PromptMemory",
):
    if not CHECKPOINT_PATH.exists():
        return None
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        s = json.load(f)
    early_fb_mem.memory = s["early_fb_mem"]
    late_fb_mem.memory  = s["late_fb_mem"]
    early_pm.memory     = [[p, sc] for p, sc in s["early_pm"]]
    late_pm.memory      = [[p, sc] for p, sc in s["late_pm"]]
    print(f"  [Resume] Round {s['round']} 이후부터 재시작  best_map={s['best_map']:.3f}")
    return (
        s["early_prompt"], s["late_prompt"],
        early_fb_mem, late_fb_mem, early_pm, late_pm,
        s["best_map"], s["no_improve"], s["global_it"], s["log"],
        s.get("best_pair_early", s["early_prompt"]),
        s.get("best_pair_late",  s["late_prompt"]),
        s.get("phase", 2),
    )


# ── 도구 컨텍스트 주입 ────────────────────────────────────────────────────────
def _build_tool_prompt(vname: str, base_prompt: str, tools: list[str],
                       web_res: str = "", seg_name: str = "") -> str:
    prompt = base_prompt
    tool_set = set(tools)
    if "title"       in tool_set: prompt = inject_title(vname, prompt)
    if "asr"         in tool_set: prompt = inject_asr(vname, prompt)
    if "channel"     in tool_set: prompt = inject_channel(vname, prompt)
    if "description" in tool_set: prompt = inject_description(vname, prompt)
    if "rag"         in tool_set: prompt = inject_rag(vname, prompt)
    if "web_search"  in tool_set: prompt = inject_web_search(vname, prompt, web_res)
    if tool_set:
        seg_desc = f"this {seg_name} frame" if seg_name else "this frame"
        prompt += (f"\n[Note: the context above describes the whole video, not just "
                   f"{seg_desc} — treat it as background, not direct visual evidence.]")
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


# ── 데이터셋 채점 (Early/Late 프레임별 개별 추론 + 도구) ────────────────────
def score_dataset(
    dataset: list[dict],
    early_prompt: str,
    late_prompt:  str,
    early_fb_mem: FeedbackMemory | None = None,
    late_fb_mem:  FeedbackMemory | None = None,
) -> list[dict]:
    """Early/Late 프레임 1장씩 추론 + tool → frame_results (segment 태그 포함)."""
    loaded: list[tuple[str, str, list]] = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    _use_e_tools = TOOLS_ENABLED and early_fb_mem is not None and len(early_fb_mem.memory) >= MIN_FB_ENTRIES
    # 실험: tool은 EARLY에만 적용 (title/channel 등은 영상 전체 정보라 source-credibility
    # 판단에 더 가까운 EARLY 쪽에서만 쓰고, LATE는 순수 시각 신호만 쓰도록 함)
    _use_l_tools = False
    web_search_videos: list[str] = []
    for vname, _, _ in loaded:
        e_t = early_fb_mem.retrieve_tools(f"early {get_video_query(vname)}") if _use_e_tools else []
        l_t = late_fb_mem.retrieve_tools(f"late {get_video_query(vname)}")   if _use_l_tools else []
        if "web_search" in e_t or "web_search" in l_t:
            web_search_videos.append(vname)
    web_data: dict[str, str] = batch_web_search(web_search_videos) if web_search_videos else {}

    all_items: list[tuple[Image.Image, str]] = []
    all_meta:  list[tuple[str, str, float, int, str]] = []

    for vname, label, frames in loaded:
        seg_lookup = _SEG_MAP.get(vname, {})
        mid        = max(1, len(frames) // 2)
        e_tools = early_fb_mem.retrieve_tools(f"early {get_video_query(vname)}") if _use_e_tools else []
        l_tools = late_fb_mem.retrieve_tools(f"late {get_video_query(vname)}")   if _use_l_tools else []
        web_res = web_data.get(vname, "")
        e_p     = _build_tool_prompt(vname, early_prompt, e_tools, web_res, seg_name="early")
        l_p     = _build_tool_prompt(vname, late_prompt,  l_tools, web_res, seg_name="late")

        for i, (sec, gt, img) in enumerate(frames):
            seg    = seg_lookup.get(sec, "early" if i < mid else "late")
            prompt = e_p if seg == "early" else l_p
            all_items.append((img, prompt))
            all_meta.append((vname, label, gt, sec, seg))

    all_preds = predict_gpt_batch(all_items, score_suffix=SCORE_SUFFIX)

    frame_results: list[dict] = []
    for (vname, label, gt, sec, seg), pred in zip(all_meta, all_preds):
        if pred is None:
            continue
        frame_results.append({
            "video":   vname, "label": label,
            "pred":    pred,  "gt":    gt,
            "kl":      bernoulli_kl(pred, gt),
            "segment": seg,   "second": sec,
        })

    return frame_results


# ── E-step (단일 구간) ────────────────────────────────────────────────────────
def _e_step_seg(
    seg_results: list[dict],
    current_prompt: str,
    fb_mem: FeedbackMemory,
    em_name: str,
    iteration: int,
) -> tuple[str, FeedbackMemory]:
    errors = [r for r in seg_results
              if (r["pred"] >= 0.5) != (r["label"] == "abnormal")]
    if not errors:
        return "", fb_mem

    clusters  = cluster_errors(errors, em_name=em_name, iteration=iteration)
    past      = fb_mem.retrieve(query="scam frame detection error")
    chunk_fbs = []

    for cid, chunk in enumerate(clusters):
        n_fn     = sum(1 for r in chunk if r["gt"] > 0.5 and r["pred"] <= 0.5)
        n_fp     = sum(1 for r in chunk if r["gt"] <= 0.5 and r["pred"] > 0.5)
        err_type = "fn" if n_fn >= n_fp else "fp"
        fb, tools, pattern = generate_feedback(
            current_prompt, chunk, past,
            em_name=em_name, iteration=iteration,
            cluster_id=cid, error_type=err_type,
        )
        fb_mem.add(fb, tools, pattern)
        chunk_fbs.append(fb)

    f_short  = "\n".join(chunk_fbs)
    f_long   = fb_mem.retrieve(query=f_short)
    feedback = evolve_feedback(f_short, f_long)
    return feedback, fb_mem


# ── M-step + Coverage Diversity ──────────────────────────────────────────────
def _m_step_seg_cov(
    seg_name: str,
    current_prompt: str,
    fixed_other_prompt: str,
    feedback: str,
    pm: PromptMemory,
    pos_examples: list[dict],
    devset: list[dict],
    this_fb_mem: FeedbackMemory,
    other_fb_mem: FeedbackMemory,
    n_candidates: int = 3,
) -> tuple[str, PromptMemory, float, dict]:
    if not feedback:
        return current_prompt, pm, 0.0, {}

    candidates = generate_prompt_candidates(current_prompt, feedback, pos_examples, n=n_candidates)

    long_top   = pm.get_top_k()
    new_prompt = crossover_prompts(candidates, long_top)

    dev_sub = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
    if seg_name == "early":
        dev_res = score_dataset(dev_sub, new_prompt, fixed_other_prompt,
                                this_fb_mem, other_fb_mem)
    else:
        dev_res = score_dataset(dev_sub, fixed_other_prompt, new_prompt,
                                other_fb_mem, this_fb_mem)

    dev_metrics = compute_metrics(dev_res)
    dev_map     = dev_metrics.get("map") or 0.0
    print(f"    MAP={dev_map:.3f}  AUC={dev_metrics.get('auc') or 0:.3f}")
    return new_prompt, pm, dev_map, dev_metrics


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run_tempo_cov_gpt(trainset: list[dict], devset: list[dict]):
    global TOOLS_ENABLED
    init_logger("tempo_cov_gpt")
    lg = get_logger()

    early_fb_mem = FeedbackMemory(top_k=3)
    late_fb_mem  = FeedbackMemory(top_k=3)
    early_pm     = PromptMemory(beam_size=BEAM_SIZE, top_k=2)
    late_pm      = PromptMemory(beam_size=BEAM_SIZE, top_k=2)

    resumed = _load_checkpoint_cov(early_fb_mem, late_fb_mem, early_pm, late_pm)

    if resumed is not None:
        (early_prompt, late_prompt,
         early_fb_mem, late_fb_mem, early_pm, late_pm,
         best_map, no_improve, global_it, log,
         best_pair_early, best_pair_late, phase) = resumed
        start_rnd = log[-1]["round"] if log else 0
        TOOLS_ENABLED = (phase == 2)
    else:
        early_prompt = INITIAL_PROMPT
        late_prompt  = INITIAL_PROMPT
        phase        = 1
        TOOLS_ENABLED = False
        dev_sub      = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
        init_res     = score_dataset(dev_sub, early_prompt, late_prompt)
        init_metrics = compute_metrics(init_res)
        init_map     = init_metrics.get("map") or 0.0
        early_pm.add(early_prompt, init_map)
        late_pm.add(late_prompt,   init_map)
        print(f"  [Init] dev_map={init_map:.3f}  auc={init_metrics.get('auc') or 0:.3f}  "
              f"f1={init_metrics.get('f1') or 0:.3f}  acc={init_metrics.get('acc') or 0:.3f}")
        best_map  = init_map
        no_improve = 0
        log        = []
        global_it  = 0
        start_rnd  = 0
        best_pair_early = early_prompt
        best_pair_late  = late_prompt

    print(f"  [Phase {phase}] tools_enabled={TOOLS_ENABLED}")

    for rnd in range(start_rnd, MAX_ROUNDS):
        print(f"\n{'═'*55}")
        print(f"  Round {rnd + 1}/{MAX_ROUNDS}  [Phase {phase}, tools_enabled={TOOLS_ENABLED}]")
        print(f"{'═'*55}")

        for seg_name in ("early", "late"):
            global_it += 1
            cur_prompt   = early_prompt if seg_name == "early" else late_prompt
            fixed_prompt = late_prompt  if seg_name == "early" else early_prompt
            fb_mem       = early_fb_mem if seg_name == "early" else late_fb_mem
            other_fb_mem = late_fb_mem  if seg_name == "early" else early_fb_mem
            pm           = early_pm     if seg_name == "early" else late_pm

            print(f"\n  ── {seg_name.upper()} 최적화 (상대 구간 고정) ──")

            train_sub     = random.sample(trainset, min(MINIBATCH, len(trainset)))
            train_results = score_dataset(train_sub, early_prompt, late_prompt,
                                          early_fb_mem, late_fb_mem)
            seg_results   = [r for r in train_results if r["segment"] == seg_name]
            train_metrics = compute_metrics(train_results)
            train_map     = train_metrics.get("map") or 0.0
            n_errors      = sum(1 for r in seg_results
                                if (r["pred"] >= 0.5) != (r["label"] == "abnormal"))
            print(f"  [E-step] train_map={train_map:.3f}  {seg_name}_errors={n_errors}")

            feedback, fb_mem = _e_step_seg(
                seg_results, cur_prompt, fb_mem,
                em_name=f"tempo_cov_gpt_{seg_name}", iteration=global_it,
            )

            pos = [r for r in seg_results
                   if (r["pred"] >= 0.5) == (r["label"] == "abnormal")][:5]

            new_prompt, pm, dev_map, dev_metrics = _m_step_seg_cov(
                seg_name, cur_prompt, fixed_prompt,
                feedback, pm, pos, devset,
                fb_mem, other_fb_mem,
            )
            print(f"  [M-step] dev_map={dev_map:.3f}  auc={dev_metrics.get('auc') or 0:.3f}  "
                  f"f1={dev_metrics.get('f1') or 0:.3f}  acc={dev_metrics.get('acc') or 0:.3f}  "
                  f"prompt={new_prompt[:55]}...")

            if seg_name == "early":
                early_prompt = new_prompt
                early_fb_mem = fb_mem
                early_pm     = pm
            else:
                late_prompt = new_prompt
                late_fb_mem = fb_mem
                late_pm     = pm

            if lg:
                lg.log_iteration(f"tempo_cov_gpt_{seg_name}", global_it,
                                 train_map, dev_map, n_errors, 0, new_prompt)

            log.append({
                "round":     rnd + 1,
                "seg":       seg_name,
                "train_map": train_map,
                "dev_map":   dev_map,
                "dev_auc":   dev_metrics.get("auc"),
                "dev_f1":    dev_metrics.get("f1"),
                "dev_acc":   dev_metrics.get("acc"),
                "prompt":    new_prompt,
                "feedback":  feedback,
            })

        # round-end 비교는 매번 같은 (전체) dev셋 사용 — 라운드 간 noise 제거.
        # (mini-eval은 후보 탐색용이라 그대로 랜덤 샘플 유지)
        full_dev      = score_dataset(
            devset,
            early_prompt, late_prompt, early_fb_mem, late_fb_mem,
        )
        round_metrics = compute_metrics(full_dev)
        round_map     = round_metrics.get("map") or 0.0
        print(f"\n  [Round {rnd+1} 완료] dev_map={round_map:.3f}  "
              f"f1={round_metrics.get('f1') or 0:.3f}  acc={round_metrics.get('acc') or 0:.3f}")

        if round_map > best_map:
            best_map   = round_map
            no_improve = 0
            best_pair_early = early_prompt
            best_pair_late  = late_prompt
        elif round_map >= best_map - PATIENCE_EPS:
            # best_map보다 살짝 낮은 정도(noise 범위)는 "실패"로 치지 않음 —
            # best_map/best_pair는 갱신하지 않되 patience는 소모하지 않음
            no_improve = 0
        else:
            no_improve += 1

        if phase == 1 and (no_improve >= PATIENCE or (rnd + 1) >= PHASE1_MAX_ROUNDS):
            print(f"  [Phase 1→2] Temporal 단계 종료 (Round {rnd+1}, no_improve={no_improve}) "
                  f"— tool 활성화, patience 리셋")
            phase         = 2
            TOOLS_ENABLED = True
            no_improve    = 0
        elif phase == 2 and no_improve >= PATIENCE:
            print(f"  Early stopping (patience={PATIENCE})")
            _save_checkpoint_cov(
                rnd + 1, global_it,
                early_prompt, late_prompt,
                early_fb_mem, late_fb_mem,
                early_pm, late_pm,
                best_map, no_improve, log,
                best_pair_early, best_pair_late, phase,
            )
            break

        _save_checkpoint_cov(
            rnd + 1, global_it,
            early_prompt, late_prompt,
            early_fb_mem, late_fb_mem,
            early_pm, late_pm,
            best_map, no_improve, log,
            best_pair_early, best_pair_late, phase,
        )

    best_early, best_late = best_pair_early, best_pair_late
    if lg:
        lg.close()
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    return best_early or early_prompt, best_late or late_prompt, early_fb_mem, late_fb_mem, log


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 데이터셋 로드 ===")
    gt_list  = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    print("\n=== TEMPO-COV-Total Coordinate Descent (GPT) 시작 ===")
    best_early, best_late, early_fb_mem, late_fb_mem, log = run_tempo_cov_gpt(trainset, devset)

    print(f"\n최적 Early 프롬프트:\n  {best_early}")
    print(f"\n최적 Late  프롬프트:\n  {best_late}")

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

    print("▶ TEMPO-COV-Total - GPT (최적화 후)")
    best_dev_res = score_dataset(devset, best_early, best_late, early_fb_mem, late_fb_mem)
    best_thr     = find_best_threshold(best_dev_res)
    best_res     = score_dataset(testset, best_early, best_late, early_fb_mem, late_fb_mem)
    best_m       = compute_metrics(best_res)
    best_m_o     = compute_metrics(best_res, best_thr)
    print(f"  AUC={_fmt(best_m['auc'])}  MAP={_fmt(best_m['map'])}  "
          f"F1={_fmt(best_m['f1'])}  ACC={_fmt(best_m['acc'])}  F1_opt={_fmt(best_m_o['f1'])}")

    result = {
        "method":            "TEMPO-COV-Total-GPT",
        "gpt_model":         os.environ.get("GPT_MODEL", "gpt-4o-mini"),
        "best_early_prompt": best_early,
        "best_late_prompt":  best_late,
        "baseline_test":     base_m,
        "best_test":         best_m,
        "best_test_f1_opt":  best_m_o["f1"],
        "best_test_acc_opt": best_m_o["acc"],
        "log":               log,
    }
    out_path = OUT_DIR / OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
