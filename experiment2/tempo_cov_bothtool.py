"""
tempo_cov.py — TEMPO-COV: Temporal EM + Tool (Phase 2)

추론: 프레임 1장씩 개별 추론
      Early 프레임 → early_prompt, Late 프레임 → late_prompt
      Phase 2에서 툴(title/channel/ASR 등)은 EARLY에만 적용

최적화:
  Phase 1 (rounds 1–PHASE1_MAX_ROUNDS): 툴 없이 Early/Late Coordinate Descent
    - 3개 후보 생성 → 전부 crossover (mini-eval 없음)
    - balanced feedback (FP+FN 동시 처리, 진동 방지)
    - cross-segment temporal 피드백
    - 라운드 끝 beam으로 최적 프롬프트 복원
  Phase 2 (이후 rounds): 툴 활성화
    - PHASE2_EVOLVE=True:  프롬프트도 계속 진화 ("시각 먼저, 모호할 때 툴")
    - PHASE2_EVOLVE=False: 프롬프트 frozen, fb_mem 툴 라우팅만 업데이트

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MM_LIMIT=1 python experiment/tempo_cov.py
"""

import json, re, os, random, cv2, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

# segmentation.json 로드: {video_name: {second: "early"/"late"}}
_SEG_PATH = Path(__file__).parent.parent / "data" / "segmentation.json"
_SEG_MAP: dict[str, dict[int, str]] = {}
# 고정 early:late 비율 오버라이드용 (ablation 스크립트에서 monkey-patch). None이면 기존 50:50 폴백.
_FIXED_RATIO: float | None = None
if _SEG_PATH.exists():
    for _e in json.load(open(_SEG_PATH, encoding="utf-8")):
        _SEG_MAP[_e["video_name"]] = {fs["second"]: fs["segment"] for fs in _e["frame_segments"]}
    print(f"segmentation.json 로드: {len(_SEG_MAP)}개 영상")

import numpy as np
from PIL import Image
from vllm import SamplingParams

from utils import (
    VIDEO_DIR, OUT_DIR,
    _llm, _processor,
    compute_metrics, find_best_threshold,
    split_data, load_gt, bernoulli_kl,
)
from em_core_tool2 import (
    FeedbackMemory, PromptMemory,
    evolve_feedback,
    generate_balanced_feedback, generate_temporal_feedback,
    get_video_query,
    generate_prompt_candidates, crossover_prompts,
    init_logger, get_logger,
)
from tools import (
    inject_title, inject_asr, inject_channel, inject_description,
    inject_rag, inject_web_search, batch_web_search,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────
MAX_ROUNDS      = 12
BEAM_SIZE       = 3
MINIBATCH       = 32
DEV_SUBSAMPLE   = 50
PATIENCE        = 2
PATIENCE_EPS    = 0.01
TRAIN_RATIO     = 0.6
DEV_RATIO       = 0.2
SEED            = 42
OUTPUT          = "tempo_cov_evolve_result.json"    # PHASE2_EVOLVE=True
# OUTPUT        = "tempo_cov_frozen_result.json"   # PHASE2_EVOLVE=False 때 이걸로 교체
MIN_FB_ENTRIES  = 1   # fb_mem 엔트리 이 수 이상일 때만 툴 라우팅 활성화 (Round 2부터)

PHASE1_MAX_ROUNDS  = 6      # Round 6까지 temporal only, Round 7부터 tool 활성화
TOOLS_ENABLED      = False  # Phase 1=False(tool 비활성), Phase 2=True. run_tempo_cov_total2b가 갱신.
PHASE2_EVOLVE      = True   # True: Phase 2에서 프롬프트도 진화 / False: 프롬프트 frozen, 툴만 추가

FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8
VLLM_BATCH   = 32

SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."

INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)

CHECKPOINT_PATH = OUT_DIR / ("tempo_cov_evolve_checkpoint.json" if PHASE2_EVOLVE
                             else "tempo_cov_frozen_checkpoint.json")


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
    best_map_ckpt = s.get("best_map") or s.get("best_auc") or 0.0
    print(f"  [Resume] Round {s['round']} 이후부터 재시작  best_map={best_map_ckpt:.3f}")
    return (
        s["early_prompt"], s["late_prompt"],
        early_fb_mem, late_fb_mem, early_pm, late_pm,
        best_map_ckpt, s["no_improve"], s["global_it"], s["log"],
        s.get("best_pair_early", s["early_prompt"]),
        s.get("best_pair_late",  s["late_prompt"]),
        s.get("phase", 1),
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
        prompt += (f"\n[Supplementary context: The information above describes the full video. "
                   f"Base your judgment primarily on the visual content in this "
                   f"{seg_name or 'current'} frame. "
                   f"Use this context only as supporting evidence when visual analysis "
                   f"alone is ambiguous or inconclusive.]")
    return prompt


# ── 프레임별 배치 추론 (1장씩) ────────────────────────────────────────────────
def _predict_batch(items: list[tuple[Image.Image, str]]) -> list[float | None]:
    """(PIL Image, prompt) 리스트 → 프레임별 float 스코어 리스트 (1장씩 추론)."""
    results: list[float | None] = []
    sp = SamplingParams(temperature=0.0, max_tokens=10)

    for start in range(0, len(items), VLLM_BATCH):
        chunk  = items[start:start + VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            img_np = np.array(img)   # PIL → numpy: vLLM v1 멀티모달 캐시 hash 안정성
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


_FRAME_CACHE: dict[str, list] = {}  # vname → [(sec, gt, img), ...]


def _load_video(vid: dict) -> tuple[str, str, list[tuple[int, float, Image.Image]]]:
    vname  = vid["video_name"]
    label  = vid["label"]
    if vname in _FRAME_CACHE:
        return vname, label, _FRAME_CACHE[vname]
    vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
    if not vfiles:
        return vname, label, []
    gt_map = {fp["second"]: fp["probability"] for fp in vid["frame_probabilities"]}
    raw    = _extract_frames(vfiles[0])
    frames = [(sec, gt_map.get(sec, float(label == "abnormal")), img) for sec, img in raw]
    _FRAME_CACHE[vname] = frames
    return vname, label, frames


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
    # tempo_cov.py는 tool을 EARLY에만 적용하도록 고정돼 있지만, 이 사본은 "2d" 체크포인트
    # (예전에 EARLY+LATE 둘 다 tool을 켰던 버전) test 재현용으로 LATE도 동일 조건으로 켠다.
    _use_l_tools = TOOLS_ENABLED and late_fb_mem is not None and len(late_fb_mem.memory) >= MIN_FB_ENTRIES
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
        ratio      = _FIXED_RATIO if _FIXED_RATIO is not None else 0.5
        mid        = max(1, min(len(frames) - 1, round(len(frames) * ratio))) if len(frames) > 1 else 1
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

    all_preds = _predict_batch(all_items)

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


# ── E-step (단일 구간) — FP/FN 균형 피드백으로 진동 방지 ──────────────────────
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

    fp_errors = [r for r in errors if r["gt"] <= 0.5 and r["pred"] >= 0.5]
    fn_errors = [r for r in errors if r["gt"] >  0.5 and r["pred"] <  0.5]

    # FP/FN 균등 샘플: 각 최대 8개
    k          = min(len(fp_errors), len(fn_errors), 8)
    fp_sample  = random.sample(fp_errors, k) if k > 0 else fp_errors[:8]
    fn_sample  = random.sample(fn_errors, k) if k > 0 else fn_errors[:8]

    past     = fb_mem.retrieve(query="scam frame detection error")
    fb, tools, pattern = generate_balanced_feedback(
        current_prompt, fp_sample, fn_sample, past,
        em_name=em_name, iteration=iteration,
    )
    fb_mem.add(fb, tools, pattern)

    f_long   = fb_mem.retrieve(query=fb)
    feedback = evolve_feedback(fb, f_long)
    return feedback, fb_mem



# ── M-step (Early/Late 구간별, COV diversity) ─────────────────────────────────
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
def run_tempo_cov_total2b(trainset: list[dict], devset: list[dict]):
    global TOOLS_ENABLED
    init_logger("tempo_cov_total2b")
    lg = get_logger()

    early_fb_mem = FeedbackMemory(top_k=3)
    late_fb_mem  = FeedbackMemory(top_k=3)
    early_pm     = PromptMemory(beam_size=BEAM_SIZE, top_k=2)
    late_pm      = PromptMemory(beam_size=BEAM_SIZE, top_k=2)
    # 라운드 끝 full-devset 채점 결과 캐시: (early_prompt, late_prompt) → frame_results.
    # top-3 앙상블 threshold 계산 시 재사용해 devset 재추론을 피하기 위함.
    dev_result_cache: dict[tuple[str, str], list[dict]] = {}

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
        init_auc     = init_metrics.get("auc") or 0.0
        init_map     = init_metrics.get("map") or 0.0
        early_pm.add(early_prompt, init_map)
        late_pm.add(late_prompt,   init_map)
        print(f"  [Init] dev_auc={init_auc:.3f}  dev_map={init_map:.3f}  "
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

            # FP/FN 균등 샘플링: abnormal/normal 각 MINIBATCH//2개
            _k         = MINIBATCH // 2
            _abnormal  = [v for v in trainset if v["label"] == "abnormal"]
            _normal    = [v for v in trainset if v["label"] == "normal"]
            train_sub  = (random.sample(_abnormal, min(_k, len(_abnormal))) +
                          random.sample(_normal,   min(_k, len(_normal))))
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
                em_name=f"tempo_cov2b_{seg_name}", iteration=global_it,
            )

            pos = [r for r in seg_results
                   if (r["pred"] >= 0.5) == (r["label"] == "abnormal")][:5]

            if phase == 1:
                new_prompt, pm, dev_map, dev_metrics = _m_step_seg_cov(
                    seg_name, cur_prompt, fixed_prompt,
                    feedback, pm, pos, devset,
                    fb_mem, other_fb_mem,
                )
                dev_auc = dev_metrics.get("auc") or 0.0
                print(f"  [M-step] map={dev_map:.3f}  auc={dev_auc:.3f}  "
                      f"f1={dev_metrics.get('f1') or 0:.3f}  acc={dev_metrics.get('acc') or 0:.3f}  "
                      f"prompt={new_prompt[:55]}...")
            else:
                if PHASE2_EVOLVE:
                    # Phase 2: 툴 켜진 상태에서 프롬프트도 진화 ("시각 먼저, 불확실하면 툴" 방향)
                    new_prompt, pm, dev_map, dev_metrics = _m_step_seg_cov(
                        seg_name, cur_prompt, fixed_prompt,
                        feedback, pm, pos, devset,
                        fb_mem, other_fb_mem,
                    )
                else:
                    # Phase 2: 프롬프트 frozen, 툴 routing만 업데이트
                    new_prompt = cur_prompt
                    _dev_sub = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
                    if seg_name == "early":
                        _dev_res = score_dataset(_dev_sub, new_prompt, fixed_prompt, fb_mem, other_fb_mem)
                    else:
                        _dev_res = score_dataset(_dev_sub, fixed_prompt, new_prompt, other_fb_mem, fb_mem)
                    dev_metrics = compute_metrics(_dev_res)
                dev_map = dev_metrics.get("map") or 0.0
                dev_auc = dev_metrics.get("auc") or 0.0
                print(f"  [Phase 2 tool eval] map={dev_map:.3f}  auc={dev_auc:.3f}  "
                      f"f1={dev_metrics.get('f1') or 0:.3f}  acc={dev_metrics.get('acc') or 0:.3f}")

            if seg_name == "early":
                early_prompt = new_prompt
                early_fb_mem = fb_mem
                early_pm     = pm
            else:
                late_prompt = new_prompt
                late_fb_mem = fb_mem
                late_pm     = pm

            if lg:
                lg.log_iteration(f"tempo_cov_total2b_{seg_name}", global_it,
                                 train_map, dev_auc, n_errors, 0, new_prompt)

            log.append({
                "round":     rnd + 1,
                "seg":       seg_name,
                "train_map": train_map,
                "dev_map":   dev_map,
                "dev_auc":   dev_auc,
                "dev_f1":    dev_metrics.get("f1"),
                "dev_acc":   dev_metrics.get("acc"),
                "prompt":    new_prompt,
                "feedback":  feedback,
            })

        # ── Cross-segment temporal 피드백 (두 구간 모두 업데이트 후) ──────────────
        # train_results는 마지막 seg("late") 루프에서 남은 값 재사용 — 추가 inference 없음
        if phase in (1, 2):
            _e_errs = [r for r in train_results if r["segment"] == "early"
                       and (r["pred"] >= 0.5) != (r["label"] == "abnormal")]
            _l_errs = [r for r in train_results if r["segment"] == "late"
                       and (r["pred"] >= 0.5) != (r["label"] == "abnormal")]
            if _e_errs or _l_errs:
                _past_cross = early_fb_mem.retrieve("temporal scam pattern cross segment")
                early_ref, late_ref = generate_temporal_feedback(
                    early_prompt, late_prompt, _e_errs, _l_errs,
                    _past_cross, em_name="tempo_cov2b_cross", iteration=global_it,
                )
                if early_ref:
                    early_fb_mem.add(early_ref, [], "temporal cross-segment early")
                if late_ref:
                    late_fb_mem.add(late_ref, [], "temporal cross-segment late")
                print(f"  [Cross-segment] early_ref={early_ref[:60]}...")

        # round-end 비교는 매번 같은 (전체) dev셋 사용 — 라운드 간 noise 제거.
        full_dev      = score_dataset(
            devset,
            early_prompt, late_prompt, early_fb_mem, late_fb_mem,
        )
        round_metrics = compute_metrics(full_dev)
        round_map     = round_metrics.get("map") or 0.0
        round_auc     = round_metrics.get("auc") or 0.0
        print(f"\n  [Round {rnd+1} 완료] dev_map={round_map:.3f}  dev_auc={round_auc:.3f}  "
              f"f1={round_metrics.get('f1') or 0:.3f}  acc={round_metrics.get('acc') or 0:.3f}")

        dev_result_cache[(early_prompt, late_prompt)] = full_dev

        # round-end 요약을 log에 별도 저장 — 지금까지는 print만 되고 사라져서, 세그먼트별
        # noisy M-step dev_map(50개 랜덤 서브샘플, 상대 구간 고정)만 남아 "라운드별 진짜 추세"를
        # 볼 방법이 없었음. type="round_end"로 구분해서 필터링 가능하게 함.
        log.append({
            "round":     rnd + 1,
            "type":      "round_end",
            "round_map": round_map,
            "round_auc": round_auc,
            "round_f1":  round_metrics.get("f1"),
            "round_acc": round_metrics.get("acc"),
        })

        early_pm.add(early_prompt, round_map)
        late_pm.add(late_prompt, round_map)

        if round_map > best_map:
            best_map   = round_map
            no_improve = 0
            best_pair_early = early_prompt
            best_pair_late  = late_prompt
        elif round_map >= best_map - PATIENCE_EPS:
            no_improve = 0
        else:
            no_improve += 1

        if phase == 1 and (no_improve >= PATIENCE or (rnd + 1) >= PHASE1_MAX_ROUNDS):
            print(f"  [Phase 1→2] Temporal 단계 종료 (Round {rnd+1}, no_improve={no_improve}) "
                  f"— tool 활성화, patience 리셋, Phase 1 최적 프롬프트 복원")
            phase         = 2
            TOOLS_ENABLED = True
            no_improve    = 0
            early_prompt  = best_pair_early   # Phase 1 최적으로 복원
            late_prompt   = best_pair_late
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
        else:
            # Phase 1/2 계속: 양 phase 모두 beam으로 최적 프롬프트 복원 (나쁜 라운드 롤백)
            early_prompt = early_pm.get_beam()[0]
            late_prompt  = late_pm.get_beam()[0]

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
    return (best_early or early_prompt, best_late or late_prompt,
            early_fb_mem, late_fb_mem, early_pm, late_pm, dev_result_cache, log)


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 데이터셋 로드 ===")
    gt_list  = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    print("\n=== TEMPO-COV-Total Coordinate Descent 시작 ===")
    (best_early, best_late, early_fb_mem, late_fb_mem,
     early_pm, late_pm, dev_result_cache, log) = run_tempo_cov_total2b(trainset, devset)

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
        base_thr = _bl["best_threshold"]
    else:
        print("  [Warning] baseline_result.json 없음 — baseline_eval.py를 먼저 실행하세요.")
        base_m   = {"auc": None, "map": None, "f1": None, "acc": None}
        base_m_o = {"f1": None, "acc": None}
        base_thr = 0.5
    print(f"  AUC={_fmt(base_m['auc'])}  MAP={_fmt(base_m['map'])}  "
          f"F1={_fmt(base_m['f1'])}  ACC={_fmt(base_m['acc'])}  F1_opt={_fmt(base_m_o['f1'])}")

    print("▶ TEMPO-COV-Total (최적화 후) — Top-3 앙상블 (round_map 가중 평균)")
    # Top-3 프롬프트 쌍 수집 (round_map 기준 상위) — 각 쌍의 가중치는 (early_score+late_score)/2
    _top_e = sorted([(p, s) for p, s in early_pm.memory], key=lambda x: x[1], reverse=True)[:3]
    _top_l = sorted([(p, s) for p, s in late_pm.memory],  key=lambda x: x[1], reverse=True)[:3]
    ensemble_pairs = [(ep, lp, max((es + ls) / 2, 1e-6))
                      for (ep, es), (lp, ls) in zip(_top_e, _top_l)]
    if not ensemble_pairs:
        ensemble_pairs = [(best_early, best_late, 1.0)]

    best_dev_res = score_dataset(devset, best_early, best_late, early_fb_mem, late_fb_mem)
    best_thr     = find_best_threshold(best_dev_res)

    print("▶ TEMPO-COV-Total (최적화 후) — 단일 최고 프롬프트 (앙상블 없음)")
    single_test_res = score_dataset(testset, best_early, best_late, early_fb_mem, late_fb_mem)
    single_m         = compute_metrics(single_test_res)
    single_m_o       = compute_metrics(single_test_res, best_thr)
    print(f"  AUC={_fmt(single_m['auc'])}  MAP={_fmt(single_m['map'])}  "
          f"F1={_fmt(single_m['f1'])}  ACC={_fmt(single_m['acc'])}  F1_opt={_fmt(single_m_o['f1'])}")

    from collections import defaultdict

    def _weighted_ensemble_score(dataset: list[dict], cache: dict | None = None) -> list[dict]:
        """ensemble_pairs=[(ep, lp, weight), ...] → video+second 기준 가중 평균 frame_results.

        cache가 주어지면 (ep, lp) 조합의 결과를 재사용해 재추론을 피한다
        (라운드 루프에서 이미 devset 전체로 채점해둔 결과와 동일한 조합인 경우).
        """
        acc: dict[tuple, list] = defaultdict(list)
        lookup: dict[tuple, dict] = {}
        for ep, lp, w in ensemble_pairs:
            _res = (cache or {}).get((ep, lp))
            if _res is None:
                _res = score_dataset(dataset, ep, lp, early_fb_mem, late_fb_mem)
            for r in _res:
                key = (r["video"], r["second"])
                acc[key].append((r["pred"], w))
                if key not in lookup:
                    lookup[key] = r   # 메타데이터(label, gt 등)는 첫 번째 결과에서 보존
        out = []
        for key, pw_list in acc.items():
            if key in lookup:
                r = dict(lookup[key])
                wsum = sum(w for _, w in pw_list)
                r["pred"] = sum(p * w for p, w in pw_list) / wsum
                out.append(r)
        return out

    # 앙상블 threshold는 앙상블 자신의 dev 예측 분포로 별도 탐색
    # (단일 프롬프트 dev 분포 기준 threshold를 그대로 쓰면 가중 평균으로 바뀐 분포와 안 맞음)
    # dev_result_cache에 라운드 루프 중 이미 계산된 (ep, lp) 조합이 있으면 재추론 없이 재사용.
    ensemble_dev_res = _weighted_ensemble_score(devset, cache=dev_result_cache)
    ensemble_thr     = find_best_threshold(ensemble_dev_res)

    best_res = _weighted_ensemble_score(testset)

    best_m   = compute_metrics(best_res)
    best_m_o = compute_metrics(best_res, ensemble_thr)
    print(f"  앙상블 프롬프트 수: {len(ensemble_pairs)}")
    print(f"  AUC={_fmt(best_m['auc'])}  MAP={_fmt(best_m['map'])}  "
          f"F1={_fmt(best_m['f1'])}  ACC={_fmt(best_m['acc'])}  F1_opt={_fmt(best_m_o['f1'])}")

    result = {
        "method":              "TEMPO-COV-Total",
        "best_early_prompt":   best_early,
        "best_late_prompt":    best_late,
        "baseline_test":       base_m,
        "single_test":         single_m,
        "single_test_f1_opt":  single_m_o["f1"],
        "single_test_acc_opt": single_m_o["acc"],
        "best_test":           best_m,
        "best_test_f1_opt":    best_m_o["f1"],
        "best_test_acc_opt":   best_m_o["acc"],
        "n_ensemble_pairs":    len(ensemble_pairs),
        "log":                 log,
    }
    out_path = OUT_DIR / OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
