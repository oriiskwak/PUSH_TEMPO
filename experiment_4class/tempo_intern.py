"""
tempo_intern.py — 5-class TEMPO (Early/Late 구간 분리 + 도구 증강), InternVL2.5-78B-AWQ 백본.

tempo_qwen.py와 완전히 동일한 파이프라인이며, 백본만 InternVL로 교체
(experiment2/tempo_cov.py → experiment_intern/tempo_cov.py와 동일한 교체 패턴):
  - utils 대신 utils_intern 사용 (+ sys.modules['utils'] 리다이렉트로 em_core_tool2/
    em_core_mc2가 내부적으로 하는 `from utils import ...`도 InternVL로 연결)
  - _predict_batch: PIL Image를 numpy 변환 없이 그대로 전달, content는 {"type": "image"}만
    (InternVL 프로세서는 <image> 플레이스홀더 개수만 셈)

Usage:
  TP_SIZE=1 PP_SIZE=1 python experiment_4class/tempo_intern.py
"""

import json, re, os, random, cv2, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

# segmentation.json 로드
_SEG_PATH = Path(__file__).parent.parent / "data" / "segmentation.json"
_SEG_MAP: dict[str, dict[int, str]] = {}
if _SEG_PATH.exists():
    for _e in json.load(open(_SEG_PATH, encoding="utf-8")):
        _SEG_MAP[_e["video_name"]] = {fs["second"]: fs["segment"]
                                       for fs in _e["frame_segments"]}
    print(f"segmentation.json 로드: {len(_SEG_MAP)}개 영상")

import numpy as np
from PIL import Image
from vllm import SamplingParams

import utils_intern
sys.modules['utils'] = utils_intern  # em_core_tool2/em_core_mc2가 utils를 import할 때 Qwen LLM 로드 방지

from utils_intern import _llm, _processor, VIDEO_DIR
from utils_mc import (
    OUT_DIR, CLASSES, TYPE_MAP, TYPE_SUFFIX,
    load_gt_mc, split_data_mc, compute_metrics_mc,
)
from em_core_mc2 import (
    cluster_errors_mc, generate_feedback_mc, generate_prompt_candidates_mc,
    crossover_prompts_mc,
)
from em_core_tool2 import (
    FeedbackMemory, PromptMemory,
    evolve_feedback, init_logger, get_logger,
)
from tools import (
    inject_title, inject_channel,
    inject_rag, inject_web_search, batch_web_search,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────
MAX_ROUNDS      = 8
BEAM_SIZE       = 3
MINIBATCH       = 32
DEV_SUBSAMPLE   = 50
PATIENCE        = 2
TRAIN_RATIO     = 0.6
DEV_RATIO       = 0.2
SEED            = 42
OUTPUT          = "tempo_intern_result.json"

CHECKPOINT_PATH = OUT_DIR / "tempo_intern_checkpoint.json"

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


# ── 도구 컨텍스트 주입 ────────────────────────────────────────────────────────
def _build_tool_prompt(vname: str, base_prompt: str, tools: list[str],
                       web_res: str = "") -> str:
    prompt   = base_prompt
    tool_set = set(tools)
    if "title"      in tool_set: prompt = inject_title(vname, prompt)
    if "channel"    in tool_set: prompt = inject_channel(vname, prompt)
    if "rag"        in tool_set: prompt = inject_rag(vname, prompt)
    if "web_search" in tool_set: prompt = inject_web_search(vname, prompt, web_res)
    return prompt


# ── 체크포인트 ────────────────────────────────────────────────────────────────
def _save_checkpoint(
    rnd: int, global_it: int,
    early_prompt: str, late_prompt: str,
    early_fb_mem: "FeedbackMemory", late_fb_mem: "FeedbackMemory",
    early_pm: "PromptMemory", late_pm: "PromptMemory",
    best_f1: float, no_improve: int, log: list,
    best_pair_early: str, best_pair_late: str,
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
        "best_f1":      best_f1,
        "no_improve":   no_improve,
        "log":          log,
        "best_pair_early": best_pair_early,
        "best_pair_late":  best_pair_late,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_checkpoint(
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
    print(f"  [Resume] Round {s['round']} 이후부터 재시작  best_f1={s['best_f1']:.3f}")
    return (
        s["early_prompt"], s["late_prompt"],
        early_fb_mem, late_fb_mem, early_pm, late_pm,
        s["best_f1"], s["no_improve"], s["global_it"], s["log"],
        s.get("best_pair_early", s["early_prompt"]),
        s.get("best_pair_late",  s["late_prompt"]),
    )


# ── 프레임별 배치 추론 ────────────────────────────────────────────────────────
def _predict_batch(items: list[tuple[Image.Image, str]]) -> list[str | None]:
    results: list[str | None] = []
    sp = SamplingParams(temperature=0.0, max_tokens=20)

    for start in range(0, len(items), VLLM_BATCH):
        chunk  = items[start:start + VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            content = [
                {"type": "image"},
                {"type": "text",  "text":  prompt + CLASS_SUFFIX},
            ]
            text = _processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs.append({"prompt": text, "multi_modal_data": {"image": img}})

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


# ── 데이터셋 채점 (Early/Late 분리) ─────────────────────────────────────────
def score_dataset(
    dataset: list[dict],
    early_prompt: str,
    late_prompt:  str,
    early_fb_mem: FeedbackMemory | None = None,
    late_fb_mem:  FeedbackMemory | None = None,
) -> list[dict]:
    loaded = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    e_tools_sample = early_fb_mem.retrieve_tools("early scam classification") if early_fb_mem else []
    l_tools_sample = late_fb_mem.retrieve_tools("late scam classification")   if late_fb_mem  else []
    needs_web = "web_search" in e_tools_sample or "web_search" in l_tools_sample
    web_data: dict[str, str] = {}
    if needs_web:
        web_data = batch_web_search([vn for vn, _, _ in loaded])

    all_items: list[tuple[Image.Image, str]] = []
    all_meta:  list[tuple[str, str, int, str]] = []

    for vname, label, frames in loaded:
        seg_lookup = _SEG_MAP.get(vname, {})
        mid        = max(1, len(frames) // 2)
        e_tools = early_fb_mem.retrieve_tools(f"early {vname}") if early_fb_mem else []
        l_tools = late_fb_mem.retrieve_tools(f"late {vname}")   if late_fb_mem  else []
        web_res = web_data.get(vname, "")
        e_p     = _build_tool_prompt(vname, early_prompt, e_tools, web_res)
        l_p     = _build_tool_prompt(vname, late_prompt,  l_tools, web_res)

        for i, (sec, img) in enumerate(frames):
            seg    = seg_lookup.get(sec, "early" if i < mid else "late")
            prompt = e_p if seg == "early" else l_p
            all_items.append((img, prompt))
            all_meta.append((vname, label, sec, seg))

    all_preds = _predict_batch(all_items)

    frame_results = []
    for (vname, label, sec, seg), pred in zip(all_meta, all_preds):
        frame_results.append({
            "video":      vname,
            "label":      label,
            "pred_label": pred,
            "second":     sec,
            "segment":    seg,
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
              if r["pred_label"] is not None and r["pred_label"] != r["label"]]
    if not errors:
        return "", fb_mem

    confusion_clusters = cluster_errors_mc(errors)
    past      = fb_mem.retrieve(query="scam type classification error")
    chunk_fbs = []

    for cid, (conf_key, chunk) in enumerate(confusion_clusters.items()):
        true_class, pred_class = conf_key.split("→", 1)
        fb, tools, pattern = generate_feedback_mc(
            current_prompt, chunk, past,
            true_class=true_class, pred_class=pred_class,
            em_name=em_name, iteration=iteration, cluster_id=cid,
        )
        fb_mem.add(fb, tools, pattern)
        chunk_fbs.append(fb)

    f_short  = "\n".join(chunk_fbs)
    f_long   = fb_mem.retrieve(query=f_short)
    feedback = evolve_feedback(f_short, f_long)
    return feedback, fb_mem


# ── M-step (COV 없음) ────────────────────────────────────────────────────────
def _m_step_seg(
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

    candidates = generate_prompt_candidates_mc(current_prompt, feedback, pos_examples, n=n_candidates)
    long_top   = pm.get_top_k()
    new_prompt = crossover_prompts_mc(candidates, long_top)

    dev_sub = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
    if seg_name == "early":
        dev_res = score_dataset(dev_sub, new_prompt, fixed_other_prompt,
                                this_fb_mem, other_fb_mem)
    else:
        dev_res = score_dataset(dev_sub, fixed_other_prompt, new_prompt,
                                other_fb_mem, this_fb_mem)

    dev_metrics = compute_metrics_mc(dev_res)
    dev_f1      = dev_metrics.get("macro_f1") or 0.0
    print(f"    macro_F1={dev_f1:.3f}  ACC={dev_metrics.get('acc') or 0:.3f}")
    return new_prompt, pm, dev_f1, dev_metrics


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run_tempo_intern(trainset: list[dict], devset: list[dict]):
    init_logger("tempo_intern")
    lg = get_logger()

    early_fb_mem = FeedbackMemory(top_k=3)
    late_fb_mem  = FeedbackMemory(top_k=3)
    early_pm     = PromptMemory(beam_size=BEAM_SIZE, top_k=2)
    late_pm      = PromptMemory(beam_size=BEAM_SIZE, top_k=2)

    resumed = _load_checkpoint(early_fb_mem, late_fb_mem, early_pm, late_pm)

    if resumed is not None:
        (early_prompt, late_prompt,
         early_fb_mem, late_fb_mem, early_pm, late_pm,
         best_f1, no_improve, global_it, log,
         best_pair_early, best_pair_late) = resumed
        start_rnd = log[-1]["round"] if log else 0
    else:
        early_prompt = INITIAL_PROMPT
        late_prompt  = INITIAL_PROMPT

        dev_sub      = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
        init_res     = score_dataset(dev_sub, early_prompt, late_prompt)
        init_metrics = compute_metrics_mc(init_res)
        init_f1      = init_metrics.get("macro_f1") or 0.0
        early_pm.add(early_prompt, init_f1)
        late_pm.add(late_prompt,   init_f1)
        print(f"  [Init] macro_F1={init_f1:.3f}  ACC={init_metrics.get('acc') or 0:.3f}")

        best_f1    = init_f1
        no_improve = 0
        log        = []
        global_it  = 0
        start_rnd  = 0
        best_pair_early = early_prompt
        best_pair_late  = late_prompt

    for rnd in range(start_rnd, MAX_ROUNDS):
        print(f"\n{'═'*55}")
        print(f"  Round {rnd + 1}/{MAX_ROUNDS}")
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
            train_metrics = compute_metrics_mc(train_results)
            train_f1      = train_metrics.get("macro_f1") or 0.0
            n_errors      = sum(1 for r in seg_results
                                if r["pred_label"] is not None and r["pred_label"] != r["label"])
            print(f"  [E-step] train_F1={train_f1:.3f}  {seg_name}_errors={n_errors}")

            feedback, fb_mem = _e_step_seg(
                seg_results, cur_prompt, fb_mem,
                em_name=f"tempo_intern_{seg_name}", iteration=global_it,
            )

            pos = [r for r in seg_results if r["pred_label"] == r["label"]][:5]

            new_prompt, pm, dev_f1, dev_metrics = _m_step_seg(
                seg_name, cur_prompt, fixed_prompt,
                feedback, pm, pos, devset,
                fb_mem, other_fb_mem,
            )
            print(f"  [M-step] macro_F1={dev_f1:.3f}  ACC={dev_metrics.get('acc') or 0:.3f}  "
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
                lg.log_iteration(f"tempo_intern_{seg_name}", global_it,
                                 train_f1, dev_f1, n_errors, 0, new_prompt)

            log.append({
                "round":      rnd + 1,
                "seg":        seg_name,
                "train_f1":   train_f1,
                "dev_f1":     dev_f1,
                "dev_acc":    dev_metrics.get("acc"),
                "dev_per_class": dev_metrics.get("per_class_f1"),
                "prompt":     new_prompt,
                "feedback":   feedback,
            })

        # 라운드 마무리 full eval (라운드 간 noise 제거용으로 매번 동일 크기 샘플)
        full_dev      = score_dataset(
            random.sample(devset, min(DEV_SUBSAMPLE * 2, len(devset))),
            early_prompt, late_prompt, early_fb_mem, late_fb_mem,
        )
        round_metrics = compute_metrics_mc(full_dev)
        round_f1      = round_metrics.get("macro_f1") or 0.0
        print(f"\n  [Round {rnd+1} 완료] macro_F1={round_f1:.3f}  "
              f"ACC={round_metrics.get('acc') or 0:.3f}")

        early_pm.add(early_prompt, round_f1)
        late_pm.add(late_prompt, round_f1)

        if round_f1 > best_f1:
            best_f1    = round_f1
            no_improve = 0
            best_pair_early = early_prompt
            best_pair_late  = late_prompt
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})")
                _save_checkpoint(
                    rnd + 1, global_it, early_prompt, late_prompt,
                    early_fb_mem, late_fb_mem, early_pm, late_pm,
                    best_f1, no_improve, log,
                    best_pair_early, best_pair_late,
                )
                break
            else:
                # 나쁜 라운드 롤백 — beam에서 최적 프롬프트로 복원 (이진분류 tempo_cov.py와 동일)
                early_prompt = early_pm.get_beam()[0]
                late_prompt  = late_pm.get_beam()[0]

        _save_checkpoint(
            rnd + 1, global_it, early_prompt, late_prompt,
            early_fb_mem, late_fb_mem, early_pm, late_pm,
            best_f1, no_improve, log,
            best_pair_early, best_pair_late,
        )

    if lg:
        lg.close()
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    return (best_pair_early or early_prompt, best_pair_late or late_prompt,
            early_fb_mem, late_fb_mem, log)


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 5-class 데이터셋 로드 ===")
    gt_list  = load_gt_mc()
    trainset, devset, testset = split_data_mc(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    print("\n=== TEMPO 시작 ===")
    best_early, best_late, early_fb_mem, late_fb_mem, log = run_tempo_intern(trainset, devset)

    print(f"\n최적 Early 프롬프트:\n  {best_early}")
    print(f"\n최적 Late  프롬프트:\n  {best_late}")

    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    print("\n=== 테스트 평가 ===")
    print("▶ Shared Baseline (baseline_eval_mc.py 결과)")
    _bl_path = OUT_DIR / "baseline_mc_intern_result.json"
    if _bl_path.exists():
        _bl    = json.load(open(_bl_path))
        base_m = _bl["test"]
    else:
        print("  [Warning] baseline_mc_intern_result.json 없음 — baseline_eval_mc_intern.py를 먼저 실행하세요.")
        base_m = {"acc": None, "macro_f1": None}
    print(f"  ACC={_fmt(base_m['acc'])}  macro_F1={_fmt(base_m['macro_f1'])}")

    print("▶ TEMPO")
    best_res = score_dataset(testset, best_early, best_late, early_fb_mem, late_fb_mem)
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
        "method":            "TEMPO",
        "best_early_prompt": best_early,
        "best_late_prompt":  best_late,
        "baseline_test":     base_m,
        "best_test":         best_m,
        "log":               log,
    }
    out_path = OUT_DIR / OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
