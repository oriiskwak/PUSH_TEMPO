"""
temporal_intern.py — Temporal EM, InternVL2.5-78B 백본

experiment2/temporal.py의 InternVL 포팅:
  - utils_intern 사용 (Qwen LLM 로드 방지)
  - PIL 이미지 직접 전달 (numpy 변환 없음)
"""

import json, re, os, random, cv2, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

# segmentation.json 로드
_SEG_PATH = Path(__file__).parent.parent / "data" / "segmentation.json"
_SEG_MAP  = {}
if _SEG_PATH.exists():
    for _e in json.load(open(_SEG_PATH, encoding="utf-8")):
        _SEG_MAP[_e["video_name"]] = {fs["second"]: fs["segment"] for fs in _e["frame_segments"]}
    print(f"segmentation.json 로드: {len(_SEG_MAP)}개 영상")

import numpy as np
from PIL import Image
from vllm import SamplingParams

import utils_intern
sys.modules['utils'] = utils_intern

from utils_intern import (
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
PATIENCE_EPS  = 0.01
PATIENCE      = 2
TRAIN_RATIO   = 0.6
DEV_RATIO     = 0.2
SEED          = 42
OUTPUT          = "temporal_em_intern_result.json"
CHECKPOINT_PATH = OUT_DIR / "temporal_em_intern_checkpoint.json"

FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8
VLLM_BATCH   = 32

SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."

INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)


# ── 체크포인트 ────────────────────────────────────────────────────────────────
def _save_checkpoint(rnd, early_prompt, late_prompt,
                     early_fb_mem, late_fb_mem,
                     early_pm, late_pm,
                     best_map, best_pair_early, best_pair_late,
                     no_improve, log, global_it):
    state = {
        "round": rnd, "global_it": global_it,
        "early_prompt": early_prompt, "late_prompt": late_prompt,
        "early_fb_mem": early_fb_mem.memory, "late_fb_mem": late_fb_mem.memory,
        "early_pm_mem": [[p, s] for p, s in early_pm.memory],
        "late_pm_mem":  [[p, s] for p, s in late_pm.memory],
        "best_map": best_map,
        "best_pair_early": best_pair_early, "best_pair_late": best_pair_late,
        "no_improve": no_improve, "log": log,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_checkpoint(early_fb_mem, late_fb_mem, early_pm, late_pm):
    if not CHECKPOINT_PATH.exists():
        return None, early_fb_mem, late_fb_mem, early_pm, late_pm, 0.0, \
               INITIAL_PROMPT, INITIAL_PROMPT, 0, [], 0
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        state = json.load(f)
    early_fb_mem.memory = state["early_fb_mem"]
    late_fb_mem.memory  = state["late_fb_mem"]
    early_pm.memory = [[p, s] for p, s in state["early_pm_mem"]]
    late_pm.memory  = [[p, s] for p, s in state["late_pm_mem"]]
    print(f"  [Resume] Round {state['round']} 이후부터 재시작  best_map={state['best_map']:.3f}")
    return (state["round"], early_fb_mem, late_fb_mem, early_pm, late_pm,
            state["best_map"], state["best_pair_early"], state["best_pair_late"],
            state["no_improve"], state["log"], state.get("global_it", 0))


# ── 프레임별 배치 추론 (InternVL: PIL 직접 전달) ──────────────────────────────
def _predict_batch(items):
    results = []
    sp = SamplingParams(temperature=0.0, max_tokens=10)

    for start in range(0, len(items), VLLM_BATCH):
        chunk  = items[start:start + VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            content = [
                {"type": "image"},
                {"type": "text", "text": prompt + SCORE_SUFFIX},
            ]
            text = _processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs.append({"prompt": text, "multi_modal_data": {"image": img}})

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


def _extract_frames(video_path):
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


_FRAME_CACHE = {}


def _load_video(vid):
    vname = vid["video_name"]
    label = vid["label"]
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


def score_dataset(dataset, early_prompt, late_prompt):
    loaded = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    all_items = []
    all_meta  = []

    for vname, label, frames in loaded:
        seg_lookup = _SEG_MAP.get(vname, {})
        mid        = max(1, len(frames) // 2)
        for i, (sec, gt, img) in enumerate(frames):
            seg    = seg_lookup.get(sec, "early" if i < mid else "late")
            prompt = early_prompt if seg == "early" else late_prompt
            all_items.append((img, prompt))
            all_meta.append((vname, label, gt, sec, seg))

    all_preds = _predict_batch(all_items)

    frame_results = []
    for (vname, label, gt, sec, seg), pred in zip(all_meta, all_preds):
        if pred is None:
            continue
        frame_results.append({
            "video": vname, "label": label,
            "pred":  pred,  "gt":    gt,
            "kl":    bernoulli_kl(pred, gt),
            "segment": seg, "second": sec,
        })
    return frame_results


def _e_step_seg(seg_results, current_prompt, fb_mem, em_name, iteration):
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
        fb, tools = generate_feedback(
            current_prompt, chunk, past,
            em_name=em_name, iteration=iteration,
            cluster_id=cid, error_type=err_type,
        )
        fb_mem.add(fb, tools)
        chunk_fbs.append(fb)

    f_short  = "\n".join(chunk_fbs)
    f_long   = fb_mem.retrieve(query=f_short)
    feedback = evolve_feedback(f_short, f_long)
    return feedback, fb_mem


def _m_step_seg(seg_name, current_prompt, fixed_other_prompt, feedback, pm, pos_examples,
                devset, n_candidates=3):
    if not feedback:
        return current_prompt, pm, 0.0, {}

    candidates = generate_prompt_candidates(current_prompt, feedback, pos_examples, n=n_candidates)
    long_top   = pm.get_top_k()
    new_prompt = crossover_prompts(candidates, long_top)

    dev_sub = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
    if seg_name == "early":
        dev_res = score_dataset(dev_sub, new_prompt, fixed_other_prompt)
    else:
        dev_res = score_dataset(dev_sub, fixed_other_prompt, new_prompt)

    dev_metrics = compute_metrics(dev_res)
    dev_map     = dev_metrics.get("map") or 0.0

    print(f"    MAP={dev_map:.3f}  AUC={dev_metrics.get('auc') or 0:.3f}")
    return new_prompt, pm, dev_map, dev_metrics


def run_temporal_em(trainset, devset):
    init_logger("temporal_em_intern")
    lg = get_logger()

    early_fb_mem = FeedbackMemory(top_k=3)
    late_fb_mem  = FeedbackMemory(top_k=3)
    early_pm     = PromptMemory(beam_size=BEAM_SIZE, top_k=2)
    late_pm      = PromptMemory(beam_size=BEAM_SIZE, top_k=2)

    (start_rnd, early_fb_mem, late_fb_mem, early_pm, late_pm,
     best_map, best_pair_early, best_pair_late,
     no_improve, log, global_it) = _load_checkpoint(
        early_fb_mem, late_fb_mem, early_pm, late_pm)

    if start_rnd is None:
        start_rnd    = 0
        early_prompt = INITIAL_PROMPT
        late_prompt  = INITIAL_PROMPT
        dev_sub      = random.sample(devset, min(DEV_SUBSAMPLE, len(devset)))
        init_res     = score_dataset(dev_sub, early_prompt, late_prompt)
        init_metrics = compute_metrics(init_res)
        init_map     = init_metrics.get("map") or 0.0
        early_pm.add(early_prompt, init_map)
        late_pm.add(late_prompt,   init_map)
        print(f"  [Init] dev_map={init_map:.3f}  dev_auc={init_metrics.get('auc') or 0:.3f}")
        best_map        = init_map
        best_pair_early = early_prompt
        best_pair_late  = late_prompt
    else:
        early_prompt = best_pair_early
        late_prompt  = best_pair_late

    for rnd in range(start_rnd, MAX_ROUNDS):
        print(f"\n{'═'*55}")
        print(f"  Round {rnd + 1}/{MAX_ROUNDS}")
        print(f"{'═'*55}")

        for seg_name in ("early", "late"):
            global_it   += 1
            cur_prompt   = early_prompt if seg_name == "early" else late_prompt
            fixed_prompt = late_prompt  if seg_name == "early" else early_prompt
            fb_mem       = early_fb_mem if seg_name == "early" else late_fb_mem
            pm           = early_pm     if seg_name == "early" else late_pm

            print(f"\n  ── {seg_name.upper()} 최적화 ──")

            train_sub     = random.sample(trainset, min(MINIBATCH, len(trainset)))
            train_results = score_dataset(train_sub, early_prompt, late_prompt)
            seg_results   = [r for r in train_results if r["segment"] == seg_name]
            train_metrics = compute_metrics(train_results)
            train_map     = train_metrics.get("map") or 0.0
            n_errors      = sum(1 for r in seg_results
                                if (r["pred"] >= 0.5) != (r["label"] == "abnormal"))
            print(f"  [E-step] train_map={train_map:.3f}  {seg_name}_errors={n_errors}")

            feedback, fb_mem = _e_step_seg(
                seg_results, cur_prompt, fb_mem,
                em_name=f"temporal_intern_{seg_name}", iteration=global_it,
            )

            pos = [r for r in seg_results
                   if (r["pred"] >= 0.5) == (r["label"] == "abnormal")][:5]

            new_prompt, pm, dev_map, dev_metrics = _m_step_seg(
                seg_name, cur_prompt, fixed_prompt, feedback, pm, pos, devset,
            )
            dev_auc = dev_metrics.get("auc") or 0.0
            print(f"  [M-step] map={dev_map:.3f}  auc={dev_auc:.3f}  "
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
                lg.log_iteration(f"temporal_intern_{seg_name}", global_it,
                                 train_map, dev_map, n_errors, 0, new_prompt)

            log.append({
                "round": rnd + 1, "seg": seg_name,
                "train_map": train_map, "dev_map": dev_map, "dev_auc": dev_auc,
                "dev_f1": dev_metrics.get("f1"), "dev_acc": dev_metrics.get("acc"),
                "prompt": new_prompt, "feedback": feedback,
            })

        full_dev      = score_dataset(devset, early_prompt, late_prompt)
        round_metrics = compute_metrics(full_dev)
        round_auc     = round_metrics.get("auc") or 0.0
        round_map     = round_metrics.get("map") or 0.0
        print(f"\n  [Round {rnd+1} 완료] dev_map={round_map:.3f}  dev_auc={round_auc:.3f}  "
              f"f1={round_metrics.get('f1') or 0:.3f}  acc={round_metrics.get('acc') or 0:.3f}")

        early_pm.add(early_prompt, round_map)
        late_pm.add(late_prompt,   round_map)

        if round_map > best_map:
            best_map        = round_map
            no_improve      = 0
            best_pair_early = early_prompt
            best_pair_late  = late_prompt
        elif round_map >= best_map - PATIENCE_EPS:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})")
                break

        early_prompt = early_pm.get_beam()[0]
        late_prompt  = late_pm.get_beam()[0]

        _save_checkpoint(
            rnd + 1, early_prompt, late_prompt,
            early_fb_mem, late_fb_mem, early_pm, late_pm,
            best_map, best_pair_early, best_pair_late,
            no_improve, log, global_it,
        )

    if lg:
        lg.close()
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    return best_pair_early or early_prompt, best_pair_late or late_prompt, log


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 데이터셋 로드 ===")
    gt_list  = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    print("\n=== Temporal EM (InternVL) 시작 ===")
    best_early, best_late, log = run_temporal_em(trainset, devset)

    print(f"\n최적 Early 프롬프트:\n  {best_early}")
    print(f"\n최적 Late  프롬프트:\n  {best_late}")

    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    print("\n=== 테스트 평가 ===")
    best_dev_res = score_dataset(devset,   best_early, best_late)
    best_thr     = find_best_threshold(best_dev_res)
    best_res     = score_dataset(testset,  best_early, best_late)
    best_m       = compute_metrics(best_res)
    best_m_o     = compute_metrics(best_res, best_thr)
    print(f"  AUC={_fmt(best_m['auc'])}  MAP={_fmt(best_m['map'])}  "
          f"F1={_fmt(best_m['f1'])}  ACC={_fmt(best_m['acc'])}  F1_opt={_fmt(best_m_o['f1'])}")

    result = {
        "method":            "Temporal-EM-InternVL",
        "best_early_prompt": best_early,
        "best_late_prompt":  best_late,
        "best_test":         best_m,
        "best_test_f1_opt":  best_m_o["f1"],
        "best_test_acc_opt": best_m_o["acc"],
        "log":               log,
    }
    out_path = OUT_DIR / OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
