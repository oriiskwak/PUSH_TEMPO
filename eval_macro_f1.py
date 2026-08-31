"""
eval_macro_f1.py — 기존 결과 파일에서 best_prompt를 재추론하여 macro F1 추가.

Usage:
  python eval_macro_f1.py
"""

import json, sys, random
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_gt, split_data, compute_metrics

SEED       = 42
TRAIN_RATIO = 0.6
DEV_RATIO   = 0.2
OUT_DIR    = Path(__file__).parent / "results"

random.seed(SEED)
np.random.seed(SEED)

gt_list = load_gt()
_, _, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)

# ── 단일 프롬프트용 score_dataset ─────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from vllm import SamplingParams
from utils import _llm, _processor, VIDEO_DIR, bernoulli_kl

FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8
VLLM_BATCH   = 32

def _load_video(item):
    vname, label = item["video"], item["label"]
    from utils import load_frame
    vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
    if not vfiles:
        return vname, label, []
    frames = []
    cap = __import__("cv2").VideoCapture(str(vfiles[0]))
    fps = cap.get(__import__("cv2").CAP_PROP_FPS) or 1
    for sec in range(0, min(MAX_SEC, int(cap.get(__import__("cv2").CAP_PROP_FRAME_COUNT) / fps)), FRAME_STEP):
        img = load_frame(str(vfiles[0]), sec)
        if img is not None:
            frames.append((sec, img))
    cap.release()
    return vname, label, frames

def score_dataset_single(dataset, prompt):
    loaded = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    all_items, all_meta = [], []
    for vname, label, frames in loaded:
        for sec, img in frames:
            all_items.append((img, prompt))
            all_meta.append((vname, label, sec))

    sp = SamplingParams(temperature=0.0, max_tokens=10)
    results = []
    for i in range(0, len(all_items), VLLM_BATCH):
        batch = all_items[i:i+VLLM_BATCH]
        prompts_out = []
        for img, p in batch:
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]}]
            text = _processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts_out.append({"prompt": text, "multi_modal_data": {"image": [img]}})
        outputs = _llm.generate(prompts_out, sp, use_tqdm=False)
        for out in outputs:
            try:
                val = float(out.outputs[0].text.strip())
                val = max(0.0, min(1.0, val))
            except:
                val = None
            results.append(val)

    frame_results = []
    for (vname, label, sec), pred in zip(all_meta, results):
        if pred is None:
            continue
        gt = 1.0 if label == "abnormal" else 0.0
        frame_results.append({
            "video": vname, "label": label, "pred": pred,
            "gt": gt, "second": sec,
            "kl": bernoulli_kl(pred, gt),
        })
    return frame_results


# ── 처리할 결과 파일 목록 ─────────────────────────────────────────────────────
TARGETS = [
    {"file": "baseline_result.json",        "prompt_key": "initial_prompt",   "type": "single"},
    {"file": "baseline_em_result.json",     "prompt_key": "best_prompt",       "type": "single"},
    {"file": "tempo_cov_total_result.json", "prompt_key": None,                "type": "temporal"},
    {"file": "temporal_em_result.json",     "prompt_key": None,                "type": "temporal"},
    {"file": "tool_em_result.json",         "prompt_key": "best_prompt",       "type": "single"},
]

for target in TARGETS:
    path = OUT_DIR / target["file"]
    if not path.exists():
        print(f"[SKIP] {target['file']} 없음")
        continue

    d = json.load(open(path, encoding="utf-8"))

    if "f1_macro" in (d.get("best_test") or d.get("test") or {}):
        print(f"[SKIP] {target['file']} 이미 f1_macro 있음")
        continue

    print(f"\n[RUN] {target['file']}")

    if target["type"] == "single":
        prompt = d[target["prompt_key"]]
        frame_res = score_dataset_single(testset, prompt)
        metrics = compute_metrics(frame_res)
        key = "test" if "test" in d else "best_test"
        d[key]["f1_macro"] = metrics["f1_macro"]

    elif target["type"] == "temporal":
        # temporal: early/late 프롬프트 → 세그멘테이션 기반 분기
        from experiment.uniapo_total_temporal import score_dataset as score_temporal
        early_p = d["best_early_prompt"]
        late_p  = d["best_late_prompt"]
        frame_res = score_temporal(testset, early_p, late_p)
        metrics = compute_metrics(frame_res)
        d["best_test"]["f1_macro"] = metrics["f1_macro"]

    print(f"  f1_macro = {metrics['f1_macro']:.4f}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료: {path.name}")

print("\n완료.")
