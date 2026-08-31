"""
baseline_eval.py — 모든 ablation 방법론의 공통 baseline 평가

최적화 없이 INITIAL_PROMPT + 툴 없음 + 동일 split(seed=42)으로 평가.
결과를 results/baseline_result.json에 저장하고,
각 실험 파일이 이 숫자를 공유 baseline으로 사용한다.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MM_LIMIT=1 python experiment/baseline_eval.py
"""

import json, re, cv2, sys, random
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

# ── 설정 (모든 실험 파일과 동일해야 함) ──────────────────────────────────────
TRAIN_RATIO = 0.6
DEV_RATIO   = 0.2
SEED        = 42
OUTPUT      = "baseline_result.json"

FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8
VLLM_BATCH   = 32

SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."

INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)


# ── 추론 ──────────────────────────────────────────────────────────────────────
def _predict_batch(items: list[tuple[Image.Image, str]]) -> list[float | None]:
    results: list[float | None] = []
    sp = SamplingParams(temperature=0.0, max_tokens=10)

    for start in range(0, len(items), VLLM_BATCH):
        chunk  = items[start:start + VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            img_np  = np.array(img)
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
            print(f"  [Warning] vLLM error: {e}")
            results.extend([None] * len(chunk))

    return results


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
    gt_map = {fp["second"]: fp["probability"] for fp in vid["frame_probabilities"]}
    raw    = _extract_frames(vfiles[0])
    return vname, label, [
        (sec, gt_map.get(sec, float(label == "abnormal")), img) for sec, img in raw
    ]


def score_dataset(dataset: list[dict], prompt: str) -> list[dict]:
    loaded = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    all_items: list[tuple[Image.Image, str]] = []
    frame_counts: list[int] = []
    for _, _, frames in loaded:
        all_items.extend([(img, prompt) for _, _, img in frames])
        frame_counts.append(len(frames))

    all_preds = _predict_batch(all_items)

    frame_results = []
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


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 데이터셋 로드 ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    print(f"\n=== Baseline 평가 (INITIAL_PROMPT, 툴 없음) ===")

    print("▶ Dev set (threshold 탐색용)")
    dev_res = score_dataset(devset, INITIAL_PROMPT)
    dev_m   = compute_metrics(dev_res)
    best_thr = find_best_threshold(dev_res)
    print(f"  AUC={_fmt(dev_m['auc'])}  MAP={_fmt(dev_m['map'])}  "
          f"F1={_fmt(dev_m['f1'])}  ACC={_fmt(dev_m['acc'])}  best_thr={best_thr:.2f}")

    print("▶ Test set")
    test_res  = score_dataset(testset, INITIAL_PROMPT)
    test_m    = compute_metrics(test_res)
    test_m_o  = compute_metrics(test_res, best_thr)
    print(f"  AUC={_fmt(test_m['auc'])}  MAP={_fmt(test_m['map'])}  "
          f"F1={_fmt(test_m['f1'])}  ACC={_fmt(test_m['acc'])}  F1_opt={_fmt(test_m_o['f1'])}")

    result = {
        "method":         "Baseline (no optimization)",
        "initial_prompt": INITIAL_PROMPT,
        "split": {
            "train": len(trainset),
            "dev":   len(devset),
            "test":  len(testset),
            "seed":  SEED,
        },
        "dev":  dev_m,
        "test": test_m,
        "test_f1_opt":  test_m_o["f1"],
        "test_acc_opt": test_m_o["acc"],
        "best_threshold": best_thr,
    }

    out_path = OUT_DIR / OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {out_path}")
    print("\n모든 ablation 실험 파일의 baseline으로 이 결과를 사용하세요.")
