"""
vanilla_cot_gpt.py — Vanilla+CoT baseline (GPT 백엔드).

baseline_gpt.py와 동일한 평가 구조(최적화 없음, 동일 split seed=42)지만,
프레임당 0.0-1.0 float를 곧바로 뱉게 하는 대신 모델에게 먼저 시각적 근거를
단계별로 추론하게 한 다음 마지막에 점수를 내놓도록 요청한다 (Chain-of-Thought).

출력에 추론 텍스트가 섞이므로 _predict_batch는 max_tokens를 늘리고, 첫 번째로
등장하는 숫자가 아니라 응답의 "마지막" 숫자를 최종 점수로 파싱한다
(추론 중간에 다른 숫자가 언급될 수 있기 때문).

Usage:
  OPENAI_API_KEY=... GPT_MODEL=gpt-4o-mini python gpt_experiment/vanilla_cot_gpt.py
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

# ── 설정 (baseline_gpt.py와 동일해야 함) ─────────────────────────────────────
TRAIN_RATIO = 0.6
DEV_RATIO   = 0.2
SEED        = 42
OUTPUT      = "vanilla_cot_gpt4omini_result.json"

FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8

COT_MAX_TOKENS = 400

SCORE_SUFFIX = (
    "\nThink step by step: first describe the visual evidence you see in this "
    "frame (on-screen text, UI elements, people, context), then briefly reason "
    "about whether that evidence matches common cyber scam patterns "
    "(urgency, unrealistic returns, impersonation, requests for money/personal info). "
    "Finally, end your response with the probability as a single number 0.0-1.0."
)

INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)


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

    all_preds = predict_gpt_batch(
        all_items, score_suffix=SCORE_SUFFIX,
        max_tokens=COT_MAX_TOKENS, use_last_match=True,
    )

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

    print(f"\n=== Vanilla+CoT 평가 - GPT (INITIAL_PROMPT + CoT, 최적화 없음) ===")

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
        "method":         "Vanilla-CoT-GPT (no optimization)",
        "gpt_model":      os.environ.get("GPT_MODEL", "gpt-4o-mini"),
        "initial_prompt": INITIAL_PROMPT,
        "score_suffix":   SCORE_SUFFIX,
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
