"""
baseline_eval_mc_intern.py — 5-class 분류 공통 baseline 평가, InternVL2.5-78B-AWQ 백본.

baseline_eval_mc.py(Qwen)와 완전히 동일한 파이프라인이며 백본만 InternVL로 교체
(tempo_qwen.py → tempo_intern.py와 동일한 교체 패턴). 최적화 없이 INITIAL_PROMPT +
동일 split(seed=42)으로 평가. 결과를 results_4class/baseline_mc_intern_result.json에 저장.

Usage:
  TP_SIZE=1 PP_SIZE=1 python experiment_4class/baseline_eval_mc_intern.py
"""

import json, re, cv2, sys, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from vllm import SamplingParams

import utils_intern
sys.modules['utils'] = utils_intern  # utils_mc/em_core 계열이 utils를 import할 때 Qwen LLM 로드 방지

from utils_intern import _llm, _processor, VIDEO_DIR
from utils_mc import (
    OUT_DIR, CLASSES, TYPE_MAP, TYPE_SUFFIX,
    load_gt_mc, split_data_mc, compute_metrics_mc,
)

# ── 설정 (모든 5-class 실험 파일과 동일해야 함) ───────────────────────────────
TRAIN_RATIO = 0.6
DEV_RATIO   = 0.2
SEED        = 42
OUTPUT      = "baseline_mc_intern_result.json"

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


# ── 추론 ──────────────────────────────────────────────────────────────────────
def _predict_batch(items: list[tuple[Image.Image, str]]) -> list[str | None]:
    """(PIL Image, prompt) 리스트 → 프레임별 클래스 예측 리스트."""
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
    raw = _extract_frames(vfiles[0])
    return vname, label, [(sec, img) for sec, img in raw]


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


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 5-class 데이터셋 로드 ===")
    gt_list = load_gt_mc()
    trainset, devset, testset = split_data_mc(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")
    for c in CLASSES:
        n = sum(1 for v in gt_list if v["label"] == c)
        print(f"  {c}: {n}개")

    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    print(f"\n=== Baseline 평가 (INITIAL_PROMPT, 최적화 없음) ===")

    print("▶ Dev set")
    dev_res = score_dataset(devset, INITIAL_PROMPT)
    dev_m   = compute_metrics_mc(dev_res)
    print(f"  ACC={_fmt(dev_m['acc'])}  macro_F1={_fmt(dev_m['macro_f1'])}")
    for c in CLASSES:
        print(f"    {c}: F1={_fmt(dev_m['per_class_f1'][c])}")

    print("▶ Test set")
    test_res = score_dataset(testset, INITIAL_PROMPT)
    test_m   = compute_metrics_mc(test_res)
    print(f"  ACC={_fmt(test_m['acc'])}  macro_F1={_fmt(test_m['macro_f1'])}")
    for c in CLASSES:
        print(f"    {c}: F1={_fmt(test_m['per_class_f1'][c])}")
    print("  Confusion (행=true, 열=pred):")
    header = "         " + "  ".join(f"{c[:4]:>4}" for c in CLASSES)
    print(f"  {header}")
    for tc in CLASSES:
        row = "  ".join(f"{test_m['confusion'][tc][pc]:>4}" for pc in CLASSES)
        print(f"  {tc[:4]:>4}  {row}")

    result = {
        "method":         "Baseline-5class-Intern (no optimization)",
        "initial_prompt": INITIAL_PROMPT,
        "split": {
            "train": len(trainset),
            "dev":   len(devset),
            "test":  len(testset),
            "seed":  SEED,
        },
        "dev":  dev_m,
        "test": test_m,
    }

    out_path = OUT_DIR / OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {out_path}")
    print("\nInternVL 5-class 실험 파일들의 baseline으로 이 결과를 사용하세요.")
