"""
vanilla_mc.py — Qwen2.5-VL 5-class baseline (최적화 없음)

Usage:
  CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 python experiment_4class/baselines/qwen/vanilla_mc.py
"""
import json, cv2, sys, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_TEMPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_TEMPO_ROOT))

import numpy as np
from PIL import Image
from vllm import SamplingParams

from utils import _llm, _processor, VIDEO_DIR
from utils_mc import (
    OUT_DIR, CLASSES, TYPE_MAP, TYPE_SUFFIX,
    load_gt_mc, split_data_mc, compute_metrics_mc,
)

TRAIN_RATIO  = 0.6
DEV_RATIO    = 0.2
SEED         = 42
OUTPUT       = "vanilla_mc_qwen_result.json"
FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8
VLLM_BATCH   = 32

INITIAL_PROMPT = (
    "Classify this video frame into one of five fraud categories.\n"
    "- Type-A\n- Type-B\n- Type-C\n- Type-D\n- Type-E"
)


def _predict_batch(items: list[tuple]) -> list:
    results = []
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    for start in range(0, len(items), VLLM_BATCH):
        chunk = items[start:start + VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            img_np = np.array(img)
            content = [
                {"type": "image", "image": img_np},
                {"type": "text",  "text":  prompt + TYPE_SUFFIX},
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
                matched = next((c for c in CLASSES if c in raw), None)
                if matched is None:
                    matched = next((cls for key, cls in TYPE_MAP.items() if key in raw), None)
                results.append(matched)
        except Exception as e:
            print(f"  [Warning] vLLM error: {e}")
            results.extend([None] * len(chunk))
    return results


def _extract_frames(video_path: Path) -> list:
    cap = cv2.VideoCapture(str(video_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sec = min(int(total / fps_v), MAX_SEC)
    out = []
    for sec in range(0, n_sec, FRAME_STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_v))
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out.append((sec, img.resize((448, 448), Image.LANCZOS)))
    cap.release()
    return out


def _load_video(vid: dict) -> tuple:
    vname = vid["video_name"]
    label = vid["label"]
    vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
    if not vfiles:
        return vname, label, []
    return vname, label, _extract_frames(vfiles[0])


def score_dataset(dataset: list[dict], prompt: str) -> list[dict]:
    loaded = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    all_items, frame_counts = [], []
    for _, _, frames in loaded:
        all_items.extend([(img, prompt) for _, img in frames])
        frame_counts.append(len(frames))

    all_preds = _predict_batch(all_items)
    frame_results, idx = [], 0
    for (vname, label, frames), count in zip(loaded, frame_counts):
        for (sec, _), pred in zip(frames, all_preds[idx:idx + count]):
            frame_results.append({"video": vname, "label": label, "pred_label": pred, "second": sec})
        idx += count
    return frame_results


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n=== 5-class 데이터셋 로드 ===")
    gt_list = load_gt_mc()
    trainset, devset, testset = split_data_mc(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    print("\n▶ Dev set")
    dev_m = compute_metrics_mc(score_dataset(devset, INITIAL_PROMPT))
    print(f"  ACC={_fmt(dev_m['acc'])}  macro_F1={_fmt(dev_m['macro_f1'])}")

    print("▶ Test set")
    test_m = compute_metrics_mc(score_dataset(testset, INITIAL_PROMPT))
    print(f"  ACC={_fmt(test_m['acc'])}  macro_F1={_fmt(test_m['macro_f1'])}")
    for c in CLASSES:
        print(f"    {c}: F1={_fmt(test_m['per_class_f1'][c])}")

    result = {
        "method": "Vanilla-MC (Qwen, no optimization)",
        "initial_prompt": INITIAL_PROMPT,
        "split": {"train": len(trainset), "dev": len(devset), "test": len(testset), "seed": SEED},
        "dev": dev_m, "test": test_m,
    }
    out_path = OUT_DIR / OUTPUT
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {out_path}")
