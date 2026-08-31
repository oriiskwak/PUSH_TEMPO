"""
eval_fair_comparison.py — UniAPO vs NoSeg 공정 비교 평가

동일 테스트셋(SEED=42)에서 툴 injection 없이 순수 프롬프트 성능 비교.
  1. UniAPO-MC 원본 — 훈련 데이터 참조(video ID) 포함
  2. UniAPO-MC 정제 — 훈련 데이터 참조 제거
  3. TEMPO-COV-MC-NoSeg — LP fix + abstract pattern

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 MM_LIMIT=1 python experiment_4class/eval_fair_comparison.py
"""

import json, sys, cv2, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

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
FRAME_STEP   = 2
MAX_SEC      = 150
LOAD_WORKERS = 8
VLLM_BATCH   = 32
CLASS_SUFFIX = TYPE_SUFFIX

# ── 프롬프트 정의 ──────────────────────────────────────────────────────────────
# 주의: 아래 3개는 5-class(정상 포함) 전환 이전, 4-class(사기 유형만) 최적화 런의
# 결과물을 그대로 박아넣은 것이라 정상(Type-E) 처리가 전혀 없음. CLASSES/utils_mc.py가
# 5-class로 바뀌어도 이 프롬프트들은 정상 영상에 대해 항상 오답을 내므로,
# 5-class 기준으로 새로 최적화된 best prompt로 교체하기 전까지는 비교 결과가 무효함.

# UniAPO 원본: 훈련셋 특정 비디오 ID("video=normal_L2SifDf52Yg") 포함
UNIAPO_ORIGINAL = (
    "Classify each video frame into one of four fraud categories: "
    "보이스피싱, 투자사기, 로맨스스캠, or 부업사기. "
    "For 보이스피싱, focus on visuals of phone calls, call centers, or individuals "
    "speaking into microphones, and urgent commands. "
    'For 투자사기, look for repetitive numeric data, financial platform references, '
    'and phrases like "27연타 27승." '
    "For 로맨스스캠, focus on text-based platforms, chat interfaces, or online dating "
    "visuals, and emotional manipulation. "
    "For 부업사기, identify visuals of low-effort tasks like data entry, keyboards, "
    "mobile apps, paperwork, and emphasis on daily income or hourly wages. "
    "Personal success stories also suggest 부업사기. "
    "Analyze the scenario structure and communication medium for accurate classification. "
    "Ensure the model outputs ONLY one category per frame. "
    "Maintain correctly classified frames from video=normal_L2SifDf52Yg."
)

# UniAPO 정제: 훈련 데이터 참조만 제거, 나머지 동일
UNIAPO_CLEAN = (
    "Classify each video frame into one of four fraud categories: "
    "보이스피싱, 투자사기, 로맨스스캠, or 부업사기. "
    "For 보이스피싱, focus on visuals of phone calls, call centers, or individuals "
    "speaking into microphones, and urgent commands. "
    'For 투자사기, look for repetitive numeric data, financial platform references, '
    'and phrases like "27연타 27승." '
    "For 로맨스스캠, focus on text-based platforms, chat interfaces, or online dating "
    "visuals, and emotional manipulation. "
    "For 부업사기, identify visuals of low-effort tasks like data entry, keyboards, "
    "mobile apps, paperwork, and emphasis on daily income or hourly wages. "
    "Personal success stories also suggest 부업사기. "
    "Analyze the scenario structure and communication medium for accurate classification. "
    "Ensure the model outputs ONLY one category per frame."
)

# TEMPO-COV-MC-NoSeg 최적 프롬프트 (LP fix + abstract pattern)
NOSEG_BEST = (
    "Classify each video frame into one of four scam types: "
    "보이스피싱, 투자사기, 로맨스스캠, 부업사기. "
    "For 로맨스스캠, identify progressive relationship building, personal interactions, "
    "and emotional manipulation. "
    "For 투자사기, look for financial promises, urgent transactions, professional settings, "
    "financial charts, and investment promotions. "
    "For 부업사기, detect quick money offers, low-effort jobs, and immediate gains. "
    "For 보이스피싱, focus on urgency, direct calls, and impersonation. "
    "Analyze abstract visual and contextual patterns to differentiate between similar scams. "
    "Prioritize financial visuals for 투자사기, quick money and job offers for 부업사기, "
    "urgency and direct communication for 보이스피싱, and romantic interactions for 로맨스스캠. "
    "Ensure each frame outputs only one class name based on the strongest visual and "
    "contextual indicators. "
    "The output should be just one class name per frame: "
    "보이스피싱 / 투자사기 / 로맨스스캠 / 부업사기."
)

PROMPTS = {
    "UniAPO-MC (original, w/ train ref)": UNIAPO_ORIGINAL,
    "UniAPO-MC (cleaned, no train ref)":  UNIAPO_CLEAN,
    "TEMPO-COV-MC-NoSeg":                 NOSEG_BEST,
}


# ── 영상 로드 / 추론 ───────────────────────────────────────────────────────────
def _extract_frames(video_path: Path):
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


def _load_video(vid: dict):
    vname  = vid["video_name"]
    label  = vid["label"]
    vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
    if not vfiles:
        return vname, label, []
    return vname, label, _extract_frames(vfiles[0])


def _predict_batch(items: list) -> list:
    results = []
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    for start in range(0, len(items), VLLM_BATCH):
        chunk  = items[start:start + VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            img_np  = np.array(img)
            content = [
                {"type": "image", "image": img_np},
                {"type": "text",  "text":  prompt + CLASS_SUFFIX},
            ]
            text = _processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs.append({"prompt": text, "multi_modal_data": {"image": img_np}})
        try:
            outs = _llm.generate(inputs, sp, use_tqdm=False)
            for out in outs:
                raw     = out.outputs[0].text.strip()
                matched = next((c for c in CLASSES if c in raw), None)
                if matched is None:
                    matched = next((cls for key, cls in TYPE_MAP.items() if key in raw), None)
                results.append(matched)
        except Exception as e:
            print(f"  [Warning] {e}")
            results.extend([None] * len(chunk))
    return results


def score_prompt(dataset: list, prompt: str) -> list:
    loaded = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_load_video, v): v for v in dataset}):
            vname, label, frames = fut.result()
            if frames:
                loaded.append((vname, label, frames))

    all_items   = []
    frame_counts = []
    for vname, label, frames in loaded:
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


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    gt_list = load_gt_mc()
    _, _, testset = split_data_mc(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Test set: {len(testset)} videos\n")

    all_results = {}
    for name, prompt in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"  평가: {name}")
        print(f"{'='*60}")
        res = score_prompt(testset, prompt)
        m   = compute_metrics_mc(res)
        all_results[name] = m
        print(f"  ACC={m['acc']:.4f}  macro_F1={m['macro_f1']:.4f}")
        for c in CLASSES:
            print(f"    {c}: F1={m['per_class_f1'][c]:.4f}")
        print("  Confusion (행=true, 열=pred):")
        header = "         " + "  ".join(f"{c[:4]:>4}" for c in CLASSES)
        print(f"  {header}")
        for tc in CLASSES:
            row = "  ".join(f"{m['confusion'][tc][pc]:>4}" for pc in CLASSES)
            print(f"  {tc[:4]:>4}  {row}")

    print(f"\n{'='*60}")
    print("  최종 비교 요약")
    print(f"{'='*60}")
    for name, m in all_results.items():
        print(f"  {name}")
        print(f"    macro_F1={m['macro_f1']:.4f}  ACC={m['acc']:.4f}")

    out_path = OUT_DIR / "fair_comparison_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {name: {
                "acc":           m["acc"],
                "macro_f1":      m["macro_f1"],
                "per_class_f1":  m["per_class_f1"],
                "confusion":     m["confusion"],
             } for name, m in all_results.items()},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n결과 저장: {out_path}")
