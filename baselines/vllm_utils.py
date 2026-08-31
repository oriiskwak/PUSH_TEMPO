"""
vllm_utils.py — EvoPrompt / APO 공용 vLLM 추론 유틸리티
TEMPO_SERVER/utils.py 의 프레임 처리·추론 방식과 동일하게 구현.

환경변수:
    TP_SIZE : tensor_parallel_size (기본 2)
"""
import cv2
import os
import re
import numpy as np
from pathlib import Path
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ── 모델 설정 ─────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

# TEMPO와 동일한 상수
MAX_SEC               = 150   # 최대 프레임 추출 초
FRAME_SCORE_THRESHOLD = 0.5   # 프레임 레벨 abnormal 임계값
SCAM_RATIO_THRESHOLD  = 0.3   # 영상 레벨 scam_ratio 임계값 (TEMPO 기본값)
VLLM_BATCH            = 128   # vLLM generate 당 최대 프레임 수

# ── vLLM 초기화 (모듈 임포트 시 1회) ─────────────────────────────────────────
_tp_size       = int(os.environ.get("TP_SIZE",        "2"))
_max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
_max_num_seqs  = int(os.environ.get("MAX_NUM_SEQS",  "2"))
print(f"[vllm_utils] Qwen2.5-VL-72B 로드 중 (tp={_tp_size}, max_model_len={_max_model_len}, max_num_seqs={_max_num_seqs})...")

_llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=_tp_size,
    dtype="float16",
    quantization="awq_marlin",
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 4},   # APO gradient 호출: 최대 4장
    max_model_len=_max_model_len,
    max_num_seqs=_max_num_seqs,
)
_processor  = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
_v_sampling = SamplingParams(temperature=0.0, max_tokens=10)
print("[vllm_utils] 모델 로드 완료.")

# 추론 시 항상 추가하는 출력 형식 지시문 (TEMPO INIT_PROMPT 마지막 줄과 동일)
SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."


# ── 프레임 추출 — TEMPO load_frame 방식과 동일 ───────────────────────────────
def extract_frames_1fps(video_path: Path, frames_dir: Path,
                        max_sec: int = MAX_SEC, step: int = 2) -> list[str]:
    """2초 간격(TEMPO 동일)으로 프레임을 JPEG 캐시에 저장하고 경로 리스트 반환."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    stem  = video_path.stem
    cap   = cv2.VideoCapture(str(video_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sec = min(int(total / fps_v), max_sec)
    cap.release()

    seconds = list(range(0, n_sec, step))

    # 모두 캐시되어 있으면 바로 반환
    paths = [frames_dir / f"{stem}_s{sec:04d}.jpg" for sec in seconds]
    if all(p.exists() for p in paths):
        return [str(p) for p in paths]

    cap   = cv2.VideoCapture(str(video_path))
    saved = []
    for sec in seconds:
        p = frames_dir / f"{stem}_s{sec:04d}.jpg"
        if p.exists():
            saved.append(str(p))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_v))
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((448, 448), Image.LANCZOS)
            img.save(str(p), "JPEG", quality=85)
            saved.append(str(p))
    cap.release()
    return saved


def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ── 비전 배치 추론 — TEMPO predict_qwen_batch와 동일 ─────────────────────────
def predict_vision_batch(items: list[tuple[Image.Image, str]]) -> list[float]:
    """(PIL Image, prompt) 리스트 → float 예측값 리스트."""
    results = []
    for start in range(0, len(items), VLLM_BATCH):
        chunk       = items[start:start + VLLM_BATCH]
        vllm_inputs = []
        for img, prompt in chunk:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  prompt + SCORE_SUFFIX},
            ]}]
            text = _processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vllm_inputs.append({
                "prompt":           text,
                "multi_modal_data": {"image": img},
            })
        outputs = _llm.generate(vllm_inputs, _v_sampling, use_tqdm=False)
        for out in outputs:
            raw = out.outputs[0].text.strip()
            m   = re.search(r"[0-9]*\.?[0-9]+", raw)
            results.append(max(0.0, min(1.0, float(m.group()))) if m else 0.5)
    return results


# ── 텍스트 전용 추론 — GA/DE crossover, APO gradient 생성용 ──────────────────
def predict_text(prompt: str, n: int = 1,
                 temperature: float = 0.7, max_tokens: int = 512) -> list[str]:
    """텍스트 전용 LLM 호출, n개 completions 반환."""
    messages = [{"role": "user", "content": prompt}]
    text     = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sp      = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
    outputs = _llm.generate([{"prompt": text}], sp, use_tqdm=False)
    return [c.text.strip() for c in outputs[0].outputs]


# ── 비전+텍스트 혼합 추론 — APO gradient 생성 시 이미지 포함 ─────────────────
def predict_vision_text(text_prompt: str, image_paths: list[str],
                        temperature: float = 0.7,
                        max_tokens: int = 600) -> str:
    """텍스트 프롬프트 + 이미지들 → 텍스트 생성 (APO gradient 호출용).
    limit_mm_per_prompt=4 이므로 최대 4장으로 제한.
    """
    image_paths = image_paths[:4]
    imgs    = [load_pil(fp) for fp in image_paths]
    content = [{"type": "image", "image": img} for img in imgs]
    content.append({"type": "text", "text": text_prompt})

    messages = [{"role": "user", "content": content}]
    text     = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sp      = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = _llm.generate(
        [{"prompt": text, "multi_modal_data": {"image": imgs}}],
        sp, use_tqdm=False
    )
    return outputs[0].outputs[0].text.strip()


# ── 영상 레벨 분류 — TEMPO scam_ratio 집계 방식과 동일 ───────────────────────
def classify_video(frame_paths: list[str], prompt: str) -> str:
    """단일 영상 프레임 경로 리스트 → normal/abnormal."""
    if not frame_paths:
        return "normal"
    items      = [(load_pil(fp), prompt) for fp in frame_paths]
    preds      = predict_vision_batch(items)
    scam_ratio = sum(p >= FRAME_SCORE_THRESHOLD for p in preds) / len(preds)
    return "abnormal" if scam_ratio >= SCAM_RATIO_THRESHOLD else "normal"


def get_video_score(frame_paths: list[str], prompt: str) -> float:
    """단일 영상 P(abnormal) 평균 점수 (AUC/MAP용)."""
    if not frame_paths:
        return 0.5
    items = [(load_pil(fp), prompt) for fp in frame_paths]
    preds = predict_vision_batch(items)
    return float(np.mean(preds))


# ── 데이터셋 전체 배치 추론 — TEMPO score_dataset 방식과 동일 ────────────────
def classify_dataset(dataset: list[dict], prompt: str) -> list[str]:
    """모든 영상의 프레임을 한 번에 배치 추론 → 영상별 normal/abnormal 반환.
    TEMPO score_dataset 처럼 전체 프레임을 하나의 배치로 묶어 처리.
    """
    if not dataset:
        return []

    all_items          = []
    video_frame_counts = []
    for ex in dataset:
        all_items.extend([(load_pil(fp), prompt) for fp in ex["frame_paths"]])
        video_frame_counts.append(len(ex["frame_paths"]))

    all_preds = predict_vision_batch(all_items)

    results, idx = [], 0
    for count in video_frame_counts:
        frame_preds = all_preds[idx:idx + count]
        scam_ratio  = sum(p >= FRAME_SCORE_THRESHOLD for p in frame_preds) / count if count else 0
        results.append("abnormal" if scam_ratio >= SCAM_RATIO_THRESHOLD else "normal")
        idx += count
    return results


def score_dataset(dataset: list[dict], prompt: str) -> list[float]:
    """모든 영상의 프레임을 한 번에 배치 추론 → 영상별 P(abnormal) 반환 (AUC/MAP용)."""
    if not dataset:
        return []

    all_items          = []
    video_frame_counts = []
    for ex in dataset:
        all_items.extend([(load_pil(fp), prompt) for fp in ex["frame_paths"]])
        video_frame_counts.append(len(ex["frame_paths"]))

    all_preds = predict_vision_batch(all_items)

    results, idx = [], 0
    for count in video_frame_counts:
        frame_preds = all_preds[idx:idx + count]
        results.append(float(np.mean(frame_preds)) if frame_preds else 0.5)
        idx += count
    return results
