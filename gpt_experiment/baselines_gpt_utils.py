"""
baselines_gpt_utils.py — baselines/vllm_utils.py(EvoPrompt/APO 공용 추론 유틸)의 GPT 백엔드 버전.

Qwen2.5-VL(_llm/_processor, vLLM 로컬 추론)이 맡던 역할을 gpt_backend.py(OpenAI)로
교체했다. 프레임 추출(extract_frames_1fps, 실제로는 2초 간격)과 영상 레벨 scam_ratio
집계 로직은 원본과 동일.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

from gpt_backend import predict_gpt_batch, text_generate_n, vision_feedback_generate

# TEMPO와 동일한 상수
MAX_SEC               = 150   # 최대 프레임 추출 초
FRAME_SCORE_THRESHOLD = 0.5   # 프레임 레벨 abnormal 임계값
SCAM_RATIO_THRESHOLD  = 0.3   # 영상 레벨 scam_ratio 임계값 (TEMPO 기본값)

SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."


# ── 프레임 추출 — TEMPO와 동일 (2초 간격, 448x448, JPEG quality=85) ──────────
def extract_frames_1fps(video_path: Path, frames_dir: Path,
                        max_sec: int = MAX_SEC, step: int = 2) -> list[str]:
    """2초 간격으로 프레임을 JPEG 캐시에 저장하고 경로 리스트 반환."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    stem  = video_path.stem
    cap   = cv2.VideoCapture(str(video_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sec = min(int(total / fps_v), max_sec)
    cap.release()

    seconds = list(range(0, n_sec, step))

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


# ── 비전 배치 추론 ────────────────────────────────────────────────────────────
def predict_vision_batch(items: list[tuple[Image.Image, str]]) -> list[float]:
    """(PIL Image, prompt) 리스트 → float 예측값 리스트."""
    preds = predict_gpt_batch(items, score_suffix=SCORE_SUFFIX)
    return [p if p is not None else 0.5 for p in preds]


# ── 텍스트 전용 추론 — GA/DE crossover, APO gradient 생성용 ──────────────────
def predict_text(prompt: str, n: int = 1,
                 temperature: float = 0.7, max_tokens: int = 512) -> list[str]:
    """텍스트 전용 LLM 호출, n개 completions 반환."""
    return text_generate_n(prompt, n=n, temperature=temperature, max_tokens=max_tokens)


# ── 비전+텍스트 혼합 추론 — APO gradient 생성 시 이미지 포함 ─────────────────
def predict_vision_text(text_prompt: str, image_paths: list[str],
                        temperature: float = 0.7,
                        max_tokens: int = 600) -> str:
    """텍스트 프롬프트 + 이미지들 → 텍스트 생성 (APO gradient 호출용)."""
    imgs = [load_pil(fp) for fp in image_paths[:4]]
    return vision_feedback_generate(text_prompt, imgs, temperature=temperature, max_tokens=max_tokens)


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


# ── 데이터셋 전체 배치 추론 ───────────────────────────────────────────────────
def classify_dataset(dataset: list[dict], prompt: str) -> list[str]:
    """모든 영상의 프레임을 한 번에 배치 추론 → 영상별 normal/abnormal 반환."""
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
