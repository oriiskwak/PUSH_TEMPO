"""
gpt_backend.py — Qwen2.5-VL(_llm/_processor) 대체용 OpenAI 백엔드.

Qwen이 맡던 두 역할을 그대로 GPT로 옮긴다:
  1. predict_gpt_batch        — 프레임별 scam 확률 예측 (predictor)
  2. text_generate            — 텍스트 전용 생성 (evolve_feedback / generate_prompt_candidates / crossover_prompts)
  3. vision_feedback_generate — 에러 프레임 기반 피드백 생성 (generate_feedback)

vision_feedback_generate는 Qwen 쪽의 grid-image 트릭(_make_grid)을 쓰지 않고 프레임을
개별 image_url로 전송한다 — grid는 vLLM의 MM_LIMIT=1(prompt당 이미지 1장) 제약을 피하기
위한 워크어라운드였고 OpenAI API에는 해당 제약이 없다.

GPT_MODEL 환경변수로 모델 교체 가능 (기본 gpt-4o-mini).
"""

import base64
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from PIL import Image

GPT_MODEL   = os.environ.get("GPT_MODEL", "gpt-4o-mini")
MAX_WORKERS = int(os.environ.get("GPT_WORKERS", "16"))
IMAGE_DETAIL = os.environ.get("GPT_IMAGE_DETAIL", "low")
MAX_RETRIES  = 3

_client = OpenAI()


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_block(img: Image.Image) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(img)}", "detail": IMAGE_DETAIL},
    }


def _chat(content, max_tokens: int, temperature: float) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = _client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  [Warning] GPT 호출 실패: {e}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


# ── 1. Predictor: 프레임별 scam 확률 ─────────────────────────────────────────

def _score_one(
    img: Image.Image, prompt: str,
    max_tokens: int = 10, use_last_match: bool = False,
) -> float | None:
    content = [{"type": "text", "text": prompt}, _image_block(img)]
    raw = _chat(content, max_tokens=max_tokens, temperature=0.0)
    if raw is None:
        return None
    matches = re.findall(r"[0-9]*\.?[0-9]+", raw)
    if not matches:
        return None
    val = matches[-1] if use_last_match else matches[0]
    return max(0.0, min(1.0, float(val)))


def predict_gpt_batch(
    items: list[tuple[Image.Image, str]], score_suffix: str = "",
    max_tokens: int = 10, use_last_match: bool = False,
) -> list[float | None]:
    """(PIL Image, prompt) 리스트 → 프레임별 float 스코어 리스트.

    vLLM의 로컬 배치 대신 ThreadPoolExecutor로 동시 API 호출 (MAX_WORKERS개씩).
    max_tokens/use_last_match: CoT처럼 추론 텍스트 뒤에 최종 숫자가 오는 경우
    max_tokens를 늘리고 use_last_match=True로 마지막 숫자를 답으로 사용.
    """
    results: list[float | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(_score_one, img, prompt + score_suffix, max_tokens, use_last_match): i
            for i, (img, prompt) in enumerate(items)
        }
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    return results


# ── 2. Optimizer: 텍스트 전용 생성 ───────────────────────────────────────────

def text_generate(prompt: str, temperature: float = 0.7, max_tokens: int = 400) -> str:
    return _chat(prompt, max_tokens=max_tokens, temperature=temperature) or ""


def text_generate_n(
    prompt: str, n: int = 1, temperature: float = 0.7, max_tokens: int = 400,
) -> list[str]:
    """n개의 독립적인 completion 반환 (APO synonym 생성 등 multi-sample 용도)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = _client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
                n=n,
            )
            return [(c.message.content or "").strip() for c in resp.choices]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  [Warning] GPT 호출 실패: {e}")
                return [""] * n
            time.sleep(1.5 * (attempt + 1))
    return [""] * n


# ── 3. Optimizer: 에러 프레임 기반 피드백 생성 (vision) ──────────────────────

def vision_feedback_generate(
    prompt: str, images: list[Image.Image],
    temperature: float = 0.7, max_tokens: int = 600,
) -> str:
    content = [{"type": "text", "text": prompt}] + [_image_block(img) for img in images[:4]]
    return _chat(content, max_tokens=max_tokens, temperature=temperature) or ""
