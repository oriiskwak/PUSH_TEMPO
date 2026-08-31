"""
tools.py — 추론 보강 도구 모음 (PANDA 스타일 외부 툴 주입)

Tool 1 — get_text_context()   : text_meta.json에서 title/ASR/OCR 조합
Tool 2 — rag_retrieve()       : scam_patterns.json에서 유사 패턴 top-K 검색
Tool 3 — web_search_tool()    : 영상 제목으로 DuckDuckGo 검색 (디스크 캐시)
Tool 4 — get_zoom_crops()     : text_meta_bbox.json의 EasyOCR bbox 기반 crop
         make_augmented_grid(): 원본 + crops → 1행 그리드 이미지
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

_BASE_DIR          = Path(__file__).parent
_TEXT_META_PATH    = _BASE_DIR / "data" / "text_meta.json"
_TEXT_META_BBOX    = _BASE_DIR / "data" / "text_meta_bbox.json"
_SCAM_PATTERNS     = _BASE_DIR / "data" / "scam_patterns.json"
_WEB_CACHE_PATH    = _BASE_DIR / "data" / "web_search_cache.json"

# lazy-loaded 전역 자원
_text_meta: dict | None = None
_text_meta_bbox: dict | None = None
_scam_patterns: list | None = None
_scam_embs = None
_embedder  = None
_web_cache: dict | None = None


# ── 내부 로더 ─────────────────────────────────────────────────────────────────

def _load_text_meta() -> dict:
    global _text_meta
    if _text_meta is None:
        _text_meta = json.load(open(_TEXT_META_PATH, encoding="utf-8")) if _TEXT_META_PATH.exists() else {}
    return _text_meta


def _load_text_meta_bbox() -> dict:
    global _text_meta_bbox
    if _text_meta_bbox is None:
        _text_meta_bbox = json.load(open(_TEXT_META_BBOX, encoding="utf-8")) if _TEXT_META_BBOX.exists() else {}
    return _text_meta_bbox


def _load_scam_patterns() -> list:
    global _scam_patterns
    if _scam_patterns is None:
        _scam_patterns = json.load(open(_SCAM_PATTERNS, encoding="utf-8")) if _SCAM_PATTERNS.exists() else []
    return _scam_patterns


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        # 가벼운 다국어 모델 — em_core의 bge-m3와 별도 로드 않도록 같은 모델 사용
        _embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
    return _embedder


def _get_scam_embs():
    global _scam_embs
    if _scam_embs is None:
        patterns = _load_scam_patterns()
        if not patterns:
            return None
        texts = [
            p["type"] + " " + p["description"] + " " + " ".join(p.get("text_cues", []))
            for p in patterns
        ]
        _scam_embs = _get_embedder().encode(texts, normalize_embeddings=True)
    return _scam_embs


def _load_web_cache() -> dict:
    global _web_cache
    if _web_cache is None:
        _web_cache = json.load(open(_WEB_CACHE_PATH, encoding="utf-8")) if _WEB_CACHE_PATH.exists() else {}
    return _web_cache


def _save_web_cache():
    if _web_cache is not None:
        json.dump(_web_cache, open(_WEB_CACHE_PATH, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)


# ── Tool 1: 텍스트 컨텍스트 ───────────────────────────────────────────────────

def get_text_context(video_name: str) -> str:
    """title + ASR + OCR을 하나의 컨텍스트 문자열로 반환."""
    meta  = _load_text_meta().get(video_name, {})
    title = meta.get("title", "")[:150]
    asr   = meta.get("asr",   "")[:300]
    ocr   = " | ".join(meta.get("ocr", []))[:200]

    parts = []
    if title: parts.append(f"Title: {title}")
    if asr:   parts.append(f"Speech (ASR): {asr}")
    if ocr:   parts.append(f"Screen text (OCR): {ocr}")
    return "\n".join(parts)


# ── Tool 2: RAG — 사기 패턴 검색 ──────────────────────────────────────────────

def rag_retrieve(video_name: str, top_k: int = 2) -> str:
    """OCR + ASR + title 기반으로 scam_patterns.json에서 유사 패턴 top-K 반환."""
    patterns = _load_scam_patterns()
    if not patterns:
        return ""

    meta = _load_text_meta().get(video_name, {})
    query = (
        meta.get("title", "") + " " +
        " ".join(meta.get("ocr", []))[:150]
    ).strip()
    if not query:
        return ""

    embs    = _get_scam_embs()
    if embs is None:
        return ""

    q_emb   = _get_embedder().encode([query], normalize_embeddings=True)[0]
    sims    = embs @ q_emb
    top_idx = np.argsort(sims)[-top_k:][::-1]

    lines = []
    for i in top_idx:
        p = patterns[i]
        lines.append(
            f"[{p['type']}] {p['description']}\n"
            f"  Visual cues: {', '.join(p.get('visual_cues', []))}\n"
            f"  Text cues: {', '.join(p.get('text_cues', []))}"
        )
    return "\n\n".join(lines)


# ── Tool 3: 웹 검색 (디스크 캐시) ─────────────────────────────────────────────

def _extract_keywords(meta: dict, max_words: int = 10) -> str:
    """title + ASR + OCR에서 핵심 키워드 추출 (중복 제거, 짧은 단어 필터)."""
    text = (
        meta.get("title", "") + " " +
        " ".join(meta.get("ocr", []))[:200]
    )
    seen, keywords = set(), []
    for word in text.split():
        word = word.strip(".,!?[]()「」『』")
        if len(word) >= 2 and word not in seen:
            seen.add(word)
            keywords.append(word)
        if len(keywords) >= max_words:
            break
    return " ".join(keywords)


def web_search_tool(video_name: str, max_results: int = 3) -> str:
    """메타데이터 키워드로 DuckDuckGo 검색 → 해당 키워드의 사기 특징 반환.

    단순 피해 사례가 아닌 키워드 기반 사기 수법·특징을 검색해
    모델에게 더 일반화된 패턴 힌트를 제공한다.
    """
    meta = _load_text_meta().get(video_name, {})
    keywords = _extract_keywords(meta)
    if not keywords:
        return ""

    cache = _load_web_cache()
    key   = keywords[:120]
    if key in cache:
        return cache[key]

    try:
        from duckduckgo_search import DDGS
        query = f"{keywords} 사기 수법 특징 주의사항"
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        snippets = [r.get("body", "")[:150] for r in results if r.get("body")]
        result_str = "\n".join(f"- {s}" for s in snippets)
    except Exception:
        result_str = ""

    cache[key] = result_str
    _save_web_cache()
    return result_str


def batch_web_search(video_names: list[str]) -> dict[str, str]:
    """여러 영상의 웹 검색 결과를 일괄 수집 (캐시 미스만 실제 검색)."""
    result = {}
    for vname in video_names:
        result[vname] = web_search_tool(vname)
    return result


# ── Tool 4: EasyOCR bbox 기반 zoom crop ─────────────────────────────────────

_easyocr_reader = None

def _get_ocr_reader():
    """EasyOCR Reader lazy 초기화 (on-the-fly 사용 시)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["ko", "en"], gpu=True, verbose=False)
    return _easyocr_reader


def _boxes_from_frame(frame_img: PILImage.Image, conf_thresh: float = 0.5) -> list[dict]:
    """EasyOCR on-the-fly bbox 추출."""
    import numpy as _np
    reader  = _get_ocr_reader()
    results = reader.readtext(_np.array(frame_img))
    return [
        {"bbox": [[int(p[0]), int(p[1])] for p in bbox], "conf": float(conf)}
        for bbox, text, conf in results
        if conf >= conf_thresh and text.strip()
    ]


def get_zoom_crops(video_name: str, second: int,
                   frame_img: PILImage.Image,
                   max_crops: int = 2,
                   min_area_ratio: float = 0.015) -> list[PILImage.Image]:
    """텍스트 밀집 영역 crop 반환.

    1) text_meta_bbox.json 캐시 우선 사용
    2) 캐시 없으면 EasyOCR on-the-fly 실행 (새로운 영상도 자동 지원)
    """
    bbox_data   = _load_text_meta_bbox().get(video_name, {})
    frame_boxes = bbox_data.get(str(second), [])

    if not frame_boxes:
        # 캐시 없으면 on-the-fly
        try:
            frame_boxes = _boxes_from_frame(frame_img)
        except Exception:
            return []

    if not frame_boxes:
        return []

    W, H     = frame_img.size
    min_area = W * H * min_area_ratio
    crops    = []

    for item in sorted(frame_boxes, key=lambda x: -x.get("conf", 0)):
        pts = item["bbox"]
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        x1  = max(0, int(min(xs)) - 20)
        y1  = max(0, int(min(ys)) - 20)
        x2  = min(W, int(max(xs)) + 20)
        y2  = min(H, int(max(ys)) + 20)

        if (x2 - x1) * (y2 - y1) < min_area:
            continue

        crops.append(frame_img.crop((x1, y1, x2, y2)))
        if len(crops) >= max_crops:
            break

    return crops


def make_augmented_grid(frame_img: PILImage.Image,
                         crops: list[PILImage.Image],
                         cell_size: int = 336) -> PILImage.Image:
    """원본 프레임 + zoom crops를 1행 그리드로 합성 (vLLM 1-image 제한 우회)."""
    all_imgs = [frame_img] + crops
    n        = len(all_imgs)
    grid     = PILImage.new("RGB", (cell_size * n, cell_size))
    for i, img in enumerate(all_imgs):
        grid.paste(img.resize((cell_size, cell_size), PILImage.LANCZOS), (i * cell_size, 0))
    return grid


# ── 프롬프트 보강 헬퍼 ────────────────────────────────────────────────────────

def inject_text_context(video_name: str, base_prompt: str) -> str:
    """Tool: text_context — title + ASR + OCR만 주입 (하위 호환용)."""
    text_ctx = get_text_context(video_name)
    if not text_ctx:
        return base_prompt
    return base_prompt + f"\n[Available Context]\n{text_ctx}"


def inject_title(video_name: str, base_prompt: str) -> str:
    """Tool: title — 영상 제목만 주입."""
    meta  = _load_text_meta().get(video_name, {})
    title = meta.get("title", "")[:150]
    if not title:
        return base_prompt
    return base_prompt + f"\n[Available Context]\nTitle: {title}"


def inject_asr(video_name: str, base_prompt: str) -> str:
    """Tool: asr — ASR(음성 인식) + OCR 텍스트만 주입."""
    meta  = _load_text_meta().get(video_name, {})
    asr   = meta.get("asr", "")[:300]
    ocr   = " | ".join(meta.get("ocr", []))[:200]
    parts = []
    if asr: parts.append(f"Speech (ASR): {asr}")
    if ocr: parts.append(f"Screen text (OCR): {ocr}")
    if not parts:
        return base_prompt
    return base_prompt + "\n[Available Context]\n" + "\n".join(parts)


def inject_channel(video_name: str, base_prompt: str) -> str:
    """Tool: channel — 채널명 + 구독자 수 주입."""
    meta    = _load_text_meta().get(video_name, {})
    channel = meta.get("channel")
    sub     = meta.get("subscriber_count")
    if not channel:
        return base_prompt
    sub_str = f"{sub:,}" if sub else "unknown"
    return base_prompt + f"\n[Available Context]\nChannel: {channel} (subscribers: {sub_str})"


def inject_rag(video_name: str, base_prompt: str) -> str:
    """Tool: rag — 유사 사기 패턴만 주입."""
    rag_ctx = rag_retrieve(video_name)
    if not rag_ctx:
        return base_prompt
    return base_prompt + f"\n[Matching Scam Patterns for Reference]\n{rag_ctx}"


def inject_web_search(video_name: str, base_prompt: str, web_results: str) -> str:
    """Tool: web_search — 웹 검색 결과만 주입."""
    if not web_results:
        return base_prompt
    return base_prompt + f"\n[Web Search Results]\n{web_results}"


def inject_description(video_name: str, base_prompt: str, max_chars: int = 300) -> str:
    """Tool: description — YouTube 영상 설명(description) 주입.
    data/fetch_description.py 실행 후 text_meta.json에 description 필드가 있어야 함.
    """
    meta = _load_text_meta().get(video_name, {})
    desc = (meta.get("description") or "").strip()[:max_chars]
    if not desc:
        return base_prompt
    return base_prompt + f"\n[Video Description]\n{desc}"


def build_augmented_prompt(video_name: str, base_prompt: str,
                            web_results: str = "") -> str:
    """전체 주입 (하위 호환용)."""
    prompt = inject_text_context(video_name, base_prompt)
    prompt = inject_rag(video_name, prompt)
    prompt = inject_web_search(video_name, prompt, web_results)
    return prompt
