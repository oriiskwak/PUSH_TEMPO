"""
em_core.py — UniAPO EM 알고리즘 핵심 구성요소

E-step:
  LF (피드백 모델): 오류 샘플 분석 → 단기 피드백 F_short 생성
  피드백 메모리 M_F: F_short + F_long → Evolver(LE)가 통합 → F^(t+1)

M-step:
  LP (프롬프트 최적화): 피드백 기반 후보 N개 생성
  프롬프트 메모리 M_P: 역사적 상위 K개 P_long → Evolver(LE) Crossover → P^(t+1)
"""

import re, json, os
from datetime import datetime
import numpy as np
from pathlib import Path
from vllm import SamplingParams
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

# em_core_tool_seedrun.py — em_core_tool.py(원본, seed 없음)의 재현성용 복제본.
# 원본엔 SamplingParams에 seed가 아예 없어서 매 실행마다 자동으로 달라지는 구조였음.
# em_core_tool2.py와 동일한 방식으로 LLM_SEED 환경변수를 추가.
_LLM_SEED = int(os.environ.get("LLM_SEED", "42"))

embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")

# 영상별 텍스트 메타 (preprocess_text.py 실행 후 생성)
_TEXT_META_PATH = Path(__file__).parent / "data" / "text_meta.json"
_text_meta: dict = json.load(open(_TEXT_META_PATH, encoding="utf-8")) if _TEXT_META_PATH.exists() else {}
if not _text_meta:
    print("[em_core] text_meta.json 없음 — 수치 클러스터링으로 폴백")

# ── EM 로거 ───────────────────────────────────────────────────────────────────

_LOG_DIR = Path(__file__).parent / "results" / "em_logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

class EMLogger:
    """이터레이션별 클러스터링·피드백·프롬프트 변화를 JSONL로 기록."""

    def __init__(self, run_name: str):
        ts   = datetime.now().strftime("%m%d_%H%M")
        self.path = _LOG_DIR / f"{run_name}_{ts}.jsonl"
        self._f   = open(self.path, "w", encoding="utf-8")
        print(f"[EMLogger] 로그 저장: {self.path.name}")

    def _write(self, record: dict):
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def log_cluster(self, em_name: str, iteration: int,
                    cluster_id: int, items: list[dict], text: str,
                    vid_stats: dict | None = None):
        """클러스터 구성 — 어떤 영상들이 묶였는지, 어떤 텍스트 기반인지."""
        videos = list({r["video"] for r in items})

        # 영상별 pass-1 통계 (mean/max/std) — run_c의 vid_stats에서 가져옴
        stats_summary = {}
        if vid_stats:
            for v in videos:
                s = vid_stats.get(v)
                if s:
                    stats_summary[v] = {
                        "mean": round(s["mean"], 3),
                        "max":  round(s["max"],  3),
                        "std":  round(s["std"],  3),
                    }

        self._write({
            "type":      "cluster",
            "em":        em_name,
            "iter":      iteration,
            "cluster":   cluster_id,
            "size":      len(items),
            "videos":    videos,
            "error_pattern": {
                "fn": sum(1 for r in items if r["gt"] > 0.5 and r["pred"] <= 0.5),
                "fp": sum(1 for r in items if r["gt"] <= 0.5 and r["pred"] > 0.5),
            },
            "pred_mean":    round(float(np.mean([r["pred"] for r in items])), 3),
            "gt_mean":      round(float(np.mean([r["gt"]   for r in items])), 3),
            "vid_stats":    stats_summary,       # pass-1 mean/max/std per video
            "embed_text":   text[:300],          # 클러스터링에 사용된 원본 텍스트
        })

    def log_feedback(self, em_name: str, iteration: int,
                     cluster_id: int, feedback: str,
                     tool: str | None = None, error_type: str = ""):
        """클러스터별 생성된 피드백."""
        self._write({
            "type":       "feedback",
            "em":         em_name,
            "iter":       iteration,
            "cluster":    cluster_id,
            "error_type": error_type,
            "tool":       tool,
            "feedback":   feedback,
        })

    def log_iteration(self, em_name: str, iteration: int,
                      train_f1: float, dev_f1: float,
                      n_errors: int, n_clusters: int,
                      best_prompt: str):
        """이터레이션 요약."""
        self._write({
            "type":       "iteration",
            "em":         em_name,
            "iter":       iteration,
            "train_f1":   round(train_f1, 4) if train_f1 else None,
            "dev_f1":     round(dev_f1,   4) if dev_f1   else None,
            "n_errors":   n_errors,
            "n_clusters": n_clusters,
            "prompt_preview": best_prompt[:300],
        })

    def close(self):
        self._f.close()


# 전역 로거 (run_*.py에서 run_em 호출 시 초기화)
_logger: EMLogger | None = None

def init_logger(run_name: str):
    global _logger
    _logger = EMLogger(run_name)

def get_logger() -> EMLogger | None:
    return _logger


def _text_generate(prompt: str, temperature: float = 0.7, max_tokens: int = 400) -> str:
    """Qwen2.5-VL 텍스트 전용 생성 (이미지 없음).
    프롬프트가 길면 앞쪽 컨텍스트를 잘라 max_model_len 초과 방지.
    """
    from utils import _llm, _processor
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 토큰 수 확인 후 초과 시 프롬프트 앞부분 truncate
    token_ids = _processor.tokenizer.encode(text)
    from utils import _llm as llm_inst
    max_len = llm_inst.llm_engine.model_config.max_model_len
    budget  = max_len - max_tokens - 50   # 여유 50토큰
    if len(token_ids) > budget:
        token_ids = token_ids[-budget:]   # 앞 자르고 최근 컨텍스트 유지
        text = _processor.tokenizer.decode(token_ids, skip_special_tokens=False)

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=_LLM_SEED)
    outputs = _llm.generate([{"prompt": text}], sp, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


# ── 메모리 클래스 ──────────────────────────────────────────────────────────────

VALID_TOOLS = {"title", "channel", "web_search", "rag"}  # asr 제외: 정상 영상 ASR이 스캠처럼 보여 방향 역전 유발


def get_video_query(vname: str) -> str:
    """영상 제목을 툴 검색 쿼리로 반환. text_meta 없으면 vname 사용."""
    title = (_text_meta.get(vname, {}).get("title") or "").strip()
    return title if title else vname


class FeedbackMemory:
    """피드백 메모리 M_F: 과거 피드백과 툴 추천을 저장하고 유사도 기반으로 검색."""

    def __init__(self, top_k: int = 3):
        self.memory: list[dict] = []   # {"feedback": str, "tools": list[str]}
        self.top_k = top_k

    def add(self, feedback: str, tools: list[str] | None = None):
        self.memory.append({"feedback": feedback, "tools": tools or []})
        lg = get_logger()
        if lg:
            lg._write({
                "type":        "memory_add",
                "feedback":    feedback[:300],
                "tools":       tools or [],
                "memory_size": len(self.memory),
            })

    def retrieve(self, query: str) -> list[str]:
        """피드백 텍스트 목록만 반환."""
        if not self.memory:
            return []
        if len(self.memory) <= self.top_k:
            return [m["feedback"] for m in self.memory]
        texts  = [m["feedback"] for m in self.memory]
        q_emb  = embedder.encode([query], normalize_embeddings=True)[0]
        m_embs = embedder.encode(texts, normalize_embeddings=True)
        sims   = m_embs @ q_emb
        top    = np.argsort(sims)[-self.top_k:][::-1]
        return [self.memory[i]["feedback"] for i in top]

    def retrieve_tools(self, query: str, sim_threshold: float = 0.5) -> list[str]:
        """유사 오류 패턴에서 툴 목록 반환 — top-k 피드백의 툴 합집합."""
        if not self.memory:
            return []
        texts  = [m["feedback"] for m in self.memory]
        q_emb  = embedder.encode([query], normalize_embeddings=True)[0]
        m_embs = embedder.encode(texts, normalize_embeddings=True)
        sims   = m_embs @ q_emb
        top    = np.argsort(sims)[-self.top_k:][::-1]
        tools: set[str] = set()
        for i in top:
            if sims[i] >= sim_threshold:
                tools.update(self.memory[i].get("tools", []))
        return list(tools)


class PromptMemory:
    """프롬프트 메모리 M_P: (prompt, dev_f1) 쌍을 F1 내림차순으로 유지."""

    def __init__(self, beam_size: int = 3, top_k: int = 2):
        self.memory: list[tuple[str, float]] = []
        self.beam_size = beam_size
        self.top_k = top_k

    def add(self, prompt: str, dev_f1: float):
        self.memory.append((prompt, dev_f1))
        self.memory.sort(key=lambda x: x[1], reverse=True)  # F1 높을수록 좋음

    def get_top_k(self) -> list[str]:
        return [p for p, _ in self.memory[:self.top_k]]

    def get_beam(self) -> list[str]:
        return [p for p, _ in self.memory[:self.beam_size]]

    def best(self) -> tuple[str, float]:
        return self.memory[0] if self.memory else ("", 0.0)


# ── E-step ────────────────────────────────────────────────────────────────────

def _dedup_error_items(error_items: list[dict], threshold: float = 0.06) -> list[dict]:
    """같은 영상 내 시각적으로 유사한 프레임 제거 → 대표 프레임만 유지.

    픽셀 차이(64×64)가 threshold 미만이면 같은 프레임으로 간주.
    글씨/화면 구성이 다른 프레임은 별도 항목으로 유지.
    """
    import sys
    _u = sys.modules.get("utils2") or sys.modules.get("utils")
    if _u is None:
        try:
            from utils import load_frame, VIDEO_DIR
        except ImportError:
            return error_items
    else:
        load_frame = _u.load_frame
        VIDEO_DIR  = _u.VIDEO_DIR

    from collections import defaultdict
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in error_items:
        by_video[r["video"]].append(r)

    result = []
    for vname, items in by_video.items():
        vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
        if not vfiles:
            result.extend(items)
            continue

        selected, selected_arrs = [], []
        for r in items:
            img = load_frame(vfiles[0], r["second"])
            if img is None:
                if not selected:
                    selected.append(r)
                continue
            arr = np.array(img.resize((64, 64))).astype(float) / 255.0
            if not selected_arrs or all(
                np.mean(np.abs(arr - s)) > threshold for s in selected_arrs
            ):
                selected.append(r)
                selected_arrs.append(arr)
        result.extend(selected)
    return result


def cluster_errors(error_items: list[dict], eps: float = 0.35,
                   em_name: str = "", iteration: int = 0,
                   vid_stats: dict | None = None) -> list[list[dict]]:
    """오류 샘플 클러스터링.

    text_meta.json 있으면: 영상 텍스트(ASR+OCR) 임베딩 기반 의미론적 클러스터링
    없으면: pred/gt 수치 기반 클러스터링 (폴백)
    """
    if not error_items:
        return []

    error_items = _dedup_error_items(error_items)
    if not error_items:
        return []

    if _text_meta:
        texts = []
        for r in error_items:
            meta = _text_meta.get(r["video"], {"ocr": []})
            combined = (
                meta.get("title", "") + " " +
                " ".join(meta.get("ocr", []))
            ).strip()
            texts.append(combined if combined else r["video"])

        embs   = embedder.encode(texts, normalize_embeddings=True)
        labels = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(embs)
    else:
        features = np.array([[r["pred"], r["gt"]] for r in error_items])
        mean, std = features.mean(axis=0), features.std(axis=0) + 1e-8
        features  = (features - mean) / std
        labels    = DBSCAN(eps=eps, min_samples=1, metric="euclidean").fit_predict(features)
        texts     = [r["video"] for r in error_items]

    clusters: dict[int, list] = {}
    for idx, lb in enumerate(labels):
        clusters.setdefault(lb, []).append(error_items[idx])

    # 로그: 클러스터별 구성 기록
    lg = get_logger()
    if lg:
        for cid, items in enumerate(clusters.values()):
            idxs = [error_items.index(item) for item in items]
            cluster_text = texts[idxs[0]] if idxs else ""
            lg.log_cluster(em_name, iteration, cid, items, cluster_text, vid_stats)

    return list(clusters.values())


def _dedup_frames(images: list, threshold: float = 0.06) -> list:
    """64×64 픽셀 차이로 유사 프레임 제거 → 대표 프레임만 반환."""
    if not images:
        return images
    selected, selected_arrs = [], []
    for img in images:
        arr = np.array(img.resize((64, 64))).astype(float) / 255.0
        if not selected_arrs or all(
            np.mean(np.abs(arr - s)) > threshold for s in selected_arrs
        ):
            selected.append(img)
            selected_arrs.append(arr)
    return selected


def _make_grid(images: list, cell_size: int = 336, max_images: int = 4) -> "Image":
    """여러 프레임을 정사각형 그리드 1장으로 만든다 (aspect ratio < 200 보장)."""
    import math
    from PIL import Image as PILImage
    images = images[:max_images]
    n    = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = PILImage.new("RGB", (cell_size * cols, cell_size * rows))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img.resize((cell_size, cell_size), PILImage.LANCZOS), (c * cell_size, r * cell_size))
    return grid


def _vision_feedback_generate(prompt: str, images: list,
                               temperature: float = 0.7, max_tokens: int = 600) -> str:
    """피드백 생성용 멀티모달 생성 — 여러 프레임을 그리드 1장으로 합쳐 전송."""
    from utils import _llm, _processor

    grid_img = _make_grid(images)

    # apply_chat_template 전에 텍스트 자르기 (토큰 인코딩/디코딩 불필요)
    max_len            = _llm.llm_engine.model_config.max_model_len
    budget             = max_len - max_tokens - 50
    IMAGE_TOKEN_BUDGET = 4000  # 2×2 grid (672×672px) ≈ 576토큰, 여유 포함
    max_text_chars     = max(100, int((budget - IMAGE_TOKEN_BUDGET) * 1.5))  # 한국어 1.5 chars/token
    if len(prompt) > max_text_chars:
        prompt = prompt[-max_text_chars:]

    content  = [{"type": "image", "image": grid_img},
                {"type": "text",  "text":  prompt}]
    messages = [{"role": "user", "content": content}]
    text     = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sp      = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=_LLM_SEED)
    mm_data = {"image": [grid_img]}
    outputs = _llm.generate([{"prompt": text, "multi_modal_data": mm_data}], sp, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


def generate_feedback(current_prompt: str, chunk: list[dict],
                       past_feedbacks: list[str],
                       em_name: str = "", iteration: int = 0,
                       cluster_id: int = 0,
                       error_type: str = "fn") -> tuple[str, str | None]:
    """LF 역할: 오류 청크 분석 → 단기 피드백 F_short 생성.

    error_type: "fn" — 사기 영상을 정상으로 예측 (탐지 강화 필요)
                "fp" — 정상 영상을 사기로 예측 (오탐 감소 필요)
    """
    import sys
    from pathlib import Path
    _u = sys.modules.get("utils2") or sys.modules.get("utils")
    if _u is None:
        from utils import load_frame, VIDEO_DIR
    else:
        load_frame = _u.load_frame
        VIDEO_DIR  = _u.VIDEO_DIR

    error_lines = []
    frame_images = []

    for r in chunk:
        meta  = _text_meta.get(r["video"], {})
        title   = meta.get("title", "")[:120]
        asr     = meta.get("asr",   "")[:200]
        ocr     = " ".join(meta.get("ocr", []))[:120]
        channel = (meta.get("channel") or "")[:80]
        subs    = meta.get("subscriber_count") or ""
        label   = r.get("label", "unknown")

        error_lines.append(
            f"- label={label}  pred={r['pred']:.2f}  gt={r['gt']:.2f}  kl={r['kl']:.3f}\n"
            f"  title:   {title}\n"
            f"  asr:     {asr}\n"
            f"  ocr:     {ocr}\n"
            f"  channel: {channel}  subscribers: {subs}"
        )

        vfiles = list(VIDEO_DIR.glob(f"{r['video']}.*"))
        if vfiles:
            img = load_frame(vfiles[0], r["second"])
            if img is not None:
                frame_images.append(img)

    error_str = "\n".join(error_lines)
    past_str  = "\n".join(f"[Past {i+1}] {f}" for i, f in enumerate(past_feedbacks)) or "None"

    if error_type == "fp":
        error_desc   = "FALSE POSITIVE — The model predicted these frames as SCAM, but they are NORMAL videos."
        objective    = (
            "- Identify what visual/textual patterns are causing the model to OVER-detect scam content\n"
            "- Suggest how to DISTINGUISH these legitimate patterns from actual scams\n"
            "- The goal is to REDUCE false alarms on this type of content"
        )
        allowed_tools = VALID_TOOLS
        tools_desc    = (
            "- title      : video title to understand content context (news/educational vs scam)\n"
            "- channel    : channel name and subscriber count to verify source credibility\n"
            "- web_search : external verification of the channel or claims in the video\n"
            "- rag        : match against known scam pattern database"
        )
    else:
        error_desc   = "FALSE NEGATIVE — The model predicted these frames as NORMAL, but they are SCAM videos."
        objective    = (
            "- Identify what scam signals the current prompt is MISSING in these frames\n"
            "- Suggest specific visual cues or instructions to better detect this type of scam\n"
            "- The goal is to INCREASE detection sensitivity for this pattern"
        )
        allowed_tools = VALID_TOOLS
        tools_desc    = (
            "- title      : video title needed to assess content context\n"
            "- channel    : channel name and subscriber count to verify source credibility\n"
            "- web_search : need external verification of claims in the video\n"
            "- rag        : need to match against known scam pattern database"
        )

    user = f"""# Task Overview
You are analyzing why a cyber-scam video frame detection prompt produces errors.

# Error Type
{error_desc}

# Current Prompt
{current_prompt}

# Past Feedbacks (from previous iterations)
{past_str}

# Error Cases
{error_str}

# Objective
Generate concise feedback that:
{objective}
- Does NOT overfit to only these specific frames
- Does NOT include specific Korean words, dates, numbers, or phrases from the error examples
- Describes ABSTRACT, GENERALIZABLE visual patterns (e.g., "financial charts with return claims" not "분할매수")

# Available Tools
{tools_desc}

# Output Format
First, judge whether improving the visual detection instructions alone can fix this error (without any tool).
Only recommend a tool if visual prompt refinement is clearly insufficient.

<policy>
[feedback content]
</policy>
<tools>[up to 2 comma-separated tool names, or "none"]</tools>"""

    if frame_images:
        text = _vision_feedback_generate(user, frame_images, temperature=0.7, max_tokens=800)
    else:
        text = _text_generate(user, temperature=0.7, max_tokens=800)

    m        = re.search(r"<policy>(.*?)</policy>", text, re.DOTALL)
    feedback = m.group(1).strip() if m else text.strip()

    t_match  = re.search(r"<tools>(.*?)</tools>", text, re.DOTALL)
    raw      = t_match.group(1) if t_match else "none"
    tools    = [t.strip().lower() for t in raw.split(",") if t.strip().lower() in allowed_tools]

    lg = get_logger()
    if lg:
        lg.log_feedback(em_name, iteration, cluster_id, feedback,
                        tool=",".join(tools) if tools else "none", error_type=error_type)

    return feedback, tools


MAX_FB_CHARS = 800   # 피드백 하나당 최대 글자 수


def _trunc(text: str, max_chars: int = MAX_FB_CHARS) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


def evolve_feedback(f_short: str, f_long_list: list[str]) -> str:
    """LE 역할: 단기(F_short) + 장기(F_long) 피드백 통합 → F^(t+1)."""
    if not f_long_list:
        return _trunc(f_short)
    f_short   = _trunc(f_short)
    long_str  = "\n".join(f"[Long {i+1}] {_trunc(f)}" for i, f in enumerate(f_long_list[:3]))
    user = f"""Merge the following feedback entries into one concise, non-redundant policy:

Short-term feedback (most recent):
{f_short}

Long-term feedback (historically relevant):
{long_str}

Output:
<policy>
[merged feedback]
</policy>"""

    text = _text_generate(user, temperature=0.3, max_tokens=300)
    m = re.search(r"<policy>(.*?)</policy>", text, re.DOTALL)
    return m.group(1).strip() if m else f_short


# ── M-step ────────────────────────────────────────────────────────────────────

def generate_prompt_candidates(current_prompt: str, feedback: str,
                                positive_examples: list[dict],
                                n: int = 3) -> list[str]:
    """LP 역할: 피드백 + positive examples 기반으로 프롬프트 후보 N개 생성."""
    pos_str = "\n".join(
        f"- video={r['video']} second={r['second']} pred={r['pred']:.2f} gt={r['gt']:.2f}"
        for r in positive_examples[:5]
    ) or "None"

    user = f"""# Task Overview
You are improving a prompt for a cyber-scam video frame detection model.
The model must output ONLY a single float 0.0–1.0 per frame.

# Current Prompt
{current_prompt}

# Feedback from Error Analysis
{feedback}

# Positive Examples (correctly predicted, maintain these)
{pos_str}

# Objective
Generate {n} improved prompt candidates that:
- Address the failure patterns identified in the feedback
- Maintain performance on correctly predicted cases
- Are concise (under 200 words each)
- Each must instruct the model to return ONLY a float 0.0–1.0

# Output Format
<candidate_1>
[prompt text]
</candidate_1>
<candidate_2>
[prompt text]
</candidate_2>
<candidate_3>
[prompt text]
</candidate_3>"""

    text = _text_generate(user, temperature=0.8, max_tokens=1200)
    candidates = re.findall(r"<candidate_\d+>(.*?)</candidate_\d+>", text, re.DOTALL)
    candidates = [c.strip() for c in candidates if c.strip()]
    return candidates if candidates else [current_prompt]


def crossover_prompts(p_short_list: list[str], p_long_list: list[str]) -> str:
    """LE 역할: 단기 후보(P_short) + 역사적 최우수(P_long) Crossover → P^(t+1)."""
    all_prompts = p_short_list + p_long_list
    if len(all_prompts) == 1:
        return all_prompts[0]

    prompts_str = "\n\n".join(f"[Prompt {i+1}]\n{p}" for i, p in enumerate(all_prompts))
    user = f"""Merge the following prompt candidates into one optimal detection prompt:

{prompts_str}

Requirements:
- Keep all important visual cues and instructions
- Remove redundancies and resolve contradictions
- Must instruct the model to return ONLY a float 0.0–1.0
- Max 200 words

Output:
<policy>
[merged prompt]
</policy>"""

    text = _text_generate(user, temperature=0.3, max_tokens=400)
    m = re.search(r"<policy>(.*?)</policy>", text, re.DOTALL)
    return m.group(1).strip() if m else p_short_list[0]
