"""
em_core.py — UniAPO EM 알고리즘 핵심 구성요소

E-step:
  LF (피드백 모델): 오류 샘플 분석 → 단기 피드백 F_short 생성
  피드백 메모리 M_F: F_short + F_long → Evolver(LE)가 통합 → F^(t+1)

M-step:
  LP (프롬프트 최적화): 피드백 기반 후보 N개 생성
  프롬프트 메모리 M_P: 역사적 상위 K개 P_long → Evolver(LE) Crossover → P^(t+1)
"""

import re, json
from datetime import datetime
import numpy as np
from pathlib import Path
from vllm import SamplingParams
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

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
                     cluster_id: int, feedback: str):
        """클러스터별 생성된 피드백."""
        self._write({
            "type":     "feedback",
            "em":       em_name,
            "iter":     iteration,
            "cluster":  cluster_id,
            "feedback": feedback,
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

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = _llm.generate([{"prompt": text}], sp, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


# ── 메모리 클래스 ──────────────────────────────────────────────────────────────

class FeedbackMemory:
    """피드백 메모리 M_F: 과거 피드백을 저장하고 유사도 기반으로 검색."""

    def __init__(self, top_k: int = 3):
        self.memory: list[str] = []
        self.top_k = top_k

    def add(self, feedback: str):
        self.memory.append(feedback)
        lg = get_logger()
        if lg:
            lg._write({
                "type":        "memory_add",
                "feedback":    feedback[:300],
                "tool":        None,
                "memory_size": len(self.memory),
            })

    def retrieve(self, query: str) -> list[str]:
        if not self.memory:
            return []
        if len(self.memory) <= self.top_k:
            return self.memory[:]
        q_emb  = embedder.encode([query], normalize_embeddings=True)[0]
        m_embs = embedder.encode(self.memory, normalize_embeddings=True)
        sims   = m_embs @ q_emb
        top    = np.argsort(sims)[-self.top_k:][::-1]
        return [self.memory[i] for i in top]


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

def cluster_errors(error_items: list[dict], eps: float = 0.35,
                   em_name: str = "", iteration: int = 0,
                   vid_stats: dict | None = None) -> list[list[dict]]:
    """오류 샘플 클러스터링.

    text_meta.json 있으면: 영상 텍스트(ASR+OCR) 임베딩 기반 의미론적 클러스터링
    없으면: pred/gt 수치 기반 클러스터링 (폴백)
    """
    if not error_items:
        return []

    if _text_meta:
        texts = []
        for r in error_items:
            meta = _text_meta.get(r["video"], {"asr": "", "ocr": []})
            combined = (
                meta.get("title", "") + " " +
                meta.get("asr", "") + " " +
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


def _make_grid(images: list, cell_size: int = 336) -> "Image":
    """여러 프레임을 가로로 이어붙여 1장의 그리드 이미지로 만든다."""
    from PIL import Image as PILImage
    n = len(images)
    grid = PILImage.new("RGB", (cell_size * n, cell_size))
    for i, img in enumerate(images):
        grid.paste(img.resize((cell_size, cell_size), PILImage.LANCZOS), (i * cell_size, 0))
    return grid


def _vision_feedback_generate(prompt: str, images: list,
                               temperature: float = 0.7, max_tokens: int = 600) -> str:
    """피드백 생성용 멀티모달 생성 — 여러 프레임을 그리드 1장으로 합쳐 전송."""
    from utils import _llm, _processor

    grid_img = _make_grid(images)

    content = [{"type": "image", "image": grid_img},
               {"type": "text",  "text":  prompt}]
    messages = [{"role": "user", "content": content}]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    token_ids = _processor.tokenizer.encode(text)
    max_len   = _llm.llm_engine.model_config.max_model_len
    budget    = max_len - max_tokens - 50
    if len(token_ids) > budget:
        token_ids = token_ids[-budget:]
        text = _processor.tokenizer.decode(token_ids, skip_special_tokens=False)

    sp      = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    mm_data = {"image": [grid_img]}
    outputs = _llm.generate([{"prompt": text, "multi_modal_data": mm_data}], sp, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


def generate_feedback(current_prompt: str, chunk: list[dict],
                       past_feedbacks: list[str],
                       em_name: str = "", iteration: int = 0,
                       cluster_id: int = 0,
                       error_type: str = "fn") -> str:
    """LF 역할: 오류 청크 분석 → 단기 피드백 F_short 생성.

    error_type: "fn" — 사기 영상을 정상으로 예측 (탐지 강화 필요)
                "fp" — 정상 영상을 사기로 예측 (오탐 감소 필요)
    """
    from utils import load_frame, VIDEO_DIR
    from pathlib import Path

    error_lines = []
    frame_images = []

    for r in chunk:
        meta  = _text_meta.get(r["video"], {})
        title = meta.get("title", "")[:120]
        asr   = meta.get("asr",   "")[:200]
        ocr   = " ".join(meta.get("ocr", []))[:120]
        label = r.get("label", "unknown")

        error_lines.append(
            f"- label={label}  pred={r['pred']:.2f}  gt={r['gt']:.2f}  kl={r['kl']:.3f}\n"
            f"  title: {title}\n"
            f"  asr:   {asr}\n"
            f"  ocr:   {ocr}"
        )

        vfiles = list(VIDEO_DIR.glob(f"{r['video']}.*"))
        if vfiles:
            img = load_frame(vfiles[0], r["second"])
            if img is not None:
                frame_images.append(img)

    error_str = "\n".join(error_lines)
    past_str  = "\n".join(f"[Past {i+1}] {f}" for i, f in enumerate(past_feedbacks)) or "None"

    if error_type == "fp":
        error_desc = "FALSE POSITIVE — The model predicted these frames as SCAM, but they are NORMAL videos."
        objective  = (
            "- Identify what visual/textual patterns are causing the model to OVER-detect scam content\n"
            "- Suggest how to DISTINGUISH these legitimate patterns from actual scams\n"
            "- The goal is to REDUCE false alarms on this type of content"
        )
    else:
        error_desc = "FALSE NEGATIVE — The model predicted these frames as NORMAL, but they are SCAM videos."
        objective  = (
            "- Identify what scam signals the current prompt is MISSING in these frames\n"
            "- Suggest specific visual cues or instructions to better detect this type of scam\n"
            "- The goal is to INCREASE detection sensitivity for this pattern"
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

# Output Format
<policy>
[feedback content]
</policy>"""

    if frame_images:
        text = _vision_feedback_generate(user, frame_images, temperature=0.7, max_tokens=600)
    else:
        text = _text_generate(user, temperature=0.7, max_tokens=600)

    m        = re.search(r"<policy>(.*?)</policy>", text, re.DOTALL)
    feedback = m.group(1).strip() if m else text.strip()

    lg = get_logger()
    if lg:
        lg.log_feedback(em_name, iteration, cluster_id, feedback)

    return feedback


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
