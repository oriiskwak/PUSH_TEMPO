"""
em_core_mc.py — 4-class 분류를 위한 EM 핵심 구성요소

em_core_tool.py의 binary 전용 함수들을 4-class용으로 대체.
FeedbackMemory, PromptMemory, evolve_feedback, init_logger, get_logger 는 그대로 재사용.
"""
import re
from em_core_tool import (
    FeedbackMemory, PromptMemory,
    evolve_feedback, init_logger, get_logger,
    _text_generate, _vision_feedback_generate, _text_meta,
    VALID_TOOLS,
)
from utils_mc import CLASSES, CLASSES_STR


# ── E-step ────────────────────────────────────────────────────────────────────

def cluster_errors_mc(error_items: list[dict],
                      min_size: int = 3) -> dict[str, list[dict]]:
    """4-class 오류를 confusion pair (true→pred) 기준으로 그룹화.

    min_size 미만 pair는 가장 큰 클러스터에 병합 — 샘플 부족으로 인한
    저품질 피드백 방지 (최대 12개 pair 중 일부는 매우 희소할 수 있음).
    """
    clusters: dict[str, list[dict]] = {}
    for r in error_items:
        key = f"{r['label']}→{r['pred_label']}"
        clusters.setdefault(key, []).append(r)

    if len(clusters) <= 1:
        return clusters

    large = {k: v for k, v in clusters.items() if len(v) >= min_size}
    small = {k: v for k, v in clusters.items() if len(v) <  min_size}

    if not large:
        # 모든 pair가 희소하면 가장 큰 것 하나로 합치기
        biggest = max(clusters, key=lambda k: len(clusters[k]))
        merged  = list(clusters[biggest])
        for k, v in clusters.items():
            if k != biggest:
                merged.extend(v)
        return {biggest: merged}

    # 희소 pair → 가장 큰 large 클러스터에 병합
    biggest = max(large, key=lambda k: len(large[k]))
    for v in small.values():
        large[biggest].extend(v)
    return large


def generate_feedback_mc(
    current_prompt: str,
    chunk: list[dict],
    past_feedbacks: list[str],
    true_class: str,
    pred_class: str,
    em_name: str = "",
    iteration: int = 0,
    cluster_id: int = 0,
) -> tuple[str, list[str]]:
    """LF 역할 (4-class): confusion pair 기반 피드백 생성."""
    from utils import VIDEO_DIR, load_frame

    error_lines  = []
    frame_images = []

    for r in chunk:
        meta  = _text_meta.get(r["video"], {})
        title = meta.get("title", "")[:120]
        asr   = meta.get("asr",   "")[:200]
        ocr   = " ".join(meta.get("ocr", []))[:120]
        error_lines.append(
            f"- video={r['video']}  second={r['second']}\n"
            f"  true={true_class}  pred={pred_class}\n"
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
    past_str  = "\n".join(f"[Past {i+1}] {f}"
                          for i, f in enumerate(past_feedbacks)) or "None"

    user = f"""# Task Overview
You are analyzing why a scam video type classification prompt produces errors.

# Confusion
The model predicted "{pred_class}" for videos that are actually "{true_class}".

# Current Prompt
{current_prompt}

# Past Feedbacks
{past_str}

# Error Cases
{error_str}

# Objective
Generate concise feedback that:
- Identifies visual and contextual differences between {true_class} and {pred_class}
- Suggests specific visual cues or scenario patterns that distinguish {true_class}
- Does NOT overfit to these specific frames
- Uses ABSTRACT, GENERALIZABLE patterns (not specific Korean words from examples)

# Available Tools
- title      : video title for topic context
- asr        : speech/on-screen text to identify scam-type vocabulary
- channel    : channel type (news/educational vs targeted scam channel)
- web_search : verify specific scam scenario vocabulary
- rag        : match against known scam pattern database

First judge whether improving the visual detection instructions alone can fix this error.
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

    t_match = re.search(r"<tools>(.*?)</tools>", text, re.DOTALL)
    raw     = t_match.group(1) if t_match else "none"
    tools   = [t.strip().lower() for t in raw.split(",") if t.strip().lower() in VALID_TOOLS]

    lg = get_logger()
    if lg:
        lg.log_feedback(em_name, iteration, cluster_id, feedback,
                        tool=",".join(tools) if tools else "none",
                        error_type=f"{true_class}→{pred_class}")

    return feedback, tools


# ── M-step ────────────────────────────────────────────────────────────────────

def generate_prompt_candidates_mc(current_prompt: str, feedback: str,
                                   positive_examples: list[dict],
                                   n: int = 3) -> list[str]:
    """LP 역할 (4-class): 피드백 기반 프롬프트 후보 N개 생성."""
    pos_str = "\n".join(
        f"- video={r['video']} second={r['second']} true={r['label']}"
        for r in positive_examples[:5]
    ) or "None"

    user = f"""# Task Overview
You are improving a prompt for a scam video TYPE classification model.
The model must classify each video frame into one of: {CLASSES_STR}
The model must output ONLY one class name per frame.

# Current Prompt
{current_prompt}

# Feedback from Error Analysis
{feedback}

# Positive Examples (correctly classified, maintain these)
{pos_str}

# Objective
Generate {n} improved prompt candidates that:
- Address the failure patterns identified in the feedback
- Help distinguish visually/contextually similar scam types
- Are concise (under 200 words each)
- Each must instruct the model to return ONLY one of: {CLASSES_STR}

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


def crossover_prompts_mc(p_short_list: list[str], p_long_list: list[str]) -> str:
    """LE 역할 (4-class): 프롬프트 후보 Crossover → P^(t+1)."""
    all_prompts = p_short_list + p_long_list
    if len(all_prompts) == 1:
        return all_prompts[0]

    prompts_str = "\n\n".join(f"[Prompt {i+1}]\n{p}" for i, p in enumerate(all_prompts))
    user = f"""Merge the following prompt candidates into one optimal classification prompt:

{prompts_str}

Requirements:
- Keep all important visual cues and type-distinguishing instructions
- Remove redundancies and resolve contradictions
- Must instruct the model to return ONLY one of: {CLASSES_STR}
- Max 200 words

Output:
<policy>
[merged prompt]
</policy>"""

    text = _text_generate(user, temperature=0.3, max_tokens=400)
    m = re.search(r"<policy>(.*?)</policy>", text, re.DOTALL)
    return m.group(1).strip() if m else p_short_list[0]
