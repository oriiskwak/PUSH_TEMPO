"""
build_scam_patterns.py — Qwen으로 사기 유형별 탐지 규칙 DB 생성

3개 유형 × 20개 규칙 = 60개 규칙
외부 데이터 미사용 (LLM 지식 기반) → data leakage 없음
결과: data_repo/scam_patterns.json

- build() 함수: run_base.py에서 import해 파이프라인 시작 시 자동 호출
- 직접 실행 시에도 동작: python3 build_scam_patterns.py
"""

import json, sys, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

OUT_PATH = Path(__file__).parent / "data" / "scam_patterns.json"

SCAM_TYPES = [
    {
        "id":   "voice_phishing",
        "type": "보이스피싱 (Voice Phishing)",
        "query": (
            "한국에서 발생하는 보이스피싱 사기 영상(유튜브 쇼츠, 틱톡, 릴스)을 "
            "자동으로 탐지하기 위한 규칙을 생성해줘. "
            "보이스피싱 유형에는 기관 사칭(검찰/경찰/금감원), 가족 사칭, "
            "대출 사기, 스미싱, 악성 앱 설치 유도가 포함돼."
        ),
    },
    {
        "id":   "investment_scam",
        "type": "투자 사기 (Investment Scam)",
        "query": (
            "한국에서 발생하는 투자 사기 영상(유튜브 쇼츠, 틱톡, 릴스)을 "
            "자동으로 탐지하기 위한 규칙을 생성해줘. "
            "투자 사기 유형에는 주식 리딩방 사기, 코인/NFT 투자 사기, "
            "가짜 재테크 강의, 폰지 사기, 다단계 투자 유도가 포함돼."
        ),
    },
    {
        "id":   "romance_scam",
        "type": "로맨스 스캠 (Romance Scam)",
        "query": (
            "한국에서 발생하는 로맨스 스캠 영상(유튜브 쇼츠, 틱톡, 릴스)을 "
            "자동으로 탐지하기 위한 규칙을 생성해줘. "
            "로맨스 스캠 유형에는 SNS 연인 사칭 후 투자 유도, "
            "군인/외국인 사칭, 선물 통관 사기, 결혼 사기가 포함돼."
        ),
    },
]

RULE_PROMPT_TEMPLATE = """당신은 한국 사이버 사기 탐지 전문가입니다.

아래 사기 유형에 대해 소셜미디어 영상 프레임에서 탐지하는 데 필요한 규칙을 20개 생성해주세요.

사기 유형: {scam_type}
{query}

각 규칙은 다음 정보를 포함해야 합니다:
- description: 이 규칙이 탐지하는 구체적인 상황 (1-2문장)
- visual_cues: 프레임 이미지에서 보이는 시각적 단서 (최대 5개)
- text_cues: OCR/ASR로 추출될 수 있는 텍스트 단서 (최대 5개)
- risk_level: high / medium / low

정확히 20개 규칙을 아래 JSON 배열 형식으로만 출력하세요:
<rules>
[
  {{
    "description": "...",
    "visual_cues": ["...", "..."],
    "text_cues": ["...", "..."],
    "risk_level": "high"
  }},
  ...
]
</rules>"""


def _repair_json(raw: str) -> str:
    """잘린 JSON 배열을 최대한 복구 — 완전한 객체만 추출."""
    start = raw.find("[")
    if start == -1:
        return ""
    depth, in_str, escape = 0, False, False
    objects, obj_start = [], None
    for i, ch in enumerate(raw[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 1:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 1 and obj_start is not None:
                objects.append(raw[obj_start:i + 1])
                obj_start = None
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
    return "[" + ",".join(objects) + "]" if objects else ""


def _generate_rules_for_type(scam_type: dict) -> list[dict]:
    from em_core import _text_generate
    prompt = RULE_PROMPT_TEMPLATE.format(
        scam_type=scam_type["type"],
        query=scam_type["query"],
    )
    print(f"  Qwen 추론 중: {scam_type['type']}...")
    raw = _text_generate(prompt, temperature=0.7, max_tokens=4000)

    m = re.search(r"<rules>(.*?)</rules>", raw, re.DOTALL)
    raw_json = m.group(1).strip() if m else re.search(r"\[.*", raw, re.DOTALL)
    if not isinstance(raw_json, str):
        raw_json = raw_json.group() if raw_json else ""

    if not raw_json:
        print(f"  [오류] JSON 블록 없음")
        return []

    try:
        rules = json.loads(raw_json)
        print(f"  → {len(rules)}개 규칙 생성")
        return rules
    except json.JSONDecodeError:
        repaired = _repair_json(raw_json)
        try:
            rules = json.loads(repaired)
            print(f"  → {len(rules)}개 규칙 생성 (복구)")
            return rules
        except json.JSONDecodeError as e:
            print(f"  [오류] JSON 복구 실패: {e}")
            return []


def build(out_path: Path = OUT_PATH) -> list[dict]:
    """Qwen으로 3개 유형 × 20개 규칙 DB 생성 후 저장. 생성된 패턴 리스트 반환."""
    print("Qwen으로 사기 탐지 규칙 DB 생성 시작...\n")
    patterns = []

    for scam_type in SCAM_TYPES:
        rules = _generate_rules_for_type(scam_type)
        for rule in rules:
            idx = sum(1 for p in patterns if p["id"].startswith(scam_type["id"])) + 1
            patterns.append({
                "id":          f"{scam_type['id']}_{idx:02d}",
                "type":        scam_type["type"],
                "description": rule.get("description", ""),
                "visual_cues": rule.get("visual_cues", []),
                "text_cues":   rule.get("text_cues", []),
                "risk_level":  rule.get("risk_level", "medium"),
            })

    json.dump(patterns, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n완료: {len(patterns)}개 규칙 저장 → {out_path}")
    for t in SCAM_TYPES:
        count = sum(1 for p in patterns if p["id"].startswith(t["id"]))
        print(f"  {t['type']}: {count}개")

    return patterns


if __name__ == "__main__":
    build()
