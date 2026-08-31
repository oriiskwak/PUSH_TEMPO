"""
add_normal_patterns.py — scam_patterns.json에 정상 콘텐츠 패턴 추가.
서버에서 한 번만 실행.
"""
import json
from pathlib import Path

PATTERNS_PATH = Path(__file__).parent / "scam_patterns.json"

normal_patterns = [
    {
        "type": "NORMAL_news_broadcast",
        "description": "공중파·케이블 뉴스 방송 — 전문 기자나 앵커가 스튜디오에서 사기 피해를 보도하거나 경보를 전달하는 정상 콘텐츠.",
        "visual_cues": [
            "방송사 로고 및 공식 브랜딩 (KBS, MBC, SBS, YTN 등)",
            "앵커 데스크 또는 뉴스 스튜디오 배경",
            "하단 자막 띠 (뉴스 ticker)",
            "리포터 현장 화면 + 방송사 워터마크"
        ],
        "text_cues": [
            "뉴스, 보도, 기자, 앵커, 특보",
            "피해 주의, 경보, 사기 피해 급증",
            "방송사명 + 뉴스 (예: KBS뉴스, MBC뉴스)"
        ]
    },
    {
        "type": "NORMAL_financial_education",
        "description": "금융 전문가나 유명 채널이 투자·재테크를 교육 목적으로 설명하는 정상 콘텐츠. 수익 보장이나 즉시 행동을 요구하지 않음.",
        "visual_cues": [
            "전문가 정장 차림 + 정돈된 스튜디오",
            "명확한 채널 브랜딩 및 로고",
            "교육용 인포그래픽·차트 (출처 표기 포함)",
            "여러 명의 패널 토론 형식"
        ],
        "text_cues": [
            "재테크, 금융 교육, 투자 기초, 절세",
            "주의사항, 리스크, 손실 가능성 안내",
            "구독자 수 다수, 공식 인증 채널"
        ]
    },
    {
        "type": "NORMAL_scam_awareness",
        "description": "사기 수법을 경고·예방 목적으로 시뮬레이션하거나 설명하는 정상 교육 콘텐츠. 실제 피해 사례를 극화하여 보여주는 경우 포함.",
        "visual_cues": [
            "재연 자막 (재연·연출 표시)",
            "경찰청·금감원 등 공공기관 로고",
            "교육용 자막 ('이렇게 하면 안 됩니다' 등)",
            "뉴스·다큐 형식의 인터뷰 구성"
        ],
        "text_cues": [
            "재연, 주의, 예방, 피해사례, 경보",
            "사기 수법 공개, 이렇게 속입니다",
            "금감원, 경찰청, 소비자원"
        ]
    },
    {
        "type": "NORMAL_government_public",
        "description": "정부 기관·공공기관이 배포하는 공식 안내 영상 또는 공익광고.",
        "visual_cues": [
            "정부 기관 로고 (금융감독원, 경찰청, 방통위 등)",
            "공식 디자인 가이드라인 준수 레이아웃",
            "자막에 기관명 및 출처 명시"
        ],
        "text_cues": [
            "금융감독원, 경찰청, 과학기술정보통신부",
            "공식 안내, 공익광고, 정책 홍보",
            "신고전화, 피해 접수 안내"
        ]
    },
]

if PATTERNS_PATH.exists():
    existing = json.load(open(PATTERNS_PATH, encoding="utf-8"))
    existing_types = {p["type"] for p in existing}
    to_add = [p for p in normal_patterns if p["type"] not in existing_types]
    existing.extend(to_add)
    json.dump(existing, open(PATTERNS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"추가 완료: {len(to_add)}개 정상 패턴 추가 (총 {len(existing)}개)")
else:
    json.dump(normal_patterns, open(PATTERNS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"새로 생성: {len(normal_patterns)}개 정상 패턴")
