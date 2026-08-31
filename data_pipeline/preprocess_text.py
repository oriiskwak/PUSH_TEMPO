"""
preprocess_text.py — 영상별 Title + ASR + OCR 텍스트 추출

Title: yt-dlp --print title (다운로드 없이 메타데이터만)
ASR  : openai-whisper (large-v3, CUDA)  → 전체 오디오 전사
OCR  : easyocr (ko + en, CUDA)          → gt_all.json 2초 간격 프레임

출력 : data/text_meta.json
  {
    "abnormal_xxx": {
      "title": "영상 제목",
      "asr":   "전사된 음성 텍스트 전체",
      "ocr":   ["텍스트1", "텍스트2", ...]
    },
    ...
  }

사용법:
  pip install openai-whisper easyocr yt-dlp
  python preprocess_text.py
"""

import csv, json, os, subprocess, cv2
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import whisper
import easyocr

BASE_DIR       = Path(__file__).parent.parent
DATA_DIR       = BASE_DIR / "data"
VIDEO_DIR      = Path(os.environ.get("VIDEO_DIR", str(DATA_DIR / "videos")))
GT_ALL         = DATA_DIR / "gt_results" / "gt_all.json"
CSV_PATH       = DATA_DIR / ("LABELS.csv" if (DATA_DIR / "LABELS.csv").exists() else "labels.csv")
TEXT_META_PATH = DATA_DIR / "text_meta.json"


def _extract_video_id(url: str) -> str:
    """YouTube/TikTok URL에서 영상 ID 추출."""
    parsed = urlparse(url)
    # youtube.com/watch?v=ID
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    # youtu.be/ID
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("?")[0]
    # TikTok 등: 경로 마지막 세그먼트
    return parsed.path.rstrip("/").split("/")[-1]


# ── labels.csv에서 video_name → link 매핑 (synthetic 제외)
link_map = {}
if CSV_PATH.exists():
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or row.get("라벨") or "").strip()
            link  = (row.get("link")  or row.get("링크")  or "").strip()
            if not label or not link:
                continue
            vid_id = _extract_video_id(link)
            vname  = f"{label}_{vid_id}"
            if "synth" not in vname:
                link_map[vname] = link
    print(f"labels.csv 로드: {len(link_map)}개 링크")
else:
    print("labels.csv 없음 — Title 추출 건너뜀")

# ── 기존 결과 로드 (중단 후 재시작 시 이어서 처리)
# 처음부터 다시 하려면: python preprocess_text.py --reset
import sys
RESET = "--reset" in sys.argv
if RESET and TEXT_META_PATH.exists():
    TEXT_META_PATH.unlink()
    print("기존 text_meta.json 삭제 — 처음부터 재시작")

if TEXT_META_PATH.exists():
    text_meta = json.load(open(TEXT_META_PATH, encoding="utf-8"))
    print(f"기존 결과 {len(text_meta)}개 로드 — 미처리 영상만 진행")
else:
    text_meta = {}

gt_list = json.load(open(GT_ALL, encoding="utf-8"))
# synthetic 데이터 제외
gt_map = {v["video_name"]: v for v in gt_list if "synth" not in v["video_name"]}
print(f"처리 대상: {len(gt_map)}개 영상 (synthetic 제외)")

# ── 모델 로드
print("Whisper large-v3 로드 중...")
whisper_model = whisper.load_model("large-v3", device="cuda")

print("EasyOCR (ko + en) 로드 중...")
reader = easyocr.Reader(["ko", "en"], gpu=True)

# ── 영상별 처리
total = len(gt_map)
for idx, (vname, vdata) in enumerate(gt_map.items(), 1):
    if vname in text_meta:
        print(f"[{idx}/{total}] {vname} — 이미 처리됨, 건너뜀")
        continue

    vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
    if not vfiles:
        print(f"[{idx}/{total}] {vname} — 파일 없음, 건너뜀")
        continue

    vpath = vfiles[0]
    print(f"\n[{idx}/{total}] {vname}")

    # ── Title (yt-dlp, 다운로드 없음)
    title = ""
    link = link_map.get(vname, "")
    if link:
        try:
            res = subprocess.run(
                ["yt-dlp", "--print", "title", "--skip-download", "--no-playlist", link],
                capture_output=True, text=True, timeout=15
            )
            title = res.stdout.strip()
        except Exception as e:
            print(f"  Title 실패: {e}")
    print(f"  Title: {title}")

    # ── ASR (전체 오디오)
    print("  ASR 처리 중...")
    try:
        result   = whisper_model.transcribe(str(vpath), language="ko", fp16=True)
        asr_text = result["text"].strip()
    except Exception as e:
        print(f"  ASR 실패: {e}")
        asr_text = ""
    print(f"  ASR: {asr_text[:100]}{'...' if len(asr_text) > 100 else ''}")

    # ── OCR (gt_all.json 2초 간격 프레임)
    print("  OCR 처리 중...")
    ocr_texts = []
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for f in vdata["frame_probabilities"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f["second"] * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            results = reader.readtext(frame, detail=1, paragraph=False)
            for _, text, conf in results:
                if conf >= 0.65:         # 저신뢰도 제거 (Korean OCR은 overconfident)
                    ocr_texts.append(text)
        except Exception as e:
            print(f"  OCR 프레임 실패 ({f['second']}s): {e}")

    cap.release()

    # 노이즈 필터 + 중복 제거 (순서 유지)
    import re as _re
    seen = set()
    ocr_unique = []
    for t in ocr_texts:
        t = t.strip()
        if not t or len(t) < 2:
            continue
        # 한글/영문/숫자 비율이 50% 미만이면 노이즈
        valid_chars = len(_re.findall(r"[가-힣a-zA-Z0-9]", t))
        if valid_chars / len(t) < 0.5:
            continue
        # 한글만 있는데 초성/중성 반복 패턴 (기늄, 기등 등) — 의미없는 오인식
        if _re.fullmatch(r"[가-힣]+", t) and len(t) >= 2:
            # 2글자 이상이 같은 받침 패턴으로 끝나는 경우 (기늄, 기등, 기름 등)
            # 어절이 1개이고 자모 다양성이 낮으면 제거
            unique_chars = len(set(t))
            if unique_chars / len(t) < 0.5:   # 절반 이상 중복 문자면 노이즈
                continue
        if t not in seen:
            seen.add(t)
            ocr_unique.append(t)

    print(f"  OCR: {len(ocr_unique)}개 텍스트")

    text_meta[vname] = {"title": title, "asr": asr_text, "ocr": ocr_unique}

    # 중간 저장 (영상마다)
    json.dump(text_meta, open(TEXT_META_PATH, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

print(f"\n완료: {TEXT_META_PATH}  (총 {len(text_meta)}개 영상)")
