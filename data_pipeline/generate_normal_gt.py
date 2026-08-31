"""
generate_normal_gt.py — 순수 normal 영상 GT 생성 + abnormal_ 접두사 rename

작업 1: 순수 normal (현재 prob=0.0 고정) → GPT-4o로 실제 프레임별 확률 생성 (0.0 고정을 교체하는 것)
작업 2: traded 영상 (abnormal_ + label=normal) → 기존 GPT-4o GT 그대로 두고 접두사만 abnormal_ → normal_로 rename

실행:
  python generate_normal_gt.py --dry-run       # 대상 목록 확인
  python generate_normal_gt.py --rename-only   # 접두사 rename만 (GT 생성 없이)
  python generate_normal_gt.py --gt-only       # GT 생성만 (rename 없이)
  python generate_normal_gt.py                 # 둘 다 실행
"""

import cv2, base64, json, os, re, sys, time, shutil
import numpy as np
from pathlib import Path

BASE_DIR       = Path(__file__).parent
GT_PATH        = BASE_DIR / "data" / "gt_results" / "gt_all.json"
TEXT_META_PATH = BASE_DIR / "data" / "text_meta.json"
TEXT_META_BBOX_PATH = BASE_DIR / "data" / "text_meta_bbox.json"
VIDEO_DIR      = Path(os.environ.get("VIDEO_DIR", str(Path(__file__).parent.parent / "data" / "videos")))

GPT_MODEL    = "gpt-5.4"
API_DELAY    = 0.5
JPEG_QUALITY = 85

MODE = next((a for a in sys.argv[1:] if a.startswith("--")), "")
DRY_RUN = MODE == "--dry-run"


SCAM_PROMPT = """\
You are an expert at detecting cyber scam content in social media video frames (YouTube Shorts, TikTok).

=== Video Context ===
Title: {title}
On-screen text (OCR): {ocr}

=== Task ===
Analyze this single video frame and estimate the probability it is from a cyber scam video.
Focus on VISUAL evidence. This is a video labeled as NORMAL (not a scam), so be precise.

Scam indicators: fake prize/lottery, bank/government impersonation, urgent security threats,
OTP/payment requests, phishing UI, countdown timers, unrealistic earnings, romance scam setups.

Return ONLY a single float 0.0–1.0. No explanation."""


def encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode("utf-8")


def extract_frame(vpath: Path, second: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def query_gpt(frame: np.ndarray, prompt: str) -> float | None:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    b64 = encode_frame(frame)
    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                }},
            ]}],
            max_completion_tokens=10,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"[0-9]*\.?[0-9]+", raw)
        return max(0.0, min(1.0, float(m.group()))) if m else None
    except Exception as e:
        print(f"    [ERROR] GPT 호출 실패: {e}")
        return None


# ── 작업 1: 순수 normal GT 생성 ───────────────────────────────────────────────

def get_pure_normals(gt_all: list) -> list:
    """normal_ 접두사 + prob이 모두 0.0~0.01 고정인 영상."""
    targets = []
    for v in gt_all:
        if v["label"] != "normal":
            continue
        if not v["video_name"].startswith("normal_"):
            continue
        if v.get("gt_generated"):
            continue
        probs = [f["probability"] for f in v["frame_probabilities"]]
        if all(p <= 0.01 for p in probs):
            targets.append(v)
    return targets


def generate_gt_for_normals(gt_all: list):
    targets = get_pure_normals(gt_all)
    print(f"\n[GT 생성] 순수 normal 영상: {len(targets)}개")
    if not targets:
        print("  모두 처리됨.")
        return

    text_meta = json.load(open(TEXT_META_PATH, encoding="utf-8"))
    vid_idx   = {v["video_name"]: i for i, v in enumerate(gt_all)}

    for n, vid in enumerate(targets, 1):
        vname  = vid["video_name"]
        meta   = text_meta.get(vname, {})
        title  = meta.get("title", "")
        ocr    = " | ".join(meta.get("ocr", []))[:300]

        vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
        if not vfiles:
            print(f"  [{n}/{len(targets)}] {vname} — 파일 없음, 건너뜀")
            continue

        vpath = vfiles[0]
        print(f"\n  [{n}/{len(targets)}] {vname}  title: {title[:50]}")

        new_probs = []
        for fp in vid["frame_probabilities"]:
            second = fp["second"]
            frame  = extract_frame(vpath, second)
            if frame is None:
                new_probs.append(fp)
                continue

            prompt = SCAM_PROMPT.format(
                title=title or "(없음)",
                ocr=ocr or "(없음)",
            )
            prob = query_gpt(frame, prompt)
            if prob is None:
                new_probs.append(fp)
                continue

            new_probs.append({"second": second, "probability": prob})
            print(f"    [{second:>3}s] 0.00 → {prob:.3f}")
            time.sleep(API_DELAY)

        idx = vid_idx[vname]
        gt_all[idx]["frame_probabilities"] = new_probs
        gt_all[idx]["gt_generated"] = True

        json.dump(gt_all, open(GT_PATH, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)

    print(f"\nGT 생성 완료. {GT_PATH} 업데이트됨.")


# ── 작업 2: abnormal_ → normal_ 접두사 rename ────────────────────────────────

def rename_traded_videos(gt_all: list):
    """label=normal인 abnormal_ 접두사 영상의 video_name을 normal_로 변경."""
    targets = [v for v in gt_all
               if v["label"] == "normal" and v["video_name"].startswith("abnormal_")]
    print(f"\n[Rename] abnormal_ → normal_ 대상: {len(targets)}개")

    if not targets:
        print("  대상 없음.")
        return

    text_meta = json.load(open(TEXT_META_PATH, encoding="utf-8"))
    text_meta_bbox = json.load(open(TEXT_META_BBOX_PATH, encoding="utf-8")) if TEXT_META_BBOX_PATH.exists() else {}
    renamed_meta, renamed_bbox = {}, {}

    for v in targets:
        old_name = v["video_name"]
        new_name = "normal_" + old_name[len("abnormal_"):]
        print(f"  {old_name} → {new_name}")

        # gt_all video_name 변경
        v["video_name"] = new_name

        # text_meta 키 변경
        if old_name in text_meta:
            renamed_meta[new_name] = text_meta.pop(old_name)

        # text_meta_bbox 키 변경
        if old_name in text_meta_bbox:
            renamed_bbox[new_name] = text_meta_bbox.pop(old_name)

    # text_meta 업데이트
    text_meta.update(renamed_meta)
    json.dump(text_meta, open(TEXT_META_PATH, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # text_meta_bbox 업데이트
    if renamed_bbox:
        text_meta_bbox.update(renamed_bbox)
        json.dump(text_meta_bbox, open(TEXT_META_BBOX_PATH, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)

    # gt_all 저장
    json.dump(gt_all, open(GT_PATH, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"  rename 완료. gt_all.json + text_meta.json + text_meta_bbox.json 업데이트됨.")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    gt_all = json.load(open(GT_PATH, encoding="utf-8"))

    if DRY_RUN:
        pure_normals = get_pure_normals(gt_all)
        traded = [v for v in gt_all
                  if v["label"] == "normal" and v["video_name"].startswith("abnormal_")]
        print(f"[GT 생성 대상] 순수 normal: {len(pure_normals)}개")
        for v in pure_normals[:]:
            print(f"  {v['video_name']}")
        print(f"\n[Rename 대상] abnormal_ → normal_: {len(traded)}개")
        for v in traded[:]:
            print(f"  {v['video_name']}")
        return

    # 백업
    backup = GT_PATH.parent / "gt_all_backup_before_normal_gt.json"
    if not backup.exists():
        shutil.copy(GT_PATH, backup)
        print(f"백업: {backup.name}")

    if MODE in ("", "--gt-only"):
        generate_gt_for_normals(gt_all)
        gt_all = json.load(open(GT_PATH, encoding="utf-8"))  # 재로드

    if MODE in ("", "--rename-only"):
        rename_traded_videos(gt_all)


if __name__ == "__main__":
    main()
