"""
prepare_data.py — 데이터 전처리 파이프라인
labels.csv 에 새 URL을 추가하고 이 스크립트를 실행하면:
  0. Download  : labels.csv의 URL → yt-dlp로 data/videos/ 저장
  1. GT 생성  : 각 프레임을 GPT로 채점 → data/gt_results/ 저장
  2. 재라벨링 : 프레임 확률 기준으로 label 재분류 → gt_all.json 업데이트
               사기 프레임(prob >= 0.5) 비율 < 30% 인 abnormal → normal
               사기 프레임(prob >= 0.5) 비율 >= 30% 인 normal  → abnormal
  3. Segmentation : 재라벨링 후 GT 기반 early/late 구간 분할 → data/segmentation.json 저장

labels.csv 포맷 (헤더 필수):
  label,link
  abnormal,https://www.youtube.com/shorts/xxx
  normal,https://www.tiktok.com/@user/video/yyy

이미 처리된 항목은 건너뜁니다 (videos/에 파일 있으면 skip, gt_results/에 json 있으면 skip).
"""

import csv
import cv2
import base64
import json
import os
import re
import subprocess
import time
import numpy as np
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from openai import OpenAI

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("[WARN] ruptures 미설치 → 고정 비율 구간 분할로 대체됩니다.")
    print("       pip install ruptures  으로 설치하면 더 정확한 분할이 됩니다.\n")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))
VIDEO_DIR = DATA_DIR / "videos"
GT_DIR    = DATA_DIR / "gt_results"
GT_ALL    = GT_DIR / "gt_all.json"
SEG_PATH  = DATA_DIR / "segmentation.json"
CSV_PATH  = DATA_DIR / "LABELS.csv"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)

GPT_MODEL      = "gpt-5.4"
API_DELAY      = 2.0
JPEG_QUALITY   = 85
MAX_DURATION_SEC = 150

SCAM_FRAME_THR  = 0.5   # 프레임을 사기로 보는 확률 임계값
RELABEL_RATIO   = 0.10  # 재라벨링 기준 비율 (10%)

openai_client = OpenAI()

# ─────────────────────────────────────────────────────────────────────────────
# Step 0 : Download
# ─────────────────────────────────────────────────────────────────────────────
def extract_video_id(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)

    if "youtube.com/shorts/" in url:
        return parsed.path.split("/shorts/")[-1].split("?")[0]

    if "tiktok.com" in url:
        for part in parsed.path.split("/"):
            if part.isdigit():
                return part

    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]

    return parsed.path.rstrip("/").split("/")[-1].split("?")[0]

def download_video(url: str, label: str, video_id: str) -> Path | None:
    stem = f"{label}_{video_id}"
    existing = list(VIDEO_DIR.glob(f"{stem}.*"))
    if existing:
        print(f"  [SKIP] 이미 존재: {existing[0].name}")
        return existing[0]

    print(f"  [DL] {stem}")
    output_template = str(VIDEO_DIR / f"{stem}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--match-filter", "duration <= 150",
        "-f", ("bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]"
               "/bestvideo[vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]"
               "/bestvideo[ext=mp4][vcodec!=av01]+bestaudio[ext=m4a]"
               "/best[ext=mp4]/best"),
        "--merge-output-format", "mp4",
        "-o", output_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] 다운로드 실패:\n{result.stderr[-400:]}")
        return None

    downloaded = list(VIDEO_DIR.glob(f"{stem}.*"))
    if downloaded:
        print(f"  [OK] 저장됨: {downloaded[0].name}")
        return downloaded[0]

    if "does not pass filter" in result.stdout or not downloaded:
        print(f"  [SKIP] 150초 초과 영상 — 건너뜀")
    return None

def _read_csv_rows() -> list[tuple[str, str]]:
    """CSV에서 (label, link) 리스트 반환. 한글/영문 컬럼명 모두 지원."""
    rows = []
    seen_ids: set[str] = set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label") or row.get("라벨") or "").strip()
            link  = (row.get("link")  or row.get("링크")  or "").strip()
            if not label or not link:
                continue
            vid_id = extract_video_id(link)
            if vid_id in seen_ids:
                continue
            seen_ids.add(vid_id)
            rows.append((label, link))
    return rows


def download_all() -> list[Path]:
    print("=" * 55)
    print("  Step 0: 영상 다운로드 (labels.csv)")
    print("=" * 55)

    if not CSV_PATH.exists():
        print(f"[ERROR] {CSV_PATH} 없음 — labels.csv를 data/ 폴더에 넣어주세요.")
        return []

    rows = [(label, link, extract_video_id(link)) for label, link in _read_csv_rows()]
    # 중복 로그 출력은 _read_csv_rows에서 이미 처리됨

    print(f"CSV 항목 {len(rows)}개 (중복 제거 후)\n")

    downloaded = []
    for i, (label, url, vid_id) in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {label} | {vid_id}")
        path = download_video(url, label, vid_id)
        if path:
            downloaded.append(path)

    print(f"\n  다운로드 완료: {len(downloaded)}/{len(rows)}개")
    return downloaded

# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────
def reencode_h264(video_path: Path) -> bool:
    tmp = video_path.with_suffix(".tmp.mp4")
    print(f"    [FFMPEG] H.264 재인코딩 중... {video_path.name}")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-c:v", "libx264", "-crf", "23", "-preset", "fast",
         "-c:a", "aac", "-b:a", "128k",
         str(tmp)],
        capture_output=True,
    )
    if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        video_path.unlink()
        tmp.rename(video_path)
        print(f"    [FFMPEG] 완료 → {video_path.name}")
        return True
    tmp.unlink(missing_ok=True)
    print(f"    [FFMPEG ERROR] 재인코딩 실패: {result.stderr[-200:]}")
    return False

def extract_frames_05fps(video_path: Path) -> list[tuple[int, np.ndarray]]:
    """0.5fps (2초당 1프레임) 추출."""
    def _read(path: Path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, round(fps * 2))
        frames, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                actual_second = int(idx / fps)
                frames.append((actual_second, frame))
            idx += 1
        cap.release()
        return frames

    frames = _read(video_path)
    if not frames:
        if reencode_h264(video_path):
            frames = _read(video_path)
    return frames

def infer_label(stem: str) -> str:
    if stem.startswith("abnormal_"):
        return "abnormal"
    if stem.startswith("normal_"):
        return "normal"
    return "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 : GT 생성
# ─────────────────────────────────────────────────────────────────────────────
SCAM_PROMPT = """You are an expert at detecting cyber scam content in social media video frames (YouTube Shorts, TikTok).

Analyze this single video frame and estimate the probability that it is from a cyber scam video.

Cyber scam indicators:
- Fake prize/lottery/giveaway notifications
- Impersonation of banks, government agencies, or law enforcement
- Urgent warnings about account security, legal threats, or fines
- Suspicious financial transaction or wire transfer interfaces
- Phishing UI elements, fake login screens
- Countdown timers, high-pressure urgency cues
- Requests for personal info, OTP codes, or money transfers
- Unrealistic investment or earning claims

Non-scam examples: news reports, educational content, legitimate banking tutorials,
official government announcements, product reviews, daily vlogs.

Return ONLY a single float between 0.0 and 1.0. No explanation.
0.0 = definitely NOT scam
1.0 = definitely scam"""

def encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode()

def query_gpt(frame: np.ndarray, retries: int = 4) -> float | None:
    b64 = encode_frame(frame)
    for attempt in range(retries):
        try:
            resp = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": SCAM_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                ]}],
                max_completion_tokens=10,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"[0-9]*\.?[0-9]+", raw)
            return max(0.0, min(1.0, float(m.group()))) if m else None
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"    [GPT ERROR] {e} → {wait}초 후 재시도 ({attempt+1}/{retries})")
            time.sleep(wait)
    return None

def process_video_gt(video_path: Path) -> dict | None:
    stem = video_path.stem
    out_path = GT_DIR / f"{stem}_gt.json"

    if out_path.exists():
        print(f"  [SKIP] {stem} — 이미 처리됨")
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    label = infer_label(stem)
    print(f"\n  [GT] {stem}  (label={label})")

    frames = extract_frames_05fps(video_path)
    if not frames:
        print("    [ERROR] 프레임 추출 실패")
        return None
    frames = [(s, f) for s, f in frames if s <= MAX_DURATION_SEC]
    if not frames:
        print(f"    [SKIP] {MAX_DURATION_SEC}초 이내 프레임 없음")
        return None
    print(f"    프레임 수: {len(frames)}  (≤{MAX_DURATION_SEC}초)")

    frame_results = []
    for sec, frame in frames:
        prob = query_gpt(frame)
        if prob is None:
            print(f"    [{sec:>3}s] GPT 실패 — 건너뜀")
            continue
        frame_results.append({"second": sec, "probability": prob})
        print(f"    [{sec:>3}s] {prob:.3f}")
        time.sleep(API_DELAY)

    if not frame_results:
        print("    [ERROR] 채점된 프레임 없음")
        return None

    probs = [r["probability"] for r in frame_results]
    arr   = np.array(probs)
    result = {
        "video_name":          stem,
        "label":               label,
        "num_frames_scored":   len(frame_results),
        "frame_probabilities": frame_results,
        "distribution": {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "median": float(np.median(arr)),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"    저장 → {out_path.name}")
    return result

def load_csv_stems() -> list[tuple[str, str]]:
    if not CSV_PATH.exists():
        return []
    return [(label, f"{label}_{extract_video_id(link)}")
            for label, link in _read_csv_rows()]

def generate_gt_all() -> list[dict]:
    print("\n" + "=" * 55)
    print("  Step 1: GT 생성")
    print("=" * 55)

    csv_stems = load_csv_stems()
    if not csv_stems:
        print(f"[ERROR] {CSV_PATH} 없음 또는 비어 있음")
        return []

    video_files = []
    missing = []
    for label, stem in csv_stems:
        matches = list(VIDEO_DIR.glob(f"{stem}.*"))
        if matches:
            video_files.append(matches[0])
        else:
            missing.append(stem)

    if missing:
        print(f"[WARN] 다운로드 안 된 영상 {len(missing)}개: {missing}")

    extra = [p for p in VIDEO_DIR.glob("*.mp4")
             if p.stem not in {s for _, s in csv_stems}]
    if extra:
        print(f"[WARN] CSV에 없는 영상 {len(extra)}개 무시됨: {[p.name for p in extra]}")

    if not video_files:
        print(f"[ERROR] {VIDEO_DIR} 에 처리할 영상 없음")
        return []

    print(f"CSV 항목 {len(csv_stems)}개 → 처리 대상 {len(video_files)}개\n")

    csv_stem_set = {stem for _, stem in csv_stems}
    seen = {}
    for gt_file in sorted(GT_DIR.glob("*_gt.json")):
        stem = gt_file.stem.replace("_gt", "")
        if stem in csv_stem_set:
            seen[stem] = json.load(open(gt_file, encoding="utf-8"))
    all_results = list(seen.values())
    print(f"  기존 GT 로드: {len(all_results)}개")

    for vf in sorted(video_files):
        result = process_video_gt(vf)
        if result and result.get("video_name") not in seen:
            seen[result["video_name"]] = result
            all_results.append(result)

    with open(GT_ALL, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  gt_all.json 저장 ({len(all_results)}개 영상)")
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 : 재라벨링
# ─────────────────────────────────────────────────────────────────────────────
def relabel_by_frame_ratio(
    gt_data: list[dict],
    scam_prob_thr: float = SCAM_FRAME_THR,
    relabel_ratio: float = RELABEL_RATIO,
) -> list[dict]:
    """프레임 사기 확률 비율 기준으로 label 재분류.

    · abnormal 영상인데 사기 프레임 비율 < relabel_ratio → normal로 변경
    · normal  영상인데 사기 프레임 비율 >= relabel_ratio → abnormal로 변경
    """
    print("\n" + "=" * 55)
    print(f"  Step 2: 재라벨링  (사기 프레임 기준 prob>={scam_prob_thr:.1f}, 비율 기준={relabel_ratio:.0%})")
    print("=" * 55)

    relabeled = []
    n_abn_to_norm = 0
    n_norm_to_abn = 0

    n_skipped = 0
    for video in gt_data:
        # 이미 재라벨링된 항목은 건너뜀 (original_label 필드가 있으면 기존 데이터)
        if "original_label" in video:
            relabeled.append(video)
            n_skipped += 1
            continue

        probs = [fp["probability"] for fp in video["frame_probabilities"]]
        if not probs:
            relabeled.append(video)
            continue

        scam_ratio = sum(1 for p in probs if p >= scam_prob_thr) / len(probs)
        orig_label = video["label"]
        new_label  = orig_label

        if orig_label == "abnormal" and scam_ratio < relabel_ratio:
            new_label = "normal"
            n_abn_to_norm += 1
            print(f"  [abnormal→normal] {video['video_name'][:45]}  "
                  f"사기프레임={scam_ratio:.1%} ({sum(1 for p in probs if p>=scam_prob_thr)}/{len(probs)})")

        elif orig_label == "normal" and scam_ratio >= relabel_ratio:
            new_label = "abnormal"
            n_norm_to_abn += 1
            print(f"  [normal→abnormal] {video['video_name'][:45]}  "
                  f"사기프레임={scam_ratio:.1%} ({sum(1 for p in probs if p>=scam_prob_thr)}/{len(probs)})")

        updated = dict(video)
        updated["label"]            = new_label
        updated["original_label"]   = orig_label
        updated["scam_frame_ratio"] = round(scam_ratio, 4)
        relabeled.append(updated)

    print(f"\n  재라벨링 결과:")
    print(f"    기존 데이터 유지   : {n_skipped}개 (스킵)")
    print(f"    abnormal → normal : {n_abn_to_norm}개")
    print(f"    normal → abnormal : {n_norm_to_abn}개")
    print(f"    변경 없음          : {len(gt_data) - n_skipped - n_abn_to_norm - n_norm_to_abn}개")

    n_abn = sum(1 for v in relabeled if v["label"] == "abnormal")
    n_nor = sum(1 for v in relabeled if v["label"] == "normal")
    print(f"\n  최종 구성: abnormal={n_abn}  normal={n_nor}  total={len(relabeled)}")

    with open(GT_ALL, "w", encoding="utf-8") as f:
        json.dump(relabeled, f, indent=2, ensure_ascii=False)
    print(f"  gt_all.json 업데이트 완료")

    return relabeled

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 : Segmentation  (early / late 2구간)
# ─────────────────────────────────────────────────────────────────────────────
def detect_breakpoint(probs: list[float]) -> int:
    """ruptures로 최적 분할점 탐지. 실패 시 중간점 반환."""
    if HAS_RUPTURES and len(probs) >= 4:
        signal = np.array(probs).reshape(-1, 1)
        try:
            algo = rpt.Dynp(model="rbf", min_size=2, jump=1).fit(signal)
            bkps = algo.predict(n_bkps=1)
            return bkps[0] - 1
        except Exception:
            pass
    return len(probs) // 2

def segment_video(video_data: dict) -> dict:
    probs   = [f["probability"] for f in video_data["frame_probabilities"]]
    seconds = [f["second"]      for f in video_data["frame_probabilities"]]
    n       = len(probs)

    bp     = detect_breakpoint(probs)
    method = "dynamic_2seg" if HAS_RUPTURES and n >= 4 else "fixed_2seg"

    seg_labels = ["early"] * n
    for i in range(bp + 1, n):
        seg_labels[i] = "late"

    seg_summary = {}
    for seg in set(seg_labels):
        idxs = [i for i, s in enumerate(seg_labels) if s == seg]
        seg_summary[seg] = {
            "n_frames":     len(idxs),
            "mean_gt_prob": float(np.mean([probs[i] for i in idxs])),
            "frame_range":  [seconds[idxs[0]], seconds[idxs[-1]]],
        }

    return {
        "video_name":   video_data["video_name"],
        "label":        video_data["label"],
        "n_frames":     n,
        "method":       method,
        "n_segments":   2,
        "breakpoint":   bp,
        "frame_segments": [
            {"second": seconds[i], "segment": seg_labels[i], "gt_prob": probs[i]}
            for i in range(n)
        ],
        "segment_summary": seg_summary,
    }

def generate_segmentation(gt_data: list[dict]):
    print("\n" + "=" * 55)
    print("  Step 3: Segmentation")
    print("=" * 55)

    results = []
    for video in gt_data:
        seg = segment_video(video)
        results.append(seg)

        segs = list(seg["segment_summary"].keys())
        print(f"  [{seg['label']:8}] {seg['video_name'][:35]:35} "
              f"n={seg['n_frames']:3}  {seg['method']:15}  {segs}")
        for s, info in seg["segment_summary"].items():
            print(f"             {s:6}: {info['n_frames']:2}프레임  "
                  f"GT mean={info['mean_gt_prob']:.3f}  "
                  f"sec {info['frame_range'][0]}~{info['frame_range'][1]}")

    with open(SEG_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    method_counts = Counter(r["method"] for r in results)
    print(f"\n  segmentation.json 저장 ({len(results)}개 영상)")
    print("  방법별:", dict(method_counts))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  데이터 전처리 파이프라인")
    print(f"  labels : {CSV_PATH}")
    print(f"  videos : {VIDEO_DIR}")
    print(f"  GPT    : {GPT_MODEL}")
    print("=" * 55 + "\n")

    # Step 0: Download
    download_all()

    # Step 1: GT 생성
    gt_data = generate_gt_all()
    if not gt_data:
        print("[ERROR] GT 생성 실패. 종료합니다.")
        return

    # Step 2: 재라벨링
    gt_data = relabel_by_frame_ratio(gt_data)

    # Step 3: Segmentation (재라벨링된 gt_data 기준)
    generate_segmentation(gt_data)

    print("\n" + "=" * 55)
    print("  전처리 완료")
    print(f"  gt_all.json       → {GT_ALL}")
    print(f"  segmentation.json → {SEG_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()
