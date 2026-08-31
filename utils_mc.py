"""
utils_mc.py — 5-class 사기 유형 분류 공용 유틸리티 (사기 유형 4종 + 정상)

5개 클래스: 보이스피싱, 투자사기, 로맨스스캠, 부업사기, 정상
621개 영상 전체 (abnormal + normal 모두) 사용, 정상 여부는 GT-relabeled v["label"] 기준
"""
import json, re, csv, random, os, cv2
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
GT_ALL    = DATA_DIR / "gt_results" / "gt_all.json"
VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", str(BASE_DIR / "data" / "videos")))
OUT_DIR   = BASE_DIR / "results_4class"
OUT_DIR.mkdir(exist_ok=True)

CLASSES     = ["보이스피싱", "투자사기", "로맨스스캠", "부업사기", "정상"]
CLASSES_STR = " / ".join(CLASSES)

# 레이블 블라인드: VLM에는 Type-A/B/C/D/E로 제시, 내부 메트릭은 한국어 유지
TYPE_MAP = {
    "Type-A": "보이스피싱",
    "Type-B": "투자사기",
    "Type-C": "로맨스스캠",
    "Type-D": "부업사기",
    "Type-E": "정상",
}
TYPE_SUFFIX = "\nReturn ONLY one of: Type-A / Type-B / Type-C / Type-D / Type-E"


# ── 데이터 유틸 ───────────────────────────────────────────────────────────────

def _extract_vid_id(link: str) -> str | None:
    """YouTube (watch?v=, /shorts/, youtu.be/) 및 TikTok URL에서 video ID 추출."""
    m = re.search(r'(?:v=|/shorts/|youtu\.be/)([a-zA-Z0-9_-]{11})', link)
    if m:
        return m.group(1)
    m = re.search(r'tiktok\.com/.*/video/(\d+)', link)
    if m:
        return m.group(1)
    return None


def load_gt_mc(max_sec: int = 150) -> list[dict]:
    """LABELS.csv + gt_all.json 조인 → 5-class(사기 유형 4종 + 정상) 레이블 붙은 영상 목록."""
    # LABELS.csv: video_id → scam_type (YouTube + TikTok 모두 처리)
    # 주의: normal 영상도 LABELS.csv에 사기 유형 테마 태그가 붙어 있으므로
    # (예: "normal,투자사기,..."), 정상/사기 판정은 이 테마 태그가 아니라
    # GT-relabeled v["label"](binary 트랙과 동일 기준)로 먼저 확인한다.
    type_map: dict[str, str] = {}
    with open(DATA_DIR / "LABELS.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row[0] == "라벨":
                continue
            link   = row[2] if len(row) > 2 else ""
            vid_id = _extract_vid_id(link)
            if vid_id:
                type_map[vid_id] = row[1]

    data = json.load(open(GT_ALL, encoding="utf-8"))
    result = []
    for v in data:
        vname = v["video_name"]
        if v["label"] == "normal":
            scam_type = "정상"
        else:
            # video_name 형식: "abnormal_YTID" / "normal_YTID"
            yt_id     = vname.split("_", 1)[1] if "_" in vname else vname
            scam_type = type_map.get(yt_id)
        if scam_type not in CLASSES:
            continue
        fps = [f for f in v["frame_probabilities"] if f["second"] <= max_sec]
        if not fps:
            continue
        result.append({
            "video_name":          vname,
            "label":               scam_type,
            "frame_probabilities": fps,
        })
    return result


def split_data_mc(gt_list: list[dict],
                  train_ratio: float = 0.6,
                  dev_ratio:   float = 0.2,
                  seed:        int   = 42):
    """4-class stratified split → (train, dev, test)."""
    random.seed(seed)
    by_class: dict[str, list] = {c: [] for c in CLASSES}
    for v in gt_list:
        if v["label"] in by_class:
            by_class[v["label"]].append(v)

    train, dev, test = [], [], []
    for c in CLASSES:
        lst = by_class[c][:]
        random.shuffle(lst)
        n_train = int(len(lst) * train_ratio)
        n_dev   = int(len(lst) * dev_ratio)
        train += lst[:n_train]
        dev   += lst[n_train:n_train + n_dev]
        test  += lst[n_train + n_dev:]

    return train, dev, test


def load_frame(video_path: Path, second: int) -> "Image.Image | None":
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


# ── 메트릭 ────────────────────────────────────────────────────────────────────

def aggregate_video_pred(frame_preds: list[str | None]) -> str:
    """프레임별 예측 클래스 다수결 → 영상 레벨 클래스."""
    valid = [p for p in frame_preds if p is not None]
    if not valid:
        return "정상"
    return Counter(valid).most_common(1)[0][0]


def compute_metrics_mc(frame_results: list[dict]) -> dict:
    """프레임 결과 집계 → acc + macro F1 + per-class F1 + confusion matrix."""
    if not frame_results:
        return {
            "acc":          None,
            "macro_f1":     None,
            "per_class_f1": {c: None for c in CLASSES},
            "confusion":    {},
        }

    vid_map: dict[str, dict] = {}
    for r in frame_results:
        vid_map.setdefault(r["video"], {"preds": [], "label": r["label"]})
        if r.get("pred_label") is not None:
            vid_map[r["video"]]["preds"].append(r["pred_label"])

    true_labels, pred_labels = [], []
    for v in vid_map.values():
        true_labels.append(v["label"])
        pred_labels.append(aggregate_video_pred(v["preds"]))

    acc      = float(accuracy_score(true_labels, pred_labels))
    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro",
                               labels=CLASSES, zero_division=0))
    per_f1   = f1_score(true_labels, pred_labels, average=None,
                         labels=CLASSES, zero_division=0)
    per_class_f1 = {c: float(per_f1[i]) for i, c in enumerate(CLASSES)}

    # confusion matrix: {true_class: {pred_class: count}}
    confusion: dict[str, dict[str, int]] = {c: {p: 0 for p in CLASSES} for c in CLASSES}
    for t, p in zip(true_labels, pred_labels):
        if t in confusion and p in confusion[t]:
            confusion[t][p] += 1

    return {
        "acc":          acc,
        "macro_f1":     macro_f1,
        "per_class_f1": per_class_f1,
        "confusion":    confusion,
    }
