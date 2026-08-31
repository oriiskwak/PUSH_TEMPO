"""
utils_gpt.py — utils.py의 데이터 유틸만 분리 (Qwen/vLLM 모델 로드 없음).

utils.py를 그대로 import하면 모듈 최상단의 vLLM LLM(...) 로드가 항상 실행돼
GPU에 Qwen2.5-VL-72B가 올라간다. predictor/optimizer를 모두 GPT로 쓰는
실험(uniapo_gpt.py, tool_gpt.py 등)에서는 이게 필요 없으므로, 그 로드를
거치지 않는 별도 모듈로 분리했다. 내용은 utils.py의 데이터 유틸 함수와 동일.
"""

import json, random, os
import numpy as np
from pathlib import Path
from PIL import Image
BASE_DIR  = Path(__file__).parent.parent   # gpt_experiment/ 한 단계 위 = 프로젝트 루트 (TEMPO/)
DATA_DIR  = BASE_DIR / "data"
GT_ALL    = DATA_DIR / "gt_results" / "gt_all.json"
VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", str(BASE_DIR / "data" / "videos")))
OUT_DIR   = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)


def load_gt(max_sec: int = 150):
    data = json.load(open(GT_ALL, encoding="utf-8"))
    filtered = []
    for v in data:
        v["frame_probabilities"] = [f for f in v["frame_probabilities"] if f["second"] <= max_sec]
        if v["frame_probabilities"]:
            filtered.append(v)
    return filtered


def split_data(gt_list, train_ratio=0.6, dev_ratio=0.2, seed=42):
    """abnormal/normal 균형 유지 stratified split → train / dev / test."""
    random.seed(seed)
    abnormal = [v for v in gt_list if v["label"] == "abnormal"]
    normal   = [v for v in gt_list if v["label"] == "normal"]

    def _split(lst):
        lst = lst[:]
        random.shuffle(lst)
        n_train = int(len(lst) * train_ratio)
        n_dev   = int(len(lst) * dev_ratio)
        return lst[:n_train], lst[n_train:n_train + n_dev], lst[n_train + n_dev:]

    ab_tr, ab_dv, ab_te = _split(abnormal)
    no_tr, no_dv, no_te = _split(normal)
    return ab_tr + no_tr, ab_dv + no_dv, ab_te + no_te


def load_frame(video_path: Path, second: int) -> Image.Image | None:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


def bernoulli_kl(p: float, q: float, eps: float = 1e-7) -> float:
    p = max(eps, min(1 - eps, p))
    q = max(eps, min(1 - eps, q))
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


# compute_metrics/find_best_threshold는 metrics_core.py(single source of truth)에서 가져옴 —
# 이 파일은 utils.py의 사본이라 예전엔 로직도 그대로 복제돼 있었는데, 여러 곳에 흩어진 사본이
# 서로 조금씩 갈라지는 문제(utils_intern.py 등)가 발견돼서 metrics_core.py로 통합함.
import sys
sys.path.insert(0, str(BASE_DIR))
from metrics_core import (
    compute_metrics_from_frames as compute_metrics,
    find_best_threshold_from_frames as find_best_threshold,
    RATIO_THRESHOLD,
)
