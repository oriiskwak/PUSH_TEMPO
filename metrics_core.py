"""
metrics_core.py — 이진 탐지(scam/normal) 평가 지표의 단일 정의처(single source of truth).

이전에 utils.py / utils_intern.py / baselines/eval_metrics.py / baselines/utils_gpt.py /
gpt_experiment/utils_gpt.py 5곳에 compute_metrics/find_best_threshold가 각자 정의돼 있었고,
실제로 아래처럼 서로 동작이 갈라져 있었다 (리뷰에서 발견):
  - utils_intern.py의 compute_metrics는 kl/f1_macro/*_ratio 필드가 빠진 축약판
  - utils_intern.py의 find_best_threshold는 grid step이 0.05(19개 후보)로
    utils.py의 0.01(90개 후보)보다 성김
  - baselines/eval_metrics.py는 클래스가 하나뿐인 미정의 구간에서 None 대신 0.0을 반환

이 파일 하나로 통합하고, 나머지 5곳은 여기서 re-export만 한다.
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

RATIO_THRESHOLD = 0.10  # data_pipeline/prepare_data.py의 RELABEL_RATIO(GT 재라벨링 기준)와 동일 비율


def compute_metrics_from_frames(frame_results: list[dict], threshold: float = 0.5) -> dict:
    """프레임 결과 집계 → KL(프레임 레벨) + AUC + MAP + F1 + ACC(영상 레벨, 평균 기반)
    + *_ratio(영상 레벨, RATIO_THRESHOLD 기준 비율 기반) 반환.

    score=mean(pred) 기반 값은 EM 탐색 라운드 선택("map" 키)에 그대로 쓰인다.
    *_ratio는 GT 재라벨링("프레임의 RATIO_THRESHOLD 이상이 scam이면 영상도 scam")과
    동일한 철학으로 scam 프레임 비율을 score로 쓴 별도 값.
    """
    empty = {"kl": None, "auc": None, "map": None, "f1": None, "f1_macro": None, "acc": None,
             "auc_ratio": None, "map_ratio": None, "f1_ratio": None, "acc_ratio": None}
    if not frame_results:
        return empty

    has_kl  = "kl" in frame_results[0]
    mean_kl = float(np.mean([r["kl"] for r in frame_results])) if has_kl else None

    vid_map: dict = defaultdict(lambda: {"preds": [], "label": None})
    for r in frame_results:
        vid_map[r["video"]]["preds"].append(r["pred"])
        vid_map[r["video"]]["label"] = r["label"]

    scores       = [float(np.mean(v["preds"])) for v in vid_map.values()]
    ratio_scores = [sum(1 for p in v["preds"] if p >= 0.5) / len(v["preds"]) for v in vid_map.values()]
    true_labels  = [1 if v["label"] == "abnormal" else 0 for v in vid_map.values()]
    pred_labels       = [1 if s >= threshold      else 0 for s in scores]
    pred_labels_ratio = [1 if s >= RATIO_THRESHOLD else 0 for s in ratio_scores]

    if len(set(true_labels)) < 2:
        return {**empty, "kl": mean_kl}

    return {
        "kl":        mean_kl,
        "auc":       float(roc_auc_score(true_labels, scores)),
        "map":       float(average_precision_score(true_labels, scores)),
        "f1":        float(f1_score(true_labels, pred_labels, zero_division=0)),
        "f1_macro":  float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "acc":       float(accuracy_score(true_labels, pred_labels)),
        "auc_ratio": float(roc_auc_score(true_labels, ratio_scores)),
        "map_ratio": float(average_precision_score(true_labels, ratio_scores)),
        "f1_ratio":  float(f1_score(true_labels, pred_labels_ratio, zero_division=0)),
        "acc_ratio": float(accuracy_score(true_labels, pred_labels_ratio)),
    }


def compute_metrics_from_preds(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict:
    """이미 분류된 (y_true, y_pred, y_score)에서 AUC/MAP/F1/ACC 계산 — APO/ERM/EvoPrompt처럼
    프레임 집계를 직접 하지 않고 video-level 예측을 이미 가진 baseline 호출부용.
    compute_metrics_from_frames와 동일한 sklearn 호출 + 동일한 미정의(단일 클래스) 처리를 공유."""
    if len(set(y_true)) < 2:
        return {"auc": None, "map": None, "f1": None, "acc": None}
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "map": float(average_precision_score(y_true, y_score)),
        "f1":  float(f1_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


def _search_threshold(true_labels: list[int], scores: list[float]) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = [1 if s >= t else 0 for s in scores]
        f = f1_score(true_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(round(t, 2))
    return best_t


def find_best_threshold_from_frames(frame_results: list[dict]) -> float:
    """dev set 결과에서 F1 최대화 threshold 탐색 (0.05~0.95, step 0.01, 90개 후보)."""
    if not frame_results:
        return 0.5
    vid_map: dict = defaultdict(lambda: {"preds": [], "label": None})
    for r in frame_results:
        vid_map[r["video"]]["preds"].append(r["pred"])
        vid_map[r["video"]]["label"] = r["label"]
    scores      = [float(np.mean(v["preds"])) for v in vid_map.values()]
    true_labels = [1 if v["label"] == "abnormal" else 0 for v in vid_map.values()]
    return _search_threshold(true_labels, scores)


def find_best_threshold_from_preds(y_true: list[int], y_score: list[float]) -> float:
    """(y_true, y_score)에서 F1 최대화 threshold 탐색 — find_best_threshold_from_frames와
    동일 grid(0.05~0.95, step 0.01)."""
    return _search_threshold(y_true, y_score)
