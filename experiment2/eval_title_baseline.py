"""
eval_title_baseline.py — 영상 제목(title) 텍스트만 이용한 이진분류(scam/normal) baseline.

sentence-transformer(BAAI/bge-m3, 다국어·한국어 지원, CPU)로 제목을 임베딩하고
logistic regression으로 분류. utils.py의 load_gt/split_data(stratified, SEED=42)를
그대로 복제해서 기존 Qwen 트랙 결과들(baseline_em_result.json 등)과 동일한
train/dev/test split을 사용 — 무거운 vLLM 모델 로드(utils.py 직접 import) 없이
GT_ALL/split 로직만 복제해서 GPU 없이 몇 분 안에 끝나도록 함.

Usage:
  python experiment2/eval_title_baseline.py
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from metrics_core import find_best_threshold_from_preds

DATA_DIR  = ROOT / "data"
GT_ALL    = DATA_DIR / "gt_results" / "gt_all.json"
TEXT_META = DATA_DIR / "text_meta.json"
OUT_PATH  = ROOT / "results" / "qwen" / "title_baseline_result.json"

TRAIN_RATIO = 0.6
DEV_RATIO   = 0.2
SEED        = 42
EMB_MODEL   = "BAAI/bge-m3"


# ── utils.py의 load_gt/split_data를 그대로 복제 (무거운 vLLM 로드 없이) ──────
def load_gt(max_sec: int = 150) -> list[dict]:
    data = json.load(open(GT_ALL, encoding="utf-8"))
    filtered = []
    for v in data:
        v["frame_probabilities"] = [f for f in v["frame_probabilities"] if f["second"] <= max_sec]
        if v["frame_probabilities"]:
            filtered.append(v)
    return filtered


def split_data(gt_list: list[dict], train_ratio: float = 0.6,
               dev_ratio: float = 0.2, seed: int = 42):
    """abnormal/normal 균형 유지 stratified split → train / dev / test (utils.py와 동일)."""
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


def _fmt(v):
    return f"{v:.4f}" if v is not None else "None"


def evaluate(y_true, y_score, threshold):
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "map": float(average_precision_score(y_true, y_score)),
        "f1":  float(f1_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


if __name__ == "__main__":
    print("=== 데이터셋 로드 (utils.py와 동일한 stratified split, SEED=42) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    text_meta = json.load(open(TEXT_META, encoding="utf-8"))

    def _titles_labels(dataset):
        titles, labels, names, dropped = [], [], [], 0
        for v in dataset:
            meta = text_meta.get(v["video_name"])
            title = (meta or {}).get("title", "").strip()
            if not title:
                dropped += 1
                continue
            titles.append(title)
            labels.append(1 if v["label"] == "abnormal" else 0)
            names.append(v["video_name"])
        return titles, labels, names, dropped

    train_titles, train_y, _,          train_drop = _titles_labels(trainset)
    dev_titles,   dev_y,   _,          dev_drop   = _titles_labels(devset)
    test_titles,  test_y,  test_names, test_drop  = _titles_labels(testset)
    print(f"제목 있음: train={len(train_titles)}(누락 {train_drop})  "
          f"dev={len(dev_titles)}(누락 {dev_drop})  test={len(test_titles)}(누락 {test_drop})")

    print(f"\n=== 제목 임베딩 ({EMB_MODEL}, CPU) ===")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMB_MODEL, device="cpu")

    X_train = embedder.encode(train_titles, normalize_embeddings=True, show_progress_bar=True)
    X_dev   = embedder.encode(dev_titles,   normalize_embeddings=True, show_progress_bar=True)
    X_test  = embedder.encode(test_titles,  normalize_embeddings=True, show_progress_bar=True)

    print("\n=== Logistic Regression 학습 ===")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(X_train, train_y)

    dev_score  = clf.predict_proba(X_dev)[:, 1]
    test_score = clf.predict_proba(X_test)[:, 1]

    dev_m_05  = evaluate(dev_y,  dev_score,  0.5)
    test_m_05 = evaluate(test_y, test_score, 0.5)

    best_thr = find_best_threshold_from_preds(dev_y, dev_score.tolist())
    dev_m_opt  = evaluate(dev_y,  dev_score,  best_thr)
    test_m_opt = evaluate(test_y, test_score, best_thr)

    # channel-disjoint 서브셋 검증: train과 채널이 겹치지 않는 test 영상만으로 재평가
    cd_path = ROOT / "data" / "channel_disjoint_analysis.json"
    clean_m_05 = None
    if cd_path.exists():
        clean_names = set(json.load(open(cd_path, encoding="utf-8"))["clean_test_video_names"])
        clean_idx = [i for i, n in enumerate(test_names) if n in clean_names]
        if clean_idx:
            clean_y     = [test_y[i] for i in clean_idx]
            clean_score = test_score[clean_idx]
            clean_m_05  = evaluate(clean_y, clean_score, 0.5)
            print(f"\n[channel-clean 서브셋, n={len(clean_idx)}/{len(test_names)}]")
            print(f"▶ Test-clean (thr=0.5)  AUC={_fmt(clean_m_05['auc'])}  MAP={_fmt(clean_m_05['map'])}  "
                  f"F1={_fmt(clean_m_05['f1'])}  ACC={_fmt(clean_m_05['acc'])}")

    print(f"\n▶ Dev (threshold=0.5)   AUC={_fmt(dev_m_05['auc'])}  MAP={_fmt(dev_m_05['map'])}  "
          f"F1={_fmt(dev_m_05['f1'])}  ACC={_fmt(dev_m_05['acc'])}")
    print(f"▶ Test (threshold=0.5)  AUC={_fmt(test_m_05['auc'])}  MAP={_fmt(test_m_05['map'])}  "
          f"F1={_fmt(test_m_05['f1'])}  ACC={_fmt(test_m_05['acc'])}")
    print(f"\n[dev 최적 threshold={best_thr:.2f}]")
    print(f"▶ Test (threshold=opt)  AUC={_fmt(test_m_opt['auc'])}  MAP={_fmt(test_m_opt['map'])}  "
          f"F1={_fmt(test_m_opt['f1'])}  ACC={_fmt(test_m_opt['acc'])}")

    result = {
        "method": "Title-only Baseline (bge-m3 embedding + Logistic Regression)",
        "embedding_model": EMB_MODEL,
        "split": {"train": len(train_titles), "dev": len(dev_titles), "test": len(test_titles),
                  "seed": SEED, "note": "utils.py(Qwen)와 동일한 stratified split"},
        "dev_test_0.5":  {"dev": dev_m_05,  "test": test_m_05},
        "best_threshold": best_thr,
        "dev_test_opt":  {"dev": dev_m_opt, "test": test_m_opt},
        "channel_clean_test": clean_m_05,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH}")
