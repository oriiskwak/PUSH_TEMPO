"""
eval_title_ngram_interpret.py — title-only baseline이 어떤 단어/n-gram에 의존하는지
해석. bge-m3 dense embedding은 coefficient가 특정 단어에 대응되지 않으므로,
같은 train/test split으로 TF-IDF(word + char n-gram) + Logistic Regression을
별도로 학습해 top coefficient(가중치가 큰 feature)를 직접 뽑는다.
eval_title_baseline.py(임베딩 버전)는 건드리지 않음 — 해석 전용 보조 모델.

Usage:
  python experiment2/eval_title_ngram_interpret.py
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

GT_ALL    = ROOT / "data" / "gt_results" / "gt_all.json"
TEXT_META = ROOT / "data" / "text_meta.json"
SEED      = 42


def load_gt(max_sec: int = 150):
    data = json.load(open(GT_ALL, encoding="utf-8"))
    filtered = []
    for v in data:
        v["frame_probabilities"] = [f for f in v["frame_probabilities"] if f["second"] <= max_sec]
        if v["frame_probabilities"]:
            filtered.append(v)
    return filtered


def split_data(gt_list, train_ratio=0.6, dev_ratio=0.2, seed=42):
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


if __name__ == "__main__":
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, 0.6, 0.2, SEED)
    text_meta = json.load(open(TEXT_META, encoding="utf-8"))

    def titles_labels(dataset):
        t, y = [], []
        for v in dataset:
            title = (text_meta.get(v["video_name"], {}) or {}).get("title", "").strip()
            if not title:
                continue
            t.append(title)
            y.append(1 if v["label"] == "abnormal" else 0)
        return t, y

    train_t, train_y = titles_labels(trainset)
    test_t,  test_y  = titles_labels(testset)

    for analyzer_name, kwargs in [
        ("word (whitespace 토큰, 해시태그/이모지 포함)", dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 1), min_df=2)),
        ("char (2~4-gram, 특수문자/부분단어 패턴 포착)", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3)),
    ]:
        print(f"\n{'='*70}\n[{analyzer_name}]\n{'='*70}")
        vec = TfidfVectorizer(**kwargs)
        X_train = vec.fit_transform(train_t)
        X_test  = vec.transform(test_t)

        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        clf.fit(X_train, train_y)

        test_score = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(test_y, test_score)
        mAP = average_precision_score(test_y, test_score)
        print(f"(참고용 TF-IDF baseline 성능) test AUC={auc:.4f}  mAP={mAP:.4f}  (임베딩 버전과 비슷하면 해석 타당)")

        feat_names = np.array(vec.get_feature_names_out())
        coefs = clf.coef_[0]
        top_abnormal = np.argsort(coefs)[::-1][:20]
        top_normal   = np.argsort(coefs)[:20]

        print("\n▶ '사기(abnormal)' 방향 가중치 상위 20개:")
        for i in top_abnormal:
            print(f"    {feat_names[i]!r:30s}  coef={coefs[i]:+.3f}")

        print("\n▶ '정상(normal)' 방향 가중치 상위 20개:")
        for i in top_normal:
            print(f"    {feat_names[i]!r:30s}  coef={coefs[i]:+.3f}")
