"""
eval_visual_baseline.py — 순수 시각 정보만 이용한 이진분류(scam/normal) baseline.

VLM 프롬프트 최적화(CYREN/UniAPO) 없이, CYREN과 동일한 프레임 샘플링(FRAME_STEP=2초
간격, MAX_SEC=150초 → 최대 75프레임/영상)으로 뽑은 프레임을 SigLIP(또는 CLIP)으로
임베딩(평균 pooling)하고 logistic regression으로 분류. title/channel 등 메타데이터는
전혀 쓰지 않음 — "VLM 프롬프트 최적화 vs 단순 이미지 분류기"라는 공정한 대비를
보여주기 위한 baseline. eval_title_baseline.py와 동일하게 utils.py(Qwen)의
stratified split(SEED=42)을 그대로 복제해서 사용.

Usage:
  python experiment2/eval_visual_baseline.py --model siglip   # 기본값
  python experiment2/eval_visual_baseline.py --model clip
"""
import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from metrics_core import find_best_threshold_from_preds

GT_ALL    = ROOT / "data" / "gt_results" / "gt_all.json"
VIDEO_DIR = ROOT / "data" / "videos"

TRAIN_RATIO = 0.6
DEV_RATIO   = 0.2
SEED        = 42
MAX_SEC     = 150   # CYREN(tempo_cov.py)의 MAX_SEC과 동일
FRAME_STEP  = 2     # CYREN(tempo_cov.py)의 FRAME_STEP과 동일 (최대 75프레임/영상)
EMBED_BATCH = 32

MODEL_NAMES = {
    "siglip": "google/siglip-base-patch16-224",
    "clip":   "openai/clip-vit-base-patch32",
}


# ── utils.py의 load_gt/split_data 복제 (eval_title_baseline.py와 동일) ──────
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


def _sample_frames(video_path: Path, step: int = FRAME_STEP,
                    max_sec: int = MAX_SEC) -> list[Image.Image]:
    """CYREN(experiment2/tempo_cov.py)의 프레임 추출과 동일: 0초부터 step(2초) 간격,
    max_sec(150초)까지 → 최대 75프레임/영상."""
    cap   = cv2.VideoCapture(str(video_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sec = min(int(total / fps_v), max_sec) if fps_v > 0 else max_sec

    imgs = []
    for sec in range(0, max(n_sec, 1), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_v))
        ret, frame = cap.read()
        if ret:
            imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return imgs


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["siglip", "clip"], default="siglip")
    args = parser.parse_args()
    model_name = MODEL_NAMES[args.model]
    out_path = ROOT / "results" / "qwen" / f"visual_{args.model}_baseline_result.json"

    print("=== 데이터셋 로드 (utils.py와 동일한 stratified split, SEED=42) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== {model_name} 로드 (device={device}) ===")
    if args.model == "siglip":
        from transformers import SiglipModel, SiglipProcessor
        model     = SiglipModel.from_pretrained(model_name).to(device).eval()
        processor = SiglipProcessor.from_pretrained(model_name)
    else:
        from transformers import CLIPModel, CLIPProcessor
        model     = CLIPModel.from_pretrained(model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_video(vname: str) -> np.ndarray | None:
        vfiles = list(VIDEO_DIR.glob(f"{vname}.*"))
        if not vfiles:
            return None
        imgs = _sample_frames(vfiles[0])
        if not imgs:
            return None
        chunk_feats = []
        for start in range(0, len(imgs), EMBED_BATCH):
            chunk = imgs[start:start + EMBED_BATCH]
            inputs = processor(images=chunk, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            chunk_feats.append(feats)
        all_feats = torch.cat(chunk_feats, dim=0)
        return all_feats.mean(dim=0).cpu().numpy()

    def embed_set(dataset, name):
        X, y, names, dropped = [], [], [], 0
        for i, v in enumerate(dataset, 1):
            emb = embed_video(v["video_name"])
            if emb is None:
                dropped += 1
                continue
            X.append(emb)
            y.append(1 if v["label"] == "abnormal" else 0)
            names.append(v["video_name"])
            if i % 50 == 0 or i == len(dataset):
                print(f"  [{name}] {i}/{len(dataset)}  (누락 {dropped})")
        return np.stack(X), np.array(y), names, dropped

    print(f"\n=== 프레임 추출(CYREN과 동일: {FRAME_STEP}초 간격, 최대 {MAX_SEC}초) + 임베딩 ===")
    X_train, y_train, _,          _ = embed_set(trainset, "train")
    X_dev,   y_dev,   _,          _ = embed_set(devset,   "dev")
    X_test,  y_test,  test_names, _ = embed_set(testset,  "test")

    print("\n=== Logistic Regression 학습 ===")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(X_train, y_train)

    dev_score  = clf.predict_proba(X_dev)[:, 1]
    test_score = clf.predict_proba(X_test)[:, 1]

    dev_m_05  = evaluate(y_dev,  dev_score,  0.5)
    test_m_05 = evaluate(y_test, test_score, 0.5)

    best_thr   = find_best_threshold_from_preds(y_dev.tolist(), dev_score.tolist())
    test_m_opt = evaluate(y_test, test_score, best_thr)

    cd_path = ROOT / "data" / "channel_disjoint_analysis.json"
    clean_m_05 = None
    if cd_path.exists():
        clean_names = set(json.load(open(cd_path, encoding="utf-8"))["clean_test_video_names"])
        clean_idx = [i for i, n in enumerate(test_names) if n in clean_names]
        if clean_idx:
            clean_y     = y_test[clean_idx]
            clean_score = test_score[clean_idx]
            clean_m_05  = evaluate(clean_y, clean_score, 0.5)
            print(f"\n[channel-clean 서브셋, n={len(clean_idx)}/{len(test_names)}]")
            print(f"▶ Test-clean (thr=0.5)  AUC={_fmt(clean_m_05['auc'])}  MAP={_fmt(clean_m_05['map'])}  "
                  f"F1={_fmt(clean_m_05['f1'])}  ACC={_fmt(clean_m_05['acc'])}")

    print(f"\n▶ Dev (thr=0.5)   AUC={_fmt(dev_m_05['auc'])}  MAP={_fmt(dev_m_05['map'])}  "
          f"F1={_fmt(dev_m_05['f1'])}  ACC={_fmt(dev_m_05['acc'])}")
    print(f"▶ Test (thr=0.5)  AUC={_fmt(test_m_05['auc'])}  MAP={_fmt(test_m_05['map'])}  "
          f"F1={_fmt(test_m_05['f1'])}  ACC={_fmt(test_m_05['acc'])}")
    print(f"\n[dev 최적 threshold={best_thr:.2f}]")
    print(f"▶ Test (thr=opt)  AUC={_fmt(test_m_opt['auc'])}  MAP={_fmt(test_m_opt['map'])}  "
          f"F1={_fmt(test_m_opt['f1'])}  ACC={_fmt(test_m_opt['acc'])}")

    result = {
        "method": f"Visual-only Baseline ({args.model} embedding, mean of frames sampled every {FRAME_STEP}s up to {MAX_SEC}s, same as CYREN + Logistic Regression)",
        "embedding_model": model_name,
        "frame_step_sec": FRAME_STEP,
        "max_sec": MAX_SEC,
        "split": {"train": len(y_train), "dev": len(y_dev), "test": len(y_test),
                  "seed": SEED, "note": "utils.py(Qwen)와 동일한 stratified split"},
        "dev_test_0.5": {"dev": dev_m_05, "test": test_m_05},
        "best_threshold": best_thr,
        "dev_test_opt": {"test": test_m_opt},
        "channel_clean_test": clean_m_05,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
