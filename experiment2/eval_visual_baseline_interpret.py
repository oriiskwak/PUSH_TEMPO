"""
eval_visual_baseline_interpret.py — visual-only(SigLIP) baseline이 실제로 뭘 보고
판단하는지 해석.
1. 영상별 duration/resolution과 classifier의 예측 점수 간 상관관계 (surface confound 확인)
2. 평균 pooling + linear classifier는 프레임별 기여도가 가산적(additive)이므로
   각 영상에서 w·frame_embedding이 가장 큰(=가장 "사기스럽다"고 판단된) 프레임과
   가장 작은(=가장 "정상스럽다"고 판단된) 프레임을 골라 이미지로 저장 → 육안 확인용.
eval_visual_baseline.py는 건드리지 않고 독립 실행(같은 학습 로직을 재현).

Usage:
  python experiment2/eval_visual_baseline_interpret.py
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression

GT_ALL      = ROOT / "data" / "gt_results" / "gt_all.json"
VIDEO_DIR   = ROOT / "data" / "videos"
OUT_DIR     = ROOT / "results" / "qwen" / "visual_interpret_frames"
SEED        = 42
MAX_SEC     = 150
FRAME_STEP  = 2
EMBED_BATCH = 32
MODEL_NAME  = "google/siglip-base-patch16-224"
N_EXAMPLES  = 4   # 시각화할 영상 개수 (정상/사기 각각)


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


def sample_frames_with_seconds(video_path, step=FRAME_STEP, max_sec=MAX_SEC):
    cap   = cv2.VideoCapture(str(video_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sec = min(int(total / fps_v), max_sec) if fps_v > 0 else max_sec
    dur   = total / fps_v if fps_v else 0

    imgs, secs = [], []
    for sec in range(0, max(n_sec, 1), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_v))
        ret, frame = cap.read()
        if ret:
            imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            secs.append(sec)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return imgs, secs, dur, w, h


if __name__ == "__main__":
    print("=== 데이터셋 로드 ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, 0.6, 0.2, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== {MODEL_NAME} 로드 (device={device}) ===")
    from transformers import SiglipModel, SiglipProcessor
    model     = SiglipModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = SiglipProcessor.from_pretrained(MODEL_NAME)

    @torch.no_grad()
    def embed_frames(imgs):
        chunk_feats = []
        for start in range(0, len(imgs), EMBED_BATCH):
            chunk = imgs[start:start + EMBED_BATCH]
            inputs = processor(images=chunk, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            chunk_feats.append(feats)
        return torch.cat(chunk_feats, dim=0)  # (n_frames, dim)

    def build_set(dataset, name, keep_frame_embs=False):
        X, y, names, durs, ws, hs = [], [], [], [], [], []
        frame_store = {}  # vname -> (frame_embs tensor, seconds, imgs) 필요시만
        for i, v in enumerate(dataset, 1):
            vfiles = list(VIDEO_DIR.glob(f"{v['video_name']}.*"))
            if not vfiles:
                continue
            imgs, secs, dur, w, h = sample_frames_with_seconds(vfiles[0])
            if not imgs:
                continue
            feats = embed_frames(imgs)
            X.append(feats.mean(dim=0).cpu().numpy())
            y.append(1 if v["label"] == "abnormal" else 0)
            names.append(v["video_name"])
            durs.append(dur); ws.append(w); hs.append(h)
            if keep_frame_embs:
                frame_store[v["video_name"]] = (feats.cpu().numpy(), secs, imgs)
            if i % 50 == 0 or i == len(dataset):
                print(f"  [{name}] {i}/{len(dataset)}")
        return (np.stack(X), np.array(y), names, np.array(durs), np.array(ws), np.array(hs), frame_store)

    print("\n=== train/test 임베딩 ===")
    X_train, y_train, *_ = build_set(trainset, "train")
    X_test, y_test, test_names, test_durs, test_ws, test_hs, frame_store = build_set(
        testset, "test", keep_frame_embs=True)

    print("\n=== Logistic Regression 학습 ===")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(X_train, y_train)
    test_score = clf.predict_proba(X_test)[:, 1]
    w_vec = clf.coef_[0]
    b_vec = clf.intercept_[0]

    # ── 1. duration/resolution vs predicted score 상관관계 ──────────────────
    print(f"\n{'='*70}\n[1] Surface feature vs predicted score 상관관계 (test set, n={len(test_score)})\n{'='*70}")
    for name, vals in [("duration", test_durs), ("width", test_ws), ("height", test_hs)]:
        r_p, p_p = pearsonr(vals, test_score)
        r_s, p_s = spearmanr(vals, test_score)
        print(f"  {name:10s}  Pearson r={r_p:+.3f} (p={p_p:.4f})   Spearman rho={r_s:+.3f} (p={p_s:.4f})")

    # duration만으로 예측 점수를 얼마나 선형회귀로 설명 가능한지 (R^2)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(test_durs.reshape(-1, 1), test_score)
    r2 = lr.score(test_durs.reshape(-1, 1), test_score)
    print(f"\n  duration 하나로 predicted_score를 선형회귀했을 때 R^2 = {r2:.4f}")

    # ── 2. 프레임별 기여도 (가산적 분해: score_logit = mean_i(w·frame_i) + b) ──
    print(f"\n{'='*70}\n[2] 프레임별 기여도 상위/하위 예시 저장\n{'='*70}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 가장 확신 있게 맞춘 abnormal, normal 영상 각 N_EXAMPLES개
    correct_abn = [(n, s) for n, s, y in zip(test_names, test_score, y_test) if y == 1]
    correct_nor = [(n, s) for n, s, y in zip(test_names, test_score, y_test) if y == 0]
    top_abn = sorted(correct_abn, key=lambda x: -x[1])[:N_EXAMPLES]
    top_nor = sorted(correct_nor, key=lambda x: x[1])[:N_EXAMPLES]

    manifest = []
    for tag, examples in [("abnormal_high_score", top_abn), ("normal_low_score", top_nor)]:
        for vname, score in examples:
            frame_embs, secs, imgs = frame_store[vname]
            frame_logits = frame_embs @ w_vec + b_vec / len(secs)  # 프레임별 기여 근사
            order = np.argsort(frame_logits)
            top_idx = order[::-1][:3]   # 가장 "사기스럽다" 판단한 프레임 3장
            bot_idx = order[:3]         # 가장 "정상스럽다" 판단한 프레임 3장

            vdir = OUT_DIR / f"{tag}__{vname}"
            vdir.mkdir(parents=True, exist_ok=True)
            for rank, idx in enumerate(top_idx):
                imgs[idx].save(vdir / f"top{rank+1}_sec{secs[idx]}_logit{frame_logits[idx]:.2f}.jpg")
            for rank, idx in enumerate(bot_idx):
                imgs[idx].save(vdir / f"bot{rank+1}_sec{secs[idx]}_logit{frame_logits[idx]:.2f}.jpg")
            manifest.append({"video": vname, "video_score": float(score),
                             "top_frame_seconds": [int(secs[i]) for i in top_idx],
                             "bottom_frame_seconds": [int(secs[i]) for i in bot_idx]})
            print(f"  {tag}: {vname}  (score={score:.3f})  저장: {vdir}")

    json.dump(manifest, open(OUT_DIR / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n프레임 이미지 + manifest 저장: {OUT_DIR}")
