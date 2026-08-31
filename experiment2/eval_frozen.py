"""
eval_frozen.py — tempo_cov_frozen 최적 프롬프트로 테스트 평가만 실행.

서버에서 frozen 실행이 메모리 오류로 test eval이 저장 안 됐을 때 재실행용.
체크포인트가 있으면 pm 로드해 Top-3 앙상블, 없으면 단일 프롬프트만.

Usage:
  cd /home/.../TEMPO
  CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 python experiment2/eval_frozen.py
"""
import json, random, sys, os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tempo_cov                          # 전역 TOOLS_ENABLED 등 공유
tempo_cov.TOOLS_ENABLED = False          # Phase1 최적 프롬프트 → tools 없이 평가

from tempo_cov import score_dataset, OUT_DIR, TRAIN_RATIO, DEV_RATIO, SEED
from utils import split_data, load_gt, compute_metrics, find_best_threshold
from em_core_tool2 import FeedbackMemory, PromptMemory

random.seed(SEED)
np.random.seed(SEED)

# ── 최적 프롬프트 (frozen 실행 Round 12 출력 기준) ────────────────────────────
BEST_EARLY = (
    "Predict the probability (0.0-1.0) that this video frame is part of a cyber scam. "
    "Distinguish between educational content and actual scams using visual analysis. "
    "Legitimate videos feature formal presentation styles, clear disclaimers, official "
    "branding, structured layouts, and explicit warnings. Scam frames may display friendly "
    "or authoritative tones, urgent calls to action, ambiguous or exaggerated claims, "
    "overpromising rewards, and hidden or misleading source information. Look for "
    "trust-building tactics and vague claims aiming to manipulate viewer actions. Contrast "
    "these with formal, structured videos that include disclaimers. Focus on early frames "
    "with subtle trust-building tactics, hints of overpromising, and unclear or exaggerated "
    "claims. Key visual cues include misleading source details and manipulation intent. "
    "Ensure the analysis captures the essence of deceptive intentions versus legitimate "
    "content, emphasizing transparency and clarity. "
    "The output must be ONLY a float between 0.0 and 1.0."
)

BEST_LATE = (
    "Predict a single float (0.0–1.0) indicating the probability of a cyber scam in "
    "this video frame. Focus on identifying red flags such as urgent calls to action, "
    "unrealistic promises, emotional manipulation, and requests for personal information. "
    "Analyze visual elements like rapid scene changes, secretive settings, and erratic "
    "interactions. Exclude formal setups like news desks, police stations, and content "
    "emphasizing awareness and prevention. Disregard videos featuring identifiable "
    "individuals in professional attire, structured dialogues, and educational text overlays "
    "about scams. Concentrate on signs of manipulation or misleading information that suggest "
    "a scam. Ensure high accuracy by distinguishing between educational content and actual "
    "scams through visual patterns and intent. The model should prioritize analyzing these "
    "elements while disregarding educational content about scams. "
    "Output must be a single float between 0.0 and 1.0."
)

# ── 체크포인트에서 pm 복원 (없으면 단일 프롬프트만) ──────────────────────────
CKPT_PATH = OUT_DIR / "tempo_cov_frozen_checkpoint.json"
early_pm = PromptMemory()
late_pm  = PromptMemory()
if CKPT_PATH.exists():
    ck = json.load(open(CKPT_PATH))
    early_pm.memory = [(p, s) for p, s in ck.get("early_pm", [])]
    late_pm.memory  = [(p, s) for p, s in ck.get("late_pm",  [])]
    print(f"체크포인트 로드: early_pm={len(early_pm.memory)}  late_pm={len(late_pm.memory)}")
else:
    early_pm.memory = [(BEST_EARLY, 1.0)]
    late_pm.memory  = [(BEST_LATE,  1.0)]
    print("체크포인트 없음 — 단일 프롬프트로만 앙상블 구성")

# FeedbackMemory (tools=False이므로 retrieve_tools 호출 안 됨)
early_fb_mem = FeedbackMemory()
late_fb_mem  = FeedbackMemory()

# ── 데이터 ───────────────────────────────────────────────────────────────────
gt_list = load_gt()
_, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
print(f"Dev={len(devset)}  Test={len(testset)}")

def _fmt(v): return f"{v:.3f}" if v is not None else "None"

# ── 단일 최고 프롬프트 ────────────────────────────────────────────────────────
print("\n▶ TEMPO-COV-Frozen — 단일 최고 프롬프트 (tools=OFF)")
dev_res    = score_dataset(devset,  BEST_EARLY, BEST_LATE, early_fb_mem, late_fb_mem)
best_thr   = find_best_threshold(dev_res)
single_res = score_dataset(testset, BEST_EARLY, BEST_LATE, early_fb_mem, late_fb_mem)
single_m   = compute_metrics(single_res)
single_m_o = compute_metrics(single_res, best_thr)
print(f"  AUC={_fmt(single_m['auc'])}  MAP={_fmt(single_m['map'])}  "
      f"F1={_fmt(single_m['f1'])}  ACC={_fmt(single_m['acc'])}  F1_opt={_fmt(single_m_o['f1'])}")

# ── Top-3 앙상블 ──────────────────────────────────────────────────────────────
print("\n▶ TEMPO-COV-Frozen — Top-3 앙상블")
_top_e = sorted(early_pm.memory, key=lambda x: x[1], reverse=True)[:3]
_top_l = sorted(late_pm.memory,  key=lambda x: x[1], reverse=True)[:3]
ensemble_pairs = [(ep, lp, max((es + ls) / 2, 1e-6))
                  for (ep, es), (lp, ls) in zip(_top_e, _top_l)]
if not ensemble_pairs:
    ensemble_pairs = [(BEST_EARLY, BEST_LATE, 1.0)]


def _ensemble_score(dataset):
    acc: dict = defaultdict(list)
    lookup: dict = {}
    for ep, lp, w in ensemble_pairs:
        for r in score_dataset(dataset, ep, lp, early_fb_mem, late_fb_mem):
            key = (r["video"], r["second"])
            acc[key].append((r["pred"], w))
            if key not in lookup:
                lookup[key] = r
    out = []
    for key, pw in acc.items():
        if key in lookup:
            r = dict(lookup[key])
            wsum = sum(w for _, w in pw)
            r["pred"] = sum(p * w for p, w in pw) / wsum
            out.append(r)
    return out


ens_dev = _ensemble_score(devset)
ens_thr = find_best_threshold(ens_dev)
ens_res = _ensemble_score(testset)
ens_m   = compute_metrics(ens_res)
ens_m_o = compute_metrics(ens_res, ens_thr)
print(f"  앙상블 수: {len(ensemble_pairs)}")
print(f"  AUC={_fmt(ens_m['auc'])}  MAP={_fmt(ens_m['map'])}  "
      f"F1={_fmt(ens_m['f1'])}  ACC={_fmt(ens_m['acc'])}  F1_opt={_fmt(ens_m_o['f1'])}")

# ── 저장 ─────────────────────────────────────────────────────────────────────
result = {
    "method":              "TEMPO-COV-Frozen",
    "best_early_prompt":   BEST_EARLY,
    "best_late_prompt":    BEST_LATE,
    "single_test":         single_m,
    "single_test_f1_opt":  single_m_o["f1"],
    "single_test_acc_opt": single_m_o["acc"],
    "best_test":           ens_m,
    "best_test_f1_opt":    ens_m_o["f1"],
    "best_test_acc_opt":   ens_m_o["acc"],
    "n_ensemble_pairs":    len(ensemble_pairs),
}
out_path = OUT_DIR / "tempo_cov_frozen_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n결과 저장: {out_path}")
