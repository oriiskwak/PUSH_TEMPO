"""
erm_gpt.py — baselines/erm_video.py(ERM: Exemplar-Guided Reflection with Memory,
Yan et al. 2025 ACL)의 GPT 백엔드 버전.

Qwen2.5-VL(vllm_utils.py) 대신 GPT(gpt_backend.py 경유 baselines_gpt_utils.py, 기본 gpt-4o-mini)로
추론/텍스트 생성을 교체했다. 알고리즘(Feedback Memory + Exemplar Factory +
exemplar-guided reflection)은 동일.

config 기본값은 실제로 results_intern/erm_intern_result.json을 만든 값
(erm_video.py/erm_intern.py 자체 argparse 기본값이 아니라, 그 실행에서 CLI로
오버라이드했던 값)과 동일하게 맞춰서, Qwen/InternVL과 "같은 옵션"으로 비교 가능하게 함.

Usage:
  OPENAI_API_KEY=... GPT_MODEL=gpt-4o-mini python gpt_experiment/erm_gpt.py
"""
import argparse
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))

OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

from eval_metrics import compute_metrics, print_metrics, find_best_threshold
from baselines_gpt_utils import (
    extract_frames_1fps, classify_dataset, score_dataset,
    predict_text, predict_vision_text,
    MAX_SEC, FRAME_SCORE_THRESHOLD, SCAM_RATIO_THRESHOLD,
)
from utils_gpt import load_gt, split_data, VIDEO_DIR

RANDOM_SEED = 42
TRAIN_RATIO = 0.6
DEV_RATIO   = 0.2

INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)

# ── 라운드별 JSONL 로거 (다른 GPT 실험들과 같은 규칙) ────────────────────────
_LOG_DIR = OUT_DIR / "em_logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)


class _RoundLogger:
    def __init__(self, run_name: str):
        ts        = datetime.now().strftime("%m%d_%H%M")
        self.path = _LOG_DIR / f"{run_name}_{ts}.jsonl"
        self._f   = open(self.path, "a", encoding="utf-8")
        print(f"[Logger] 로그 저장: {self.path}")

    def write(self, record: dict):
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ── 메타 프롬프트 (erm_video.py와 동일) ──────────────────────────────────────
REFLECTION_META = """\
You are optimizing a zero-shot classifier prompt for detecting cyber scam video frames.

Current prompt: "{prompt}"

The following frames were misclassified (attached images above):
{error_desc}

Step 1: Select {n_e} diverse and representative misclassified examples. For each provide:
  - A text description of what is visually shown
  - The correct label (normal / abnormal)
  - A detailed solution process explaining why it should be classified that way

Step 2: Based on those exemplars, generate {n_f} specific actionable feedbacks explaining \
why the current prompt failed.

Output format:
<exemplars>
[description | label | solution process]
...
</exemplars>
<feedbacks>
<feedback>feedback text</feedback>
...
</feedbacks>"""

OPTIMIZATION_META = """\
You are improving a zero-shot classifier prompt for detecting cyber scam video frames.

Current prompt: "{prompt}"

The prompt failed because: {feedback}

Relevant historical feedbacks:
{memory_feedbacks}

Generate an improved prompt that addresses these issues. \
The prompt must instruct the model to return ONLY a single float 0.0–1.0.

Output the improved prompt wrapped with <prompt> and </prompt>."""

MULTI_FEEDBACK_META = """\
You are improving a zero-shot classifier prompt for detecting cyber scam video frames.

Current prompt: "{prompt}"

Multiple issues identified:
{feedbacks}

Generate an improved prompt that addresses all issues. \
The prompt must instruct the model to return ONLY a single float 0.0–1.0.

Output the improved prompt wrapped with <prompt> and </prompt>."""


# ── 유틸 (erm_video.py와 동일) ────────────────────────────────────────────────
def _softmax(scores: list[float], tau: float = 1.0) -> list[float]:
    shifted = [(s / tau) for s in scores]
    max_s   = max(shifted)
    exps    = [math.exp(s - max_s) for s in shifted]
    total   = sum(exps)
    return [e / total for e in exps]


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _parse_prompt(text: str, fallback: str) -> str:
    m = re.search(r"<prompt>(.*?)</prompt>", text, re.DOTALL)
    return m.group(1).strip() if m else fallback


# ── Feedback Memory (erm_video.py와 동일) ────────────────────────────────────
class FeedbackMemory:
    def __init__(self, max_size: int = 20, beta: float = 0.7,
                 theta: float = 0.3, tau: float = 1.0):
        self.items    = []
        self.max_size = max_size
        self.beta     = beta
        self.theta    = theta
        self.tau      = tau

    def add(self, text: str, init_priority: float = 0.5):
        for item in self.items:
            if _jaccard(item["text"], text) > 0.8:
                return
        self.items.append({"text": text, "priority": init_priority})
        if len(self.items) > self.max_size:
            self.items.sort(key=lambda x: x["priority"], reverse=True)
            self.items = self.items[:self.max_size]

    def retrieve(self, n: int) -> list[str]:
        if not self.items:
            return []
        n     = min(n, len(self.items))
        probs = _softmax([it["priority"] for it in self.items], self.tau)
        idxs  = np.random.choice(len(self.items), size=n, replace=False, p=probs)
        return [self.items[i]["text"] for i in idxs]

    def update(self, texts: list[str], improved: bool):
        indicator = 1.0 if improved else 0.0
        for item in self.items:
            if item["text"] in texts:
                item["priority"] = (1 - self.beta) * item["priority"] + self.beta * indicator
        self.items = [it for it in self.items if it["priority"] >= self.theta]


# ── Exemplar Factory (erm_video.py와 동일) ───────────────────────────────────
class ExemplarFactory:
    def __init__(self, max_size: int = 30, beta: float = 0.7,
                 theta: float = 0.3, tau: float = 1.0):
        self.items    = []
        self.max_size = max_size
        self.beta     = beta
        self.theta    = theta
        self.tau      = tau

    def add(self, text: str, label: str, init_priority: float = 0.5):
        self.items.append({"text": text, "label": label, "priority": init_priority})
        if len(self.items) > self.max_size:
            self.items.sort(key=lambda x: x["priority"], reverse=True)
            self.items = self.items[:self.max_size]

    def retrieve_for_prompt(self, n: int = 3) -> str:
        if not self.items:
            return ""
        n     = min(n, len(self.items))
        probs = _softmax([it["priority"] for it in self.items], self.tau)
        idxs  = np.random.choice(len(self.items), size=n, replace=False, p=probs)
        lines = [
            f"Example {k+1} [Label: {self.items[i]['label']}]: {self.items[i]['text']}"
            for k, i in enumerate(idxs)
        ]
        return "\n\nReference examples from similar videos:\n" + "\n".join(lines)

    def update(self, texts: list[str], improved: bool):
        indicator = 1.0 if improved else 0.0
        for item in self.items:
            if item["text"] in texts:
                item["priority"] = (1 - self.beta) * item["priority"] + self.beta * indicator
        self.items = [it for it in self.items if it["priority"] >= self.theta]


# ── 데이터셋 로드 — utils_gpt.load_gt()+split_data()로 다른 모든 GPT 방법과
# 동일한 train/dev/test 구성을 쓰고, ERM이 쓰는 frame_paths 캐시 포맷으로
# 변환만 한다 (apo_gpt.py/evoprompt_gpt.py의 load_dataset_split과 동일 패턴) ──
def load_dataset_split(
    frames_dir: Path, train_ratio: float = TRAIN_RATIO,
    dev_ratio: float = DEV_RATIO, seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, train_ratio, dev_ratio, seed)

    def _convert(split: list[dict]) -> list[dict]:
        items = []
        for v in split:
            vfiles = list(VIDEO_DIR.glob(f"{v['video_name']}.*"))
            if not vfiles:
                continue
            frame_paths = extract_frames_1fps(vfiles[0], frames_dir)
            if not frame_paths:
                continue
            items.append({"video_name": v["video_name"], "frame_paths": frame_paths, "label": v["label"]})
        return items

    return _convert(trainset), _convert(devset), _convert(testset)


# ── 프롬프트 평가 (erm_video.py와 동일) ──────────────────────────────────────
def evaluate_prompt(prompt: str, dataset: list[dict]) -> tuple[float, list[dict], dict]:
    y_score = score_dataset(dataset, prompt)
    y_true  = [1 if ex["label"] == "abnormal" else 0 for ex in dataset]
    y_pred  = [1 if s >= FRAME_SCORE_THRESHOLD else 0 for s in y_score]
    preds   = ["abnormal" if p == 1 else "normal" for p in y_pred]
    errors  = [ex for ex, p in zip(dataset, preds) if p != ex["label"]]
    m       = compute_metrics(y_true, y_pred, y_score)
    return m["map"] or 0.0, errors, m


# ── Exemplar-Guided Reflection (erm_video.py와 동일) ─────────────────────────
def exemplar_guided_reflection(
    prompt: str, errors: list[dict], n_e: int, n_f: int
) -> tuple[list[dict], list[str]]:
    sample    = random.sample(errors, min(n_e, len(errors)))
    img_paths = [ex["frame_paths"][0] for ex in sample]

    error_desc = "\n".join(
        f"Frame {i+1}: True label = {ex['label']}"
        for i, ex in enumerate(sample)
    )
    meta = REFLECTION_META.format(
        prompt=prompt, error_desc=error_desc, n_e=n_e, n_f=n_f
    )
    out = predict_vision_text(meta, img_paths, temperature=0.7, max_tokens=900)

    exemplars = []
    m = re.search(r"<exemplars>(.*?)</exemplars>", out, re.DOTALL)
    if m:
        for line in m.group(1).strip().splitlines():
            line = line.strip("[]- \t")
            if "|" in line:
                parts = line.split("|")
                desc  = parts[0].strip()
                label = "abnormal" if "abnormal" in parts[1].lower() else "normal"
                if desc:
                    exemplars.append({"text": desc, "label": label})

    feedbacks = [f.strip() for f in
                 re.findall(r"<feedback>(.*?)</feedback>", out, re.DOTALL)
                 if f.strip()]
    return exemplars, feedbacks


# ── Feedback 적용 → 새 프롬프트 생성 (erm_video.py와 동일) ──────────────────
def apply_feedback(prompt: str, feedback: str, mem_feedbacks: list[str]) -> str:
    mem_str = "\n".join(f"- {f}" for f in mem_feedbacks) if mem_feedbacks else "None"
    meta    = OPTIMIZATION_META.format(
        prompt=prompt, feedback=feedback, memory_feedbacks=mem_str
    )
    out = predict_text(meta, n=1, temperature=0.7, max_tokens=512)[0]
    return _parse_prompt(out, prompt)


def apply_multi_feedback(prompt: str, feedbacks: list[str]) -> str:
    fb_str = "\n".join(f"{i+1}. {f}" for i, f in enumerate(feedbacks))
    meta   = MULTI_FEEDBACK_META.format(prompt=prompt, feedbacks=fb_str)
    out    = predict_text(meta, n=1, temperature=0.7, max_tokens=512)[0]
    return _parse_prompt(out, prompt)


# ── ERM 최적화 루프 (erm_video.py와 동일) ────────────────────────────────────
def run_erm(trainset: list[dict], valset: list[dict], config: dict):
    beam_size    = config["beam_size"]
    rounds       = config["rounds"]
    n_e          = config["n_exemplars"]
    n_f          = config["n_feedbacks"]
    n_fb_mem     = config["n_feedback_from_memory"]
    n_ex_inf     = config["n_exemplars_inference"]
    mb_size      = config["minibatch_size"]
    val_sub_n    = config.get("val_subsample", 15)
    patience     = config.get("patience",     2)
    patience_eps = config.get("patience_eps", 0.01)

    fb_mem  = FeedbackMemory(max_size=config["mem_max_size"])
    ex_fact = ExemplarFactory(max_size=config["ex_max_size"])
    logger  = _RoundLogger(config.get("run_name", "erm_gpt"))

    partial_path = OUT_DIR / "erm_gpt4omini_partial_log.json"
    if partial_path.exists():
        with open(partial_path, encoding="utf-8") as _f:
            saved = json.load(_f)
        beam        = saved.get("beam",        [saved["best_prompt"]])
        beam_scores = saved.get("beam_scores", [saved["best_dev_map"]])
        log         = saved.get("log",         [])
        fb_mem.items     = saved.get("fb_mem_items",     [])
        ex_fact.items    = saved.get("ex_fact_items",    [])
        start_rnd        = saved.get("last_round",       0) + 1
        no_improve_count = saved.get("no_improve_count", 0)
        best_ever_map    = saved.get("best_ever_map",    max(beam_scores))
        print(f"  [Resume] Round {start_rnd}부터 재시작  best_map={best_ever_map:.3f}  no_improve={no_improve_count}")
    else:
        beam        = [INITIAL_PROMPT]
        beam_scores = [evaluate_prompt(INITIAL_PROMPT, valset)[0]]
        print(f"초기 val_map={beam_scores[0]:.3f}")
        log              = []
        start_rnd        = 1
        no_improve_count = 0
        best_ever_map    = max(beam_scores)

    for rnd in range(start_rnd, rounds + 1):
        print(f"\n=== Round {rnd}/{rounds} ===")
        new_cands = list(beam)
        round_fbs = []
        round_exs = []

        for prompt in beam:
            mb = random.sample(trainset, min(mb_size, len(trainset)))
            _, errors, _ = evaluate_prompt(prompt, mb)
            if not errors:
                continue

            exemplars, feedbacks = exemplar_guided_reflection(prompt, errors, n_e, n_f)
            print(f"  exemplars={len(exemplars)}, feedbacks={len(feedbacks)}")

            for fb in feedbacks:
                logger.write({
                    "type": "feedback", "round": rnd,
                    "prompt_preview": prompt[:200], "feedback": fb,
                })

            round_exs.extend(exemplars)
            round_fbs.extend(feedbacks)

            mem_fbs = fb_mem.retrieve(n_fb_mem)

            for fb in feedbacks:
                new_cands.append(apply_feedback(prompt, fb, mem_fbs))

            if len(feedbacks) > 1:
                all_fbs = feedbacks + mem_fbs
                new_cands.append(apply_multi_feedback(prompt, all_fbs))

        for ex in round_exs:
            ex_fact.add(ex["text"], ex["label"])

        new_cands = list(set(new_cands))
        print(f"  후보 {len(new_cands)}개 평가 중 (val_subsample={val_sub_n})...")

        val_sub   = random.sample(valset, min(val_sub_n, len(valset)))
        ex_suffix = ex_fact.retrieve_for_prompt(n_ex_inf)
        scored = []
        for cand in new_cands:
            augmented = cand + ex_suffix
            map_score, _, _ = evaluate_prompt(augmented, val_sub)
            scored.append((map_score, cand))

        scored.sort(reverse=True)
        beam        = [s[1] for s in scored[:beam_size]]
        beam_scores = [s[0] for s in scored[:beam_size]]

        best_score = beam_scores[0]
        if best_score > best_ever_map:
            best_ever_map    = best_score
            no_improve_count = 0
        elif best_score >= best_ever_map - patience_eps:
            no_improve_count = 0
        else:
            no_improve_count += 1
        improved = best_score > best_ever_map - patience_eps

        _, _, dev_m = evaluate_prompt(beam[0] + ex_suffix, valset)
        print(f"  dev_map={dev_m['map'] or 0:.3f}  auc={dev_m['auc'] or 0:.3f}  f1={dev_m['f1'] or 0:.3f}  "
              f"{'↑' if improved else f'- ({no_improve_count}/{patience})'}")

        for fb in round_fbs:
            fb_mem.add(fb)
        fb_mem.update(round_fbs, improved)
        ex_fact.update([ex["text"] for ex in round_exs], improved)

        round_record = {
            "round":        rnd,
            "dev_map":      dev_m["map"],
            "dev_auc":      dev_m["auc"],
            "dev_f1":       dev_m.get("f1_macro") or dev_m.get("f1", 0.0),
            "dev_acc":      dev_m["acc"],
            "prompt":       beam[0],
            "feedback":     round_fbs[0] if round_fbs else "",
            "n_candidates": len(new_cands),
            "improved":     improved,
        }
        log.append(round_record)
        logger.write({"type": "iteration", **round_record})

        partial = {
            "method":            "ERM-GPT",
            "best_prompt":       beam[0],
            "best_dev_map":      max(beam_scores),
            "log":               log,
            "beam":              beam,
            "beam_scores":       beam_scores,
            "fb_mem_items":      fb_mem.items,
            "ex_fact_items":     ex_fact.items,
            "last_round":        rnd,
            "no_improve_count":  no_improve_count,
            "best_ever_map":     best_ever_map,
        }
        with open(partial_path, "w", encoding="utf-8") as _f:
            json.dump(partial, _f, ensure_ascii=False, indent=2)
        print(f"  [저장] {partial_path}")

        if no_improve_count >= patience:
            print(f"  [조기종료] {patience}라운드 연속 개선 없음 → 종료 (round {rnd}/{rounds})")
            break

    if partial_path.exists():
        partial_path.unlink()
    logger.close()

    ex_suffix   = ex_fact.retrieve_for_prompt(n_ex_inf)
    best_prompt = beam[0] + ex_suffix
    return best_prompt, beam[0], log


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir",             default=str(Path(__file__).parent.parent / "data" / "frames_cache_2s"))
    parser.add_argument("--output",                 default=str(OUT_DIR / "erm_gpt4omini_result.json"))
    # 기본값은 erm_video.py 자체 기본값이 아니라, 실제로 results_intern/erm_intern_result.json을
    # 만든 실행에서 쓰인 config와 동일 (Qwen/InternVL과 "같은 옵션" 비교를 위해).
    parser.add_argument("--rounds",                 type=int,   default=12)
    parser.add_argument("--beam_size",              type=int,   default=2)
    parser.add_argument("--minibatch_size",         type=int,   default=16)
    parser.add_argument("--n_exemplars",            type=int,   default=1)
    parser.add_argument("--n_feedbacks",            type=int,   default=2)
    parser.add_argument("--n_feedback_from_memory", type=int,   default=1)
    parser.add_argument("--n_exemplars_inference",  type=int,   default=3)
    parser.add_argument("--mem_max_size",           type=int,   default=20)
    parser.add_argument("--ex_max_size",            type=int,   default=30)
    parser.add_argument("--patience",               type=int,   default=2,
                        help="조기종료: N라운드 연속 val_map 미개선 시 중단 (erm_intern.py와 동일)")
    parser.add_argument("--patience_eps",           type=float, default=0.01,
                        help="조기종료 내성: 이 미만 하락은 개선으로 간주 (erm_intern.py와 동일)")
    parser.add_argument("--val_subsample",          type=int,   default=15,
                        help="후보 선택 시 val 서브샘플 크기 (erm_intern.py와 동일)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    config = vars(args)

    print("\n=== 데이터셋 로드 & 2초 간격 프레임 추출 (공유 split 사용) ===")
    trainset, valset, testset = load_dataset_split(Path(args.frames_dir))
    print(f"분할 → train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    print("\n=== ERM-GPT 최적화 시작 ===")
    best_prompt, best_prompt_base, log = run_erm(trainset, valset, config)

    print(f"\n=== 최적 프롬프트 ===\n{best_prompt_base}\n")

    def get_test_metrics(prompt, threshold=0.5):
        y_score = score_dataset(testset, prompt)
        y_true  = [1 if ex["label"] == "abnormal" else 0 for ex in testset]
        y_pred  = [1 if s >= threshold else 0 for s in y_score]
        return compute_metrics(y_true, y_pred, y_score)

    print("=== 최종 테스트 평가 ===")
    baseline_m = get_test_metrics(INITIAL_PROMPT)
    best_m     = get_test_metrics(best_prompt)

    val_true   = [1 if ex["label"] == "abnormal" else 0 for ex in valset]
    val_score  = score_dataset(valset, best_prompt)
    best_thr   = find_best_threshold(val_true, val_score)
    best_m_opt = get_test_metrics(best_prompt, threshold=best_thr)
    best_m["f1_opt"]         = best_m_opt["f1"]
    best_m["acc_opt"]        = best_m_opt["acc"]
    best_m["best_threshold"] = best_thr

    print_metrics(baseline_m, label="Baseline")
    print_metrics(best_m,     label="ERM-GPT")

    output = {
        "method":            "ERM-GPT",
        "gpt_model":         os.environ.get("GPT_MODEL", "gpt-4o-mini"),
        "best_prompt":       best_prompt,
        "best_prompt_base":  best_prompt_base,
        "initial_prompt":    INITIAL_PROMPT,
        "baseline_test":     baseline_m,
        "best_test":         best_m,
        "best_test_f1_opt":  best_m["f1_opt"],
        "best_test_acc_opt": best_m["acc_opt"],
        "config":            config,
        "log":               log,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
