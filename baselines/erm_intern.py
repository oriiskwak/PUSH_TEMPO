"""
ERM-Video: Exemplar-Guided Reflection with Memory — vLLM 직접 방식 적용
Based on: Yan et al. 2025 "Efficient and Accurate Prompt Optimization:
          the Benefit of Memory in Exemplar-Guided Reflection" (ACL 2025)

추론: TEMPO와 동일하게 vLLM 직접 호출, 2초 간격 프레임, scam_ratio 집계.

사용법:
    TP_SIZE=1 PP_SIZE=2 python erm_intern.py --video_dir /path/to/videos
"""
import argparse
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent.parent / "results_intern"
OUT_DIR.mkdir(exist_ok=True)

# ── 라운드별 JSONL 로거 (다른 실험들의 em_core.EMLogger와 같은 규칙:
#    results/em_logs/{run_name}_{MMDD_HHMM}.jsonl, 매 줄 즉시 flush) ──
# em_core.py를 직접 import하면 SentenceTransformer/DBSCAN 등 불필요한 게 같이 로드돼서
# 여기선 가벼운 버전을 자체 구현. 파일이 append+flush라 중간 저장(erm_partial_log.json)과
# 별개로, 그 시점까지의 raw 기록이 크래시와 무관하게 디스크에 남음.
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

# vLLM(tensor_parallel_size>1)이 CUDA 초기화 이후 워커를 spawn으로 띄울 때, 워커
# 부트스트랩 과정이 이 스크립트를 "__mp_main__"로 다시 import함. 아래 import들이
# (지금처럼) 최상단에 무조건 실행되면, 그 재-import 때마다 utils.py/vllm_utils.py의
# 모듈 레벨 `_llm = LLM(...)`이 또 실행돼서 워커가 또 워커를 spawn하려다 죽음
# (RuntimeError: An attempt has been made to start a new process before the
#  current process has finished its bootstrapping phase).
# `if __name__ == "__main__":`으로 감싸면 재-import 시엔 "__mp_main__" != "__main__"라
# 이 블록 자체가 스킵되어 재귀가 끊김. (모듈 최상단의 if 블록이라 여기서 만든 이름들도
# 그대로 모듈 전역이 되어 evaluate_prompt() 등 다른 함수에서 평소처럼 참조 가능)
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))  # TEMPO 루트: vllm_utils_intern 위치
    from utils_gpt import load_gt, split_data, VIDEO_DIR

    from eval_metrics import compute_metrics, print_metrics, find_best_threshold
    from vllm_utils_intern import (
        extract_frames_1fps, classify_dataset, score_dataset,
        predict_text, predict_vision_text,
        FRAME_SCORE_THRESHOLD,
    )

RANDOM_SEED = 42

INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)

# ── 메타 프롬프트 (논문 3가지 메타 프롬프트 대응) ────────────────────────────
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


# ── 유틸 ─────────────────────────────────────────────────────────────────────
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


# ── Feedback Memory ───────────────────────────────────────────────────────────
class FeedbackMemory:
    """논문 Algorithm 1 Feedback Memory 대응."""

    def __init__(self, max_size: int = 20, beta: float = 0.7,
                 theta: float = 0.3, tau: float = 1.0):
        self.items    = []   # [{"text": str, "priority": float}]
        self.max_size = max_size
        self.beta     = beta
        self.theta    = theta
        self.tau      = tau

    def add(self, text: str, init_priority: float = 0.5):
        # 유사 피드백 중복 방지
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
        n      = min(n, len(self.items))
        probs  = _softmax([it["priority"] for it in self.items], self.tau)
        idxs   = np.random.choice(len(self.items), size=n, replace=False, p=probs)
        return [self.items[i]["text"] for i in idxs]

    def update(self, texts: list[str], improved: bool):
        indicator = 1.0 if improved else 0.0
        for item in self.items:
            if item["text"] in texts:
                item["priority"] = (1 - self.beta) * item["priority"] + self.beta * indicator
        # Forgetting: θ 미만 제거
        self.items = [it for it in self.items if it["priority"] >= self.theta]


# ── Exemplar Factory ──────────────────────────────────────────────────────────
class ExemplarFactory:
    """논문 Algorithm 2 Exemplar Factory 대응."""

    def __init__(self, max_size: int = 30, beta: float = 0.7,
                 theta: float = 0.3, tau: float = 1.0):
        self.items    = []   # [{"text": str, "label": str, "priority": float}]
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
        """우선순위 기반 exemplar n개를 프롬프트 suffix로 반환."""
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


# ── 데이터셋 로드 ─────────────────────────────────────────────────────────────
# 라벨은 파일명 접두사(수집 시점)가 아니라 gt_all.json의 GT-relabel(v["label"])
# 기준 — TEMPO 본체(tempo_cov_total*.py 등)와 동일한 ground truth를 써야
# baseline 비교가 공정해짐.
_GT_PATH = Path(__file__).parent.parent / "data" / "gt_results" / "gt_all.json"
_GT_LABEL_MAP = {v["video_name"]: v["label"]
                for v in json.load(open(_GT_PATH, encoding="utf-8"))}


def load_dataset(video_dir: Path, frames_dir: Path) -> list[dict]:
    items, skipped = [], 0
    video_files    = sorted(video_dir.glob("*.mp4"))
    print(f"총 MP4: {len(video_files)}개")
    for i, vf in enumerate(video_files, 1):
        name  = vf.name
        label = _GT_LABEL_MAP.get(vf.stem)
        if label not in ("normal", "abnormal"): continue
        print(f"  [{i:3d}/{len(video_files)}] {name[:50]}", end="\r")
        frame_paths = extract_frames_1fps(vf, frames_dir)
        if not frame_paths:
            skipped += 1
            continue
        items.append({"video_name": vf.stem, "frame_paths": frame_paths, "label": label})
    print()
    n_norm = sum(1 for it in items if it["label"] == "normal")
    print(f"로드 완료: {len(items)}개  (normal={n_norm}, abnormal={len(items)-n_norm}, 스킵={skipped})")
    return items


def _stratified_split(dataset: list[dict], train_ratio: float = 0.6,
                      dev_ratio: float = 0.2, seed: int = RANDOM_SEED):
    """label 균형을 유지하는 stratified split → train / val / test
    (apo_video.py / evoprompt_video.py와 동일한 방식, baseline 간 split 통일)."""
    random.seed(seed)
    normal   = [x for x in dataset if x["label"] == "normal"]
    abnormal = [x for x in dataset if x["label"] == "abnormal"]

    def _split(lst):
        lst = lst[:]
        random.shuffle(lst)
        n_tr = int(len(lst) * train_ratio)
        n_dv = int(len(lst) * dev_ratio)
        return lst[:n_tr], lst[n_tr:n_tr + n_dv], lst[n_tr + n_dv:]

    no_tr, no_dv, no_te = _split(normal)
    ab_tr, ab_dv, ab_te = _split(abnormal)
    train = no_tr + ab_tr;  random.shuffle(train)
    val   = no_dv + ab_dv;  random.shuffle(val)
    test  = no_te + ab_te;  random.shuffle(test)
    return train, val, test


# ── 프롬프트 평가 ─────────────────────────────────────────────────────────────
def evaluate_prompt(prompt: str, dataset: list[dict]) -> tuple[float, list[dict], dict]:
    """MAP + 오분류 예시 + 전체 메트릭 반환. score_dataset 1회 호출로 통합."""
    y_score = score_dataset(dataset, prompt)
    y_true  = [1 if ex["label"] == "abnormal" else 0 for ex in dataset]
    y_pred  = [1 if s >= FRAME_SCORE_THRESHOLD else 0 for s in y_score]
    preds   = ["abnormal" if p == 1 else "normal" for p in y_pred]
    errors  = [ex for ex, p in zip(dataset, preds) if p != ex["label"]]
    m       = compute_metrics(y_true, y_pred, y_score)
    return m["map"] or 0.0, errors, m


# ── Exemplar-Guided Reflection ────────────────────────────────────────────────
def exemplar_guided_reflection(
    prompt: str, errors: list[dict], n_e: int, n_f: int
) -> tuple[list[dict], list[str]]:
    """오분류 프레임 이미지 + 메타 프롬프트 → exemplars & feedbacks 생성."""
    sample  = random.sample(errors, min(n_e, len(errors)))
    img_paths = [ex["frame_paths"][0] for ex in sample]   # 오류당 첫 프레임 (max 4장)

    error_desc = "\n".join(
        f"Frame {i+1}: True label = {ex['label']}"
        for i, ex in enumerate(sample)
    )
    meta = REFLECTION_META.format(
        prompt=prompt, error_desc=error_desc, n_e=n_e, n_f=n_f
    )
    out = predict_vision_text(meta, img_paths, temperature=0.7, max_tokens=900)

    # Exemplar 파싱
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

    # Feedback 파싱
    feedbacks = [f.strip() for f in
                 re.findall(r"<feedback>(.*?)</feedback>", out, re.DOTALL)
                 if f.strip()]
    return exemplars, feedbacks


# ── Feedback 적용 → 새 프롬프트 생성 ─────────────────────────────────────────
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


# ── ERM 최적화 루프 ───────────────────────────────────────────────────────────
def run_erm(trainset: list[dict], valset: list[dict], config: dict):
    beam_size   = config["beam_size"]
    rounds      = config["rounds"]
    n_e         = config["n_exemplars"]
    n_f         = config["n_feedbacks"]
    n_fb_mem    = config["n_feedback_from_memory"]
    n_ex_inf    = config["n_exemplars_inference"]
    mb_size      = config["minibatch_size"]
    val_sub_n    = config.get("val_subsample", 15)   # TEMPO DEV_SUBSAMPLE과 동일 개념
    patience     = config.get("patience",     2)
    patience_eps = config.get("patience_eps", 0.01)

    fb_mem  = FeedbackMemory(max_size=config["mem_max_size"])
    ex_fact = ExemplarFactory(max_size=config["ex_max_size"])
    logger  = _RoundLogger(config.get("run_name", "erm_intern"))

    partial_path = OUT_DIR / "erm_intern_partial_log.json"
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
        prev_best    = max(beam_scores)
        new_cands    = list(beam)
        round_fbs    = []
        round_exs    = []

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
            no_improve_count = 0   # TEMPO와 동일: ε 미만 하락은 무시
        else:
            no_improve_count += 1
        improved = best_score > best_ever_map - patience_eps

        # 상세 메트릭
        _, _, dev_m = evaluate_prompt(beam[0] + ex_suffix, valset)
        print(f"  dev_map={dev_m['map'] or 0:.3f}  auc={dev_m['auc'] or 0:.3f}  f1={dev_m['f1'] or 0:.3f}  {'↑' if improved else f'- ({no_improve_count}/{patience})'}")

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

        # 라운드마다 중간 저장 (resume용 전체 상태 포함)
        partial = {
            "method":          "ERM",
            "best_prompt":     beam[0],
            "best_dev_map":    max(beam_scores),
            "log":             log,
            "beam":            beam,
            "beam_scores":     beam_scores,
            "fb_mem_items":    fb_mem.items,
            "ex_fact_items":   ex_fact.items,
            "last_round":      rnd,
            "no_improve_count": no_improve_count,
            "best_ever_map":   best_ever_map,
        }
        with open(partial_path, "w", encoding="utf-8") as _f:
            json.dump(partial, _f, ensure_ascii=False, indent=2)
        print(f"  [저장] {partial_path}")

        if no_improve_count >= patience:
            print(f"  [조기종료] {patience}라운드 연속 개선 없음 → 종료 (round {rnd}/{rounds})")
            break

    # 완료 후 partial log 삭제
    if partial_path.exists():
        partial_path.unlink()
    logger.close()

    # 최종 best prompt에 exemplar suffix 붙여서 반환
    ex_suffix   = ex_fact.retrieve_for_prompt(n_ex_inf)
    best_prompt = beam[0] + ex_suffix
    return best_prompt, beam[0], log


# ── 최종 상세 평가 ────────────────────────────────────────────────────────────
def eval_detailed(prompt: str, testset: list[dict], label: str = "") -> None:
    preds_str = classify_dataset(testset, prompt)
    y_score   = score_dataset(testset, prompt)
    y_true    = [1 if ex["label"] == "abnormal" else 0 for ex in testset]
    y_pred    = [1 if p == "abnormal" else 0 for p in preds_str]
    print_metrics(compute_metrics(y_true, y_pred, y_score), label=label)


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",              required=True)
    parser.add_argument("--frames_dir",             default="./frames_cache_2s")
    parser.add_argument("--output",                 default=str(OUT_DIR / "erm_intern_result.json"))
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
                        help="조기종료: N라운드 연속 val_map 미개선 시 중단 (TEMPO와 동일)")
    parser.add_argument("--patience_eps",           type=float, default=0.01,
                        help="조기종료 내성: 이 미만 하락은 개선으로 간주 (TEMPO와 동일)")
    parser.add_argument("--val_subsample",          type=int,   default=15,
                        help="후보 선택 시 val 서브샘플 크기 (TEMPO DEV_SUBSAMPLE과 동일 개념)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    config = vars(args)

    print("\n=== 데이터셋 로드 & 2초 간격 프레임 추출 ===")
    gt_list = load_gt()
    trainset_meta, valset_meta, testset_meta = split_data(gt_list, 0.6, 0.2, RANDOM_SEED)

    # video_name → frame_paths 매핑 생성 (TEMPO와 동일한 split 사용)
    all_data = load_dataset(Path(args.video_dir), Path(args.frames_dir))
    frame_map = {ex["video_name"]: ex for ex in all_data}

    def _filter(meta_list):
        return [frame_map[m["video_name"]] for m in meta_list if m["video_name"] in frame_map]

    trainset = _filter(trainset_meta)
    valset   = _filter(valset_meta)
    testset  = _filter(testset_meta)
    print(f"분할 → train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    print("\n=== ERM 최적화 시작 ===")
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

    # threshold 재보정: val_map으로 프롬프트를 고른 것과 별개로, 고정 0.5 threshold가
    # 최적화된 점수 분포와 안 맞아 F1/ACC가 왜곡될 수 있음 (TEMPO 계열과 동일 기준으로 맞춤).
    val_true      = [1 if ex["label"] == "abnormal" else 0 for ex in valset]
    val_score     = score_dataset(valset, best_prompt)
    best_thr      = find_best_threshold(val_true, val_score)
    best_m_opt    = get_test_metrics(best_prompt, threshold=best_thr)
    best_m["f1_opt"]         = best_m_opt["f1"]
    best_m["acc_opt"]        = best_m_opt["acc"]
    best_m["best_threshold"] = best_thr

    print_metrics(baseline_m, label="Baseline")
    print_metrics(best_m,     label="ERM")

    output = {
        "method":              "ERM",
        "best_prompt":         best_prompt,
        "best_prompt_base":    best_prompt_base,
        "initial_prompt":      INITIAL_PROMPT,
        "baseline_test":       baseline_m,
        "best_test":           best_m,
        "best_test_f1_opt":    best_m["f1_opt"],
        "best_test_acc_opt":   best_m["acc_opt"],
        "config":              config,
        "log":                 log,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
