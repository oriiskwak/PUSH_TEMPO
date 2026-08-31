"""
EvoPrompt-Video: EvoPrompt GA/DE — vLLM 직접 방식 적용
Based on: https://github.com/beeevita/EvoPrompt

원본 텍스트 입력 → 비디오 프레임 입력으로 확장.
추론: TEMPO와 동일하게 vLLM 직접 호출, 2초 간격 프레임, scam_ratio 집계.
데이터셋은 utils_gpt.load_gt()+split_data()로 다른 모든 방법과 동일한 train/dev/test를
사용한다 (공정한 baseline 비교).

사용법:
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 python evoprompt_video.py --mode ga
"""
import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# ── 스텝별 JSONL 로거 (다른 실험들의 em_core.EMLogger와 같은 규칙:
#    results/em_logs/{run_name}_{MMDD_HHMM}.jsonl, 매 줄 즉시 flush) ──
_LOG_DIR = Path(__file__).parent.parent / "results" / "em_logs"
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
# 최상단에 무조건 실행되면, 재-import 때마다 vllm_utils.py의 모듈 레벨
# `_llm = LLM(...)`이 또 실행돼서 워커가 또 워커를 spawn하려다 죽는다
# (RuntimeError: bootstrapping phase). `if __name__ == "__main__":`으로 감싸면
# 재-import 시 "__mp_main__" != "__main__"라 이 블록이 스킵되어 재귀가 끊긴다.
if __name__ == "__main__":
    from vllm_utils import (
        extract_frames_1fps, classify_dataset, score_dataset, predict_text,
        MAX_SEC, SCAM_RATIO_THRESHOLD,
    )
    from utils_gpt import load_gt, split_data, VIDEO_DIR
    from eval_metrics import find_best_threshold

RANDOM_SEED = 42
TRAIN_RATIO = 0.6
DEV_RATIO   = 0.2


def _compute_metrics(y_true: list, y_pred: list, y_score: list) -> dict:
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "auc":      round(float(roc_auc_score(y_true, y_score)), 4),
        "map":      round(float(average_precision_score(y_true, y_score)), 4),
    }


def _print_metrics(metrics: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Accuracy : {metrics['accuracy']:.4f}")
    print(f"  {prefix}F1 Macro : {metrics['f1_macro']:.4f}")
    print(f"  {prefix}AUC      : {metrics['auc']:.4f}")
    print(f"  {prefix}MAP      : {metrics['map']:.4f}")
    if "f1_opt" in metrics:
        print(f"  {prefix}F1_opt   : {metrics['f1_opt']:.4f}  (threshold={metrics.get('best_threshold')})")
    if "acc_opt" in metrics:
        print(f"  {prefix}ACC_opt  : {metrics['acc_opt']:.4f}")


def load_dataset_split(
    frames_dir: Path, train_ratio: float = TRAIN_RATIO,
    dev_ratio: float = DEV_RATIO, seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """utils_gpt.load_gt()+split_data()로 다른 모든 방법과 동일한 train/dev/test 구성을
    쓰고, EvoPrompt가 쓰는 frame_paths 캐시 포맷으로 변환만 한다 (독자적인 *.mp4 glob +
    자체 split을 쓰면 비디오 구성이 미묘하게 달라져 baseline 비교가 불공정해짐)."""
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
            items.append({"frame_paths": frame_paths, "label": v["label"]})
        return items

    return _convert(trainset), _convert(devset), _convert(testset)


# ── 초기 프롬프트 (다른 실험 baseline과 동일) ─────────────────────────────────
INITIAL_PROMPT = (
    "Predict the probability (0.0-1.0) that this video contains a cyber scam."
)

# ── EvoPrompt 공식 GA 템플릿 (template_ga.py 기반, 원본 유지) ────────────────
GA_TEMPLATE = """\
Please follow the instruction step-by-step to generate a better prompt.
1. Cross over the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt \
bracketed with <prompt> and </prompt>.
"""

# ── EvoPrompt 공식 DE 템플릿 (template_de.py v2 기반, 원본 유지) ─────────────
DE_TEMPLATE = """\
Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2.
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Combine the differences with the Prompt 3 and generate a new prompt.
Prompt 3: <prompt0>
3. Mutate the prompt generated in Step 2 and generate a final prompt \
bracketed with <prompt> and </prompt>.
"""

# 유효한 프롬프트 판단 기준: 사기 탐지 관련 키워드 최소 1개 포함
_SCAM_KEYWORDS = {"scam", "fraud", "cyber", "abnormal", "phishing", "probability",
                  "estimate", "detect", "suspicious", "fake", "impersonation"}


def _is_valid_prompt(text: str) -> bool:
    """태스크와 무관한 쓰레기 프롬프트 필터링."""
    if len(text) < 40:
        return False
    words = set(text.lower().split())
    return bool(words & _SCAM_KEYWORDS)

def generate_initial_population(seed: str, size: int) -> list[str]:
    """씨앗 프롬프트 1개 → LLM으로 size-1개 변형 생성 → size개 초기 집단 반환."""
    if size <= 1:
        return [seed]
    query = (
        f"Generate {size - 1} paraphrased variations of the following prompt. "
        f"Keep the same task (cyber scam detection in video frames) and meaning "
        f"but use different wording. Each variation must instruct the model to "
        f"return ONLY a single float 0.0–1.0.\n\n"
        f"Original prompt:\n{seed}\n\n"
        f"Wrap each variation with <prompt> and </prompt>."
    )
    out      = predict_text(query, n=1, temperature=1.0, max_tokens=800)[0]
    variants = [m.group(1).strip()
                for m in re.finditer(r"<prompt>(.*?)</prompt>", out, re.DOTALL)
                if _is_valid_prompt(m.group(1).strip())]
    population = [seed] + variants[:size - 1]
    # 유효한 변형이 부족하면 seed로 채움
    while len(population) < size:
        population.append(seed)
    return population




# ── 프롬프트 파싱 + 유효성 검사 ──────────────────────────────────────────────
def parse_prompt(text: str, fallback: str) -> str:
    m = re.search(r"<prompt>(.*?)</prompt>", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if _is_valid_prompt(candidate):
            return candidate
    return fallback


# ── 후보 선택용 점수 (MAP 기반 — 클래스 불균형에 강건) ──────────────────────
def evaluate_prompt(prompt: str, dataset: list[dict], n_eval: int) -> float:
    """MAP 반환 — accuracy 대신 사용하여 클래스 불균형 문제 해결."""
    subset  = random.sample(dataset, min(n_eval, len(dataset)))
    y_score = score_dataset(subset, prompt)
    y_true  = [1 if ex["label"] == "abnormal" else 0 for ex in subset]
    if len(set(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


# ── 전체 데이터셋 완전 평가 (MAP / AUC / F1 / ACC) ───────────────────────────
def evaluate_full(prompt: str, dataset: list[dict], threshold: float = 0.5) -> dict:
    y_score = score_dataset(dataset, prompt)
    y_true  = [1 if ex["label"] == "abnormal" else 0 for ex in dataset]
    y_pred  = [1 if s >= threshold else 0 for s in y_score]
    if len(set(y_true)) < 2:
        return {"map": 0.0, "auc": 0.0, "f1": 0.0, "acc": 0.0}
    return {
        "map": round(float(average_precision_score(y_true, y_score)), 4),
        "auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "f1":  round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "acc": round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 4),
    }


# ── GA 교배 (EvoPrompt GA_TEMPLATE 적용) ─────────────────────────────────────
def ga_crossover(p1: str, p2: str) -> str:
    query = GA_TEMPLATE.replace("<prompt1>", p1).replace("<prompt2>", p2)
    out   = predict_text(query, n=1, temperature=1.0, max_tokens=300)[0]
    return parse_prompt(out, p1)


# ── DE 변이 (EvoPrompt DE_TEMPLATE 적용) ─────────────────────────────────────
def de_mutate(p0: str, p1: str, p2: str) -> str:
    query = (DE_TEMPLATE
             .replace("<prompt0>", p0)
             .replace("<prompt1>", p1)
             .replace("<prompt2>", p2))
    out = predict_text(query, n=1, temperature=1.0, max_tokens=300)[0]
    return parse_prompt(out, p0)


# ── 룰렛 휠 선택 ─────────────────────────────────────────────────────────────
def roulette_wheel(scores: list[float]) -> int:
    total = sum(scores)
    if total == 0:
        return random.randrange(len(scores))
    r, cumulative = random.uniform(0, total), 0.0
    for i, s in enumerate(scores):
        cumulative += s
        if cumulative >= r:
            return i
    return len(scores) - 1


# ── GA 진화 루프 ──────────────────────────────────────────────────────────────
def run_ga(population, scores, trainset, valset, budget, n_eval,
           partial_path=None, start_step=0, log=None, logger=None,
           patience=2, patience_eps=0.01):
    print(f"\n[GA] budget={budget}, popsize={len(population)}, patience={patience}")
    if log is None:
        log = []
    best_ever_ga = max(scores) if scores else 0.0
    no_improve_ga = 0
    for step in range(start_step, budget):
        step_sample = random.sample(trainset, min(n_eval, len(trainset)))
        scores = [evaluate_prompt(p, step_sample, n_eval) for p in population]

        i1 = roulette_wheel(scores)
        i2 = roulette_wheel(scores)
        while i2 == i1:
            i2 = roulette_wheel(scores)

        child       = ga_crossover(population[i1], population[i2])
        child_score = evaluate_prompt(child, step_sample, n_eval)

        worst_idx = scores.index(min(scores))
        if child_score >= scores[worst_idx]:
            population[worst_idx] = child
            scores[worst_idx]     = child_score

        best_score  = max(scores)
        avg_score   = sum(scores) / len(scores)
        best_prompt = population[scores.index(best_score)]

        dev_m   = evaluate_full(best_prompt, valset)
        cur_dev = dev_m["map"]
        if cur_dev > best_ever_ga:
            best_ever_ga  = cur_dev
            no_improve_ga = 0
        elif cur_dev >= best_ever_ga - patience_eps:
            no_improve_ga = 0
        else:
            no_improve_ga += 1
        print(f"  step {step+1:2d}/{budget}  train_map={best_score:.3f}  "
              f"dev_map={cur_dev:.3f}  dev_f1={dev_m['f1']:.3f}  no_improve={no_improve_ga}/{patience}")
        step_record = {
            "step":        step + 1,
            "train_map":   round(best_score, 4),
            "train_avg":   round(avg_score, 4),
            "dev_map":     cur_dev,
            "dev_auc":     dev_m["auc"],
            "dev_f1":      dev_m["f1"],
            "dev_acc":     dev_m["acc"],
            "best_prompt": best_prompt,
            "feedback":    "",
        }
        log.append(step_record)
        if logger is not None:
            logger.write({"type": "iteration", **step_record})
        if partial_path:
            with open(partial_path, "w", encoding="utf-8") as _f:
                json.dump({"population": population, "scores": scores,
                           "log": log, "last_step": step + 1}, _f,
                          ensure_ascii=False, indent=2)
        if no_improve_ga >= patience:
            print(f"  [조기종료] {patience}스텝 연속 미개선 → 종료 (step {step+1}/{budget})")
            break
    return population, scores, log


# ── DE 진화 루프 ──────────────────────────────────────────────────────────────
def run_de(population, scores, trainset, valset, budget, n_eval,
           partial_path=None, start_step=0, log=None, logger=None,
           patience=2, patience_eps=0.01):
    popsize = len(population)
    print(f"\n[DE] budget={budget}, popsize={popsize}, patience={patience}")
    if log is None:
        log = []
    best_ever_de  = max(scores) if scores else 0.0
    no_improve_de = 0
    for step in range(start_step, budget):
        step_sample = random.sample(trainset, min(n_eval, len(trainset)))
        scores = [evaluate_prompt(p, step_sample, n_eval) for p in population]

        for i in range(popsize):
            others = [j for j in range(popsize) if j != i]
            a, b   = random.sample(others, 2)

            mutant       = de_mutate(population[i], population[a], population[b])
            mutant_score = evaluate_prompt(mutant, step_sample, n_eval)

            if mutant_score >= scores[i]:
                population[i] = mutant
                scores[i]     = mutant_score

        best_score  = max(scores)
        avg_score   = sum(scores) / len(scores)
        best_prompt = population[scores.index(best_score)]

        dev_m   = evaluate_full(best_prompt, valset)
        cur_dev = dev_m["map"]
        if cur_dev > best_ever_de:
            best_ever_de  = cur_dev
            no_improve_de = 0
        elif cur_dev >= best_ever_de - patience_eps:
            no_improve_de = 0
        else:
            no_improve_de += 1
        print(f"  step {step+1:2d}/{budget}  train_map={best_score:.3f}  "
              f"dev_map={cur_dev:.3f}  dev_f1={dev_m['f1']:.3f}  no_improve={no_improve_de}/{patience}")
        step_record = {
            "step":        step + 1,
            "train_map":   round(best_score, 4),
            "train_avg":   round(avg_score, 4),
            "dev_map":     cur_dev,
            "dev_auc":     dev_m["auc"],
            "dev_f1":      dev_m["f1"],
            "dev_acc":     dev_m["acc"],
            "best_prompt": best_prompt,
            "feedback":    "",
        }
        log.append(step_record)
        if logger is not None:
            logger.write({"type": "iteration", **step_record})
        if partial_path:
            with open(partial_path, "w", encoding="utf-8") as _f:
                json.dump({"population": population, "scores": scores,
                           "log": log, "last_step": step + 1}, _f,
                          ensure_ascii=False, indent=2)
        if no_improve_de >= patience:
            print(f"  [조기종료] {patience}스텝 연속 미개선 → 종료 (step {step+1}/{budget})")
            break
    return population, scores, log


# ── 최종 상세 평가 ────────────────────────────────────────────────────────────
def eval_detailed(prompt: str, testset: list[dict], label: str = "") -> None:
    """F1 / AUC / MAP 상세 평가."""
    preds_str = classify_dataset(testset, prompt)
    y_score   = score_dataset(testset, prompt)
    y_true    = [1 if ex["label"] == "abnormal" else 0 for ex in testset]
    y_pred    = [1 if p == "abnormal" else 0 for p in preds_str]
    _print_metrics(_compute_metrics(y_true, y_pred, y_score), label=label)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", default="./frames_cache_2s")
    parser.add_argument("--mode",       default="ga", choices=["ga", "de"])
    parser.add_argument("--budget",      type=int,   default=12)
    parser.add_argument("--popsize",     type=int,   default=5)
    parser.add_argument("--n_eval",      type=int,   default=10)
    parser.add_argument("--output",      default="evoprompt_result.json")
    parser.add_argument("--patience",    type=int,   default=2,
                        help="조기종료: N스텝 연속 dev_map 미개선 시 중단 (TEMPO와 동일)")
    parser.add_argument("--patience_eps", type=float, default=0.01,
                        help="조기종료 내성: 이 미만 하락은 개선으로 간주 (TEMPO와 동일)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    # ── 데이터셋 (다른 모든 방법과 동일한 split — 공정한 baseline 비교) ────────
    print("\n=== 데이터셋 로드 & 2초 간격 프레임 추출 (공유 split 사용) ===")
    trainset, valset, testset = load_dataset_split(Path(args.frames_dir))
    n_norm_tr = sum(1 for x in trainset if x["label"] == "normal")
    n_norm_va = sum(1 for x in valset   if x["label"] == "normal")
    n_norm_te = sum(1 for x in testset  if x["label"] == "normal")
    print(f"분할 → train={len(trainset)}(normal={n_norm_tr}), "
          f"val={len(valset)}(normal={n_norm_va}), "
          f"test={len(testset)}(normal={n_norm_te})")

    partial_path = Path(args.output).parent / f"evoprompt_{args.mode}_partial_log.json"
    logger = _RoundLogger(f"evoprompt_{args.mode}")

    if partial_path.exists():
        with open(partial_path, encoding="utf-8") as _f:
            saved = json.load(_f)
        population = saved["population"]
        scores     = saved["scores"]
        saved_log  = saved.get("log", [])
        start_step = saved.get("last_step", 0)
        print(f"  [Resume] step {start_step}부터 재시작  best_score={max(scores):.3f}")
    else:
        # ── 초기 집단 생성 (INITIAL_PROMPT 1개 → LLM 변형 popsize개) ──────────
        print("\n=== 초기 집단 생성 (LLM 변형) ===")
        population = generate_initial_population(INITIAL_PROMPT, args.popsize)
        print(f"  생성된 집단: {len(population)}개")

        # ── 초기 집단 평가 ────────────────────────────────────────────────────
        print("\n=== 초기 집단 평가 ===")
        scores = []
        for i, p in enumerate(population):
            s = evaluate_prompt(p, trainset, args.n_eval)
            scores.append(s)
            print(f"  [{i+1}/{args.popsize}] score={s:.3f}  {p[:60]}...")
        saved_log  = []
        start_step = 0

        # 초기 집단 생성+평가까지는 GA/DE 루프 진입 전이라 체크포인트가 없었음 —
        # 여기서 한 번 저장해두면 이 단계 이후 중단돼도 처음부터(population 재생성+재평가)
        # 다시 할 필요 없이 step 0부터 재시작 가능.
        with open(partial_path, "w", encoding="utf-8") as _f:
            json.dump({"population": population, "scores": scores,
                       "log": saved_log, "last_step": 0}, _f,
                      ensure_ascii=False, indent=2)

    # ── 진화 ──────────────────────────────────────────────────────────────────
    print(f"\n=== EvoPrompt {args.mode.upper()} 최적화 시작 ===")
    if args.mode == "ga":
        population, scores, log = run_ga(
            population, scores, trainset, valset, args.budget, args.n_eval,
            partial_path=partial_path, start_step=start_step, log=saved_log,
            logger=logger, patience=args.patience, patience_eps=args.patience_eps)
    else:
        population, scores, log = run_de(
            population, scores, trainset, valset, args.budget, args.n_eval,
            partial_path=partial_path, start_step=start_step, log=saved_log,
            logger=logger, patience=args.patience, patience_eps=args.patience_eps)

    if partial_path.exists():
        partial_path.unlink()
    logger.close()

    # ── 최종 best prompt (dev MAP 기준으로 재선택) ────────────────────────────
    best_log    = max(log, key=lambda x: x["dev_map"])
    best_prompt = best_log["best_prompt"]
    best_train  = best_log["train_map"]
    print(f"\n최적 프롬프트 (dev_map={best_log['dev_map']:.3f}, step={best_log['step']}):\n  {best_prompt}")

    # ── 테스트 평가: baseline vs best ─────────────────────────────────────────
    print("\n=== 최종 테스트 평가 (MAP / AUC / F1 / ACC) ===")
    baseline_test = evaluate_full(INITIAL_PROMPT, testset)
    best_test     = evaluate_full(best_prompt,    testset)

    # threshold 재보정: dev_map으로 프롬프트를 고른 것과 별개로, 고정 0.5 threshold가
    # 최적화된 점수 분포와 안 맞아 F1/ACC가 왜곡될 수 있음 (TEMPO 계열과 동일 기준으로 맞춤).
    val_score  = score_dataset(valset, best_prompt)
    val_true   = [1 if ex["label"] == "abnormal" else 0 for ex in valset]
    best_thr   = find_best_threshold(val_true, val_score)
    best_test_opt = evaluate_full(best_prompt, testset, threshold=best_thr)
    best_test["f1_opt"]         = best_test_opt["f1"]
    best_test["acc_opt"]        = best_test_opt["acc"]
    best_test["best_threshold"] = best_thr

    print("▶ Baseline")
    _print_metrics({
        "accuracy": baseline_test["acc"], "f1_macro": baseline_test["f1"],
        "auc": baseline_test["auc"],      "map":      baseline_test["map"],
    }, label="Baseline")
    print(f"▶ EvoPrompt-{args.mode.upper()}")
    _print_metrics({
        "accuracy": best_test["acc"], "f1_macro": best_test["f1"],
        "auc": best_test["auc"],      "map":      best_test["map"],
        "f1_opt": best_test["f1_opt"], "acc_opt": best_test["acc_opt"],
        "best_threshold": best_test["best_threshold"],
    }, label=f"EvoPrompt-{args.mode.upper()}")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    output = {
        "method":           f"EvoPrompt-{args.mode.upper()}",
        "best_prompt":      best_prompt,
        "baseline_prompt":  INITIAL_PROMPT,
        "baseline_test":    baseline_test,
        "best_test":        best_test,
        "best_test_f1_opt":  best_test["f1_opt"],
        "best_test_acc_opt": best_test["acc_opt"],
        "train_map":        round(best_train, 4),
        "mode":             args.mode,
        "log":              log,
        "final_population": [{"prompt": p, "score": s}
                              for p, s in zip(population, scores)],
        "config": {
            "budget":  args.budget,
            "popsize": args.popsize,
            "n_eval":  args.n_eval,
        },
        "split": {"train": len(trainset), "val": len(valset), "test": len(testset)},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
