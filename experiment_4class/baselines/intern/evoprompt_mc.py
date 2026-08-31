"""
evoprompt_mc.py — EvoPrompt GA/DE 5-class 최적화, InternVL2.5-78B-AWQ

Usage:
  CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=1 PP_SIZE=2 python experiment_4class/baselines/intern/evoprompt_mc.py \
      --mode ga --frames_dir ./frames_cache_2s
"""
import argparse, json, random, re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

_OUT_DIR = Path(__file__).parent.parent.parent.parent / "results_4class"
_OUT_DIR.mkdir(exist_ok=True)
_LOG_DIR = _OUT_DIR / "em_logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_VLLM_BATCH = 32

RANDOM_SEED    = 42
TRAIN_RATIO    = 0.6
DEV_RATIO      = 0.2
INITIAL_PROMPT = (
    "Classify this video frame into one of five fraud categories.\n"
    "- Type-A\n- Type-B\n- Type-C\n- Type-D\n- Type-E"
)

GA_TEMPLATE = """\
Please follow the instruction step-by-step to generate a better prompt.
1. Cross over the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt \
bracketed with <prompt> and </prompt>.
"""

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

_FRAUD_KEYWORDS = {
    "fraud", "scam", "phishing", "classify", "category", "type",
    "investment", "romance", "part-time", "voice", "normal", "detect",
}


class _RoundLogger:
    def __init__(self, run_name: str):
        ts = datetime.now().strftime("%m%d_%H%M")
        self.path = _LOG_DIR / f"{run_name}_{ts}.jsonl"
        self._f = open(self.path, "a", encoding="utf-8")
        print(f"[Logger] 로그 저장: {self.path}")

    def write(self, record: dict):
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def _is_valid_prompt(text: str) -> bool:
    if len(text) < 40:
        return False
    return bool(set(text.lower().split()) & _FRAUD_KEYWORDS)


def parse_prompt(text: str, fallback: str) -> str:
    m = re.search(r"<prompt>(.*?)</prompt>", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if _is_valid_prompt(candidate):
            return candidate
    return fallback


# ── 추론 함수 (InternVL: PIL 직접 전달) ──────────────────────────────────────

def _predict_batch_mc(items: list[tuple]) -> list:
    from vllm import SamplingParams
    results = []
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    for start in range(0, len(items), _VLLM_BATCH):
        chunk = items[start:start + _VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            text = _apply_chat_template(prompt + TYPE_SUFFIX, n_images=1)
            inputs.append({"prompt": text, "multi_modal_data": {"image": img}})
        try:
            outs = _llm.generate(inputs, sp, use_tqdm=False)
            for out in outs:
                raw = out.outputs[0].text.strip()
                matched = next((c for c in CLASSES if c in raw), None)
                if matched is None:
                    matched = next((cls for k, cls in TYPE_MAP.items() if k in raw), None)
                results.append(matched)
        except Exception as e:
            print(f"  [Warning] vLLM error: {e}")
            results.extend([None] * len(chunk))
    return results


def classify_dataset_mc(dataset: list[dict], prompt: str) -> list[str]:
    from PIL import Image as _Image
    all_items, frame_counts = [], []
    for ex in dataset:
        imgs = [_Image.open(fp) for fp in ex["frame_paths"]]
        all_items.extend([(img, prompt) for img in imgs])
        frame_counts.append(len(ex["frame_paths"]))
    all_preds = _predict_batch_mc(all_items)
    video_preds, idx = [], 0
    for count in frame_counts:
        preds_slice = [p for p in all_preds[idx:idx + count] if p is not None]
        video_preds.append(Counter(preds_slice).most_common(1)[0][0] if preds_slice else CLASSES[0])
        idx += count
    return video_preds


def evaluate_prompt_mc(prompt: str, dataset: list[dict], n_eval: int) -> float:
    from sklearn.metrics import f1_score
    subset = random.sample(dataset, min(n_eval, len(dataset)))
    preds = classify_dataset_mc(subset, prompt)
    labels = [ex["label"] for ex in subset]
    return float(f1_score(labels, preds, labels=CLASSES, average="macro", zero_division=0))


def evaluate_full_mc(prompt: str, dataset: list[dict]) -> dict:
    from sklearn.metrics import f1_score
    preds = classify_dataset_mc(dataset, prompt)
    labels = [ex["label"] for ex in dataset]
    macro_f1 = float(f1_score(labels, preds, labels=CLASSES, average="macro", zero_division=0))
    per_cls = {c: float(f1_score(labels, preds, labels=[c], average="macro", zero_division=0))
               for c in CLASSES}
    acc = sum(t == p for t, p in zip(labels, preds)) / max(len(labels), 1)
    return {"macro_f1": round(macro_f1, 4), "per_class_f1": per_cls, "acc": round(acc, 4)}


def generate_initial_population(seed: str, size: int) -> list[str]:
    if size <= 1:
        return [seed]
    query = (
        f"Generate {size - 1} paraphrased variations of the following prompt. "
        f"Keep the same task (5-class fraud categorization of video frames) and meaning "
        f"but use different wording. Each variation must instruct the model to "
        f"return ONLY one of: Type-A / Type-B / Type-C / Type-D / Type-E.\n\n"
        f"Original prompt:\n{seed}\n\n"
        f"Wrap each variation with <prompt> and </prompt>."
    )
    out = predict_text(query, n=1, temperature=1.0, max_tokens=800)[0]
    variants = [m.group(1).strip()
                for m in re.finditer(r"<prompt>(.*?)</prompt>", out, re.DOTALL)
                if _is_valid_prompt(m.group(1).strip())]
    population = [seed] + variants[:size - 1]
    while len(population) < size:
        population.append(seed)
    return population


def ga_crossover(p1: str, p2: str) -> str:
    query = GA_TEMPLATE.replace("<prompt1>", p1).replace("<prompt2>", p2)
    out = predict_text(query, n=1, temperature=1.0, max_tokens=300)[0]
    return parse_prompt(out, p1)


def de_mutate(p0: str, p1: str, p2: str) -> str:
    query = (DE_TEMPLATE.replace("<prompt0>", p0)
             .replace("<prompt1>", p1).replace("<prompt2>", p2))
    out = predict_text(query, n=1, temperature=1.0, max_tokens=300)[0]
    return parse_prompt(out, p0)


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


def run_ga(population, scores, trainset, valset, budget, n_eval,
           partial_path=None, start_step=0, log=None, logger=None,
           patience=2, patience_eps=0.01):
    print(f"\n[GA] budget={budget}, popsize={len(population)}, patience={patience}")
    if log is None:
        log = []
    best_ever, no_improve = max(scores) if scores else 0.0, 0
    for step in range(start_step, budget):
        step_sample = random.sample(trainset, min(n_eval, len(trainset)))
        scores = [evaluate_prompt_mc(p, step_sample, n_eval) for p in population]
        i1 = roulette_wheel(scores)
        i2 = roulette_wheel(scores)
        while i2 == i1:
            i2 = roulette_wheel(scores)
        child = ga_crossover(population[i1], population[i2])
        child_score = evaluate_prompt_mc(child, step_sample, n_eval)
        worst_idx = scores.index(min(scores))
        if child_score >= scores[worst_idx]:
            population[worst_idx], scores[worst_idx] = child, child_score

        best_score = max(scores)
        best_prompt = population[scores.index(best_score)]
        dev_m = evaluate_full_mc(best_prompt, valset)
        cur_dev = dev_m["macro_f1"]
        if cur_dev > best_ever:
            best_ever, no_improve = cur_dev, 0
        elif cur_dev >= best_ever - patience_eps:
            no_improve = 0
        else:
            no_improve += 1
        print(f"  step {step+1:2d}/{budget}  train_f1={best_score:.3f}  dev_f1={cur_dev:.3f}  no_improve={no_improve}/{patience}")

        rec = {"step": step+1, "train_f1": round(best_score, 4), "dev_f1": cur_dev,
               "dev_acc": dev_m["acc"], "best_prompt": best_prompt}
        log.append(rec)
        if logger:
            logger.write({"type": "iteration", **rec})
        if partial_path:
            with open(partial_path, "w", encoding="utf-8") as _f:
                json.dump({"population": population, "scores": scores, "log": log, "last_step": step+1},
                          _f, ensure_ascii=False, indent=2)
        if no_improve >= patience:
            print(f"  [조기종료] {patience}스텝 연속 미개선")
            break
    return population, scores, log


def run_de(population, scores, trainset, valset, budget, n_eval,
           partial_path=None, start_step=0, log=None, logger=None,
           patience=2, patience_eps=0.01):
    popsize = len(population)
    print(f"\n[DE] budget={budget}, popsize={popsize}, patience={patience}")
    if log is None:
        log = []
    best_ever, no_improve = max(scores) if scores else 0.0, 0
    for step in range(start_step, budget):
        step_sample = random.sample(trainset, min(n_eval, len(trainset)))
        scores = [evaluate_prompt_mc(p, step_sample, n_eval) for p in population]
        for i in range(popsize):
            others = [j for j in range(popsize) if j != i]
            a, b = random.sample(others, 2)
            mutant = de_mutate(population[i], population[a], population[b])
            mutant_score = evaluate_prompt_mc(mutant, step_sample, n_eval)
            if mutant_score >= scores[i]:
                population[i], scores[i] = mutant, mutant_score

        best_score = max(scores)
        best_prompt = population[scores.index(best_score)]
        dev_m = evaluate_full_mc(best_prompt, valset)
        cur_dev = dev_m["macro_f1"]
        if cur_dev > best_ever:
            best_ever, no_improve = cur_dev, 0
        elif cur_dev >= best_ever - patience_eps:
            no_improve = 0
        else:
            no_improve += 1
        print(f"  step {step+1:2d}/{budget}  train_f1={best_score:.3f}  dev_f1={cur_dev:.3f}  no_improve={no_improve}/{patience}")

        rec = {"step": step+1, "train_f1": round(best_score, 4), "dev_f1": cur_dev,
               "dev_acc": dev_m["acc"], "best_prompt": best_prompt}
        log.append(rec)
        if logger:
            logger.write({"type": "iteration", **rec})
        if partial_path:
            with open(partial_path, "w", encoding="utf-8") as _f:
                json.dump({"population": population, "scores": scores, "log": log, "last_step": step+1},
                          _f, ensure_ascii=False, indent=2)
        if no_improve >= patience:
            print(f"  [조기종료] {patience}스텝 연속 미개선")
            break
    return population, scores, log


if __name__ == "__main__":
    import sys
    _TEMPO_ROOT = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(_TEMPO_ROOT))

    from vllm_utils_intern import _llm, _apply_chat_template, extract_frames_1fps, predict_text  # noqa: F401
    from utils_mc import CLASSES, TYPE_MAP, TYPE_SUFFIX, VIDEO_DIR, load_gt_mc, split_data_mc  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir",   default="./frames_cache_2s")
    parser.add_argument("--mode",         default="ga", choices=["ga", "de"])
    parser.add_argument("--budget",       type=int, default=12)
    parser.add_argument("--popsize",      type=int, default=5)
    parser.add_argument("--n_eval",       type=int, default=10)
    parser.add_argument("--output",       default=str(_OUT_DIR / "evoprompt_mc_intern_result.json"))
    parser.add_argument("--patience",     type=int, default=2)
    parser.add_argument("--patience_eps", type=float, default=0.01)
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n=== 5-class 데이터셋 로드 & 프레임 추출 ===")
    gt_list = load_gt_mc()
    trainset_meta, valset_meta, testset_meta = split_data_mc(gt_list, TRAIN_RATIO, DEV_RATIO, RANDOM_SEED)

    def _convert(meta_list):
        items = []
        for v in meta_list:
            vfiles = list(VIDEO_DIR.glob(f"{v['video_name']}.*"))
            if not vfiles:
                continue
            frame_paths = extract_frames_1fps(vfiles[0], Path(args.frames_dir))
            if frame_paths:
                items.append({"frame_paths": frame_paths, "label": v["label"]})
        return items

    trainset = _convert(trainset_meta)
    valset   = _convert(valset_meta)
    testset  = _convert(testset_meta)
    print(f"분할 → train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    partial_path = _OUT_DIR / f"evoprompt_mc_{args.mode}_intern_partial_log.json"
    logger = _RoundLogger(f"evoprompt_mc_{args.mode}_intern")

    if partial_path.exists():
        saved = json.load(open(partial_path, encoding="utf-8"))
        population, scores = saved["population"], saved["scores"]
        saved_log, start_step = saved.get("log", []), saved.get("last_step", 0)
        print(f"  [Resume] step {start_step}부터 재시작  best_f1={max(scores):.3f}")
    else:
        print("\n=== 초기 집단 생성 ===")
        population = generate_initial_population(INITIAL_PROMPT, args.popsize)
        scores = [evaluate_prompt_mc(p, trainset, args.n_eval) for p in population]
        saved_log, start_step = [], 0
        with open(partial_path, "w", encoding="utf-8") as _f:
            json.dump({"population": population, "scores": scores, "log": [], "last_step": 0},
                      _f, ensure_ascii=False, indent=2)

    print(f"\n=== EvoPrompt-{args.mode.upper()} MC (InternVL) 최적화 시작 ===")
    if args.mode == "ga":
        population, scores, log = run_ga(population, scores, trainset, valset, args.budget,
                                         args.n_eval, partial_path, start_step, saved_log,
                                         logger, args.patience, args.patience_eps)
    else:
        population, scores, log = run_de(population, scores, trainset, valset, args.budget,
                                         args.n_eval, partial_path, start_step, saved_log,
                                         logger, args.patience, args.patience_eps)

    if partial_path.exists():
        partial_path.unlink()
    logger.close()

    best_log = max(log, key=lambda x: x["dev_f1"])
    best_prompt = best_log["best_prompt"]
    print(f"\n최적 프롬프트 (dev_f1={best_log['dev_f1']:.3f}):\n  {best_prompt[:120]}")

    baseline_test = evaluate_full_mc(INITIAL_PROMPT, testset)
    best_test = evaluate_full_mc(best_prompt, testset)
    print(f"Baseline  macro_F1={baseline_test['macro_f1']:.3f}  acc={baseline_test['acc']:.3f}")
    print(f"EvoPrompt macro_F1={best_test['macro_f1']:.3f}  acc={best_test['acc']:.3f}")

    result = {
        "method": f"EvoPrompt-{args.mode.upper()}-MC (InternVL)",
        "best_prompt": best_prompt, "baseline_prompt": INITIAL_PROMPT,
        "baseline_test": baseline_test, "best_test": best_test,
        "mode": args.mode, "log": log,
        "final_population": [{"prompt": p, "score": s} for p, s in zip(population, scores)],
        "config": {"budget": args.budget, "popsize": args.popsize, "n_eval": args.n_eval},
        "split": {"train": len(trainset), "val": len(valset), "test": len(testset)},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {args.output}")
