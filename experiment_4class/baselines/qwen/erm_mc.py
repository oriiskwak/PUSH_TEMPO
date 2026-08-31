"""
erm_mc.py — ERM (Exemplar-Guided Reflection with Memory) 5-class, Qwen2.5-VL

Usage:
  CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 python experiment_4class/baselines/qwen/erm_mc.py \
      --frames_dir ./frames_cache_2s
"""
import argparse, json, math, random, re
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

REFLECTION_META = """\
You are optimizing a zero-shot classifier prompt for categorizing video frames \
into five fraud categories (Type-A: voice phishing, Type-B: investment fraud, \
Type-C: romance scam, Type-D: part-time job scam, Type-E: normal).

Current prompt: "{prompt}"

The following frames were misclassified (attached images above):
{error_desc}

Step 1: Select {n_e} diverse and representative misclassified examples. For each provide:
  - A text description of what is visually shown
  - The correct label (Type-A / Type-B / Type-C / Type-D / Type-E)
  - A detailed solution process explaining why it should be classified that way

Step 2: Based on those exemplars, generate {n_f} specific actionable feedbacks \
explaining why the current prompt failed.

Output format:
<exemplars>
[description | Type-X | solution process]
...
</exemplars>
<feedbacks>
<feedback>feedback text</feedback>
...
</feedbacks>"""

OPTIMIZATION_META = """\
You are improving a zero-shot classifier prompt for categorizing video frames into \
five fraud categories (Type-A: voice phishing / Type-B: investment fraud / \
Type-C: romance scam / Type-D: part-time job scam / Type-E: normal).

Current prompt: "{prompt}"

The prompt failed because: {feedback}

Relevant historical feedbacks:
{memory_feedbacks}

Generate an improved prompt that addresses these issues. \
The prompt must instruct the model to return ONLY one of: Type-A / Type-B / Type-C / Type-D / Type-E.

Output the improved prompt wrapped with <prompt> and </prompt>."""

MULTI_FEEDBACK_META = """\
You are improving a zero-shot classifier prompt for categorizing video frames into \
five fraud categories (Type-A: voice phishing / Type-B: investment fraud / \
Type-C: romance scam / Type-D: part-time job scam / Type-E: normal).

Current prompt: "{prompt}"

Multiple issues identified:
{feedbacks}

Generate an improved prompt that addresses all issues. \
The prompt must instruct the model to return ONLY one of: Type-A / Type-B / Type-C / Type-D / Type-E.

Output the improved prompt wrapped with <prompt> and </prompt>."""


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


def _softmax(scores: list[float], tau: float = 1.0) -> list[float]:
    shifted = [s / tau for s in scores]
    max_s = max(shifted)
    exps = [math.exp(s - max_s) for s in shifted]
    total = sum(exps)
    return [e / total for e in exps]


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def _parse_prompt(text: str, fallback: str) -> str:
    m = re.search(r"<prompt>(.*?)</prompt>", text, re.DOTALL)
    return m.group(1).strip() if m else fallback


class FeedbackMemory:
    def __init__(self, max_size=20, beta=0.7, theta=0.3, tau=1.0):
        self.items, self.max_size = [], max_size
        self.beta, self.theta, self.tau = beta, theta, tau

    def add(self, text: str, init_priority: float = 0.5):
        if any(_jaccard(it["text"], text) > 0.8 for it in self.items):
            return
        self.items.append({"text": text, "priority": init_priority})
        if len(self.items) > self.max_size:
            self.items.sort(key=lambda x: x["priority"], reverse=True)
            self.items = self.items[:self.max_size]

    def retrieve(self, n: int) -> list[str]:
        if not self.items:
            return []
        n = min(n, len(self.items))
        probs = _softmax([it["priority"] for it in self.items], self.tau)
        idxs = np.random.choice(len(self.items), size=n, replace=False, p=probs)
        return [self.items[i]["text"] for i in idxs]

    def update(self, texts: list[str], improved: bool):
        indicator = 1.0 if improved else 0.0
        for it in self.items:
            if it["text"] in texts:
                it["priority"] = (1 - self.beta) * it["priority"] + self.beta * indicator
        self.items = [it for it in self.items if it["priority"] >= self.theta]


class ExemplarFactory:
    def __init__(self, max_size=30, beta=0.7, theta=0.3, tau=1.0):
        self.items, self.max_size = [], max_size
        self.beta, self.theta, self.tau = beta, theta, tau

    def add(self, text: str, label: str, init_priority: float = 0.5):
        self.items.append({"text": text, "label": label, "priority": init_priority})
        if len(self.items) > self.max_size:
            self.items.sort(key=lambda x: x["priority"], reverse=True)
            self.items = self.items[:self.max_size]

    def retrieve_for_prompt(self, n: int = 3) -> str:
        if not self.items:
            return ""
        n = min(n, len(self.items))
        probs = _softmax([it["priority"] for it in self.items], self.tau)
        idxs = np.random.choice(len(self.items), size=n, replace=False, p=probs)
        lines = [f"Example {k+1} [Label: {self.items[i]['label']}]: {self.items[i]['text']}"
                 for k, i in enumerate(idxs)]
        return "\n\nReference examples from similar videos:\n" + "\n".join(lines)

    def update(self, texts: list[str], improved: bool):
        indicator = 1.0 if improved else 0.0
        for it in self.items:
            if it["text"] in texts:
                it["priority"] = (1 - self.beta) * it["priority"] + self.beta * indicator
        self.items = [it for it in self.items if it["priority"] >= self.theta]


# ── 추론 함수 (module level — 실제 호출은 __main__ guard 이후) ─────────────────

def _predict_batch_mc(items: list[tuple]) -> list:
    from vllm import SamplingParams
    results = []
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    for start in range(0, len(items), _VLLM_BATCH):
        chunk = items[start:start + _VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            img_np = np.array(img)
            content = [
                {"type": "image", "image": img_np},
                {"type": "text",  "text":  prompt + TYPE_SUFFIX},
            ]
            text = _processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs.append({"prompt": text, "multi_modal_data": {"image": img_np}})
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


def evaluate_prompt_mc(prompt: str, dataset: list[dict]) -> tuple[float, list[dict], dict]:
    from sklearn.metrics import f1_score
    preds = classify_dataset_mc(dataset, prompt)
    labels = [ex["label"] for ex in dataset]
    errors = [ex for ex, p in zip(dataset, preds) if p != ex["label"]]
    f1 = float(f1_score(labels, preds, labels=CLASSES, average="macro", zero_division=0))
    per_cls = {c: float(f1_score(labels, preds, labels=[c], average="macro", zero_division=0))
               for c in CLASSES}
    acc = sum(t == p for t, p in zip(labels, preds)) / max(len(labels), 1)
    return f1, errors, {"macro_f1": f1, "per_class_f1": per_cls, "acc": acc}


def exemplar_guided_reflection_mc(prompt, errors, n_e, n_f):
    sample = random.sample(errors, min(n_e, len(errors)))
    img_paths = [ex["frame_paths"][0] for ex in sample]
    error_desc = "\n".join(f"Frame {i+1}: True label = {ex['label']}"
                           for i, ex in enumerate(sample))
    meta = REFLECTION_META.format(prompt=prompt, error_desc=error_desc, n_e=n_e, n_f=n_f)
    out = predict_vision_text(meta, img_paths, temperature=0.7, max_tokens=900)

    exemplars = []
    m = re.search(r"<exemplars>(.*?)</exemplars>", out, re.DOTALL)
    if m:
        for line in m.group(1).strip().splitlines():
            line = line.strip("[]- \t")
            if "|" in line:
                parts = line.split("|")
                desc = parts[0].strip()
                label_text = parts[1].strip() if len(parts) > 1 else ""
                cls = next((v for k, v in TYPE_MAP.items() if k in label_text), None)
                if not cls:
                    cls = next((c for c in CLASSES if c in label_text), CLASSES[0])
                if desc:
                    exemplars.append({"text": desc, "label": cls})

    feedbacks = [f.strip() for f in
                 re.findall(r"<feedback>(.*?)</feedback>", out, re.DOTALL) if f.strip()]
    return exemplars, feedbacks


def apply_feedback(prompt: str, feedback: str, mem_feedbacks: list[str]) -> str:
    mem_str = "\n".join(f"- {f}" for f in mem_feedbacks) if mem_feedbacks else "None"
    meta = OPTIMIZATION_META.format(prompt=prompt, feedback=feedback, memory_feedbacks=mem_str)
    out = predict_text(meta, n=1, temperature=0.7, max_tokens=512)[0]
    return _parse_prompt(out, prompt)


def apply_multi_feedback(prompt: str, feedbacks: list[str]) -> str:
    fb_str = "\n".join(f"{i+1}. {f}" for i, f in enumerate(feedbacks))
    meta = MULTI_FEEDBACK_META.format(prompt=prompt, feedbacks=fb_str)
    out = predict_text(meta, n=1, temperature=0.7, max_tokens=512)[0]
    return _parse_prompt(out, prompt)


def run_erm_mc(trainset, valset, config, partial_path):
    beam_size  = config["beam_size"]
    rounds     = config["rounds"]
    n_e        = config["n_exemplars"]
    n_f        = config["n_feedbacks"]
    n_fb_mem   = config["n_feedback_from_memory"]
    n_ex_inf   = config["n_exemplars_inference"]
    mb_size    = config["minibatch_size"]
    val_sub_n  = config.get("val_subsample", 15)
    patience   = config.get("patience", 2)
    pat_eps    = config.get("patience_eps", 0.01)

    fb_mem  = FeedbackMemory(max_size=config["mem_max_size"])
    ex_fact = ExemplarFactory(max_size=config["ex_max_size"])
    logger  = _RoundLogger("erm_mc_qwen")

    if partial_path.exists():
        saved = json.load(open(partial_path, encoding="utf-8"))
        beam, beam_scores = saved.get("beam", [saved["best_prompt"]]), saved.get("beam_scores", [saved["best_dev_f1"]])
        log, fb_mem.items, ex_fact.items = saved.get("log", []), saved.get("fb_mem_items", []), saved.get("ex_fact_items", [])
        start_rnd, no_improve_count, best_ever = saved.get("last_round", 0) + 1, saved.get("no_improve_count", 0), saved.get("best_ever_f1", max(beam_scores))
        print(f"  [Resume] Round {start_rnd}부터 재시작  best_f1={best_ever:.3f}")
    else:
        beam = [INITIAL_PROMPT]
        beam_scores = [evaluate_prompt_mc(INITIAL_PROMPT, valset)[0]]
        log, start_rnd, no_improve_count, best_ever = [], 1, 0, max(beam_scores)
        print(f"초기 val_f1={beam_scores[0]:.3f}")

    for rnd in range(start_rnd, rounds + 1):
        print(f"\n=== Round {rnd}/{rounds} ===")
        new_cands, round_fbs, round_exs = list(beam), [], []

        for prompt in beam:
            mb = random.sample(trainset, min(mb_size, len(trainset)))
            _, errors, _ = evaluate_prompt_mc(prompt, mb)
            if not errors:
                continue
            exemplars, feedbacks = exemplar_guided_reflection_mc(prompt, errors, n_e, n_f)
            print(f"  exemplars={len(exemplars)}, feedbacks={len(feedbacks)}")
            for fb in feedbacks:
                logger.write({"type": "feedback", "round": rnd, "feedback": fb})
            round_exs.extend(exemplars)
            round_fbs.extend(feedbacks)
            mem_fbs = fb_mem.retrieve(n_fb_mem)
            for fb in feedbacks:
                new_cands.append(apply_feedback(prompt, fb, mem_fbs))
            if len(feedbacks) > 1:
                new_cands.append(apply_multi_feedback(prompt, feedbacks + mem_fbs))

        for ex in round_exs:
            ex_fact.add(ex["text"], ex["label"])

        new_cands = list(set(new_cands))
        val_sub = random.sample(valset, min(val_sub_n, len(valset)))
        ex_suffix = ex_fact.retrieve_for_prompt(n_ex_inf)
        scored = []
        for cand in new_cands:
            f1, _, _ = evaluate_prompt_mc(cand + ex_suffix, val_sub)
            scored.append((f1, cand))
        scored.sort(reverse=True)
        beam = [s[1] for s in scored[:beam_size]]
        beam_scores = [s[0] for s in scored[:beam_size]]

        best_score = beam_scores[0]
        improved = best_score > best_ever - pat_eps
        if best_score > best_ever:
            best_ever, no_improve_count = best_score, 0
        elif improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

        _, _, dev_m = evaluate_prompt_mc(beam[0] + ex_suffix, valset)
        print(f"  dev_f1={dev_m['macro_f1']:.3f}  acc={dev_m['acc']:.3f}  {'↑' if improved else f'- ({no_improve_count}/{patience})'}")

        for fb in round_fbs:
            fb_mem.add(fb)
        fb_mem.update(round_fbs, improved)
        ex_fact.update([ex["text"] for ex in round_exs], improved)

        round_rec = {"round": rnd, "dev_f1": dev_m["macro_f1"], "dev_acc": dev_m["acc"], "prompt": beam[0], "improved": improved}
        log.append(round_rec)
        logger.write({"type": "iteration", **round_rec})

        partial = {"best_prompt": beam[0], "best_dev_f1": max(beam_scores), "log": log,
                   "beam": beam, "beam_scores": beam_scores, "fb_mem_items": fb_mem.items,
                   "ex_fact_items": ex_fact.items, "last_round": rnd,
                   "no_improve_count": no_improve_count, "best_ever_f1": best_ever}
        with open(partial_path, "w", encoding="utf-8") as _f:
            json.dump(partial, _f, ensure_ascii=False, indent=2)

        if no_improve_count >= patience:
            print(f"  [조기종료] {patience}라운드 연속 미개선")
            break

    if partial_path.exists():
        partial_path.unlink()
    logger.close()

    ex_suffix = ex_fact.retrieve_for_prompt(n_ex_inf)
    return beam[0] + ex_suffix, beam[0], log


if __name__ == "__main__":
    import sys
    _TEMPO_ROOT = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(_TEMPO_ROOT))
    sys.path.insert(0, str(_TEMPO_ROOT / "baselines"))

    from vllm_utils import _llm, _processor, extract_frames_1fps, predict_text, predict_vision_text  # noqa: F401
    from utils_mc import CLASSES, TYPE_MAP, TYPE_SUFFIX, VIDEO_DIR, load_gt_mc, split_data_mc, compute_metrics_mc  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir",             default="./frames_cache_2s")
    parser.add_argument("--output",                 default=str(_OUT_DIR / "erm_mc_qwen_result.json"))
    parser.add_argument("--rounds",                 type=int, default=12)
    parser.add_argument("--beam_size",              type=int, default=2)
    parser.add_argument("--minibatch_size",         type=int, default=16)
    parser.add_argument("--n_exemplars",            type=int, default=1)
    parser.add_argument("--n_feedbacks",            type=int, default=2)
    parser.add_argument("--n_feedback_from_memory", type=int, default=1)
    parser.add_argument("--n_exemplars_inference",  type=int, default=3)
    parser.add_argument("--mem_max_size",           type=int, default=20)
    parser.add_argument("--ex_max_size",            type=int, default=30)
    parser.add_argument("--patience",               type=int, default=2)
    parser.add_argument("--patience_eps",           type=float, default=0.01)
    parser.add_argument("--val_subsample",          type=int, default=15)
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    config = vars(args)

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
                items.append({"video_name": v["video_name"], "frame_paths": frame_paths, "label": v["label"]})
        return items

    trainset = _convert(trainset_meta)
    valset   = _convert(valset_meta)
    testset  = _convert(testset_meta)
    print(f"분할 → train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    partial_path = _OUT_DIR / "erm_mc_qwen_partial_log.json"
    print("\n=== ERM-MC 최적화 시작 ===")
    best_prompt, best_base, log = run_erm_mc(trainset, valset, config, partial_path)

    print("\n=== 최종 테스트 평가 ===")
    def _fmt(v): return f"{v:.3f}"
    _, _, base_m = evaluate_prompt_mc(INITIAL_PROMPT, testset)
    _, _, best_m = evaluate_prompt_mc(best_prompt, testset)
    print(f"Baseline  macro_F1={_fmt(base_m['macro_f1'])}  acc={_fmt(base_m['acc'])}")
    print(f"ERM-MC    macro_F1={_fmt(best_m['macro_f1'])}  acc={_fmt(best_m['acc'])}")

    result = {
        "method": "ERM-MC (Qwen)", "best_prompt": best_prompt,
        "best_prompt_base": best_base, "initial_prompt": INITIAL_PROMPT,
        "baseline_test": base_m, "best_test": best_m, "config": config, "log": log,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {args.output}")
