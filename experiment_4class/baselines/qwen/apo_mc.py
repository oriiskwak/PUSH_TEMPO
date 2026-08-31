"""
apo_mc.py — APO (ProTeGi) 5-class 최적화, Qwen2.5-VL

Usage:
  CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 python experiment_4class/baselines/qwen/apo_mc.py \
      --frames_dir ./frames_cache_2s
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


# ── 추론 함수 ─────────────────────────────────────────────────────────────────

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


def scorer_01_mc(prompts: list[str], eval_exs: list[dict]) -> list[float]:
    from sklearn.metrics import f1_score
    scores = []
    for prompt in prompts:
        preds = classify_dataset_mc(eval_exs, prompt)
        labels = [ex["label"] for ex in eval_exs]
        scores.append(float(f1_score(labels, preds, labels=CLASSES, average="macro", zero_division=0)))
    return scores


# ── APO 컴포넌트 ───────────────────────────────────────────────────────────────

class UCBBanditEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, prompts, exs, rounds, num_prompts_per_round, samples_per_eval):
        n = len(prompts)
        counts = np.zeros(n)
        scores = np.zeros(n)
        c = self.config.get("c", 1.0)

        def choose(t, k):
            if counts.sum() == 0:
                return random.sample(range(n), min(k, n))
            avg = np.divide(scores, counts, out=np.zeros(n), where=counts != 0)
            ucb = avg + c * np.sqrt(np.log(max(t, 1)) / (counts + 1e-3))
            return np.argsort(ucb)[::-1][:k].tolist()

        for ri in range(rounds):
            chosen_idx = choose(ri, min(num_prompts_per_round, n))
            chosen_prompts = [prompts[i] for i in chosen_idx]
            sample_data = random.sample(exs, min(samples_per_eval, len(exs)))
            batch_scores = scorer_01_mc(chosen_prompts, sample_data)
            for i, s in zip(chosen_idx, batch_scores):
                counts[i] += samples_per_eval
                scores[i] += s * samples_per_eval

        avg = np.divide(scores, counts, out=np.zeros(n), where=counts != 0)
        return avg.tolist()


class BruteForceEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, prompts, exs, rounds, num_prompts_per_round, samples_per_eval):
        sample_size = max(4, min(len(exs), int(self.config["eval_budget"] / max(len(prompts), 1))))
        eval_exs = random.sample(exs, min(sample_size, len(exs)))
        return scorer_01_mc(prompts, eval_exs)


class VideoFraudTaskMC:
    def evaluate(self, prompt, exs, n=None):
        subset = exs[:n] if n else exs
        preds_str = classify_dataset_mc(subset, prompt)
        frame_paths_list = [ex["frame_paths"] for ex in subset]
        labels = [ex["label"] for ex in subset]
        acc = sum(p == l for p, l in zip(preds_str, labels)) / max(len(subset), 1)
        return acc, frame_paths_list, labels, preds_str


class ProTeGiMC:
    def __init__(self, config, evaluator_fn, bf_eval):
        self.opt, self.evaluator_fn, self.bf_eval = config, evaluator_fn, bf_eval

    def _sample_error_str(self, frame_paths_list, labels, preds, n=4):
        error_idxs = [i for i, (l, p) in enumerate(zip(labels, preds)) if l != p]
        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))
        error_string, error_frame_paths = "", []
        for k, i in enumerate(sample_idxs):
            error_string += f"## Example {k+1}\nLabel: {labels[i]}\nPrediction: {preds[i]}\n\n"
            error_frame_paths.append(frame_paths_list[i][0])
        return error_string.strip(), error_frame_paths

    def parse_tagged_text(self, text, start_tag, end_tag):
        results = []
        while True:
            s = text.find(start_tag)
            if s == -1:
                break
            e = text.find(end_tag, s)
            if e == -1:
                break
            results.append(text[s + len(start_tag):e].strip())
            text = text[e + len(end_tag):]
        return results

    def _get_gradients(self, prompt, error_string, error_frame_paths, num_feedbacks=5, n=1):
        gradient_prompt = (
            f'I\'m trying to write a zero-shot classifier prompt for categorizing video frames '
            f'into five fraud categories (Type-A through Type-E).\n\n'
            f'My current prompt is:\n"{prompt}"\n\n'
            f'But this prompt gets the following examples wrong:\n{error_string}\n'
            f'The frames from these misclassified videos are attached above.\n\n'
            f'Give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.\n'
            f'Wrap each reason with <START> and <END>'
        )
        feedbacks = []
        for _ in range(n):
            out = predict_vision_text(gradient_prompt, error_frame_paths, temperature=0.7, max_tokens=600)
            feedbacks += self.parse_tagged_text(out, "<START>", "<END>")
        return feedbacks

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
        transformation_prompt = (
            f'I\'m trying to write a zero-shot classifier for 5-class fraud detection.\n\n'
            f'My current prompt is:\n"{prompt}"\n\n'
            f'But it gets the following examples wrong:\n{error_str}\n\n'
            f'Based on these examples the problem with this prompt is that {feedback_str}\n\n'
            f'Based on the above information, I wrote {steps_per_gradient} different improved prompts.\n'
            f'Each prompt must instruct the model to return ONLY one of: '
            f'Type-A / Type-B / Type-C / Type-D / Type-E.\n'
            f'Each prompt is wrapped with <START> and <END>.\n\n'
            f'The {steps_per_gradient} new prompts are:'
        )
        res = predict_text(transformation_prompt, n=n, temperature=0.7, max_tokens=800)
        new_prompts = []
        for r in res:
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3):
        rewriter_prompt = (
            f"Generate a variation of the following instruction while keeping the semantic meaning.\n\n"
            f"Input: {prompt_section}\n\nOutput:"
        )
        return [x for x in predict_text(rewriter_prompt, n=n, temperature=0.7, max_tokens=400) if x]

    def get_gradients(self, prompt, task, frame_paths_list, labels, preds):
        prompt_feedbacks = []
        for _ in range(self.opt["n_gradients"]):
            error_string, error_frame_paths = self._sample_error_str(
                frame_paths_list, labels, preds, n=self.opt["errors_per_gradient"]
            )
            gradients = self._get_gradients(prompt, error_string, error_frame_paths,
                                            self.opt["gradients_per_error"], n=1)
            prompt_feedbacks += [(g, error_string) for g in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts, task, train_exs):
        minibatch = random.sample(train_exs, min(self.opt["minibatch_size"], len(train_exs)))
        new_prompts = []
        for prompt in prompts:
            _, frame_paths_list, labels, preds = task.evaluate(prompt, minibatch)
            new_task_sections = []
            if self.opt["n_gradients"] > 0:
                gradients = self.get_gradients(prompt, task, frame_paths_list, labels, preds)
                for feedback, error_string in gradients:
                    new_task_sections += self.apply_gradient(prompt, error_string, feedback,
                                                             self.opt["steps_per_gradient"])
            mc_sampled = []
            if self.opt["mc_samples_per_step"] > 0:
                for sect in new_task_sections + [prompt]:
                    mc_sampled += self.generate_synonyms(sect, n=self.opt["mc_samples_per_step"])
            new_sections = list(set(new_task_sections + mc_sampled))
            if len(new_sections) > self.opt["max_expansion_factor"]:
                new_sections = random.sample(new_sections, self.opt["max_expansion_factor"])
            new_prompts += new_sections
        new_prompts = list(set(new_prompts + prompts))
        return new_prompts

    def score_candidates(self, prompts, train_exs):
        if len(prompts) == 1:
            return [1.0]
        return self.evaluator_fn(prompts, train_exs,
                                 rounds=self.opt["eval_rounds"],
                                 num_prompts_per_round=self.opt["eval_prompts_per_round"],
                                 samples_per_eval=self.opt["samples_per_eval"])


if __name__ == "__main__":
    import sys
    _TEMPO_ROOT = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(_TEMPO_ROOT))
    sys.path.insert(0, str(_TEMPO_ROOT / "baselines"))

    from vllm_utils import _llm, _processor, extract_frames_1fps, predict_text, predict_vision_text  # noqa: F401
    from utils_mc import CLASSES, TYPE_MAP, TYPE_SUFFIX, VIDEO_DIR, load_gt_mc, split_data_mc  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir",             default="./frames_cache_2s")
    parser.add_argument("--output",                 default=str(_OUT_DIR / "apo_mc_qwen_result.json"))
    parser.add_argument("--rounds",                 type=int, default=12)
    parser.add_argument("--beam_size",              type=int, default=4)
    parser.add_argument("--minibatch_size",         type=int, default=16)
    parser.add_argument("--n_gradients",            type=int, default=1)
    parser.add_argument("--errors_per_gradient",    type=int, default=4)
    parser.add_argument("--gradients_per_error",    type=int, default=1)
    parser.add_argument("--steps_per_gradient",     type=int, default=1)
    parser.add_argument("--mc_samples_per_step",    type=int, default=2)
    parser.add_argument("--max_expansion_factor",   type=int, default=8)
    parser.add_argument("--evaluator",              default="ucb", choices=["bf", "ucb"])
    parser.add_argument("--eval_rounds",            type=int, default=2)
    parser.add_argument("--eval_prompts_per_round", type=int, default=8)
    parser.add_argument("--samples_per_eval",       type=int, default=4)
    parser.add_argument("--c",                      type=float, default=1.0)
    parser.add_argument("--patience",               type=int, default=2)
    parser.add_argument("--patience_eps",           type=float, default=0.01)
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    config = {
        "rounds": args.rounds, "beam_size": args.beam_size,
        "minibatch_size": args.minibatch_size, "n_gradients": args.n_gradients,
        "errors_per_gradient": args.errors_per_gradient,
        "gradients_per_error": args.gradients_per_error,
        "steps_per_gradient": args.steps_per_gradient,
        "mc_samples_per_step": args.mc_samples_per_step,
        "max_expansion_factor": args.max_expansion_factor,
        "evaluator": args.evaluator, "eval_rounds": args.eval_rounds,
        "eval_prompts_per_round": args.eval_prompts_per_round,
        "samples_per_eval": args.samples_per_eval,
        "eval_budget": args.samples_per_eval * args.eval_rounds * args.eval_prompts_per_round,
        "c": args.c,
    }

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

    task = VideoFraudTaskMC()
    evaluator_fn = UCBBanditEvaluator(config) if args.evaluator == "ucb" else BruteForceEvaluator(config)
    optimizer = ProTeGiMC(config, evaluator_fn, BruteForceEvaluator(config))
    logger = _RoundLogger("apo_mc_qwen")

    partial_path = _OUT_DIR / "apo_mc_qwen_partial_log.json"
    if partial_path.exists():
        saved = json.load(open(partial_path, encoding="utf-8"))
        candidates = saved.get("candidates", [saved["best_prompt"]])
        scores = saved.get("scores", [saved["best_dev_f1"]])
        round_logs = saved.get("log", [])
        best_dev_f1 = saved.get("best_dev_f1", 0.0)
        best_prompt_so_far = saved.get("best_prompt", INITIAL_PROMPT)
        start_rnd = saved.get("last_round", 0) + 1
        no_improve = saved.get("no_improve", 0)
        best_ever = saved.get("best_ever", best_dev_f1)
        print(f"  [Resume] Round {start_rnd}부터 재시작  best_f1={best_dev_f1:.3f}")
    else:
        candidates = [INITIAL_PROMPT]
        scores = [0.0]
        round_logs, best_dev_f1, best_prompt_so_far = [], 0.0, INITIAL_PROMPT
        start_rnd, no_improve, best_ever = 0, 0, 0.0

    from sklearn.metrics import f1_score

    for rnd in range(start_rnd, config["rounds"] + 1):
        print(f"\n=== Round {rnd}/{config['rounds']} ===")
        if rnd > 0:
            candidates = optimizer.expand_candidates(candidates, task, trainset)
            print(f"  확장된 후보: {len(candidates)}개")

        scores = optimizer.score_candidates(candidates, trainset)
        paired = sorted(zip(scores, candidates), reverse=True)
        scores, candidates = [list(x) for x in zip(*paired)]
        candidates = candidates[:config["beam_size"]]
        scores = scores[:config["beam_size"]]
        print(f"  Best train_f1: {scores[0]:.3f}  |  {candidates[0][:80]}...")

        dev_preds = classify_dataset_mc(valset, candidates[0])
        dev_labels = [ex["label"] for ex in valset]
        dev_f1 = float(f1_score(dev_labels, dev_preds, labels=CLASSES, average="macro", zero_division=0))
        per_cls = {c: float(f1_score(dev_labels, dev_preds, labels=[c], average="macro", zero_division=0)) for c in CLASSES}
        acc = sum(t == p for t, p in zip(dev_labels, dev_preds)) / len(dev_labels)
        dev_m = {"macro_f1": round(dev_f1, 4), "per_class_f1": per_cls, "acc": round(acc, 4)}
        print(f"  dev_f1={dev_m['macro_f1']:.3f}  acc={dev_m['acc']:.3f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_prompt_so_far = candidates[0]
        if dev_f1 > best_ever:
            best_ever, no_improve = dev_f1, 0
        elif dev_f1 >= best_ever - args.patience_eps:
            no_improve = 0
        else:
            no_improve += 1

        round_record = {"round": rnd, "dev_f1": dev_m["macro_f1"], "dev_acc": dev_m["acc"],
                        "prompt": candidates[0], "no_improve": no_improve}
        round_logs.append(round_record)
        logger.write({"type": "iteration", **round_record})

        partial = {"best_prompt": best_prompt_so_far, "best_dev_f1": best_dev_f1,
                   "log": round_logs, "candidates": candidates, "scores": scores,
                   "last_round": rnd, "no_improve": no_improve, "best_ever": best_ever}
        with open(partial_path, "w", encoding="utf-8") as _f:
            json.dump(partial, _f, ensure_ascii=False, indent=2)
        print(f"  [저장] no_improve={no_improve}/{args.patience}")

        if no_improve >= args.patience:
            print(f"  [조기종료] {args.patience}라운드 연속 미개선")
            break

    if partial_path.exists():
        partial_path.unlink()
    logger.close()

    print(f"\n=== 최적 프롬프트 ===\n{best_prompt_so_far}\n")
    print("=== 최종 테스트 평가 ===")

    def _eval_test(prompt):
        preds = classify_dataset_mc(testset, prompt)
        labels = [ex["label"] for ex in testset]
        f1 = float(f1_score(labels, preds, labels=CLASSES, average="macro", zero_division=0))
        per_c = {c: float(f1_score(labels, preds, labels=[c], average="macro", zero_division=0)) for c in CLASSES}
        acc = sum(t == p for t, p in zip(labels, preds)) / len(labels)
        return {"macro_f1": round(f1, 4), "per_class_f1": per_c, "acc": round(acc, 4)}

    baseline_m = _eval_test(INITIAL_PROMPT)
    best_m = _eval_test(best_prompt_so_far)
    print(f"Baseline  macro_F1={baseline_m['macro_f1']:.3f}  acc={baseline_m['acc']:.3f}")
    print(f"APO-MC    macro_F1={best_m['macro_f1']:.3f}  acc={best_m['acc']:.3f}")

    result = {
        "method": "APO-MC (Qwen)", "best_prompt": best_prompt_so_far,
        "initial_prompt": INITIAL_PROMPT, "baseline_test": baseline_m,
        "best_test": best_m, "config": config, "log": round_logs,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {args.output}")
