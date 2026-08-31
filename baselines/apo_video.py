"""
APO-Video: Automatic Prompt Optimization (ProTeGi) — vLLM 직접 방식 적용
Based on: https://github.com/microsoft/LMOps/tree/main/prompt_optimization
Paper: "Automatic Prompt Optimization with 'Gradient Descent' and Beam Search" (EMNLP 2023)

추론: TEMPO와 동일하게 vLLM 직접 호출, 2초 간격 프레임, scam_ratio 집계.
데이터셋은 utils_gpt.load_gt()+split_data()로 다른 모든 방법과 동일한 train/dev/test를
사용한다 (공정한 baseline 비교).

사용법:
    TP_SIZE=2 python apo_video.py
"""
import argparse
import json
import random
import re
from datetime import datetime

import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── 라운드별 JSONL 로거 (다른 실험들의 em_core.EMLogger와 같은 규칙:
#    results/em_logs/{run_name}_{MMDD_HHMM}.jsonl, 매 줄 즉시 flush) ──
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
# 최상단에 무조건 실행되면, 재-import 때마다 vllm_utils.py의 모듈 레벨
# `_llm = LLM(...)`이 또 실행돼서 워커가 또 워커를 spawn하려다 죽는다
# (RuntimeError: bootstrapping phase). `if __name__ == "__main__":`으로 감싸면
# 재-import 시 "__mp_main__" != "__main__"라 이 블록이 스킵되어 재귀가 끊긴다.
if __name__ == "__main__":
    from eval_metrics import compute_metrics, print_metrics, find_best_threshold
    from vllm_utils import (
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


# ── 데이터셋 로드 — utils_gpt.load_gt()+split_data()로 다른 모든 방법과 동일한
# train/dev/test 구성을 쓰고, APO가 쓰는 frame_paths 캐시 포맷으로 변환만 한다
# (독자적인 *.mp4 glob + 자체 split을 쓰면 비디오 구성이 미묘하게 달라져
# baseline 비교가 불공정해짐).
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
            items.append({"frame_paths": frame_paths, "label": v["label"]})
        return items

    return _convert(trainset), _convert(devset), _convert(testset)




# ── 스코러 (MAP 기반) ────────────────────────────────────────────────────────
def scorer_map(prompts: list[str], eval_exs: list[dict]) -> list[float]:
    """각 프롬프트의 MAP 점수 반환."""
    scores = []
    for prompt in prompts:
        y_score = score_dataset(eval_exs, prompt)
        y_true  = [1 if ex["label"] == "abnormal" else 0 for ex in eval_exs]
        y_pred  = [1 if s >= 0.5 else 0 for s in y_score]
        m       = compute_metrics(y_true, y_pred, y_score)
        scores.append(m["map"] or 0.0)
    return scores


def scorer_01(prompts: list[str], eval_exs: list[dict]) -> list[float]:
    """각 프롬프트의 정확도 점수 반환 (UCB 내부용)."""
    n_exs = len(eval_exs)
    scores = []
    for prompt in prompts:
        preds   = classify_dataset(eval_exs, prompt)
        correct = sum(p == ex["label"] for p, ex in zip(preds, eval_exs))
        scores.append(correct / n_exs)
    return scores


# ── UCB 평가기 (원본 evaluators.py UCBBanditEvaluator 대응) ──────────────────
class UCBBanditEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, prompts: list[str], exs: list[dict],
                 rounds: int, num_prompts_per_round: int,
                 samples_per_eval: int) -> list[float]:
        n      = len(prompts)
        counts = np.zeros(n)
        scores = np.zeros(n)
        c      = self.config.get("c", 1.0)

        def choose(t: int, k: int) -> list[int]:
            if counts.sum() == 0:
                return random.sample(range(n), min(k, n))
            avg = np.divide(scores, counts, out=np.zeros(n), where=counts != 0)
            ucb = avg + c * np.sqrt(np.log(max(t, 1)) / (counts + 1e-3))
            return np.argsort(ucb)[::-1][:k].tolist()

        for ri in range(rounds):
            chosen_idx     = choose(ri, min(num_prompts_per_round, n))
            chosen_prompts = [prompts[i] for i in chosen_idx]
            sample_data    = random.sample(exs, min(samples_per_eval, len(exs)))
            batch_scores   = scorer_01(chosen_prompts, sample_data)
            for i, s in zip(chosen_idx, batch_scores):
                counts[i] += samples_per_eval
                scores[i] += s * samples_per_eval

        avg = np.divide(scores, counts, out=np.zeros(n), where=counts != 0)
        return avg.tolist()


# ── Brute Force 평가기 (원본 evaluators.py BruteForceEvaluator 대응) ──────────
class BruteForceEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, prompts: list[str], exs: list[dict],
                 rounds: int, num_prompts_per_round: int,
                 samples_per_eval: int) -> list[float]:
        sample_size = max(4, min(len(exs),
                                 int(self.config["eval_budget"] / max(len(prompts), 1))))
        eval_exs = random.sample(exs, min(sample_size, len(exs)))
        return scorer_01(prompts, eval_exs)


# ── 태스크 (원본 tasks.py 대응) ───────────────────────────────────────────────
class VideoFraudTask:
    def stringify_prediction(self, pred: int) -> str:
        return "abnormal" if pred == 1 else "normal"

    def evaluate(self, prompt: str, exs: list[dict],
                 n: int = None) -> tuple:
        """
        반환: (accuracy, frame_paths_list, labels, preds)
        원본 tasks.evaluate() 시그니처 유지.
        """
        subset           = exs[:n] if n else exs
        preds_str        = classify_dataset(subset, prompt)
        frame_paths_list = [ex["frame_paths"] for ex in subset]
        labels           = [1 if ex["label"] == "abnormal" else 0 for ex in subset]
        preds            = [1 if p == "abnormal" else 0 for p in preds_str]
        acc              = sum(l == p for l, p in zip(labels, preds)) / len(subset)
        return acc, frame_paths_list, labels, preds


# ── ProTeGi 옵티마이저 (원본 optimizers.py ProTeGi 대응) ─────────────────────
class ProTeGi:
    def __init__(self, config: dict, evaluator_fn, bf_eval):
        self.opt          = config
        self.evaluator_fn = evaluator_fn
        self.bf_eval      = bf_eval

    def _sample_error_str(self, frame_paths_list: list, labels: list,
                          preds: list, task: VideoFraudTask,
                          n: int = 4) -> tuple[str, list]:
        """오분류 예시 에러 문자열 + 해당 frame_paths 반환."""
        error_idxs  = [i for i, (l, p) in enumerate(zip(labels, preds)) if l != p]
        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        error_string      = ""
        error_frame_paths = []
        for k, i in enumerate(sample_idxs):
            error_string += f"## Example {k+1}\n"
            error_string += (f"Label: {task.stringify_prediction(labels[i])}\n"
                             f"Prediction: {task.stringify_prediction(preds[i])}\n\n")
            # 에러 예시당 첫 번째 프레임 1장 포함 (limit_mm_per_prompt=4 기준)
            error_frame_paths.append(frame_paths_list[i][0])
        return error_string.strip(), error_frame_paths

    def parse_tagged_text(self, text: str, start_tag: str, end_tag: str) -> list[str]:
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

    def _get_gradients(self, prompt: str, error_string: str,
                       error_frame_paths: list[str],
                       num_feedbacks: int = 5, n: int = 1) -> list[str]:
        """오분류 프레임 이미지 + 에러 문자열 → gradient 생성 (비전+텍스트 호출)."""
        gradient_prompt = (
            f'I\'m trying to write a zero-shot classifier prompt.\n\n'
            f'My current prompt is:\n"{prompt}"\n\n'
            f'But this prompt gets the following examples wrong:\n{error_string}\n'
            f'The frames from these misclassified videos are attached above.\n\n'
            f'give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.\n'
            f'Wrap each reason with <START> and <END>'
        )
        feedbacks = []
        for _ in range(n):
            out = predict_vision_text(gradient_prompt, error_frame_paths,
                                      temperature=0.7, max_tokens=600)
            feedbacks += self.parse_tagged_text(out, "<START>", "<END>")
        return feedbacks

    def apply_gradient(self, prompt: str, error_str: str, feedback_str: str,
                       steps_per_gradient: int, n: int = 1) -> list[str]:
        """gradient 적용 → 개선된 프롬프트 생성."""
        transformation_prompt = (
            f'I\'m trying to write a zero-shot classifier.\n\n'
            f'My current prompt is:\n"{prompt}"\n\n'
            f'But it gets the following examples wrong:\n{error_str}\n\n'
            f'Based on these examples the problem with this prompt is that {feedback_str}\n\n'
            f'Based on the above information, I wrote {steps_per_gradient} different improved prompts.\n'
            f'Each prompt is wrapped with <START> and <END>.\n\n'
            f'The {steps_per_gradient} new prompts are:'
        )
        res         = predict_text(transformation_prompt, n=n,
                                   temperature=0.7, max_tokens=800)
        new_prompts = []
        for r in res:
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
        return new_prompts

    def generate_synonyms(self, prompt_section: str, n: int = 3) -> list[str]:
        """의미를 유지하는 프롬프트 변형 생성."""
        rewriter_prompt = (
            f"Generate a variation of the following instruction while keeping the semantic meaning.\n\n"
            f"Input: {prompt_section}\n\nOutput:"
        )
        new_instructions = predict_text(rewriter_prompt, n=n,
                                        temperature=0.7, max_tokens=400)
        return [x for x in new_instructions if x]

    def get_gradients(self, prompt: str, task_section: str,
                      task: VideoFraudTask,
                      frame_paths_list: list, labels: list,
                      preds: list) -> list[tuple]:
        """n_gradients번 에러 샘플링 후 gradient 추출."""
        prompt_feedbacks = []
        for _ in range(self.opt["n_gradients"]):
            error_string, error_frame_paths = self._sample_error_str(
                frame_paths_list, labels, preds, task,
                n=self.opt["errors_per_gradient"]
            )
            gradients = self._get_gradients(
                task_section, error_string, error_frame_paths,
                self.opt["gradients_per_error"], n=1
            )
            prompt_feedbacks += [(g, error_string) for g in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts: list[str], task: VideoFraudTask,
                          train_exs: list[dict]) -> list[str]:
        """gradient 적용 + synonym 생성으로 후보 확장."""
        minibatch = random.sample(train_exs, min(self.opt["minibatch_size"], len(train_exs)))

        new_prompts = []
        for prompt in prompts:
            task_section = prompt.strip()

            _, frame_paths_list, labels, preds = task.evaluate(prompt, minibatch)

            new_task_sections = []
            if self.opt["n_gradients"] > 0:
                gradients = self.get_gradients(
                    prompt, task_section, task, frame_paths_list, labels, preds
                )
                for feedback, error_string in gradients:
                    tmp = self.apply_gradient(
                        task_section, error_string, feedback,
                        self.opt["steps_per_gradient"]
                    )
                    new_task_sections += tmp

            mc_sampled_task_sections = []
            if self.opt["mc_samples_per_step"] > 0:
                for sect in new_task_sections + [task_section]:
                    mc_sects = self.generate_synonyms(sect, n=self.opt["mc_samples_per_step"])
                    mc_sampled_task_sections += mc_sects

            new_sections    = list(set(new_task_sections + mc_sampled_task_sections))
            tmp_new_prompts = new_sections

            if len(tmp_new_prompts) > self.opt["max_expansion_factor"]:
                tmp_new_prompts = random.sample(tmp_new_prompts,
                                                k=self.opt["max_expansion_factor"])
            new_prompts += tmp_new_prompts

        new_prompts += prompts
        new_prompts  = list(set(new_prompts))
        return new_prompts

    def score_candidates(self, prompts: list[str], train_exs: list[dict]) -> list[float]:
        """각 후보 프롬프트를 평가해 점수 반환."""
        if len(prompts) == 1:
            return [1.0]
        return self.evaluator_fn(
            prompts, train_exs,
            rounds=self.opt["eval_rounds"],
            num_prompts_per_round=self.opt["eval_prompts_per_round"],
            samples_per_eval=self.opt["samples_per_eval"],
        )


# ── 최종 상세 평가 ────────────────────────────────────────────────────────────
def eval_detailed(prompt: str, testset: list[dict], label: str = "") -> None:
    """F1 / AUC / MAP 상세 평가."""
    preds_str = classify_dataset(testset, prompt)
    y_score   = score_dataset(testset, prompt)
    y_true    = [1 if ex["label"] == "abnormal" else 0 for ex in testset]
    y_pred    = [1 if p == "abnormal" else 0 for p in preds_str]
    print_metrics(compute_metrics(y_true, y_pred, y_score), label=label)


# ── 메인 (원본 main.py 구조 유지) ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir",             default="./frames_cache_2s")
    parser.add_argument("--output",                 default=str(OUT_DIR / "apo_result.json"))

    # 원본 main.py 하이퍼파라미터
    parser.add_argument("--rounds",                 type=int,   default=12)
    parser.add_argument("--beam_size",              type=int,   default=4)
    parser.add_argument("--n_test_exs",             type=int,   default=None)
    parser.add_argument("--minibatch_size",         type=int,   default=16)
    parser.add_argument("--n_gradients",            type=int,   default=1)
    parser.add_argument("--errors_per_gradient",    type=int,   default=4)
    parser.add_argument("--gradients_per_error",    type=int,   default=1)
    parser.add_argument("--steps_per_gradient",     type=int,   default=1)
    parser.add_argument("--mc_samples_per_step",    type=int,   default=2)
    parser.add_argument("--max_expansion_factor",   type=int,   default=8)
    parser.add_argument("--evaluator",              default="ucb", choices=["bf", "ucb"])
    parser.add_argument("--eval_rounds",            type=int,   default=2)
    parser.add_argument("--eval_prompts_per_round", type=int,   default=8)
    parser.add_argument("--samples_per_eval",       type=int,   default=4)
    parser.add_argument("--c",                      type=float, default=1.0)
    parser.add_argument("--patience",               type=int,   default=2,
                        help="조기종료: N라운드 연속 dev_map 미개선 시 중단 (TEMPO와 동일)")
    parser.add_argument("--patience_eps",           type=float, default=0.01,
                        help="조기종료 내성: 이 미만 하락은 개선으로 간주 (TEMPO와 동일)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    config = {
        "rounds":                args.rounds,
        "beam_size":             args.beam_size,
        "minibatch_size":        args.minibatch_size,
        "n_gradients":           args.n_gradients,
        "errors_per_gradient":   args.errors_per_gradient,
        "gradients_per_error":   args.gradients_per_error,
        "steps_per_gradient":    args.steps_per_gradient,
        "mc_samples_per_step":   args.mc_samples_per_step,
        "max_expansion_factor":  args.max_expansion_factor,
        "evaluator":             args.evaluator,
        "eval_rounds":           args.eval_rounds,
        "eval_prompts_per_round":args.eval_prompts_per_round,
        "samples_per_eval":      args.samples_per_eval,
        "eval_budget":           args.samples_per_eval * args.eval_rounds * args.eval_prompts_per_round,
        "c":                     args.c,
    }

    # ── 데이터셋 (다른 모든 방법과 동일한 split — 공정한 baseline 비교) ────────
    print("\n=== 데이터셋 로드 & 2초 간격 프레임 추출 (공유 split 사용) ===")
    trainset, valset, testset = load_dataset_split(Path(args.frames_dir))
    print(f"분할 → train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    # ── 컴포넌트 초기화 ───────────────────────────────────────────────────────
    task = VideoFraudTask()

    if args.evaluator == "ucb":
        evaluator_fn = UCBBanditEvaluator(config)
    else:
        evaluator_fn = BruteForceEvaluator(config)

    bf_eval   = BruteForceEvaluator(config)
    optimizer = ProTeGi(config, evaluator_fn, bf_eval)

    logger = _RoundLogger("apo")

    partial_path = OUT_DIR / "apo_partial_log.json"
    if partial_path.exists():
        with open(partial_path, encoding="utf-8") as _f:
            saved = json.load(_f)
        candidates         = saved.get("candidates",   [saved["best_prompt"]])
        scores             = saved.get("scores",       [saved["best_dev_map"]])
        round_logs         = saved.get("log",          [])
        best_dev_map       = saved.get("best_dev_map", 0.0)
        best_prompt_so_far = saved.get("best_prompt",  INITIAL_PROMPT)
        start_rnd          = saved.get("last_round",   0) + 1
        no_improve         = saved.get("no_improve",   0)
        best_ever          = saved.get("best_ever",    best_dev_map)
        print(f"  [Resume] Round {start_rnd}부터 재시작  best_map={best_dev_map:.3f}  no_improve={no_improve}")
    else:
        candidates         = [INITIAL_PROMPT]
        scores             = [0.0]
        round_logs         = []
        best_dev_map       = 0.0
        best_prompt_so_far = INITIAL_PROMPT
        start_rnd          = 0
        no_improve         = 0
        best_ever          = 0.0

    patience     = args.patience
    patience_eps = args.patience_eps

    # ── 최적화 루프 ───────────────────────────────────────────────────────────
    for rnd in range(start_rnd, config["rounds"] + 1):
        print(f"\n=== Round {rnd}/{config['rounds']} ===")

        if rnd > 0:
            candidates = optimizer.expand_candidates(candidates, task, trainset)
            print(f"  확장된 후보: {len(candidates)}개")

        scores = optimizer.score_candidates(candidates, trainset)
        paired     = sorted(zip(scores, candidates), reverse=True)
        scores, candidates = map(list, zip(*paired))
        candidates = candidates[:config["beam_size"]]
        scores     = scores[:config["beam_size"]]
        print(f"  Best train_score: {scores[0]:.3f}  |  {candidates[0][:80]}...")

        # dev MAP 평가
        dev_metrics_list = scorer_map(candidates[:1], valset)
        dev_map = dev_metrics_list[0]
        y_score = score_dataset(valset, candidates[0])
        y_true  = [1 if ex["label"] == "abnormal" else 0 for ex in valset]
        y_pred  = [1 if s >= 0.5 else 0 for s in y_score]
        dev_m   = compute_metrics(y_true, y_pred, y_score)
        print(f"  dev_map={dev_m['map'] or 0:.3f}  auc={dev_m['auc'] or 0:.3f}  f1={dev_m['f1'] or 0:.3f}")

        cur_map = dev_m["map"] or 0.0
        if cur_map > best_dev_map:
            best_dev_map       = cur_map
            best_prompt_so_far = candidates[0]
        if cur_map > best_ever:
            best_ever  = cur_map
            no_improve = 0
        elif cur_map >= best_ever - patience_eps:
            no_improve = 0
        else:
            no_improve += 1

        round_record = {
            "round":      rnd,
            "dev_map":    dev_m["map"],
            "dev_auc":    dev_m["auc"],
            "dev_f1":     dev_m["f1"],
            "dev_acc":    dev_m["acc"],
            "prompt":     candidates[0],
        }
        round_logs.append(round_record)
        logger.write({"type": "iteration", **round_record})

        # 라운드마다 중간 저장 (resume용 전체 상태 포함)
        partial = {
            "method":       "APO",
            "best_prompt":  best_prompt_so_far,
            "best_dev_map": best_dev_map,
            "log":          round_logs,
            "candidates":   candidates,
            "scores":       scores,
            "last_round":   rnd,
            "no_improve":   no_improve,
            "best_ever":    best_ever,
        }
        with open(partial_path, "w", encoding="utf-8") as _f:
            json.dump(partial, _f, ensure_ascii=False, indent=2)
        print(f"  [저장] {partial_path}  no_improve={no_improve}/{patience}")

        if no_improve >= patience:
            print(f"  [조기종료] {patience}라운드 연속 미개선 → 종료 (round {rnd}/{config['rounds']})")
            break

    # 완료 후 partial log 삭제
    if partial_path.exists():
        partial_path.unlink()
    logger.close()

    # ── 최종 결과 ─────────────────────────────────────────────────────────────
    print(f"\n=== 최적 프롬프트 ===\n{best_prompt_so_far}\n")

    def get_test_metrics(prompt, threshold=0.5):
        y_score = score_dataset(testset, prompt)
        y_true  = [1 if ex["label"] == "abnormal" else 0 for ex in testset]
        y_pred  = [1 if s >= threshold else 0 for s in y_score]
        return compute_metrics(y_true, y_pred, y_score)

    print("=== 최종 테스트 평가 ===")
    baseline_m = get_test_metrics(INITIAL_PROMPT)
    best_m     = get_test_metrics(best_prompt_so_far)

    # threshold 재보정: val_map으로 프롬프트를 고른 것과 별개로, 고정 0.5 threshold가
    # 최적화된 점수 분포와 안 맞아 F1/ACC가 왜곡될 수 있음 (TEMPO 계열과 동일 기준으로 맞춤).
    val_true      = [1 if ex["label"] == "abnormal" else 0 for ex in valset]
    val_score     = score_dataset(valset, best_prompt_so_far)
    best_thr      = find_best_threshold(val_true, val_score)
    best_m_opt    = get_test_metrics(best_prompt_so_far, threshold=best_thr)
    best_m["f1_opt"]         = best_m_opt["f1"]
    best_m["acc_opt"]        = best_m_opt["acc"]
    best_m["best_threshold"] = best_thr

    print_metrics(baseline_m, label="Baseline")
    print_metrics(best_m,     label="APO")

    output = {
        "method":            "APO",
        "best_prompt":       best_prompt_so_far,
        "initial_prompt":    INITIAL_PROMPT,
        "baseline_test":     baseline_m,
        "best_test":         best_m,
        "best_test_f1_opt":  best_m["f1_opt"],
        "best_test_acc_opt": best_m["acc_opt"],
        "config":            config,
        "log":               round_logs,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
