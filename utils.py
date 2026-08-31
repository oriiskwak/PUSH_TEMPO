"""
utils.py — uniAPO_qwen shared utilities
Predictor : Qwen/Qwen2.5-VL-72B-Instruct-AWQ via vLLM (2× A6000, tensor_parallel_size=2)
Optimizer : GPT-4o (em_core.py 에서 사용, 변경 없음)
"""
import json, re, cv2, random, os, math
os.environ.setdefault("VLLM_USE_V1", "0")
import numpy as np
from pathlib import Path
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

MODEL_NAME = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
GT_ALL    = DATA_DIR / "gt_results" / "gt_all.json"
VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", str(BASE_DIR / "data" / "videos")))
OUT_DIR   = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── vLLM 모델 초기화 (모듈 로드 시 1회) ──────────────────────────────────────
_tp_size  = int(os.environ.get("TP_SIZE",   "2"))
_mm_limit = int(os.environ.get("MM_LIMIT",  "1"))   # 프롬프트당 최대 이미지 수
print(f"Qwen2.5-VL-72B 로드 중 (tensor_parallel_size={_tp_size}, mm_limit={_mm_limit})...")
_llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=_tp_size,
    dtype="float16",
    quantization="awq_marlin",
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": _mm_limit},
    max_model_len=8192 if _tp_size == 1 else 24576,
    max_num_seqs=2    if _tp_size == 1 else 4,
)
_processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
_sampling  = SamplingParams(temperature=0.0, max_tokens=10)
print("모델 로드 완료.")

VLLM_BATCH = 32    # vLLM generate 당 최대 프레임 수 (GPU OOM 방지)


# ── 데이터 유틸 ───────────────────────────────────────────────────────────────

def load_gt(max_sec: int = 150):
    data = json.load(open(GT_ALL, encoding="utf-8"))
    filtered = []
    for v in data:
        v["frame_probabilities"] = [f for f in v["frame_probabilities"] if f["second"] <= max_sec]
        if v["frame_probabilities"]:
            filtered.append(v)
    return filtered


def split_data(gt_list, train_ratio=0.6, dev_ratio=0.2, seed=42):
    """abnormal/normal 균형 유지 stratified split → train / dev / test."""
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


def load_frame(video_path: Path, second: int) -> Image.Image | None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


def bernoulli_kl(p: float, q: float, eps: float = 1e-7) -> float:
    p = max(eps, min(1 - eps, p))
    q = max(eps, min(1 - eps, q))
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


# compute_metrics/find_best_threshold는 metrics_core.py(single source of truth)에서 가져옴 —
# 예전엔 이 파일에 직접 정의돼 있었는데, utils_intern.py/baselines 쪽 사본들과 조금씩 동작이
# 갈라져 있던 게 발견돼서 metrics_core.py 하나로 통합함.
from metrics_core import (
    compute_metrics_from_frames as compute_metrics,
    find_best_threshold_from_frames as find_best_threshold,
    RATIO_THRESHOLD,
)


# ── Qwen2.5-VL 배치 예측 ──────────────────────────────────────────────────────

def predict_qwen_batch(items: list[tuple[Image.Image, str]]) -> list[float | None]:
    """vLLM 배치 추론: (PIL Image, prompt) 리스트 → 예측값 리스트.

    VLLM_BATCH 단위로 나눠 처리해 CPU 메모리를 안전하게 유지.
    vLLM continuous batching이 4× A6000 GPU throughput을 자동 최적화.
    """
    results: list[float | None] = []

    for start in range(0, len(items), VLLM_BATCH):
        chunk = items[start:start + VLLM_BATCH]
        vllm_inputs = []

        for img, prompt in chunk:
            img = img.resize((448, 448), Image.LANCZOS)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  prompt},
            ]}]
            text = _processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vllm_inputs.append({
                "prompt":           text,
                "multi_modal_data": {"image": img},
            })

        outputs = _llm.generate(vllm_inputs, _sampling, use_tqdm=False)
        for out in outputs:
            raw = out.outputs[0].text.strip()
            m   = re.search(r"[0-9]*\.?[0-9]+", raw)
            results.append(max(0.0, min(1.0, float(m.group()))) if m else None)

    return results


def predict_qwen_batch_with_entropy(
    items: list[tuple[Image.Image, str]]
) -> list[tuple[float | None, float | None]]:
    """Pass 1용: score + top-token entropy 반환.

    entropy가 높을수록 모델이 불확실 → TEMPO inference 분기 기준으로 사용.
    Returns: [(score, entropy), ...]
    """
    _sampling_lp = SamplingParams(temperature=0.0, max_tokens=10, logprobs=20)
    results: list[tuple[float | None, float | None]] = []

    for start in range(0, len(items), VLLM_BATCH):
        chunk = items[start:start + VLLM_BATCH]
        vllm_inputs = []

        for img, prompt in chunk:
            img = img.resize((448, 448), Image.LANCZOS)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  prompt},
            ]}]
            text = _processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vllm_inputs.append({
                "prompt":           text,
                "multi_modal_data": {"image": img},
            })

        outputs = _llm.generate(vllm_inputs, _sampling_lp, use_tqdm=False)
        for out in outputs:
            raw = out.outputs[0].text.strip()
            m   = re.search(r"[0-9]*\.?[0-9]+", raw)
            score = max(0.0, min(1.0, float(m.group()))) if m else None

            # 첫 번째 토큰의 logprob 분포로 entropy 계산
            entropy = None
            if out.outputs[0].logprobs:
                lp_dict = out.outputs[0].logprobs[0]   # {token_id: logprob}
                lps     = [lp.logprob for lp in lp_dict.values()]
                if lps:
                    probs   = [math.exp(lp) for lp in lps]
                    total   = sum(probs)
                    probs   = [p / total for p in probs]
                    entropy = -sum(p * math.log(p + 1e-9) for p in probs)

            results.append((score, entropy))

    return results
