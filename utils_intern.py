"""
utils_intern.py — utils.py와 동일한 인터페이스, InternVL2.5-78B-AWQ 백본.

tempo_cov.py 상단의
    from utils import VIDEO_DIR, OUT_DIR, _llm, _processor, ...
를
    from utils_intern import VIDEO_DIR, OUT_DIR, _llm, _processor, ...
으로만 바꾸면 InternVL로 스위칭됨.

핵심 차이: InternVL은 AutoProcessor 대신 AutoTokenizer + <image> 플레이스홀더.
_processor를 래퍼 클래스로 동일 인터페이스 제공.
"""
import json, re, cv2, random, os, math
os.environ["VLLM_USE_V1"] = "0"
import numpy as np
from pathlib import Path
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ── utils.py와 동일한 경로 설정 ───────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
GT_ALL    = DATA_DIR / "gt_results" / "gt_all.json"
VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", str(BASE_DIR / "data" / "videos")))
OUT_DIR   = BASE_DIR / "results_intern"
OUT_DIR.mkdir(exist_ok=True)

# ── InternVL 모델 설정 ────────────────────────────────────────────────────────
MODEL_NAME         = str(BASE_DIR / "hf_models" / "InternVL2_5-78B-AWQ")
_tp_size           = int(os.environ.get("TP_SIZE",               "1"))
_pp_size           = int(os.environ.get("PP_SIZE",               "2"))   # 로컬 v0.20.1에서 PP=2 정상동작, 서버 v0.11.2는 PP=4 에러 있음
_max_model_len     = int(os.environ.get("MAX_MODEL_LEN",        "8192"))
_max_num_seqs      = int(os.environ.get("MAX_NUM_SEQS",         "2"))
_max_dynamic_patch = int(os.environ.get("INTERN_MAX_DYNAMIC_PATCH", "6"))
_gpu_mem_util      = float(os.environ.get("GPU_MEM_UTIL",       "0.95"))  # ViT fp16 복제로 VRAM 빡빡

print(f"[utils_intern] InternVL2.5-78B 로드 중 (tp={_tp_size}, pp={_pp_size}, max_len={_max_model_len}, util={_gpu_mem_util})...")

_llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=_tp_size,
    pipeline_parallel_size=_pp_size,
    dtype="float16",
    # "awq_marlin"(AWQMarlinConfig)이 아니라 "awq"(AWQConfig)를 써야 함: 이 체크포인트는
    # LLM 부분만 양자화돼 있고 비전 타워(InternViT)는 fp16인데, vLLM의 InternVL
    # _patch_quant_config()가 비전 타워를 modules_to_not_convert에 넣어주는 예외처리를
    # isinstance(quant_config, AWQConfig)로만 체크해서 AWQMarlinConfig는 빠뜨림
    # (vLLM InternVL 구현 버그). "awq"도 내부적으로 marlin 커널을 자동 사용하므로 속도
    # 손해는 없음. (TP>1에서 겪었다는 group_size 정렬 오류는 TP=1에서는 해당 없음)
    quantization="awq",
    gpu_memory_utilization=_gpu_mem_util,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1},
    max_model_len=_max_model_len,
    max_num_seqs=_max_num_seqs,
)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, fix_mistral_regex=True)
_sampling  = SamplingParams(temperature=0.0, max_tokens=10)
print("[utils_intern] 모델 로드 완료.")


class _InternVLProcessor:
    """
    tempo_cov.py가 쓰는 _processor.apply_chat_template() 인터페이스를 InternVL용으로 래핑.
    messages = [{"role": "user", "content": [{"type":"image",...}, {"type":"text",...}]}]
    → InternVL의 <image>\n{text} 포맷으로 변환.
    """
    def __init__(self):
        self.tokenizer = _tokenizer

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        n_images, text = 0, ""
        for msg in messages:
            if msg["role"] != "user":
                continue
            content = msg["content"]
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        n_images += 1
                    elif item.get("type") == "text":
                        text = item["text"]
            else:
                text = content
        image_tags   = "\n".join(["<image>"] * n_images)
        full_content = f"{image_tags}\n{text}" if n_images else text
        return _tokenizer.apply_chat_template(
            [{"role": "user", "content": full_content}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


_processor = _InternVLProcessor()

# ── utils.py와 동일한 유틸 함수들 (복사) ──────────────────────────────────────
def bernoulli_kl(p: float, q: float, eps: float = 1e-7) -> float:
    p = max(eps, min(1 - eps, p))
    q = max(eps, min(1 - eps, q))
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


# compute_metrics/find_best_threshold는 metrics_core.py(single source of truth)에서 가져옴 —
# 이 파일엔 예전에 kl/f1_macro/*_ratio 필드가 빠진 축약판이 따로 정의돼 있었고,
# find_best_threshold도 grid step이 0.05(19개 후보)로 utils.py의 0.01(90개 후보)보다
# 성겼던 게 리뷰에서 발견돼서 metrics_core.py로 통합함 (thresholds 커스텀 인자는 아무 호출부도
# 안 쓰길래 통합 시그니처에서 제거).
from metrics_core import (
    compute_metrics_from_frames as compute_metrics,
    find_best_threshold_from_frames as find_best_threshold,
)


def load_frame(video_path: Path, second: int) -> Image.Image | None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


def load_gt() -> list[dict]:
    with open(GT_ALL, encoding="utf-8") as f:
        return json.load(f)


def split_data(data: list, train_ratio: float, dev_ratio: float,
               seed: int = 42) -> tuple:
    rng = random.Random(seed)
    d   = data[:]
    rng.shuffle(d)
    n_train = int(len(d) * train_ratio)
    n_dev   = int(len(d) * dev_ratio)
    return d[:n_train], d[n_train:n_train + n_dev], d[n_train + n_dev:]
