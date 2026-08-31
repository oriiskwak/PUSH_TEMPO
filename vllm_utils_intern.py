"""
vllm_utils_intern.py — vllm_utils.py(Qwen2.5-VL-72B)와 동일한 인터페이스를 갖는
InternVL2.5-78B-AWQ 백본 버전. 실험 스크립트에서
    import vllm_utils as backend        # Qwen
    import vllm_utils_intern as backend # InternVL
처럼 import만 바꿔서 백본을 스위칭할 수 있도록 함수 시그니처를 그대로 맞췄다.

InternVL은 Qwen2.5-VL과 달리 AutoProcessor 멀티모달 파이프라인이 없고,
프롬프트 문자열에 리터럴 "<image>" 플레이스홀더를 넣는 방식(vLLM 내장
InternVLChatModel 처리기가 dynamic-tiling/IMG_CONTEXT 토큰 확장을 담당)이라
_apply_chat_template()와 프롬프트 조립 부분만 Qwen 버전과 다르다.

환경변수:
    TP_SIZE             : tensor_parallel_size (기본 2)
    INTERN_MAX_DYNAMIC_PATCH : 이미지 1장당 최대 dynamic tile 수 (기본 6, 모델 기본값은 12)
                                — 4장 조합(predict_vision_text)에서 컨텍스트가 터지는 것을 막기 위해 절반으로 낮춤
"""
import cv2
import os
os.environ.setdefault("VLLM_USE_V1", "0")
import re
import numpy as np
from pathlib import Path
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ── vLLM InternVL 비디오 프로세서 호환성 패치 ────────────────────────────────
# InternVL2.5의 LLM 백본이 Qwen2.5라서, 비디오를 전혀 안 쓰는데도 vLLM이 프로파일링
# 단계에서 InternVLVideoProcessor를 자동 생성한다. 이때 mm_processor_kwargs로 넘긴
# max_dynamic_patch가 (이미지 프로세서용인데) 비디오 프로세서 생성자에도 그대로 전달돼
# "unexpected keyword argument 'max_dynamic_patch'"로 죽는다. 비디오는 실제로 안 쓰므로
# 해당 인자만 무시하도록 생성자를 감싼다.
from vllm.model_executor.models.internvl import InternVLVideoProcessor as _IVVideoProcessor
_orig_video_processor_init = _IVVideoProcessor.__init__
def _patched_video_processor_init(self, *args, **kwargs):
    kwargs.pop("max_dynamic_patch", None)
    kwargs.pop("dynamic_image_size", None)
    kwargs.pop("use_thumbnail", None)
    _orig_video_processor_init(self, *args, **kwargs)
_IVVideoProcessor.__init__ = _patched_video_processor_init

# ── 모델 설정 ─────────────────────────────────────────────────────────────────
MODEL_NAME = str(Path(__file__).parent / "hf_models" / "InternVL2_5-78B-AWQ")

# TEMPO와 동일한 상수 (vllm_utils.py와 동일하게 유지)
MAX_SEC               = 100
FRAME_SCORE_THRESHOLD = 0.5
SCAM_RATIO_THRESHOLD  = 0.3
VLLM_BATCH            = 128

# ── vLLM 초기화 (모듈 임포트 시 1회) ─────────────────────────────────────────
_tp_size            = int(os.environ.get("TP_SIZE",        "1"))   # AWQ group_size 제약으로 TP>1 불가
_pp_size            = int(os.environ.get("PP_SIZE",        "2"))   # PP=2 로컬 2xA6000 기본값
_max_model_len      = int(os.environ.get("MAX_MODEL_LEN", "8192"))
_max_num_seqs       = int(os.environ.get("MAX_NUM_SEQS",  "2"))
_max_dynamic_patch  = int(os.environ.get("INTERN_MAX_DYNAMIC_PATCH", "6"))
_gpu_mem_util       = float(os.environ.get("GPU_MEM_UTIL", "0.95"))
print(f"[vllm_utils_intern] InternVL2.5-78B 로드 중 (tp={_tp_size}, pp={_pp_size}, "
      f"max_model_len={_max_model_len}, max_num_seqs={_max_num_seqs}, max_dynamic_patch={_max_dynamic_patch})...")

_llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=_tp_size,
    pipeline_parallel_size=_pp_size,
    dtype="float16",
    # "awq_marlin"(AWQMarlinConfig)이 아니라 "awq"(AWQConfig)를 써야 함: 이 체크포인트는
    # LLM 부분만 양자화돼 있고 비전 타워(InternViT)는 fp16인데, vLLM의 InternVL
    # _patch_quant_config()가 비전 타워를 modules_to_not_convert에 넣어주는 예외처리를
    # isinstance(quant_config, AWQConfig)로만 체크해서 AWQMarlinConfig는 빠뜨림
    # (vLLM 0.19.1 InternVL 구현 버그). "awq"도 내부적으로 marlin 커널을 자동 사용하므로
    # 속도 손해는 없음.
    quantization="awq",
    gpu_memory_utilization=_gpu_mem_util,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 4},   # APO gradient 호출: 최대 4장 (Qwen과 동일)
    max_model_len=_max_model_len,
    max_num_seqs=_max_num_seqs,
    mm_processor_kwargs={"max_dynamic_patch": _max_dynamic_patch},
)
_tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
_v_sampling = SamplingParams(temperature=0.0, max_tokens=10)
print("[vllm_utils_intern] 모델 로드 완료.")

SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."


def _apply_chat_template(text: str, n_images: int) -> str:
    """InternVL 프롬프트: 이미지 개수만큼 <image> 플레이스홀더를 앞에 붙인다."""
    image_tags = "\n".join(["<image>"] * n_images)
    content    = f"{image_tags}\n{text}" if n_images else text
    messages   = [{"role": "user", "content": content}]
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── 프레임 추출 (vllm_utils.py와 동일) ───────────────────────────────────────
def extract_frames_1fps(video_path: Path, frames_dir: Path,
                        max_sec: int = MAX_SEC, step: int = 2) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    stem  = video_path.stem
    cap   = cv2.VideoCapture(str(video_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sec = min(int(total / fps_v), max_sec)
    cap.release()

    seconds = list(range(0, n_sec, step))

    paths = [frames_dir / f"{stem}_s{sec:04d}.jpg" for sec in seconds]
    if all(p.exists() for p in paths):
        return [str(p) for p in paths]

    cap   = cv2.VideoCapture(str(video_path))
    saved = []
    for sec in seconds:
        p = frames_dir / f"{stem}_s{sec:04d}.jpg"
        if p.exists():
            saved.append(str(p))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_v))
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((448, 448), Image.LANCZOS)
            img.save(str(p), "JPEG", quality=85)
            saved.append(str(p))
    cap.release()
    return saved


def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ── 비전 배치 추론 ────────────────────────────────────────────────────────────
def predict_vision_batch(items: list[tuple[Image.Image, str]]) -> list[float]:
    """(PIL Image, prompt) 리스트 → float 예측값 리스트."""
    results = []
    for start in range(0, len(items), VLLM_BATCH):
        chunk       = items[start:start + VLLM_BATCH]
        vllm_inputs = []
        for img, prompt in chunk:
            text = _apply_chat_template(prompt + SCORE_SUFFIX, n_images=1)
            vllm_inputs.append({
                "prompt":           text,
                "multi_modal_data": {"image": img},
            })
        outputs = _llm.generate(vllm_inputs, _v_sampling, use_tqdm=False)
        for out in outputs:
            raw = out.outputs[0].text.strip()
            m   = re.search(r"[0-9]*\.?[0-9]+", raw)
            results.append(max(0.0, min(1.0, float(m.group()))) if m else 0.5)
    return results


# ── 텍스트 전용 추론 — GA/DE crossover, APO gradient 생성용 ──────────────────
def predict_text(prompt: str, n: int = 1,
                 temperature: float = 0.7, max_tokens: int = 512) -> list[str]:
    text    = _apply_chat_template(prompt, n_images=0)
    sp      = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
    outputs = _llm.generate([{"prompt": text}], sp, use_tqdm=False)
    return [c.text.strip() for c in outputs[0].outputs]


# ── 비전+텍스트 혼합 추론 — APO gradient 생성 시 이미지 포함 ─────────────────
def predict_vision_text(text_prompt: str, image_paths: list[str],
                        temperature: float = 0.7,
                        max_tokens: int = 600) -> str:
    image_paths = image_paths[:4]
    imgs = [load_pil(fp) for fp in image_paths]
    text = _apply_chat_template(text_prompt, n_images=len(imgs))

    sp      = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = _llm.generate(
        [{"prompt": text, "multi_modal_data": {"image": imgs}}],
        sp, use_tqdm=False
    )
    return outputs[0].outputs[0].text.strip()


# ── 영상 레벨 분류 ────────────────────────────────────────────────────────────
def classify_video(frame_paths: list[str], prompt: str) -> str:
    if not frame_paths:
        return "normal"
    items      = [(load_pil(fp), prompt) for fp in frame_paths]
    preds      = predict_vision_batch(items)
    scam_ratio = sum(p >= FRAME_SCORE_THRESHOLD for p in preds) / len(preds)
    return "abnormal" if scam_ratio >= SCAM_RATIO_THRESHOLD else "normal"


def get_video_score(frame_paths: list[str], prompt: str) -> float:
    if not frame_paths:
        return 0.5
    items = [(load_pil(fp), prompt) for fp in frame_paths]
    preds = predict_vision_batch(items)
    return float(np.mean(preds))


# ── 데이터셋 전체 배치 추론 ───────────────────────────────────────────────────
def classify_dataset(dataset: list[dict], prompt: str) -> list[str]:
    if not dataset:
        return []

    all_items          = []
    video_frame_counts = []
    for ex in dataset:
        all_items.extend([(load_pil(fp), prompt) for fp in ex["frame_paths"]])
        video_frame_counts.append(len(ex["frame_paths"]))

    all_preds = predict_vision_batch(all_items)

    results, idx = [], 0
    for count in video_frame_counts:
        frame_preds = all_preds[idx:idx + count]
        scam_ratio  = sum(p >= FRAME_SCORE_THRESHOLD for p in frame_preds) / count if count else 0
        results.append("abnormal" if scam_ratio >= SCAM_RATIO_THRESHOLD else "normal")
        idx += count
    return results


def score_dataset(dataset: list[dict], prompt: str) -> list[float]:
    if not dataset:
        return []

    all_items          = []
    video_frame_counts = []
    for ex in dataset:
        all_items.extend([(load_pil(fp), prompt) for fp in ex["frame_paths"]])
        video_frame_counts.append(len(ex["frame_paths"]))

    all_preds = predict_vision_batch(all_items)

    results, idx = [], 0
    for count in video_frame_counts:
        frame_preds = all_preds[idx:idx + count]
        results.append(float(np.mean(frame_preds)) if frame_preds else 0.5)
        idx += count
    return results
