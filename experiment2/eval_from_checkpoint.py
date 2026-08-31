"""
eval_from_checkpoint.py — 중단/에러된 run의 체크포인트에서 테스트 결과 복원

사용법:
  # 체크포인트 자동 탐색 (frozen 우선, 없으면 evolve)
  python experiment2/eval_from_checkpoint.py

  # 특정 체크포인트 직접 지정
  python experiment2/eval_from_checkpoint.py --checkpoint results/tempo_cov_frozen_checkpoint.json
"""

import json, sys, argparse, os, re, cv2, random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from vllm import SamplingParams

from utils import (
    VIDEO_DIR, OUT_DIR,
    _llm, _processor,
    compute_metrics, find_best_threshold,
    split_data, load_gt,
)
from em_core_tool2 import FeedbackMemory, PromptMemory

# ── 설정 (tempo_cov.py와 동일) ───────────────────────────────────────────────
TRAIN_RATIO  = 0.6
DEV_RATIO    = 0.2
SEED         = 42
FRAME_STEP   = 2
MAX_SEC      = 150
VLLM_BATCH   = 32
BEAM_SIZE    = 3
SCORE_SUFFIX = "\nReturn ONLY a single float 0.0–1.0."

_SEG_PATH = Path(__file__).parent.parent / "data" / "segmentation.json"
_SEG_MAP: dict[str, dict[int, str]] = {}
if _SEG_PATH.exists():
    for _e in json.load(open(_SEG_PATH, encoding="utf-8")):
        _SEG_MAP[_e["video_name"]] = {fs["second"]: fs["segment"] for fs in _e["frame_segments"]}


# ── 프레임 추출 / 추론 (tempo_cov.py에서 복사) ───────────────────────────────
def _extract_frames(video_path: Path) -> list[tuple[int, Image.Image]]:
    cap   = cv2.VideoCapture(str(video_path))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sec = min(int(total / fps_v), MAX_SEC)
    out   = []
    for sec in range(0, n_sec, FRAME_STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_v))
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out.append((sec, img.resize((448, 448), Image.LANCZOS)))
    cap.release()
    return out


def _predict_batch(items: list[tuple[Image.Image, str]]) -> list[float | None]:
    results: list[float | None] = []
    sp = SamplingParams(temperature=0.0, max_tokens=10)
    for start in range(0, len(items), VLLM_BATCH):
        chunk  = items[start:start + VLLM_BATCH]
        inputs = []
        for img, prompt in chunk:
            img_np = np.array(img)
            content = [
                {"type": "image", "image": img_np},
                {"type": "text",  "text":  prompt + SCORE_SUFFIX},
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
                m   = re.search(r"[0-9]*\.?[0-9]+", raw)
                results.append(max(0.0, min(1.0, float(m.group()))) if m else None)
        except Exception as e:
            print(f"  [Warning] {e}")
            results.extend([None] * len(chunk))
    return results


def score_dataset(dataset, early_prompt, late_prompt, early_fb_mem=None, late_fb_mem=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    frame_map = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_extract_frames, VIDEO_DIR / f"{r['video']}.mp4"): r for r in dataset}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                frames = fut.result()
                frame_map[r["video"]] = (r, frames)
            except Exception as e:
                print(f"  [Warning] {r['video']}: {e}")

    items, meta = [], []
    for vname, (r, frames) in frame_map.items():
        seg_map = _SEG_MAP.get(vname, {})
        for sec, img in frames:
            seg = seg_map.get(sec, "early")
            prompt = early_prompt if seg == "early" else late_prompt
            items.append((img, prompt))
            meta.append((r, sec, seg))

    preds = _predict_batch(items)

    from collections import defaultdict
    video_frames: dict[str, list] = defaultdict(list)
    for (r, sec, seg), pred in zip(meta, preds):
        if pred is not None:
            video_frames[r["video"]].append({"second": sec, "pred": pred, "label": r["label"]})

    for vname, frames in video_frames.items():
        r = frame_map[vname][0]
        for fr in frames:
            results.append({
                "video":  vname,
                "second": fr["second"],
                "pred":   fr["pred"],
                "label":  r["label"],
                "gt":     r["label"],
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="체크포인트 경로 (기본: results/tempo_cov_frozen_checkpoint.json 또는 evolve)")
    parser.add_argument("--early-prompt", type=str, default=None,
                        help="Early 프롬프트 직접 입력 (체크포인트 없을 때)")
    parser.add_argument("--late-prompt", type=str, default=None,
                        help="Late 프롬프트 직접 입력 (체크포인트 없을 때)")
    parser.add_argument("--output", type=str, default="tempo_cov_recovered_result.json")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    print("=== 데이터 로드 ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Dev={len(devset)}  Test={len(testset)}")

    # ── 프롬프트 / pm 복원 ────────────────────────────────────────────────────
    early_pm = PromptMemory(beam_size=BEAM_SIZE, top_k=2)
    late_pm  = PromptMemory(beam_size=BEAM_SIZE, top_k=2)
    early_fb_mem = FeedbackMemory()
    late_fb_mem  = FeedbackMemory()

    best_early = args.early_prompt
    best_late  = args.late_prompt

    # 체크포인트 탐색
    if args.checkpoint:
        ck_paths = [Path(args.checkpoint)]
    else:
        ck_paths = [
            OUT_DIR / "tempo_cov_frozen_checkpoint.json",
            OUT_DIR / "tempo_cov_evolve_checkpoint.json",
            OUT_DIR / "tempo_cov_total2b_checkpoint.json",
        ]

    ck_loaded = False
    for ck_path in ck_paths:
        if ck_path.exists():
            print(f"체크포인트 로드: {ck_path}")
            ck = json.load(open(ck_path, encoding="utf-8"))

            # 새 포맷 (experiment2) vs 구 포맷 (experiment)
            if "best_pair_early" in ck:
                best_early = best_early or ck.get("best_pair_early") or ck.get("early_prompt")
                best_late  = best_late  or ck.get("best_pair_late")  or ck.get("late_prompt")
            else:
                best_early = best_early or ck.get("early_prompt")
                best_late  = best_late  or ck.get("late_prompt")

            # pm 복원
            if "early_pm" in ck:
                early_pm.memory = [[p, s] for p, s in ck["early_pm"]]
            elif "early_pm_data" in ck:
                early_pm.memory = [[p, s] for p, s in ck["early_pm_data"]]
            if "late_pm" in ck:
                late_pm.memory = [[p, s] for p, s in ck["late_pm"]]
            elif "late_pm_data" in ck:
                late_pm.memory = [[p, s] for p, s in ck["late_pm_data"]]

            # fb_mem 복원
            if "early_fb_mem" in ck:
                early_fb_mem.memory = ck["early_fb_mem"]
            if "late_fb_mem" in ck:
                late_fb_mem.memory = ck["late_fb_mem"]

            print(f"  best_map (checkpoint) = {ck.get('best_map', 'N/A')}")
            print(f"  round={ck.get('round')}  phase={ck.get('phase')}")
            ck_loaded = True
            break

    if not ck_loaded:
        print("[Info] 체크포인트 없음 — --early-prompt / --late-prompt 로 직접 지정하세요.")

    if not best_early or not best_late:
        print("[Error] 프롬프트를 알 수 없습니다. --early-prompt 와 --late-prompt 를 지정하세요.")
        sys.exit(1)

    print(f"\nEarly 프롬프트: {best_early[:80]}...")
    print(f"Late  프롬프트: {best_late[:80]}...")

    def _fmt(v): return f"{v:.3f}" if v is not None else "None"

    # ── 단일 최고 프롬프트 평가 ──────────────────────────────────────────────
    print("\n=== Dev → threshold 탐색 ===")
    dev_res = score_dataset(devset, best_early, best_late, early_fb_mem, late_fb_mem)
    best_thr = find_best_threshold(dev_res)
    print(f"  best_threshold={best_thr:.3f}")

    print("\n=== Test 평가 (단일 프롬프트) ===")
    test_res = score_dataset(testset, best_early, best_late, early_fb_mem, late_fb_mem)
    single_m   = compute_metrics(test_res)
    single_m_o = compute_metrics(test_res, best_thr)
    print(f"  AUC={_fmt(single_m['auc'])}  MAP={_fmt(single_m['map'])}  "
          f"F1={_fmt(single_m['f1'])}  ACC={_fmt(single_m['acc'])}  "
          f"F1_opt={_fmt(single_m_o['f1'])}")

    # ── Top-3 앙상블 (pm 데이터 있을 때) ────────────────────────────────────
    ensemble_pairs = []
    if early_pm.memory and late_pm.memory:
        _top_e = sorted(early_pm.memory, key=lambda x: x[1], reverse=True)[:3]
        _top_l = sorted(late_pm.memory,  key=lambda x: x[1], reverse=True)[:3]
        ensemble_pairs = [(ep, lp, max((es + ls) / 2, 1e-6))
                          for (ep, es), (lp, ls) in zip(_top_e, _top_l)]

    if not ensemble_pairs:
        ensemble_pairs = [(best_early, best_late, 1.0)]

    print(f"\n=== Test 평가 (Top-{len(ensemble_pairs)} 앙상블) ===")

    acc: dict = defaultdict(list)
    lookup: dict = {}
    for ep, lp, w in ensemble_pairs:
        if ep == best_early and lp == best_late:
            _res = test_res
        else:
            _res = score_dataset(testset, ep, lp, early_fb_mem, late_fb_mem)
        for r in _res:
            key = (r["video"], r["second"])
            acc[key].append((r["pred"], w))
            if key not in lookup:
                lookup[key] = r

    # 앙상블 dev threshold
    acc_dev: dict = defaultdict(list)
    lookup_dev: dict = {}
    for ep, lp, w in ensemble_pairs:
        if ep == best_early and lp == best_late:
            _r = dev_res
        else:
            _r = score_dataset(devset, ep, lp, early_fb_mem, late_fb_mem)
        for r in _r:
            key = (r["video"], r["second"])
            acc_dev[key].append((r["pred"], w))
            if key not in lookup_dev:
                lookup_dev[key] = r

    def _merge(acc_map, lookup_map):
        out = []
        for key, pw_list in acc_map.items():
            if key in lookup_map:
                r = dict(lookup_map[key])
                wsum = sum(w for _, w in pw_list)
                r["pred"] = sum(p * w for p, w in pw_list) / wsum
                out.append(r)
        return out

    ensemble_dev_res  = _merge(acc_dev, lookup_dev)
    ensemble_thr      = find_best_threshold(ensemble_dev_res)
    ensemble_test_res = _merge(acc, lookup)
    ens_m   = compute_metrics(ensemble_test_res)
    ens_m_o = compute_metrics(ensemble_test_res, ensemble_thr)
    print(f"  AUC={_fmt(ens_m['auc'])}  MAP={_fmt(ens_m['map'])}  "
          f"F1={_fmt(ens_m['f1'])}  ACC={_fmt(ens_m['acc'])}  "
          f"F1_opt={_fmt(ens_m_o['f1'])}")

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    result = {
        "method":              "TEMPO-COV-Recovered",
        "best_early_prompt":   best_early,
        "best_late_prompt":    best_late,
        "single_test":         single_m,
        "single_test_f1_opt":  single_m_o["f1"],
        "single_test_acc_opt": single_m_o["acc"],
        "best_test":           ens_m,
        "best_test_f1_opt":    ens_m_o["f1"],
        "best_test_acc_opt":   ens_m_o["acc"],
        "n_ensemble_pairs":    len(ensemble_pairs),
    }
    out_path = OUT_DIR / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
