"""
eval_early_late_tool_test.py — "tool을 EARLY에만 적용"(2b, 현재 방식) vs
"tool을 EARLY+LATE 둘 다 적용"(2d, 예전 방식)의 실제 test set 성능 비교.

results/tempo_cov_total2b_checkpoint.json / tempo_cov_total2d_checkpoint.json
(둘 다 8/5, round 5/round 3에서 멈춘 채로 남아있던 체크포인트)의 early_prompt/
late_prompt/early_fb_mem/late_fb_mem을 그대로 꺼내 test set(121개)에 재채점한다.
재학습 없음. 2b는 tempo_cov.py(EARLY만 tool) 그대로, 2d는 tempo_cov_bothtool.py
(EARLY+LATE 둘 다 tool)로 채점 — 두 체크포인트 각각 원래 학습됐던 tool 정책과
동일한 정책으로 재채점해야 공정한 비교가 됨.

주의: 두 체크포인트 모두 round 5/3에서 멈춘 중간 상태라 원래 실험이 끝까지
완주됐을 때와 다를 수 있음 — "그 시점까지의 진행 상황"을 비교하는 것.

Usage:
  CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 python experiment2/eval_early_late_tool_test.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
ROOT = Path(__file__).parent.parent
FRAMES_CACHE_DIR = ROOT / "data" / "frames_cache_2s"


def _fmt(v):
    return f"{v:.4f}" if v is not None else "None"


def _load_fb_mem(ckpt, key, FeedbackMemory):
    m = FeedbackMemory(top_k=3)
    m.memory = ckpt[key]
    return m


def _make_cached_loader(frame_step: int, max_sec: int):
    """data/frames_cache_2s의 사전 추출된 448x448 JPG를 읽어 원본 mp4 디코딩을
    건너뛴다 (tempo_cov.py의 _load_video 대체용, 원본 파일은 안 건드림)."""
    from PIL import Image

    def _cached_load_video(vid):
        vname = vid["video_name"]
        label = vid["label"]
        gt_map = {fp["second"]: fp["probability"] for fp in vid["frame_probabilities"]}
        frames = []
        for sec in range(0, max_sec, frame_step):
            p = FRAMES_CACHE_DIR / f"{vname}_s{sec:04d}.jpg"
            if not p.exists():
                continue
            img = Image.open(p).convert("RGB")
            gt = gt_map.get(sec, float(label == "abnormal"))
            frames.append((sec, gt, img))
        return vname, label, frames

    return _cached_load_video


if __name__ == "__main__":
    from em_core_tool2 import FeedbackMemory
    from metrics_core import compute_metrics_from_frames

    print("=== 데이터셋 로드 (동일 SEED=42 split) ===")

    results = {}

    for tag, ckpt_name, module_name in [
        ("2d_early_late",  "tempo_cov_total2d_checkpoint.json", "tempo_cov_bothtool"),
        ("2b_early_only", "tempo_cov_total2b_checkpoint.json", "tempo_cov"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"=== {tag} ({module_name}.py 정책으로 재채점) ===")
        ckpt = json.load(open(ROOT / "results" / ckpt_name, encoding="utf-8"))
        print(f"  체크포인트: round={ckpt['round']}  best_map(dev)={ckpt.get('best_map')}")

        if module_name == "tempo_cov":
            import experiment2.tempo_cov as tc
        else:
            import experiment2.tempo_cov_bothtool as tc

        # 원본 mp4 디코딩 대신 사전 추출된 프레임 캐시(data/frames_cache_2s)를 읽도록 교체
        # (score_dataset은 모듈 전역 _load_video를 참조하므로 패치가 그대로 반영됨)
        tc._load_video = _make_cached_loader(tc.FRAME_STEP, tc.MAX_SEC)

        from utils import split_data, load_gt
        gt_list = load_gt()
        trainset, devset, testset = split_data(gt_list, tc.TRAIN_RATIO, tc.DEV_RATIO, tc.SEED)
        if tag == "2b_early_only":
            print(f"  Test={len(testset)}")

        early_fb_mem = _load_fb_mem(ckpt, "early_fb_mem", FeedbackMemory)
        late_fb_mem  = _load_fb_mem(ckpt, "late_fb_mem",  FeedbackMemory)
        early_prompt = ckpt["early_prompt"]
        late_prompt  = ckpt["late_prompt"]
        print(f"  early_fb_mem entries={len(early_fb_mem.memory)}  late_fb_mem entries={len(late_fb_mem.memory)}")

        tc.TOOLS_ENABLED = True
        res = tc.score_dataset(testset, early_prompt, late_prompt, early_fb_mem, late_fb_mem)
        m = compute_metrics_from_frames(res)
        print(f"  AUC={_fmt(m['auc'])}  MAP={_fmt(m['map'])}  F1={_fmt(m['f1'])}  ACC={_fmt(m['acc'])}")
        results[tag] = {"checkpoint_round": ckpt["round"], "checkpoint_dev_best_map": ckpt.get("best_map"), **m}

    print(f"\n{'=' * 60}")
    print("=== 비교 요약 (test set) ===")
    for k in ("auc", "map", "f1", "acc"):
        a = results["2b_early_only"].get(k)
        b = results["2d_early_late"].get(k)
        print(f"  {k:>4}: EARLY만={_fmt(a)}  EARLY+LATE={_fmt(b)}  diff(EARLY만-EARLY+LATE)={_fmt(a - b) if a is not None and b is not None else 'None'}")

    out_path = ROOT / "results" / "early_late_tool_test_result.json"
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
