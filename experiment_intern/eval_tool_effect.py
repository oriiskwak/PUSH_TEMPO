"""
eval_tool_effect.py — tool_usage_result.json에서 나눈 두 그룹(도구 활성화 44개 /
비활성화 77개)에 대해, 저장된 best prompt + 실제 학습된 fb_mem으로 재채점만 해서
"도구를 켰을 때 vs 껐을 때"의 AUC/mAP/F1/ACC 차이를 본다. 재학습 없음.

4개 조건:
  A1) 활성화 44개 — 실제 CYREN 그대로 (fb_mem 그대로, 각 영상이 학습 때와 동일하게
      retrieve_tools()로 스스로 tool을 켬)
  A2) 활성화 44개 — 같은 영상, tool 강제로 끔 (fb_mem=None)
  B1) 비활성화 77개 — 원래 상태 (fb_mem=None과 동일 — 어차피 이 영상들은 실제로도
      retrieve_tools()가 빈 리스트를 반환했던 영상들)
  B2) 비활성화 77개 — 같은 영상, tool 강제로 켬 (ForceToolMemory: 항상
      [channel, title, web_search] 반환 — description/rag는 이 fb_mem에서 실질적으로
      한 번도 활성화된 적 없어서 제외)

Usage:
  CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 PP_SIZE=1 python experiment_intern/eval_tool_effect.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent


class ForceToolMemory:
    """항상 고정된 tool 목록을 반환하는 가짜 FeedbackMemory — '도구 강제 ON' 조건용."""
    def __init__(self, tools: list[str]):
        self.memory = [{"feedback": "force-on", "tools": tools, "pattern": "force-on"}]  # len>=MIN_FB_ENTRIES 통과용
        self._tools = tools

    def retrieve_tools(self, query: str) -> list[str]:
        return list(self._tools)


def _fmt(v):
    return f"{v:.4f}" if v is not None else "None"


if __name__ == "__main__":
    print("=== tool_usage_result.json에서 44개/77개 그룹 로드 ===")
    tu = json.load(open(ROOT / "results_intern" / "tool_usage_result.json", encoding="utf-8"))
    activated_names   = {r["video_name"] for r in tu["per_video"] if r["n_tools"] > 0}
    unactivated_names = {r["video_name"] for r in tu["per_video"] if r["n_tools"] == 0}
    print(f"활성화 44개 확인: {len(activated_names)}개")
    print(f"비활성화 77개 확인: {len(unactivated_names)}개")

    print("\n=== best prompt 로드 ===")
    saved = json.load(open(ROOT / "results_intern" / "tempo_cov_intern_evolve_result.json", encoding="utf-8"))
    best_early = saved["best_early_prompt"]
    best_late  = saved["best_late_prompt"]
    print(f"method: {saved.get('method')}")

    # experiment_intern.tempo_cov를 import하는 시점에 InternVL vLLM 모델이 로드됨
    import experiment_intern.tempo_cov as tc
    from utils_intern import split_data, load_gt
    from metrics_core import compute_metrics_from_frames
    from em_core_tool2 import FeedbackMemory

    print("\n=== 데이터셋 로드 (동일 SEED=42 split) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, tc.TRAIN_RATIO, tc.DEV_RATIO, tc.SEED)
    activated_set   = [v for v in testset if v["video_name"] in activated_names]
    unactivated_set = [v for v in testset if v["video_name"] in unactivated_names]
    print(f"Test 활성화 서브셋={len(activated_set)}  비활성화 서브셋={len(unactivated_set)}")

    print("\n=== 실제 학습된 fb_mem 복원 (체크포인트) ===")
    ckpt = json.load(open(ROOT / "results_intern" / "tempo_cov_intern_evolve_checkpoint.json", encoding="utf-8"))
    real_early_fb_mem = FeedbackMemory(top_k=3)
    real_early_fb_mem.memory = ckpt["early_fb_mem"]
    real_late_fb_mem = FeedbackMemory(top_k=3)
    real_late_fb_mem.memory = ckpt["late_fb_mem"]
    print(f"early_fb_mem entries: {len(real_early_fb_mem.memory)}")

    force_tool_mem = ForceToolMemory(["channel", "title", "web_search"])

    # score_dataset이 tool을 쓰려면 TOOLS_ENABLED가 True여야 함 (학습 루프 밖에서 직접
    # 호출하면 모듈 기본값 False라서 명시적으로 켜줘야 함)
    tc.TOOLS_ENABLED = True

    def _run(label, dataset, early_fb, late_fb):
        res = tc.score_dataset(dataset, best_early, best_late, early_fb, late_fb)
        m = compute_metrics_from_frames(res)
        print(f"\n▶ {label}  (n={len(dataset)}, frames={len(res)})")
        print(f"  AUC={_fmt(m['auc'])}  MAP={_fmt(m['map'])}  F1={_fmt(m['f1'])}  ACC={_fmt(m['acc'])}")
        return m

    print("\n" + "=" * 60)
    print("[그룹 A] 실제로 도구가 활성화됐던 44개 영상")
    print("=" * 60)
    a1 = _run("A1) 실제 CYREN 그대로 (tool ON, as-trained)", activated_set, real_early_fb_mem, real_late_fb_mem)
    a2 = _run("A2) 같은 44개, tool 강제 OFF", activated_set, None, None)

    print("\n" + "=" * 60)
    print("[그룹 B] 실제로 도구가 비활성화됐던 77개 영상 (대조군)")
    print("=" * 60)
    b1 = _run("B1) 원래 상태 (tool OFF, as-trained)", unactivated_set, None, None)
    b2 = _run("B2) 같은 77개, tool 강제 ON (channel+title+web_search)", unactivated_set, force_tool_mem, force_tool_mem)

    print("\n" + "=" * 60)
    print("=== 비교 요약 ===")
    print("=" * 60)
    print("\n[그룹 A: 44개, 실제로 도구를 쓴 영상들]")
    for k in ("auc", "map", "f1", "acc"):
        d = (a1[k] - a2[k]) if (a1.get(k) is not None and a2.get(k) is not None) else None
        print(f"  {k:>4}: ON={_fmt(a1.get(k))}  OFF={_fmt(a2.get(k))}  Δ(ON-OFF)={_fmt(d)}")

    print("\n[그룹 B: 77개, 실제로 도구를 안 쓴 영상들 — 대조군]")
    for k in ("auc", "map", "f1", "acc"):
        d = (b2[k] - b1[k]) if (b2.get(k) is not None and b1.get(k) is not None) else None
        print(f"  {k:>4}: OFF(원래)={_fmt(b1.get(k))}  강제ON={_fmt(b2.get(k))}  Δ(강제ON-원래)={_fmt(d)}")

    out = {
        "backbone": "InternVL2.5-78B-AWQ",
        "best_early_prompt": best_early,
        "best_late_prompt": best_late,
        "group_A_tool_activated_44": {"n": len(activated_set), "tool_on": a1, "tool_off": a2},
        "group_B_tool_unactivated_77_control": {"n": len(unactivated_set), "as_trained_off": b1, "forced_on": b2},
    }
    out_path = ROOT / "results_intern" / "tool_effect_result.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
