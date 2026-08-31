"""
plot_convergence.py — 수렴 패턴 비교
실행: /anaconda3/bin/python plot_convergence.py

비교 방법론:
  - Vanilla Baseline  (수평선)
  - Baseline EM       (UniAPO)
  - Temporal Only
  - Tool Only
  - TEMPO             (Temporal + Tool + COV)
"""

import json, collections
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

# ── 방법론 정의 (추가하려면 여기에만 넣으면 됨) ───────────────────────────────
# key: (label, color, marker, linestyle, result_file, log_type)
# log_type: "single" | "dual" (early/late) | "evoprompt" | "baseline"
METHODS = [
    # Vanilla Baseline은 논문 round-curve 비교에서 제외 (round가 없는 zero-shot 단일점이라
    # 곡선 비교엔 안 씀) — 필요하면 주석 해제. 경로도 실제로는 qwen/ 밑에 있음.
    # {
    #     "label":      "Vanilla Baseline",
    #     "color":      "#888888",
    #     "marker":     None,
    #     "linestyle":  "--",
    #     "file":       "qwen/baseline_result.json",
    #     "log_type":   "baseline",
    # },
    {
        "label":      "APO*",
        "color":      "#F44336",
        "marker":     "s",
        "linestyle":  "-",
        "file":       "baseline/apo_result.json",
        "log_type":   "single",
    },
    {
        "label":      "ERM*",
        "color":      "#FF9800",
        "marker":     "D",
        "linestyle":  "-",
        "file":       "baseline/erm_result.json",
        "log_type":   "single",
    },
    {
        "label":      "EvoPrompt*",
        "color":      "#9E9E9E",
        "marker":     "x",
        "linestyle":  ":",
        "file":       "evoprompt_result.json",
        "log_type":   "evoprompt",
    },
    {
        "label":      "UniAPO (EM)",
        "color":      "#2196F3",
        "marker":     "o",
        "linestyle":  "-",
        "file":       "qwen/baseline_em_result.json",   # 경로 수정: 루트가 아니라 qwen/ 밑에 있음
        "log_type":   "single",
    },
    # Temporal Only / Tool Only는 논문 비교에서 제외하기로 함 (2026-08-18) — 필요하면 주석 해제
    # {
    #     "label":      "Temporal Only",
    #     "color":      "#E91E63",
    #     "marker":     "*",
    #     "linestyle":  "-",
    #     "file":       "temporal_em_result.json",
    #     "log_type":   "dual",
    # },
    # {
    #     "label":      "Tool Only (v1)",
    #     "color":      "#4CAF50",
    #     "marker":     "D",
    #     "linestyle":  "--",
    #     "file":       "tool_em5b_result.json",
    #     "log_type":   "single",
    # },
    {
        "label":      "CYREN (Ours)",
        "color":      "#9C27B0",
        "marker":     "^",
        "linestyle":  "-",
        "file":       "tempo_cov_total_covfix_result.json",  # Table 1 CYREN 수치(AUC .6929/MAP .5423)와 매칭되는 파일
        "log_type":   "dual",
    },
    # GPT-4o-mini 백본 비교군 — 이번 큐웬 전용 비교에서는 제외. 필요하면 주석 해제.
    # {
    #     "label":      "Baseline EM (GPT-4o-mini)",
    #     "color":      "#795548",
    #     "marker":     "s",
    #     "linestyle":  "-",
    #     "file":       "uniapo_gpt4omini_result.json",
    #     "log_type":   "single",
    # },
    # {
    #     "label":      "Tool Only (GPT-4o-mini)",
    #     "color":      "#00BCD4",
    #     "marker":     "v",
    #     "linestyle":  "--",
    #     "file":       "tool_gpt4omini_result.json",
    #     "log_type":   "single",
    # },
]


# ── 데이터 파싱 ───────────────────────────────────────────────────────────────

def load_method(m: dict) -> dict:
    path = RESULTS / m["file"]
    if not path.exists():
        print(f"  [Skip] {m['file']} 없음")
        return None

    d = json.load(open(path))

    if m["log_type"] == "baseline":
        init_map = d["dev"]["map"]
        return {"xs": [0], "ys": [init_map], "is_baseline": True, "init": init_map}

    init_map = d.get("baseline_test", {}).get("map", 0.0)

    if m["log_type"] == "evoprompt":
        # best_test가 없음 (프롬프트가 마지막에 의미 없는 텍스트로 붕괴돼 최종 test 평가 자체가
        # 스킵됨) — log[i]["best"]가 evaluate_prompt()의 average_precision_score(=MAP)라
        # dev_map과 동일 지표이므로 그대로 사용. init_map은 baseline_test가 없어 1스텝째
        # population 평균(avg)으로 근사.
        devs = [r["best"] for r in d["log"]]
        init = d["log"][0]["avg"] if d["log"] else 0.0
        xs = list(range(len(devs) + 1))
        ys = [init] + devs
        return {"xs": xs, "ys": ys, "is_baseline": False, "init": init,
                "no_final_test": True}

    if m["log_type"] == "single":
        devs = [r["dev_map"] for r in d["log"]]
        xs = list(range(len(devs) + 1))
        ys = [init_map] + devs
        return {"xs": xs, "ys": ys, "is_baseline": False, "init": init_map}

    if m["log_type"] == "dual":
        # round_end 항목(전체 dev셋 기준, noise 제거된 round_map)이 있으면 그걸 우선 쓴다 —
        # 세그먼트별 log(dev_map)는 50개 랜덤 서브샘플 + 상대 구간 고정 상태의 M-step 평가라
        # 라운드 간 진짜 추세를 보기엔 노이즈가 커서(코드 주석: "round-end 비교는 매번 같은
        # 전체 dev셋 사용 — 라운드 간 noise 제거"), round_end가 없는 예전 결과 파일에서만
        # early/late 평균으로 근사한다.
        round_end = {r["round"]: r["round_map"]
                     for r in d["log"] if r.get("type") == "round_end"}
        if round_end:
            rounds = sorted(round_end)
            ys = [init_map] + [round_end[r] for r in rounds]
            xs = list(range(len(ys)))
            return {"xs": xs, "ys": ys, "is_baseline": False, "init": init_map}

        by_round = collections.defaultdict(dict)
        for r in d["log"]:
            if r.get("type") == "round_end":
                continue
            by_round[r["round"]][r["seg"]] = r["dev_map"]
        rounds = sorted(by_round)
        avg = [(by_round[r].get("early", 0) + by_round[r].get("late", 0)) / 2
               for r in rounds]
        early = [by_round[r].get("early", 0) for r in rounds]
        late  = [by_round[r].get("late",  0) for r in rounds]
        xs = list(range(len(avg) + 1))
        ys     = [init_map] + avg
        ys_e   = [init_map] + early
        ys_l   = [init_map] + late
        return {"xs": xs, "ys": ys, "ys_e": ys_e, "ys_l": ys_l,
                "is_baseline": False, "init": init_map,
                "approx": True}  # 세그먼트 평균 근사치임을 표시 (round_end 없는 구버전 결과)

    return None


def running_best(ys: list) -> list:
    best, out = ys[0] or 0.0, []
    for v in ys:
        v = v or 0.0
        best = max(best, v)
        out.append(best)
    return out


# ── 플롯 ──────────────────────────────────────────────────────────────────────

def plot_panel(ax, data_list, use_running: bool, title: str):
    max_x = max(d["xs"][-1] for d in data_list if d)

    for m, data in zip(METHODS, data_list):
        if data is None:
            continue

        color = m["color"]
        marker = m["marker"]
        ls = m["linestyle"]

        if data["is_baseline"]:
            ax.axhline(data["init"], color=color, linewidth=1.8, linestyle="--",
                       label=f"{m['label']} (AP={data['init']:.3f})", zorder=3)
            continue

        xs = data["xs"]
        ys = running_best(data["ys"]) if use_running else data["ys"]

        # early/late 밴드
        if "ys_e" in data and not use_running:
            import numpy as np
            ys_e = np.array(data["ys_e"], dtype=float)
            ys_l = np.array(data["ys_l"], dtype=float)
            ax.fill_between(xs, ys_e, ys_l, color=color, alpha=0.10, zorder=1)

        ms = 9 if marker == "*" else 7
        ax.plot(xs, ys, color=color, marker=marker, linestyle=ls,
                linewidth=2, markersize=ms, label=m["label"], zorder=4)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("AP", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(range(max_x + 1))
    ax.set_ylim(0.0, 0.85)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)


def main():
    data_list = [load_method(m) for m in METHODS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Convergence Pattern: dev AP per Iteration",
                 fontsize=14, fontweight="bold", y=1.01)

    plot_panel(axes[0], data_list, use_running=False, title="Raw dev AP")
    plot_panel(axes[1], data_list, use_running=True,  title="Best dev AP")

    fig.text(
        0.5, -0.04,
        "* EvoPrompt's iter 0-5 scores look high but reflect noisy small-sample AP (n_eval≈10-15)\n"
        "on semantically degenerate prompts (e.g. a single word \"and\", or off-topic text about baking) —\n"
        "not meaningful discriminative performance. See Table X for example collapsed prompts.",
        ha="center", va="top", fontsize=8.5, color="#555555", style="italic",
    )

    plt.tight_layout()
    out = Path(__file__).parent / "convergence_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"저장: {out}")
    plt.show()


if __name__ == "__main__":
    main()
