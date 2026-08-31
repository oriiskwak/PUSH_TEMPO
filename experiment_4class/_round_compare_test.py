"""
_round_compare_test.py — 일회성 진단 스크립트.

tempo_cov_mc_noseg_result_v2_5class.json에서 round 3(raw dev_f1=0.4354, 최고)과
round 4(raw dev_f1=0.4337, 실제 최종 선택됨)의 프롬프트를 똑같은 test set에
(fb_mem=None, 도구 주입 없이 동일 조건으로) 평가해서 COV 가중치 때문에 진짜
"덜 정확한" 프롬프트가 최종으로 뽑혔는지 직접 확인한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils_mc import load_gt_mc, split_data_mc, compute_metrics_mc
from tempo_qwen import score_dataset

TRAIN_RATIO, DEV_RATIO, SEED = 0.6, 0.2, 42

ROUND3_PROMPT = (
    "Classify this video frame into one of: 보이스피싱 / 투자사기 / 로맨스스캠 / 부업사기 / 정상. "
    "For 부업사기, focus on app interfaces, progress bars, financial tracking, passive income "
    "schemes, daily tasks, and app-based mechanics. For 보이스피싱, identify urgent calls to "
    "action, threatening language, immediate financial transactions, and direct communication "
    "like phone calls or text messages. For 투자사기, observe financial schemes and promises of "
    "high returns. For 로맨스스캠, note personal interactions, gift exchanges, and emotional "
    "appeals. For 정상, ensure no financial urgency or emotional manipulation. Use specific "
    "visual and contextual cues to determine the category. The model must return ONLY one of: "
    "보이스피싱 / 투자사기 / 로맨스스캠 / 부업사기 / 정상. Ensure the output is one class name per frame."
)

ROUND4_PROMPT = (
    "Classify each video frame into one of: 보이스피싱 / 투자사기 / 로맨스스캠 / 부업사기 / 정상. "
    "Legitimate content shows clear branding, detailed explanations, and a professional tone. "
    "Scams exhibit urgency, vague promises, or suspicious contacts. For 보이스피싱, identify "
    "urgent calls, threats, and direct communication like phone calls or texts. For 투자사기, "
    "spot high-return promises and financial schemes. For 로맨스스캠, note personal interactions, "
    "gift exchanges, and emotional appeals. For 부업사기, analyze app interfaces, progress bars, "
    "financial tracking, passive income schemes, and app-based mechanics. For 정상, ensure no "
    "financial urgency or manipulation. Use abstract patterns and specific visual and contextual "
    "cues to distinguish content. The model must return ONLY one of: 보이스피싱 / 투자사기 / "
    "로맨스스캠 / 부업사기 / 정상. Ensure the output is one class name per frame."
)

if __name__ == "__main__":
    gt_list = load_gt_mc()
    _, _, testset = split_data_mc(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Test set: {len(testset)}개 영상")

    for name, prompt in [("Round 3 (raw dev_f1=0.4354)", ROUND3_PROMPT),
                         ("Round 4 (raw dev_f1=0.4337, 실제 선택됨)", ROUND4_PROMPT)]:
        print(f"\n=== {name} ===")
        res = score_dataset(testset, prompt, fb_mem=None)
        m   = compute_metrics_mc(res)
        print(f"  acc={m['acc']:.4f}  macro_f1={m['macro_f1']:.4f}")
        for c, f1 in m["per_class_f1"].items():
            print(f"    {c}: F1={f1:.4f}")
