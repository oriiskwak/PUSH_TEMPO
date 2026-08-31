#!/bin/bash
# 4-class 실험 순서: baseline → uniapo_mc_v2 → tempo_qwen
# (freeze 버그 수정 후 재실행용)
# 실행: bash run_4class.sh

set -e  # 에러 발생 시 중단

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TP_SIZE=4
export MM_LIMIT=1
export VLLM_USE_V1=0

cd "$(dirname "$0")"

echo "========================================"
echo "  Step 1/3: Baseline (최적화 없음)"
echo "========================================"
python experiment_4class/baseline_eval_mc.py
echo "[완료] baseline_eval_mc"

echo ""
echo "========================================"
echo "  Step 2/3: UniAPO-MC-v2"
echo "========================================"
python experiment_4class/uniapo_mc_v2.py
echo "[완료] uniapo_mc_v2"

echo ""
echo "========================================"
echo "  Step 3/3: TEMPO (Qwen)"
echo "========================================"
python experiment_4class/tempo_qwen.py
echo "[완료] tempo_qwen"

echo ""
echo "========================================"
echo "  전체 완료. 결과: results_4class/"
echo "========================================"
