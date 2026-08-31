#!/bin/bash
# Temporal-EM (uniapo_total_temporal.py) 단독 실행
# (freeze 버그 수정 후 재실행용)
# 실행: bash run_temporal.sh

set -e  # 에러 발생 시 중단

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TP_SIZE=4
export MM_LIMIT=1
export VLLM_USE_V1=0

cd "$(dirname "$0")"

echo "========================================"
echo "  Temporal-EM (uniapo_total_temporal.py)"
echo "========================================"
python experiment/uniapo_total_temporal.py
echo "[완료] uniapo_total_temporal"
