#!/bin/bash
# InternVL UniAPO 실행 스크립트
# 현재 돌고있는 GPU 작업이 끝나면 자동으로 이어서 실행
#
# 사용법:
#   bash run_intern_uniapo.sh              # 현재 GPU 작업 기다린 후 실행
#   bash run_intern_uniapo.sh --now        # 바로 실행 (기다리지 않음)

cd /home1/ekrud02/TEMPO

if [[ "$1" != "--now" ]]; then
    if nvidia-smi | grep -q "VLLM::Worker"; then
        echo "[$(date '+%H:%M:%S')] GPU 사용 중 (VLLM worker 감지), 종료 대기 중..."
        while nvidia-smi | grep -q "VLLM::Worker"; do
            sleep 60
            echo "[$(date '+%H:%M:%S')] 아직 실행 중..."
        done
        echo "[$(date '+%H:%M:%S')] GPU 해제 확인!"
    else
        echo "[$(date '+%H:%M:%S')] GPU 비어있음, 바로 시작합니다."
    fi
fi

echo "[$(date '+%H:%M:%S')] InternVL UniAPO 시작 (PP=4, 4×A10)..."
conda run -n uniAPO --no-capture-output \
    env CUDA_VISIBLE_DEVICES=0,1,2,3 PP_SIZE=4 \
    python experiment_intern/uniapo.py
