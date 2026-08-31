#!/bin/bash
# tempo_cov → tempo 순서대로 실행

PYTHON=/anaconda3/envs/hee/bin/python
WORK_DIR=/home/labhosik4609/DG/TEMPO

export CUDA_VISIBLE_DEVICES=0,1,2
export TP_SIZE=3
export MM_LIMIT=1

LOG_DIR=$WORK_DIR/experiment/auto_logs
mkdir -p $LOG_DIR

echo "========================================"
echo "  실험 시작: $(date)"
echo "========================================"

cd $WORK_DIR

run_exp() {
    local name=$1
    local script=experiment/${name}.py
    local logfile=$LOG_DIR/${name}.log

    echo ""
    echo "--------------------------------------"
    echo "  [$name] 시작: $(date)"
    echo "--------------------------------------"

    $PYTHON $script 2>&1 | tee $logfile

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "  [$name] 완료: $(date)"
    else
        echo "  [$name] 오류 (exit=$exit_code): $(date)"
        echo "  로그: $logfile"
        exit $exit_code
    fi
}

run_exp tempo_cov
run_exp tempo

echo ""
echo "========================================"
echo "  전체 완료: $(date)"
echo "========================================"
