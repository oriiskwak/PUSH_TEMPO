#!/bin/bash
# guard_no_local_mc.sh — TEMPO/Vanilla(4-class, Qwen)가 다른 서버로 옮겨졌으므로,
# chain6이 Tool-effect 끝내고 이 둘을 로컬에서 자동 실행하려는 순간 즉시 차단.
# tool_effect 완료까지는 그대로 두고, 그 이후 프로세스만 감시.

echo "[$(date)] guard 시작 — tempo_qwen.py / vanilla_mc.py 감시 중 (10초 간격)"
while true; do
    PID=$(pgrep -f "experiment_4class/tempo_qwen.py\|experiment_4class/baselines/qwen/vanilla_mc.py" | head -1)
    if [ -n "$PID" ]; then
        echo "[$(date)] 감지됨 (PID=$PID) — 즉시 종료. TEMPO/Vanilla는 다른 서버에서 도는 걸로 결정됨."
        kill -9 "$PID" 2>/dev/null
        # chain6 wrapper도 같이 종료 (다음 stage로 안 넘어가게)
        pkill -f "run_qwen_mc_chain6.sh" 2>/dev/null
        echo "[$(date)] guard 종료."
        break
    fi
    sleep 10
done
