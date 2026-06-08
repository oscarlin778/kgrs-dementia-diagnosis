#!/bin/bash
# Run new server on port 8082 + batch evaluation
# Usage: bash run_eval_v2.sh

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate AD 2>/dev/null || true

echo "[1] Starting api_server on port 8082..."
API_PORT=8082 nohup python api_server.py > /tmp/api_server_8082.log 2>&1 &
SRV_PID=$!
echo "    PID=$SRV_PID"

echo "[2] Waiting for server ready..."
for i in $(seq 1 40); do
    if curl -sf http://localhost:8082/api/v1/health > /dev/null 2>&1; then
        echo "    Server ready (${i}x3s)"
        break
    fi
    sleep 3
done

echo "[3] Clearing old results..."
rm -rf results/batch_eval/reports results/batch_eval/summary.csv results/batch_eval/accuracy.json
mkdir -p results/batch_eval/reports

echo "[4] Running batch evaluation..."
BATCH_API_URL=http://localhost:8082 python batch_evaluate.py

echo "[5] Done. Stopping port-8082 server..."
kill $SRV_PID 2>/dev/null || true
echo "Results in: results/batch_eval/"
