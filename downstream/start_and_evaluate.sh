#!/bin/bash
# 一鍵重啟 api_server + 跑 batch 評估
# 用法：bash start_and_evaluate.sh

set -e
cd "$(dirname "$0")"

echo "=== 停止舊的 api_server ==="
pkill -f "python.*api_server" 2>/dev/null || true
pkill -f "uvicorn.*api_server" 2>/dev/null || true
sleep 3

echo "=== 啟動新的 api_server (背景) ==="
source ~/miniconda3/etc/profile.d/conda.sh && conda activate AD
nohup python api_server.py > /tmp/api_server_new.log 2>&1 &
API_PID=$!
echo "api_server PID=$API_PID"

echo "=== 等待 api_server 就緒 ==="
for i in $(seq 1 30); do
    if curl -sf http://localhost:8081/api/v1/health > /dev/null 2>&1; then
        echo "api_server 已就緒 (${i}s)"
        break
    fi
    sleep 2
    echo "  等待中... (${i}s)"
done

echo ""
echo "=== 清除舊的 batch 結果 ==="
rm -rf results/batch_eval/reports results/batch_eval/summary.csv results/batch_eval/accuracy.json
mkdir -p results/batch_eval/reports

echo "=== 開始 batch 評估 (v2) ==="
python batch_evaluate.py

echo ""
echo "=== 完成 ==="
echo "報告存於：results/batch_eval/"
echo "準確率：results/batch_eval/accuracy.json"
