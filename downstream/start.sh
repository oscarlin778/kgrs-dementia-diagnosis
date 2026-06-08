#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== KGRS 系統啟動 ==="

# 終止舊的 uvicorn（如有）
OLD_BACK=$(ps aux | grep "uvicorn api_server" | grep -v grep | awk '{print $2}')
if [ -n "$OLD_BACK" ]; then
    echo "終止舊的後端 (PID $OLD_BACK)..."
    kill $OLD_BACK 2>/dev/null
fi

# 終止舊的 Vite（如有）
OLD_VITE=$(ps aux | grep "vite --host 0.0.0.0 --port 5173" | grep -v grep | awk '{print $2}')
if [ -n "$OLD_VITE" ]; then
    echo "終止舊的前端 (PID $OLD_VITE)..."
    kill $OLD_VITE 2>/dev/null
fi

sleep 2

# 啟動後端
echo "啟動後端（port 8080）..."
conda run -n AD uvicorn api_server:app \
    --host 0.0.0.0 --port 8080 \
    > /tmp/kgrs_backend.log 2>&1 &
BACKEND_PID=$!
echo "後端 PID: $BACKEND_PID，log: /tmp/kgrs_backend.log"

# 等待後端就緒
echo -n "等待後端就緒"
for i in $(seq 1 20); do
    sleep 1
    echo -n "."
    if curl -s http://localhost:8080/api/v1/patients > /dev/null 2>&1; then
        echo " OK"
        break
    fi
done

# 啟動前端
echo "啟動前端（port 5173）..."
cd "$SCRIPT_DIR/kgrs-frontend"
npm run dev -- --host 0.0.0.0 --port 5173
