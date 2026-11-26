#!/usr/bin/env bash
# tools/start_server.sh — 노트북 실험 서버 켜기 (realtest 폴더 기준)
#   bash tools/start_server.sh
# 끄기: Ctrl+C.  기록 폴더는 logs/server_<시각>/ (켤 때마다 새 폴더)
cd "$(dirname "$0")/.."
python3 -c "import starlette, uvicorn" 2>/dev/null || { echo "설치 필요: pip install starlette 'uvicorn[standard]' pandas numpy"; exit 1; }
exec python3 -m uvicorn server.exp_server:app --host 0.0.0.0 --port 8000 --ws-max-size 33554432 --no-access-log
