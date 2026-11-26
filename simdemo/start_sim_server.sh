#!/usr/bin/env bash
# simdemo/start_sim_server.sh — 시뮬 데모용 노트북 서버 (realtest 폴더의 server/exp_server.py 를 시뮬 설정으로)
#   bash simdemo/start_sim_server.sh
# 관제 화면: http://localhost:8000/monitor   (촬영할 화면)
# 기록: logs/server_<시각>/  (fusion.csv = 차량별 판정 근거, sim_states.csv = CARLA 차량·신호, alerts.csv = 앱 전송·지연)
cd "$(dirname "$0")/.."
python3 -c "import starlette, uvicorn" 2>/dev/null || { echo "설치 필요: pip install starlette 'uvicorn[standard]' pandas numpy"; exit 1; }
export BGX_SITE=simdemo/site_sim.json
export BGX_CALIB_DIR=calib_sim          # 공원 실물 테스트 보정(calib/)과 분리
mkdir -p calib_sim
echo "노트북 IP (Pi·폰이 붙을 주소): $(hostname -I 2>/dev/null)"
exec python3 -m uvicorn server.exp_server:app --host 0.0.0.0 --port 8000 --ws-max-size 33554432
