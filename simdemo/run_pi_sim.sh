#!/usr/bin/env bash
# simdemo/run_pi_sim.sh — (Pi 에서 실행) CARLA 가상 Cam1·Cam2 영상을 받아 분석
#   cd ~/realtest && bash simdemo/run_pi_sim.sh <노트북IP>        ← 노트북 IP
# Cam1 = TCP 8765 (1280×720), Cam2 = TCP 8767 (640×480) 에서 노트북 carla_demo.py 를 기다림.
# 보정은 carla_demo.py 가 카메라 모델로 계산해서 서버 → Pi 로 자동 적용 (calib_sim/ 에 저장, 공원용 calib/ 는 그대로)
set -e
LAPTOP=${1:?"사용법: bash simdemo/run_pi_sim.sh 노트북IP"}
shift
cd "$(dirname "$0")/.."
[ -f ~/bgx-venv/bin/activate ] && source ~/bgx-venv/bin/activate
export BGX_CALIB_DIR=calib_sim
mkdir -p calib_sim
exec python3 pi/edge.py --site simdemo/site_sim.json --server "ws://$LAPTOP:8000/edge" --tag sim "$@"
