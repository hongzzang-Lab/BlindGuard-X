#!/usr/bin/env bash
# tools/sync_pi.sh — Pi 의 기록(CSV·녹화 영상)과 보정 파일을 노트북 realtest/pi_logs 로 가져옴
#   bash tools/sync_pi.sh pi@raspberrypi.local
# 실험 중간중간 돌려도 됨 (이미 받은 것은 건너뜀). 마지막에 꼭 한 번.
set -e
HOST=${1:?"사용법: bash tools/sync_pi.sh 사용자@Pi주소"}
cd "$(dirname "$0")/.."
mkdir -p pi_logs
rsync -av "$HOST:~/realtest/logs/" pi_logs/
rsync -av "$HOST:~/realtest/calib/" pi_logs/calib_from_pi/
du -sh pi_logs
echo "가져온 곳: $(pwd)/pi_logs"
