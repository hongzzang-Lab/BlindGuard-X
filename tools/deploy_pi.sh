#!/usr/bin/env bash
# tools/deploy_pi.sh — 노트북의 realtest 폴더를 Pi 로 복사 (기록·앱 패치는 빼고)
#   bash tools/deploy_pi.sh pi@raspberrypi.local      (또는 pi@10.42.0.xx)
set -e
HOST=${1:?"사용법: bash tools/deploy_pi.sh 사용자@Pi주소"}
cd "$(dirname "$0")/.."
rsync -av --exclude logs --exclude pi_logs --exclude app_patch --exclude '__pycache__' --exclude '*.avi' \
  ./ "$HOST:~/realtest/"
echo "복사 끝 → Pi 에서: source ~/bgx-venv/bin/activate && cd ~/realtest && python3 pi/check_pi.py"
