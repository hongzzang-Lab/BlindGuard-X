#!/usr/bin/env bash
# tools/replay_modes_pi.sh — (Pi에서) 녹화한 Cam1 원본 영상 하나를 6가지 조건(모드 × YOLO 실행 방식)으로 다시 분석
#   bash tools/replay_modes_pi.sh logs/pi_XXXX/video/T005_V1_cam1_raw.avi
# 끝나면: python3 analysis/pi_modes.py logs/pi_*_replay_*
set -e
VID=${1:?"사용법: bash tools/replay_modes_pi.sh <cam1_raw.avi> [edge.py 추가 옵션]"}
shift
EXTRA=("$@")
VID=$(realpath "$VID")
cd "$(dirname "$0")/.."
NAME=$(basename "$VID" .avi)
# 모드 4가지 + (adaptive·packed) × YOLO 실행 방식 2가지 = 6가지
for C in "full every" "fixed every" "adaptive every" "adaptive event" "packed every" "packed event"; do
  set -- $C "${EXTRA[@]}"
  M=$1; P=$2
  echo "== $M / $P =="
  python3 pi/edge.py --offline --no-cam2 --cam1 "$VID" --mode "$M" --policy "$P" --preview-hz 0 \
    --tag "replay_${NAME}_${M}_${P}" "${EXTRA[@]}"
done
echo "비교: python3 analysis/pi_modes.py logs/pi_*_replay_${NAME}_*"
