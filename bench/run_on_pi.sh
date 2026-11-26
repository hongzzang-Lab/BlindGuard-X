#!/usr/bin/env bash
# bench/run_on_pi.sh — (Pi 에서 실행) 공개 벤치마크(MOT15 고정 카메라 구간)로 Cam1 방식 비교
#   cd ~/realtest && bash bench/run_on_pi.sh            ← 전체 (데이터 자동 받기 약 1.3 GB, 그다음 10~20분)
#   cd ~/realtest && bash bench/run_on_pi.sh --seqs PETS09-S2L1     ← 주 평가 구간만 (3~5분)
# 먼저 pi/edge.py 는 꺼 둘 것 (같이 돌면 처리 시간이 느리게 나옴).
set -e
cd "$(dirname "$0")/.."
[ -f ~/bgx-venv/bin/activate ] && source ~/bgx-venv/bin/activate
if pgrep -f "pi/edge.py" >/dev/null; then
  echo "[주의] pi/edge.py 가 돌고 있음 → 처리 시간이 느리게 측정됨. 그 터미널에서 Ctrl+C 로 끄고 다시 실행하세요."
  exit 1
fi
exec python3 bench/mot_bench.py "$@"
