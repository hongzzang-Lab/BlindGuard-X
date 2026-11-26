#!/usr/bin/env bash
# shoot/shoot.sh — 실물 Vision 성능 평가용 촬영 (Pi 에서 실행)
#
#   bash shoot/shoot.sh cam1  걷기_가까이        # 세션 1: Cam1 을 삼각대에 (횡단보도)
#   bash shoot/shoot.sh cam2  출발_3번           # 세션 2: Cam2 를 삼각대에 (대기구역)
#   bash shoot/shoot.sh calib 보정               # Cam1 거리 보정용 (말한 자리에 서 있기)
#   bash shoot/shoot.sh cam1  걷기 40            # 맨 뒤 숫자 = 초 (없으면 Ctrl+C 로 끝냄)
#
# 어느 세션이든 카메라 두 대를 모두 켜서 분석함 (Pi 한 대가 둘을 같이 돌리는 실제 조건).
# 저장: logs/pi_날짜_시각_<세션>_<이름>/  (원본 영상 · 분석이 겹쳐진 영상 · CSV 기록)
# 노트북 관제 화면으로 같이 보려면:  SERVER=ws://노트북IP:8000/edge bash shoot/shoot.sh cam1 걷기
set -e
SES=${1:?"사용법: bash shoot/shoot.sh cam1|cam2|calib 이름 [초]"}
NAME=${2:?"이름을 붙여 주세요 (예: 걷기_가까이)"}
SEC=${3:-0}
case "$SES" in cam1|cam2|calib) ;; *) echo "첫 칸은 cam1, cam2, calib 중 하나"; exit 1;; esac
cd "$(dirname "$0")/.."
[ -f "$HOME/bgx-venv/bin/activate" ] && source "$HOME/bgx-venv/bin/activate"
ARGS=(--record-all --raw --tag "${SES}_${NAME}")
if [ -n "$SERVER" ]; then ARGS+=(--server "$SERVER"); else ARGS+=(--offline); fi
if [ "$SEC" != "0" ]; then ARGS+=(--minutes "$(python3 -c "print($SEC/60)")"); fi
echo "──────────────────────────────────────────────"
echo " 촬영: 세션 $SES · 이름 $NAME · $([ "$SEC" = "0" ] && echo 'Ctrl+C 로 끝냄' || echo "${SEC}초")"
echo " 시작 후 5초는 화면 밖에 있기 (Cam2 가 빈 배경을 배움)"
echo "──────────────────────────────────────────────"
python3 pi/edge.py "${ARGS[@]}"
