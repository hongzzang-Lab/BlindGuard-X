#!/usr/bin/env bash
# simdemo/run_demo.sh — (노트북) CARLA 시나리오 실행
#   bash simdemo/run_demo.sh --inspect                         # 1) 기하·카메라 확인 그림만 (Pi·서버 없이)
#   bash simdemo/run_demo.sh --pi <PiIP> --variant main --takes 3
#   bash simdemo/run_demo.sh --pi <PiIP> --variant stages
#   bash simdemo/run_demo.sh --pi <PiIP> --variant wait_only
# 다른 옵션: --port 2000(CARLA 포트) --clear(J189 주변 다른 차·사람 지우기) --view approach|top|none
cd "$(dirname "$0")/.."
PY=${BLIND_PY:-$HOME/miniconda3/envs/blindguard-carla/bin/python}
[ -x "$PY" ] || PY=python3
exec "$PY" simdemo/carla_demo.py "$@"
