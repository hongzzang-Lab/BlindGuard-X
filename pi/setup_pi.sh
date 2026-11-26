#!/usr/bin/env bash
# pi/setup_pi.sh — Raspberry Pi 5 준비 (인터넷 되는 곳에서, 출발 전에 한 번)
#
#   cd ~/realtest && bash pi/setup_pi.sh
#
# 하는 일
#   1) 가상환경 ~/bgx-venv (시스템 OpenCV·numpy 도 같이 보이게)
#   2) ultralytics, ncnn, websockets 설치
#   3) YOLO26n 을 NCNN(416×416)으로 내보내서 realtest/models/yolo26n_ncnn_model 에 둠
#      (이미 팀이 쓰던 yolo26n_ncnn_model 폴더가 있으면 그걸 models/ 에 복사해도 됨)
#   4) pi/check_pi.py 로 카메라·모델·속도 확인
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo "== 1. 시스템 패키지 =="
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-opencv v4l-utils rsync

echo "== 2. 가상환경 =="
if [ ! -d ~/bgx-venv ]; then
  python3 -m venv ~/bgx-venv --system-site-packages
fi
source ~/bgx-venv/bin/activate
pip install -U pip
pip install ultralytics ncnn websockets

echo "== 3. YOLO26n → NCNN 416 =="
mkdir -p "$ROOT/models"
if [ -d "$ROOT/models/yolo26n_ncnn_model" ]; then
  echo "이미 있음: models/yolo26n_ncnn_model"
else
  cd "$ROOT/models"
  yolo export model=yolo26n.pt format=ncnn imgsz=416
  cd "$ROOT"
fi

echo "== 4. 점검 =="
python3 pi/check_pi.py

echo
echo "끝. 앞으로 실행할 때는:  source ~/bgx-venv/bin/activate && cd $ROOT && python3 pi/edge.py"
