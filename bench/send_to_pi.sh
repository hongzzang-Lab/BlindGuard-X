#!/usr/bin/env bash
# bench/send_to_pi.sh — (노트북에서 실행) 벤치마크 평가에 필요한 파일을 Pi 로 보냄. 비밀번호는 한 번만 물음.
#   bash bench/send_to_pi.sh 계정@Pi주소          ← 자기 Pi 의 계정과 주소 (Pi 에서 hostname -I 로 확인)
set -e
PI=${1:?"사용법: bash bench/send_to_pi.sh 계정@Pi주소   (예: pi@192.168.0.20)"}
cd "$(dirname "$0")/.."
tar cf - bgx/cam1_pipeline.py bench/mot_bench.py bench/run_on_pi.sh bench/README.md \
  | ssh "$PI" 'mkdir -p ~/realtest && cd ~/realtest && tar xf - && ls -la bench/mot_bench.py bgx/cam1_pipeline.py'
echo
echo "[완료] Pi($PI) 로 보냄. 다음: ssh $PI  →  cd ~/realtest && bash bench/run_on_pi.sh --seqs PETS09-S2L1"
