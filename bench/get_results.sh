#!/usr/bin/env bash
# bench/get_results.sh — (노트북에서 실행) Pi 의 벤치마크 결과 폴더를 노트북 bench/results/ 로 가져옴 (영상 데이터는 안 가져옴)
#   bash bench/get_results.sh 계정@Pi주소
set -e
PI=${1:?"사용법: bash bench/get_results.sh 계정@Pi주소"}
cd "$(dirname "$0")"
mkdir -p results
ssh "$PI" 'cd ~/realtest/bench && tar cf - results' | tar xf -
echo "[완료] 가져온 결과 폴더:"
ls -1 results
