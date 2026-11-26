#!/usr/bin/env bash
# simdemo/studio.sh — (노트북) 시뮬 촬영 화면 열기.  서버가 켜져 있어야 함 (bash simdemo/start_sim_server.sh)
#   bash simdemo/studio.sh
# 촬영 화면 한 장(1920×1080 설계)에 상태 띠 · 관제(Cam1·Cam2·지도·Pi 처리) · 경고 기록이 다 들어 있음.
# OBS 에는 소스 3개만:  ① STUDIO(이 창, 맨 아래, 화면에 맞추기)  ② CARLA  위치 0,120  크기 1280×720
#                       ③ PHONE  위치 1300,170  크기 400×880      (②③ 은 '안쪽 경계에 맞춤')
# 이 창은 최소화하지 말 것 (다른 창 뒤에 있는 것은 괜찮음). 전체 화면 풀기/켜기 = F11, 닫기 = Alt+F4
URL=${1:-http://localhost:8000/studio}
if ! curl -s -o /dev/null --max-time 2 "http://localhost:8000/api/hud"; then
  echo "서버가 꺼져 있음 → 먼저 다른 터미널에서: bash simdemo/start_sim_server.sh"; exit 1
fi
for B in google-chrome google-chrome-stable chromium chromium-browser brave-browser microsoft-edge; do
  if command -v "$B" >/dev/null 2>&1; then
    nohup "$B" --app="$URL" --start-fullscreen --user-data-dir="$HOME/.bgx-studio-profile" \
      --disable-features=Translate,TranslateUI --lang=ko \
      --disable-backgrounding-occluded-windows --disable-renderer-backgrounding --disable-background-timer-throttling \
      --no-first-run --no-default-browser-check >/dev/null 2>&1 &
    echo "촬영 화면을 열었음 ($B, 창 제목 STUDIO). 전체 화면으로 안 열렸으면 그 창에서 F11"
    exit 0
  fi
done
echo "Chrome 계열 브라우저를 못 찾음 → 브라우저에서 $URL 을 열고 F11"
xdg-open "$URL" >/dev/null 2>&1 &
