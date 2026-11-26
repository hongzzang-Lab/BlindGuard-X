#!/usr/bin/env bash
# simdemo/open_hud.sh — (노트북) 촬영용 상태 띠 창 열기.  OBS 에서 창 캡처로 잡아 CARLA 화면 위쪽에 겹쳐 놓음.
#   bash simdemo/open_hud.sh            # 서버가 켜져 있어야 함 (bash simdemo/start_sim_server.sh)
# 창 제목은 "HUD". 창을 최소화하면 OBS 에 멈춘 화면이 찍히므로 최소화하지 말 것 (다른 창 뒤에 있는 것은 괜찮음).
URL=${1:-http://localhost:8000/hud}
for B in google-chrome google-chrome-stable chromium chromium-browser brave-browser microsoft-edge; do
  if command -v "$B" >/dev/null 2>&1; then
    # 따로 쓰는 프로필 + 가려져도 계속 그리게 하는 옵션
    nohup "$B" --app="$URL" --window-size=1296,176 --window-position=40,40 --user-data-dir="$HOME/.bgx-hud-profile" \
      --disable-features=Translate,TranslateUI --lang=ko \
      --disable-backgrounding-occluded-windows --disable-renderer-backgrounding --disable-background-timer-throttling \
      --no-first-run --no-default-browser-check >/dev/null 2>&1 &
    echo "상태 띠 창을 열었음 ($B). OBS: 소스 + → 창 캡처(Xcomposite) → 창 'HUD' 선택"
    exit 0
  fi
done
echo "Chrome 계열 브라우저를 못 찾음 → 브라우저 새 창에서 $URL 을 열고, 창 높이를 낮게(띠 모양) 줄여서 쓰기"
xdg-open "$URL" >/dev/null 2>&1 &
