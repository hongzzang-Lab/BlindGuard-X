#!/usr/bin/env bash
# tools/hotspot_laptop.sh — 우분투 노트북을 Wi-Fi 핫스팟으로 (노트북 주소 = 10.42.0.1)
#   BGX_HOTSPOT_PASS=<8자 이상> bash tools/hotspot_laptop.sh     # 켜기 (이름 BGX-TEST, 비밀번호는 각자 정함)
#   nmcli connection down Hotspot           # 끄기
set -e
SSID=${1:-BGX-TEST}
PASS=${2:-${BGX_HOTSPOT_PASS:?"비밀번호를 정해 주세요:  BGX_HOTSPOT_PASS=<8자 이상> bash tools/hotspot_laptop.sh"}}
IF=$(nmcli -t -f DEVICE,TYPE device | awk -F: '$2=="wifi"{print $1; exit}')
[ -n "$IF" ] || { echo "Wi-Fi 장치를 못 찾음"; exit 1; }
nmcli device wifi hotspot ifname "$IF" ssid "$SSID" password "$PASS"
sleep 2
ip -4 addr show "$IF" | grep inet
echo
echo "Pi 연결 (Pi 터미널):   sudo nmcli dev wifi connect $SSID password $PASS"
echo "폰 연결: Wi-Fi '$SSID' → '인터넷 없음' 이라도 연결 유지, 그리고 모바일 데이터 끄기"
echo "방화벽(켜져 있으면):    sudo ufw allow 8000"
echo "연결된 기기 주소 보기:  ip neigh show dev $IF"
