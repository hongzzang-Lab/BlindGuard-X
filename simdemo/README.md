# BlindGuard-X CARLA 시뮬 데모 (포스터 최종본 그림 2 그대로)

**무엇을 보여 주나**: 포스터 그림 1의 문제 상황(정차 차량에 가려 운전자가 보행자의 출발 징후를 못 봄)을
CARLA Town10HD_Opt J189 에서 재현하고, 우리 모듈을 적용했을 때 **차량 A 운전자 앱에만** 경고가 가는 모습.

```
CARLA Town10HD_Opt · J189
 ├ Cam1·Cam2 영상 ───TCP──▶ Raspberry Pi 5 (pi/edge.py, 실물 테스트와 같은 코드)
 │                            Cam1: Optical Flow + Track Memory → Adaptive ROI + ROI Packing → YOLO26n-NCNN → ByteTrack → Homography
 │                            Cam2: Polygon ROI → MOG2 → Sparse LK → 경계 접근 분석 → EMR Score
 ├ 보행자 신호 RED / GREEN / FLASH ─┐
 └ 차량 A·B: GNSS 위치 + 가상 OBU 속도 ─┴HTTP─▶ 노트북 서버 (server/exp_server.py, 차량 출처 = carla)
                                              차량별 CPA 위험 판정 (3초 예측 · 안전반경 2 m · 등속)
                                              + Cam2 EMR (조기 징후 보조 입력) + 보행자 신호
                                              └ FastAPI → WebSocket → 해당 차량 앱 (주의 / 경고 / 긴급)
```

## 판정 규칙 (팀 확정 11/21)

| 단계 | 조건 | 근거 표시 |
|---|---|---|
| **주의** | Cam2 EMR ≥ 0.6 (연속, 출발 징후) + 차량이 대기구역 쪽 차로에서 **5초 안에** 횡단보도 도착 | `Cam2 조기 징후` |
| **경고** | Cam1 CPA: **dCPA ≤ 2 m 이고 0 ≤ tCPA ≤ 3 s**<br>또는 위 '주의'가 **보행자 신호 FLASH 또는 RED 된 지 3초 이내**(신호 전환 구간) | `CPA 충돌 예측`<br>`Cam2 조기 징후` + `보행자 신호` |
| **긴급** | CPA 이면서 **tCPA ≤ 1.5 s** | `CPA 충돌 예측` |

- 단계는 1초 동안 유지한다(깜빡임 방지). 바뀔 때만 앱으로 보낸다.
- 차량 B는 반대 방향 먼 차로라서 dCPA가 크고 Cam2 주의 대상 차로도 아님 → **경고 안 함**.
- 코드: `bgx/fusion.py`. 설정: `simdemo/site_sim.json` (`risk`, `fusion`).

## 촬영용 5가지 상황 (11/22, 가짜 CARLA·실제 J189 기하·0.6배속으로 검증)

| variant | 상황 | 앱이 결정적인 이유 | 검증 결과 (차량 A 앱) | 차량 A | 차량 B |
|---|---|---|---|---|---|
| `case1_blind_spot` | 정차 트럭에 가린 무단횡단 | 운전자가 못 보는 보행자를 단계별로 알림 | 주의(Cam2) → 경고(CPA) → 긴급(CPA) | 급정지, 보행자 2.4 m 앞 | 없음 |
| `case2_signal_change` | 보행 신호가 빨강으로 바뀐 직후 뛰어 건넘 | 차도에 내려서기 전에 경고 | 경고(Cam2 조기 징후 + 보행자 신호) | 서서히 정지 | 없음 |
| `case3_red_light` | 초록불 정상 횡단 중 신호위반 차량 A (트럭·B 는 빨간불 지켜 정지) | 신호를 무시한 차에만 알림 | 주의 → 경고(CPA) | 정지 | 없음 (정지선에 섬) |
| `case4_speeding` | 50 km/h 과속 차량 | 빠를수록 더 멀리서 경고 | 주의(Cam2) → 경고(CPA) | 횡단보도 15 m 앞 정지 | 없음 |
| `case5_selective` | 같은 방향 2차로: 연석 쪽 B 가 지나가자마자 보행자가 B 뒤로 뛰어듦 (트럭 없음) | 부딪힐 차(옆 차로 A)에만 경고, 이미 지나간 B 는 경고 없음 | 주의 → 경고 | 정지 | **없음** |
| `case1_no_app` | CASE 1 대조: 운전자에게 앱 없음 (폰·관제 화면은 OBS 에서 숨기고 촬영) | 앱이 없으면 트럭에 가린 보행자를 못 보고 그대로 충돌 | (서버는 경고하지만 운전자는 못 받음) | **충돌** (30 km/h) | 없음 |
| `wait_only` | 서성이기만 함 (대조군) | 오경고 없음 | 없음 | 통과 | 없음 |

```bash
bash simdemo/run_demo.sh --pi <PiIP> --variant case1_blind_spot --takes 1 --view chase
```
시작할 때 터미널에 상황 이름과 `기대 결과` 가 나오고, 관제 화면 CARLA 칸에도 상황 이름이 보인다.
가짜 CARLA 검증: `FAKE_RT=0.6 PYTHONPATH=tests/fakecarla python3 tests/run_fake_demo.py case1_blind_spot` (인식은 정답 위치를 쓰는 가짜라 실제 Pi 결과와 다를 수 있음)

## 장면 3가지 (`simdemo/scenario_j189.json` → variants)

| variant | 장면 | 기대 결과 (가짜 CARLA·완벽 인식 시험 결과) |
|---|---|---|
| `main` | 보행 신호가 깜빡이다(14 s) 빨강(18 s). 보행자가 서성이다가 빨강 직후 서둘러 연석으로 다가가 건넘. 차량 A는 정차 트럭 옆 차로로 30 km/h | A: **경고**(Cam2 조기 징후 + 신호 전환, 보행자가 차도에 내려서기 1.3초 전) → 운전자 감속 → 충돌 없음 / B: 경고 없음 |
| `stages` | 신호가 빨강이 된 지 오래 뒤 출발 (신호 상승 없음). 운전자는 긴급에만 반응 | A: **주의**(Cam2) → **경고**(CPA) → **긴급**(CPA) → 급제동, 보행자 2 m 앞 정지 / B: 경고 없음 |
| `wait_only` | 보행자가 대기구역에서 서성이기만 함 | A·B 모두 경고 없음 (오경고 없음 확인) |

차량·보행자 움직임은 연출용 스크립트이고, **경고 판단에는 정답(GT)을 쓰지 않는다**.
판단 = Pi 의 실제 Cam1·Cam2 인식 + CARLA 차량 GNSS·OBU + 신호. 정답은 노트북 `simdemo/logs/` 에만 저장(사후 확인용).

---

## 0. 준비 (한 번)

### 노트북
```bash
cd <저장소 폴더>
python3 tests/test_sim.py            # 계산 시험 9개 ok
python3 tests/test_sim_server.py     # 서버 흐름: A 경고→긴급, B 없음 → ok test_sim_server
```

### Pi (노트북에서 복사)
```bash
bash tools/deploy_pi.sh <Pi계정>@<PiIP>      # simdemo/, bgx/netsource.py, 고친 pi/edge.py 가 같이 감
```

### 앱 (폰 A = 차량 A)
노트북 IP 로 다시 빌드한다 (Pi·폰·노트북이 같은 네트워크).
```bash
hostname -I                                       # 예: <노트북IP>
cd <앱 소스 폴더>
flutter build apk --release --dart-define=SERVER_URL=ws://<노트북IP>:8000/ws
flutter install
```
폰 화면은 노트북에서 `scrcpy` 로 띄워 같이 녹화한다 (`sudo apt install scrcpy`, USB 디버깅 켜기).

## 1. 실행 순서 (매번)

| # | 어디서 | 명령 | 확인 |
|---|---|---|---|
| 1 | 노트북 | 팀 런처로 CARLA 실행 (Town10HD_Opt). **팀 Blind 교통 스크립트는 끄기** (이 데모가 동기 모드 틱 주인) | CARLA 창 |
| 2 | 노트북 | `bash simdemo/run_demo.sh --inspect` (처음 한 번) | `simdemo/logs/sim_*/check_cam1.jpg`, `check_cam2.jpg` 에서 빨간 연석선·노란 차로선·파란 대기구역이 실제 도로와 맞는지 |
| 3 | 노트북 | `bash simdemo/start_sim_server.sh` | 브라우저 `http://localhost:8000/monitor` |
| 4 | Pi | `cd ~/realtest && bash simdemo/run_pi_sim.sh <노트북IP>` | `[net] 0.0.0.0:8765 에서 CARLA 영상 기다리는 중` / 관제 'Pi 연결' |
| 5 | 폰 A | 앱 실행 → 주행 시작 | 관제 '폰 A 주행 중' |
| 6 | 노트북 | `scrcpy` (폰 화면) + OBS 녹화 시작 | |
| 7 | 노트북 | `bash simdemo/run_demo.sh --pi <PiIP> --variant main --takes 3` | 터미널에 `[보정] Cam1 Homography 적용, Cam2 대기구역 적용`, take 마다 `첫 경고`·`A 감속` |

- 처음 실행 때 보정(Cam1 Homography, Cam2 다각형·경계선)이 **카메라 모델로 계산되어 Pi 에 자동 적용**된다. 손 보정 없음.
- take 마다 서버가 시험(T001, T002…)을 열고 Pi 가 **오버레이 영상**(`T001_SIM_cam1_overlay.avi`, `_cam2_overlay.avi`)을 녹화한다.
- take 시작 4초는 빈 장면으로 Cam2 배경을 학습한다 (관제 Cam2 'BG LEARNING' 이 사라지면 됨).
- CARLA 가 이미 J189 에 다른 차·사람을 띄워 두었으면 `--clear`.

## 2-1. 차량 A 추적 영상 (화면 녹화 없이 바로 mp4)

차량 A 뒤 8 m·위 3.2 m 에서 따라가며, 보행자(인도) 쪽으로 6° 틀어 **신호등과 대기구역 보행자가 같이 보이는** 시점.
```bash
bash simdemo/run_demo.sh --pi <PiIP> --variant main --takes 3 --chase --view chase
```
- `--chase`: take 마다 `simdemo/logs/sim_<시각>/take01_chase.mp4` 저장 (1280×720, 20 fps, **시뮬 시간 기준**이라 노트북이 느려도 영상은 실제 속도)
- 영상 위 표시: 보행 신호·차량 신호, 차량 A 속도·감속, **차량 A 앱에 실제로 온 경고 단계와 근거**(서버 응답 그대로), 차량 B 경고 여부.
  한글 글꼴(Noto CJK 또는 나눔)이 있으면 한글, 없으면 영어 (`sudo apt install fonts-noto-cjk`)
- `--view chase`: CARLA 창도 같은 시점으로 따라감 (OBS 로 같이 녹화할 때)
- 녹화 구간: 기본 = 차량 A 출발 1초 전부터 끝까지. 처음부터는 `--chase-from 0`
- 시점 조정: `scenario_j189.json` → `chase` (`back_m`, `up_m`, `pitch`, `yaw_off`, `side_m`, `fov`, `width`, `height`)
  - 신호등이 화면 위로 잘리면 `pitch` 를 -6 쪽으로, 보행자가 안 보이면 `yaw_off` 를 10 쪽으로
- 이 카메라는 촬영용일 뿐, Pi 로 가지 않고 판단에도 안 쓴다.

## 2. 촬영 화면 구성 (OBS 한 화면, 1920×1080)

```
┌──────────────────────────────┬──────────────────────┐
│ CARLA 창 (관전자 = 차량 A 뒤쪽  │ 폰 A 화면 (scrcpy)   │
│ 위에서 정차 트럭·횡단보도 보임)  │ 주의/경고/긴급 + 소리 │
├──────────────────────────────┴──────────────────────┤
│ 관제 화면 http://localhost:8000/monitor              │
│ Cam1 오버레이 · Cam2 EMR · 지도(차로·트럭·A·B·보행자) ·  │
│ 차량별 CPA 위험 판정 표(근거) · 앱 전송 로그 · 지연       │
└─────────────────────────────────────────────────────┘
```
- 관전자 시점: `--view approach`(기본, 차량 A 뒤 위) / `top`(위에서) / `none`(직접 조작).
- OBS 오디오에 폰 소리(scrcpy 2.0 이상은 소리도 넘어옴)를 넣으면 경고음까지 담긴다.
- 영상 편집 자막 예: "정차 트럭 때문에 차량 A는 보행자를 못 봄" → "Cam2: 출발 징후(EMR) + 보행 신호 빨강 전환 → 차량 A만 경고" → "Cam1 CPA: tCPA 1.5 s → 긴급" → "차량 B: 충돌 경로 아님 → 경고 안 함".

## 3. 기록 (노트북)

| 무엇 | 어디 |
|---|---|
| 서버: 판정 근거 전부 (차량별 CPA·EMR·ETA·신호·단계) | `logs/server_<시각>/fusion.csv` |
| 서버: 앱으로 보낸 경고 + 지연 + 근거 + 계기(scene/cam2/sim) | `logs/server_<시각>/alerts.csv`, `alert_acks.csv` |
| 서버: CARLA 차량 GNSS·OBU·신호 | `logs/server_<시각>/sim_states.csv` |
| 서버: Cam1·Cam2 프레임별 결과 (Pi 처리 시간) | `logs/server_<시각>/cam1_frames.csv`, `cam2_frames.csv` |
| CARLA 정답·take 요약 (첫 경고 시각, 감속 시각, 안전장치 사용 여부) | `simdemo/logs/sim_<시각>/take01_gt.csv`, `summary.json`, `geometry.json` |
| Pi 오버레이 영상·CSV | `bash tools/sync_pi.sh <Pi계정>@<PiIP>` → `pi_logs/` |

## 4. 발표 질의응답에서 그대로 말할 것

- **실제로 도는 것**: CARLA 가상 카메라 2대 영상 → Pi 5 한 대에서 Cam1·Cam2 실시간 분석 → 노트북 서버의 차량별 CPA 판정(+ Cam2 EMR·보행자 신호) → 실제 폰 앱 수신·표시. 지연은 `alerts.csv` 로 구간별 측정.
- **시뮬레이터 값인 것**: 차량 위치(CARLA GNSS)와 속도(가상 OBU), 보행자 신호(시나리오).
- **연출인 것**: 보행자·차량 움직임 일정. 운전자 반응(경고 → 감속, 긴급 → 급제동 7.5 m/s², 반응 0.3 s)은 앱 경고를 받은 운전자를 흉내 낸 것.
- **안전장치**: 경고가 없거나 늦으면 사고 장면을 막으려고 비상 정지한다. 쓰이면 `summary.json`·`trials.csv` 에 `fallback_no_alert`(시스템 실패) / `fallback_late`(늦음)로 남는다 → 숨기지 말고 그대로 보고.
- **EMR 가중치·임계값**(근접 0.4 + 차도 방향 0.4 + 지속성 0.2, 기준 0.6)과 단계 기준은 팀이 정한 값이고, 현장 데이터로 맞춘 값은 아니다.

## 5. 문제 해결

| 증상 | 해결 |
|---|---|
| `[영상] Cam1 → Pi … 접속 실패` | Pi 에서 `run_pi_sim.sh` 가 떠 있는지, Pi IP, 방화벽 (`sudo ufw allow 8765:8767/tcp` on Pi) |
| `[보정] … 실패(Pi 없음)` | 관제 'Pi 연결' 확인. Pi 의 `--server` 노트북 IP |
| 관제에 Cam1 보행자 점이 엉뚱한 곳 | `--inspect` 그림에서 격자가 도로와 맞는지. 팀 Cam1 위치가 바뀌었으면 `scenario_j189.json` cameras.cam1.source = "auto" |
| Cam2 EMR 가 안 오름 | 관제 Cam2 화면에서 파란 다각형·빨간 경계선 위치 확인. 사람이 작으면 `site_sim.json` cam2.pipeline.min_blob_px ↓ |
| 다른 차·보행자가 장면에 들어옴 | `--clear` |
| `동기 모드` 경고 / 화면이 멈춤 | 팀 Blind 교통 스크립트(blindguard_city_j189_v3.py)를 끄고 다시 |
| CARLA 가 느림 (실제 시간보다 느리게 흐름) | `-quality-level=Low`, 다른 프로그램 끄기. 시나리오 시간은 시뮬 시간 기준이라 결과는 같음 |

## 6. 파일

| 파일 | 내용 |
|---|---|
| `simdemo/carla_demo.py` | CARLA 시나리오: J189 기하 자동 추출, 카메라 2대 → Pi, 보정 자동 적용, 보행자·트럭·차량 A/B·신호, 운전자 반응, 기록 |
| `simdemo/simgeom.py` | 좌표·카메라 투영·Homography·차량 속도 계획 (CARLA 없이 시험됨) |
| `simdemo/scenario_j189.json` | 시나리오 시간표·속도·장면 3가지 |
| `simdemo/site_sim.json` | 서버·Pi 설정 (카메라 = TCP, 판정 규칙) |
| `bgx/fusion.py` | 차량별 CPA 판정 + Cam2 EMR + 보행자 신호 융합 |
| `bgx/netsource.py` | Pi: CARLA 영상 TCP 입력 (`--cam1 tcp:8765`) |
| `server/exp_server.py` | (추가) 차량 출처 `carla`, `/api/sim`, 융합 판정, `fusion.csv`·`sim_states.csv` |
| `server/monitor.html` | (추가) CARLA·보행자 신호 표시, 차량별 판정 근거 표, 차로·대기구역·정차 트럭 지도 |
| `tests/test_sim.py`, `tests/test_sim_server.py`, `tests/run_fake_demo.py` | 시험 (가짜 CARLA: `tests/fakecarla/`) |

## 촬영 구조 (11/24 추가)

영상만 봐도 상황이 읽히도록 **상태 띠(HUD)** 를 CARLA 화면 위에 겹쳐서 찍는다.

- 상태 띠: `bash simdemo/open_hud.sh` → 창 제목 "HUD". 상황 이름 · 보행/차량 신호 · 차량 A 속도와 경고 단계 · 차량 B · 결과 한 줄 · 충돌 표시.
  OBS: 소스 + → 창 캡처(Xcomposite) → 'HUD' → 변환 편집: 위치 0,0 / 경계 상자 1280×140 (안쪽 경계에 맞춤) / CARLA 소스보다 위.
- 도입 전경: `--view chase` 는 차량 A 출발 1초 전까지 높은 시점(횡단보도·트럭·대기구역)을 보여 주다가 추적 시점으로 바뀐다 (`--no-intro` 로 끔).
- 결과: A 가 멈추거나(또는 충돌·통과) 하면 띠에 결과 한 줄이 뜨고 5초 뒤 take 가 끝난다 (`end_after_result_s`).
- 촬영 전 점검: Pi · Cam1/Cam2 분석 · 폰 앱(주행 모드) · 관제 화면 · 상태 띠가 살아 있고 최신이며 최소화되지 않았는지 확인한다.
  ✘ 가 있으면 고치고 다시 실행 (`--no-check` 로 건너뜀, 띠 없이 찍을 때는 `--no-hud`).
- 끝나면 터미널에 `━━ [결과] … ✔ 기대대로 / △ / ✘` 한 줄이 나온다 (기대값은 scenario 의 `check`).
- CASE 2 는 경고 뒤 1.5 m/s² 로 서서히 서서 횡단보도 약 10 m 앞에 멈춘다 (전에는 23 m 앞).

### 촬영 화면 한 장 (/studio) — 권장

`bash simdemo/studio.sh` → 창 제목 "STUDIO" (전체 화면). 상태 띠 · 관제(Cam1·Cam2·지도·Pi 처리) · 경고 기록이 한 장에 들어 있고,
CARLA 와 폰이 들어갈 자리가 비어 있다. OBS 소스는 3개만 쓴다 (아래에서 위 순서).

| 소스 | 종류 | 변환 편집 |
|---|---|---|
| STUDIO | 창 캡처 → STUDIO | 화면에 맞추기 (Ctrl+F) · 맨 아래 |
| CARLA | 창 캡처 → CARLA | 위치 0, 120 · 안쪽 경계에 맞춤 · 1280 × 720 |
| PHONE | 창 캡처 → PHONE | 위치 1300, 170 · 안쪽 경계에 맞춤 · 400 × 880 |

'앱 없음' 대조 상황은 PHONE 소스의 눈 아이콘만 끄면 그 자리에 "앱 없음" 이 나온다 (서버도 그 상황에서는 폰으로 경고를 보내지 않는다).
이 방식에서는 관제 창(/monitor)과 띠 창(/hud)을 따로 띄울 필요가 없다. 촬영 전 점검도 STUDIO 창 하나만 본다.
