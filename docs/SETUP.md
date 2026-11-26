# 직접 돌려 보기 (SET UP)

시연 영상에 나온 장면을 직접 재현하는 방법이다. 가진 장비에 따라 세 단계로 나눴다.

| 단계 | 필요한 것 | 걸리는 시간 | 볼 수 있는 것 |
|---|---|---|---|
| [1. 노트북만](#1-노트북만-5분) | Python 3 | 5분 | 경고 판정 흐름 전체 (주의 → 경고 → 긴급), 관제 화면 |
| [2. CARLA + Raspberry Pi 5](#2-carla--raspberry-pi-5-시연-영상과-같은-구성) | Ubuntu 노트북, CARLA 0.9.16, Pi 5 | 1~2시간 | 시연 영상과 같은 장면. Pi가 실제로 영상을 분석 |
| [3. 실물 카메라](#3-실물-카메라) | Pi 5, USB 웹캠 2대, 안드로이드 폰 | 반나절 | 실제 사람을 찍어 폰으로 경고 받기 |

> 확인 범위: 1단계는 깨끗한 환경에서 다시 돌려 확인했다. 2·3단계는 우리 팀 장비(Ubuntu 노트북, CARLA 0.9.16, Raspberry Pi 5)에서만 돌려 봤다. 다른 환경에서는 손볼 곳이 있을 수 있다.

---

## 1. 노트북만 (5분)

CARLA도 Pi도 없이, **가짜 CARLA**(`tests/fakecarla/`)와 실제 서버 코드로 시나리오를 돌린다. 보행자 인식은 정답 위치를 넣어 주는 가짜이고, 차량별 CPA 판정과 경고 단계는 실제 서버 코드가 한다.

```bash
git clone https://github.com/hongzzang-Lab/BlindGuard-X.git
cd BlindGuard-X
pip install starlette "uvicorn[standard]" numpy pandas opencv-python scipy websockets
```

### 1-1. 시험이 통과하는지

```bash
python3 tests/test_core.py          # 좌표·칼만·CPA 계산 → 줄마다 ok
python3 tests/test_sim.py           # 시뮬 기하·영상 전송 → 줄마다 ok
python3 tests/test_sim_server.py    # 서버 흐름: A 경고→긴급, B 없음 → ok test_sim_server
```

### 1-2. 시나리오 돌려 보기

```bash
FAKE_RT=0.6 PYTHONPATH=tests/fakecarla python3 tests/run_fake_demo.py case1_blind_spot
```

1분쯤 뒤 터미널에 이렇게 나오면 성공이다.

```
[경고] 차량 A None→caution
[경고] 차량 A caution→warning
[경고] 차량 A warning→danger
━━ [결과] CASE 1 사각지대 무단횡단 │ A 최고 단계 긴급 │ 경고 받고 보행자 2 m 앞에서 정지 · 차량 B 경고 없음 │ ✔ 기대대로
```

도는 동안 브라우저에서 **http://127.0.0.1:9765/monitor** 를 열면 관제 화면이 보인다. Pi가 없으니 카메라 영상 칸은 비어 있다.

`FAKE_RT`는 진행 배속이다. 판정이 실제 시계로 속도를 재기 때문에 **너무 빠르게(예: 4) 돌리면 경고가 늦게 나와 실패로 끝난다.** 0.6을 권한다.

### 다른 상황

맨 뒤 이름만 바꾸면 된다.

| 이름 | 상황 | 기대 결과 (차량 A) |
|---|---|---|
| `case1_blind_spot` | 정차 트럭에 가린 무단횡단 | 주의 → 경고 → 긴급, 급정지 |
| `case2_signal_change` | 보행 신호가 빨강으로 바뀐 직후 뛰어 건넘 | 경고, 서서히 정지 |
| `case3_red_light` | 초록불 횡단 중 신호위반 차량 | 주의 → 경고, 정지 |
| `case4_speeding` | 50 km/h 과속 차량 | 주의 → 경고, 정지 |
| `case5_selective` | 같은 방향 두 차 중 부딪힐 차만 | 옆 차로 A만 경고 |
| `case1_no_app` | CASE 1 대조: 운전자에게 앱 없음 | 충돌 |
| `wait_only` | 보행자가 서성이기만 함 | 경고 없음 (오경고 확인) |

---

## 2. CARLA + Raspberry Pi 5 (시연 영상과 같은 구성)

CARLA의 가상 카메라 2대 영상을 **실제 Pi로 보내**, Pi가 YOLO·추적·Cam2 분석을 하고, 노트북 서버가 판정한다. 폰이 없어도 관제 화면에서 경고를 볼 수 있다.

```
노트북 (CARLA + 서버) ──TCP 영상──▶ Raspberry Pi 5 ──WebSocket 결과──▶ 노트북 서버 ──▶ (폰 앱)
```

### 준비물

- Ubuntu 노트북: CARLA 0.9.16과 그 Python API(`import carla`가 되는 파이썬)
- Raspberry Pi 5: 노트북과 **같은 네트워크**
- (선택) 안드로이드 폰과 운전자 앱. 앱 전체 소스는 이 저장소에 없고 수정분만 `app_patch/`에 있다

아래에서 `<노트북IP>`, `<PiIP>`, `<Pi계정>`은 자기 값으로 바꾼다. IP는 각 기기에서 `hostname -I`로 본다.

### 2-1. Pi 준비 (한 번)

노트북에서 코드를 Pi의 `~/realtest`로 복사한다. 스크립트들이 이 폴더 이름을 기준으로 쓴다.

```bash
bash tools/deploy_pi.sh <Pi계정>@<PiIP>
```

Pi에서 (인터넷이 되는 상태로):

```bash
cd ~/realtest && bash pi/setup_pi.sh
```

가상환경 `~/bgx-venv`를 만들고, `ultralytics`·`ncnn`을 설치하고, YOLO26n을 NCNN 416으로 내보내 `models/yolo26n_ncnn_model`에 둔다. 마지막 점검에서 모델 속도가 나오면 끝이다.

### 2-2. 실행 (매번)

| # | 어디서 | 명령 | 확인 |
|---|---|---|---|
| 1 | 노트북 | CARLA 실행. 지도는 `Town10HD_Opt` | CARLA 창 |
| 2 | 노트북 | `bash simdemo/run_demo.sh --inspect` (처음 한 번) | `simdemo/logs/sim_*/check_cam1.jpg`에서 빨간 연석선이 실제 도로와 맞는지 |
| 3 | 노트북 | `bash simdemo/start_sim_server.sh` | 브라우저 `http://localhost:8000/monitor` |
| 4 | Pi | `cd ~/realtest && bash simdemo/run_pi_sim.sh <노트북IP>` | `CARLA 영상 기다리는 중`, 관제 화면에 'Pi 연결' |
| 5 | 노트북 | `bash simdemo/run_demo.sh --pi <PiIP> --variant case1_blind_spot --takes 1 --view chase --no-check` | 터미널에 `[결과] … ✔ 기대대로` |

- `run_demo.sh`는 `carla` 모듈이 있는 파이썬을 써야 한다. 기본은 `python3`이고, 다른 파이썬을 쓰려면 `BLIND_PY=/경로/python bash simdemo/run_demo.sh …`.
- `--no-check`는 폰 앱·촬영 화면이 떠 있는지 보는 사전 점검을 건너뛴다. 폰까지 연결했다면 빼도 된다.
- 다른 차나 보행자가 장면에 있으면 `--clear`를 붙인다.
- 이 스크립트가 CARLA 동기 모드의 주인이 된다. 다른 교통 스크립트는 꺼 둔다.
- 관제 화면에는 Cam1·Cam2 분석 영상, 지도, 차량별 판정 근거, 경고 기록, 지연이 나온다.

### 2-3. 폰 앱까지 (선택)

앱 소스가 있다면 노트북 주소로 다시 빌드한다. `app_patch/patched/`의 세 파일(`lib/config.dart`, `lib/services/location_service.dart`, `lib/services/server_link.dart`)을 덮어쓴 뒤:

```bash
flutter build apk --release --dart-define=SERVER_URL=ws://<노트북IP>:8000/ws
flutter install
```

앱에서 주행을 시작하면 관제 화면에 '폰 A 주행 중'이 뜨고, 경고가 폰으로 간다.

서버 주소는 각자 정한다. 이 저장소의 앱 설정에는 팀 서버 주소가 없고 자리 표시(`YOUR-SERVER.example.com`)만 있다. 같은 네트워크의 노트북 서버를 쓰면 위처럼 `SERVER_URL`만 주면 되고, 클라우드에 서버를 올려 쓰려면 `server/exp_server.py`를 자기 계정의 호스팅에 배포한 뒤 그 주소를 넣는다.

### 2-4. 촬영 화면

시연 영상의 화면 구성(상태 띠, 관제, 경고 기록을 한 장에)은 `bash simdemo/studio.sh`로 연다. OBS 배치는 [`simdemo/README.md`](../simdemo/README.md)의 '촬영 구조'에 있다.

### 자주 막히는 곳

| 증상 | 해결 |
|---|---|
| `Cam1 → Pi … 접속 실패` | Pi에서 4번이 떠 있는지, Pi IP, 방화벽 (Pi에서 `sudo ufw allow 8765:8767/tcp`) |
| 관제 화면의 보행자 점이 엉뚱한 곳 | 2번의 확인 그림에서 격자가 도로와 맞는지 |
| CARLA가 느림 | `-quality-level=Low`. 시나리오는 시뮬 시간 기준이라 결과는 같다 |
| `안전장치로 비상 정지 (경고가 늦음)` | 경고가 제때 안 나온 것. 관제 화면에서 Pi 처리 속도와 Cam1 검출을 확인 |

더 자세한 설명은 [`simdemo/README.md`](../simdemo/README.md).

---

## 3. 실물 카메라

USB 웹캠 2대를 Pi에 꽂고, 실제 사람이 보행자와 차량(폰 GPS) 역할을 한다. 바닥 4점 보정, GPS 기준점 잡기, 시험 순서가 길어서 별도 문서로 뒀다.

- 전체 순서: [`실험_실행_메모.md`](실험_실행_메모.md)
- 촬영만: [`../shoot/촬영방법.md`](../shoot/촬영방법.md)

사람이 차량 역할을 하므로 **실제 차량 없이 통제된 공간에서** 한다.

---

## 공개 데이터셋 평가 다시 돌리기

[`../bench/README.md`](../bench/README.md) 참고. Pi 5에서 데이터셋마다 약 15분 걸렸다.
