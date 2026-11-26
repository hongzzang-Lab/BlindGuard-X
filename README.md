# BlindGuard-X — 2초 전 횡단보도

사각지대 횡단보도에서 보행자와의 충돌 위험을 **운전자에게 미리 알리는** 도로변 엣지 AI 시스템.
도로변 고정 카메라 2대와 Raspberry Pi 5가 보행자를 보고, 서버가 차량별로 충돌을 예측해, 위험한 차량의 폰 앱에만 경고를 보낸다.

> *Roadside edge-AI system that warns drivers before a pedestrian conflict at blind-spot crosswalks. Two fixed cameras + a Raspberry Pi 5 (YOLO26n-NCNN with ROI Packing, ByteTrack) estimate pedestrian position and velocity; a server runs per-vehicle CPA prediction and pushes alerts to the driver's phone app over WebSocket.*

팀 온곁 · 2026년 대회 출품작. 이 저장소는 프로토타입 이후 **실제로 구현하고 시연·평가에 쓴 코드**다.

---

## 구조

```
 [Cam1 횡단보도] ─┐
                  ├─ Raspberry Pi 5 (pi/edge.py)
 [Cam2 대기구역] ─┘    Cam1: Optical Flow + Track Memory → Adaptive ROI → ROI Packing(416×416)
                              → YOLO26n-NCNN → ByteTrack → Homography(바닥 좌표 m)
                       Cam2: Polygon ROI → MOG2 → Sparse LK → 경계 접근 분석 → EMR Score
                          │  보행자 위치·속도, 출발 징후
                          ▼ WebSocket
                   노트북 서버 (server/exp_server.py)
                       차량 위치·속도 (폰 GPS 또는 CARLA) → 차량별 CPA 위험 판정
                       + Cam2 조기 징후 + 보행 신호
                          │  주의 / 경고 / 긴급
                          ▼ WebSocket
                   위험한 차량의 운전자 폰 앱
```

### 판정 규칙 (`bgx/fusion.py`, `bgx/risk.py`)

| 단계 | 조건 |
|---|---|
| 주의 | Cam2 EMR ≥ 0.6 (출발 징후) + 차량이 대기구역 쪽 차로에서 5초 안에 횡단보도 도착 |
| 경고 | Cam1 CPA: dCPA ≤ 2 m 이고 0 ≤ tCPA ≤ 3 s (등속 가정), 또는 신호 전환 구간의 '주의' |
| 긴급 | CPA 이면서 tCPA ≤ 1.5 s |

같은 장면에 차량이 여럿 있어도 조건에 걸린 차량에만 보낸다.

### Cam1 Vision — ROI Packing (`bgx/cam1_pipeline.py`)

Pi 5에는 GPU가 없어 YOLO 입력을 416으로 작게 쓴다. 그러면 멀리 있는 보행자가 작게 찍혀 놓치기 쉽다. 입력을 키우는 대신 **사람이 있을 만한 영역만 잘라 416×416 한 장에 모아** YOLO를 한 번 돌린다.

ROI는 두 신호를 합쳐 만든다.

| 신호 | 잡는 대상 |
|---|---|
| Optical Flow (Farneback, 폭 160으로 줄여 계산) | 새로 들어오는 사람, 움직이는 사람 |
| Track Memory (최근 1초 안에 추적하던 사람의 예상 위치) | 추적 중에 멈췄거나 검출이 잠깐 끊긴 사람 |

둘 다 놓치는 정지 보행자는 N프레임마다 한 번 하는 전체 화면 검사로 보완한다. ROI가 한 장에 다 안 들어가면 그 프레임은 전체 화면으로 처리한다.

비교용으로 네 가지 모드가 있다: `full` · `adaptive` · `crops`(ROI별 추론) · `packed`(최종 구조).

---

## 폴더

| 폴더 | 내용 |
|---|---|
| `bgx/` | 핵심 라이브러리. Cam1·Cam2 파이프라인, ByteTrack, Homography, GPS→바닥 좌표, CPA 판정·융합 |
| `pi/` | Raspberry Pi 5 실행 프로그램 (`edge.py`), 설치·점검·보정 스크립트 |
| `server/` | 노트북 서버 (Starlette)와 웹 화면: 조작 `/control`, 관제 `/monitor`, 촬영 `/studio`, HUD, Vision 시험 |
| `simdemo/` | CARLA 0.9.16 Town10HD_Opt J189 시연. 가상 카메라 영상을 실제 Pi로 보내 같은 코드로 처리 |
| `bench/` | 공개 데이터셋(MOT15, SOMPT22) 평가 스크립트 |
| `app_patch/` | 운전자 폰 앱(Flutter)의 수정분. 앱 전체 소스는 이 저장소에 없다 |
| `shoot/`, `analysis/`, `tools/` | 실물 촬영, 기록 분석, 배포·모의 실행 도구 |
| `tests/` | 가짜 CARLA·가짜 WebSocket으로 서버와 시연 흐름을 시험 |
| `calib/`, `calib_sim/`, `site.json`, `scenarios.json` | 카메라 보정값과 현장·시나리오 설정 |
| `docs/` | 평가 결과, 선행 연구, 실험 실행 메모 |

---

## 평가 결과 요약

공개 보행자 추적 데이터셋 2종의 고정 카메라 구간에서, 모델·추적기·설정은 그대로 두고 YOLO에 넣는 화면만 바꿔 비교했다. Raspberry Pi 5, YOLO26n-NCNN 416, ByteTrack. **순위표 제출이 아닌 자체 계산**이다.

**MOT15** — 고정 카메라 6개 구간 전체 (2,315프레임)

| 방식 | MOTA | IDF1 | Recall | Precision | FPS |
|---|---|---|---|---|---|
| FULL | 52.4 | 52.9 | 62.6 | **86.9** | **30.6** |
| ADAPTIVE | 54.2 | 55.4 | 65.3 | 86.3 | 25.3 |
| OURS (ROI별 추론) | 53.1 | **59.8** | **72.9** | 79.3 | 11.5 |
| OURS_PACKED | **56.0** | 59.5 | 67.4 | 86.4 | 25.2 |

**SOMPT22** — 학습용 9개 구간, 구간당 앞 10초 (2,700프레임)

| 방식 | MOTA | IDF1 | Recall | Precision | FPS |
|---|---|---|---|---|---|
| FULL | 18.0 | 27.9 | 19.5 | **93.6** | **30.2** |
| ADAPTIVE | 19.6 | 30.0 | 21.4 | 92.9 | 24.8 |
| OURS (ROI별 추론) | **29.3** | **42.6** | **33.0** | 90.6 | 8.7 |
| OURS_PACKED | 23.5 | 35.1 | 26.1 | 91.9 | 24.3 |

- 두 데이터셋 모두에서 사람별 ROI를 쓰면 FULL 대비 Recall·IDF1이 올랐다.
- Packing은 ROI별 추론보다 2.2~2.8배 빠르고, FULL 대비 MOTA가 3.6~5.5 높았다.
- **속도는 FULL이 가장 빠르다.** 이 방식으로 얻는 것은 속도가 아니라 정확도다.
- SOMPT22의 한 구간에서는 FULL보다 나빴고, 사람이 매우 많은 구간에서는 Packing이 대부분 전체 화면으로 되돌아갔다.

구간별 표와 한계 전체는 [`docs/benchmark.md`](docs/benchmark.md), 선행 연구는 [`docs/related_work.md`](docs/related_work.md).

### 그 밖의 측정

| 무엇 | 결과 | 조건 |
|---|---|---|
| CARLA 영상 (포스터 표) | OURS_PACKED 15.41 FPS · Recall 0.6429 · ID 변경 0 / ADAPTIVE 21.14 · 0.6883 · 0 / FIXED 31.83 · 0.3636 · 1 | 240프레임 구간 하나 |
| 실물 Cam1 (웹캠 + Pi 5) | 검출률 88.5%, ID 변경 0, 약 10 FPS | 실물 촬영 영상 하나 |
| 실물 Cam2 | 출발 징후를 감지하지 못함 | 실물 시험 1회 |

---

## 직접 돌려 보기

**[`docs/SETUP.md`](docs/SETUP.md)** 에 장비별 세 단계로 정리했다. 노트북만 있어도 5분이면 경고 판정 흐름을 볼 수 있다.

| 하려는 것 | 문서 |
|---|---|
| **처음 돌려 보기 (노트북만 / CARLA + Pi / 실물)** | [`docs/SETUP.md`](docs/SETUP.md) |
| 실물(웹캠) 시험 전체 순서 | [`docs/실험_실행_메모.md`](docs/실험_실행_메모.md) — 팀이 쓰던 메모라 경로와 주소는 우리 장비 기준 |
| CARLA 시연 | [`simdemo/README.md`](simdemo/README.md) |
| 공개 데이터셋 평가 | [`bench/README.md`](bench/README.md) |
| 실물 촬영 | [`shoot/촬영방법.md`](shoot/촬영방법.md) |

빠른 확인 (노트북, CARLA·Pi 없이):

```bash
pip install starlette "uvicorn[standard]" numpy pandas opencv-python scipy websockets
python3 tests/test_core.py
FAKE_RT=0.6 PYTHONPATH=tests/fakecarla python3 tests/run_fake_demo.py case1_blind_spot
# → [결과] CASE 1 사각지대 무단횡단 │ A 최고 단계 긴급 │ … ✔ 기대대로
```

필요한 것

- Raspberry Pi 5: `ultralytics`, `ncnn`, `websockets`, OpenCV (`pi/setup_pi.sh`가 설치하고 YOLO26n을 NCNN 416으로 내보낸다)
- 노트북: Python 3, `starlette`, `uvicorn`, `numpy`, `pandas`, `opencv-python`, `scipy`
- CARLA 시연: CARLA 0.9.16

## 저장소에 없는 것

- YOLO 모델 파일 (`models/`) — `pi/setup_pi.sh`로 만든다.
- 벤치마크 데이터 — MOT15는 `bench/mot_bench.py`가 내려받는다. SOMPT22는 배포처에서 직접 받아야 한다(비상업 연구용).
- 촬영 영상과 실험 기록.
- 운전자 폰 앱 전체 소스 (`app_patch/`에 수정분만 있음).

## 한계

- 실제 도로에서 검증하지 않았다. 시뮬레이션, 웹캠 실물 시험, 공개 데이터셋 평가까지다.
- 차량 위치는 폰 GPS(또는 시뮬의 GNSS)에 의존한다. 실제 GPS 오차가 경고에 주는 영향은 충분히 검증하지 못했다.
- CPA 판정은 등속 가정이다.
- Cam2 EMR Score 식은 임시이며 실물에서 동작을 확인하지 못했다.
- 야간 장면은 검출을 확보하지 못했다.
