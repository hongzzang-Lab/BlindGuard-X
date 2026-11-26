# BlindGuard-X

[![Edge: Raspberry Pi 5](https://img.shields.io/badge/Edge-Raspberry%20Pi%205-238636.svg)](docs/vision-method.md) [![Detector: YOLO26n NCNN 416](https://img.shields.io/badge/Detector-YOLO26n%20NCNN%20416-0969da.svg)](docs/vision-method.md) [![Evaluated: MOT15 · SOMPT22](https://img.shields.io/badge/Evaluated-MOT15%20%C2%B7%20SOMPT22-8250df.svg)](docs/evaluation.md) [![Road validation: not yet](https://img.shields.io/badge/Road%20validation-Not%20yet-6e7781.svg)](#limitations)

#### BlindGuard-X is a roadside edge-AI system that warns the driver who is on a conflict course with a pedestrian the driver cannot see.

<p align="center"><img src="assets/demo/teaser_case1.gif" width="800"/></p>

> *Two seconds before the crosswalk.* A parked truck hides a pedestrian. Two roadside cameras and a Raspberry Pi 5 see them first; the server predicts the conflict and the driver's phone goes from **caution** to **warning** to **danger** before the car reaches the crosswalk. (CARLA demonstration, scenario `case1_blind_spot`; on-screen text is Korean.)

## Demo Links

| Full demo videos | Run it on a laptop | Evaluation | Related work |
|:---:|:---:|:---:|:---:|
| [7 scenarios (mp4)](#visualization-results-in-carla) | [5-minute quick start](#demo) | [MOT15 · SOMPT22](docs/evaluation.md) | [Prior ROI-packing work](docs/related-work.md) |

## Abstract

Drivers cannot react to a pedestrian they cannot see. BlindGuard-X places the sensing at the roadside instead of in the car: a fixed camera watches the crosswalk and a second one watches the waiting zone, and a Raspberry Pi 5 turns both streams into pedestrian positions, velocities and an early-departure cue. A server combines these with each vehicle's position and speed, predicts the closest point of approach (CPA) per vehicle, and sends a staged alert—caution, warning, danger—over WebSocket to **only the vehicle on a conflict course**. Because the Pi has no GPU, the detector runs at a small 416×416 input, where distant pedestrians become too small to find. We therefore select regions from optical flow and track memory and pack them into a single 416×416 canvas for one inference (ROI Packing). On two public tracking datasets this raised Recall and IDF1 over full-frame input at about 25 FPS on the Pi 5, while running about 18–20 % slower than plain full-frame detection. The system has been exercised in CARLA and on public data, not on a public road.

<p align="center"><img src="assets/poster/fig1-concept.png" width="640"/></p>

> Concept (poster figure 1; labels in Korean). ① A parked vehicle hides the pedestrian from the driver. ② A roadside camera keeps watching the crosswalk and the waiting zone. ③ When a risk is detected, only the approaching vehicle on that side is alerted.

## News

- (2026.11.26) Evaluation on MOT15 and SOMPT22 added — see [docs/evaluation.md](docs/evaluation.md).
- (2026.11.24) Seven CARLA scenarios recorded with a real Raspberry Pi 5 and phone app in the loop.
- (2026.11) Code released. Prior work on ROI packing is listed in [docs/related-work.md](docs/related-work.md); no novelty is claimed for packing itself.

## Pipeline

<p align="center"><img src="assets/poster/fig2-architecture.png" width="900"/></p>

> System architecture (poster figure 2; labels in Korean). CARLA supplies the two camera streams, pedestrian-signal state, and each vehicle's GNSS position and virtual OBU speed. The Raspberry Pi 5 analyses Cam1 (blue) and Cam2 (green); the server makes the per-vehicle CPA decision and sends caution / warning / danger to the app of the vehicle concerned.

<p align="center"><img src="assets/roi-packing.svg" width="900"/></p>

**Cam1 (crosswalk):** Optical Flow + Track Memory → Adaptive ROI → ROI Packing (416×416) → YOLO26n-NCNN → ByteTrack → Homography<br>
**Cam2 (waiting zone):** Polygon ROI → MOG2 → Sparse LK → boundary approach → EMR score<br>
**Server:** per-vehicle CPA (dCPA ≤ 2 m, 0 ≤ tCPA ≤ 3 s, constant velocity) + Cam2 early cue + signal state → caution / warning / danger

Details: [system flow](docs/system-flow.md) · [vision method](docs/vision-method.md)

<p align="center"><img src="assets/poster/app-driving.png" width="200"/>&nbsp;&nbsp;&nbsp;<img src="assets/poster/app-danger.png" width="200"/></p>

> Driver app: normal driving (left) and a danger alert (right). The radar shows where the pedestrian is relative to the vehicle's heading.

<p align="center"><img src="assets/demo/screen_case1_danger.jpg" width="800"/></p>

> Recording layout. Top: scenario, signals, and the alert state of vehicles A and B. Centre: CARLA view and the driver's real phone screen. Bottom: Cam1 and Cam2 as analysed on the Raspberry Pi 5, a top-down map, Pi processing rate, and delivery latency.

## Benchmark Performance

Same detector, tracker and settings; only the image region given to the detector changes. Raspberry Pi 5, YOLO26n-NCNN at 416 input, ByteTrack. Metrics are self-computed on training sequences with public ground truth—**not leaderboard results**.

### Results on MOT15 (six static-camera sequences, 2,315 frames)

| Input strategy | MOTA | IDF1 | Recall | Precision | IDs | FPS |
|----------------|------|------|--------|-----------|-----|-----|
| FULL | 52.4 | 52.9 | 62.6 | **86.9** | 153 | **30.6** |
| ADAPTIVE | 54.2 | 55.4 | 65.3 | 86.3 | 148 | 25.3 |
| OURS (per-ROI inference) | 53.1 | **59.8** | **72.9** | 79.3 | **137** | 11.5 |
| OURS_PACKED | **56.0** | 59.5 | 67.4 | 86.4 | 141 | 25.2 |

### Results on SOMPT22 (nine training sequences, first 10 s each, 2,700 frames)

| Input strategy | MOTA | IDF1 | Recall | Precision | IDs | FPS |
|----------------|------|------|--------|-----------|-----|-----|
| FULL | 18.0 | 27.9 | 19.5 | **93.6** | 182 | **30.2** |
| ADAPTIVE | 19.6 | 30.0 | 21.4 | 92.9 | **162** | 24.8 |
| OURS (per-ROI inference) | **29.3** | **42.6** | **33.0** | 90.6 | 295 | 8.7 |
| OURS_PACKED | 23.5 | 35.1 | 26.1 | 91.9 | 243 | 24.3 |

Per-person ROI raised Recall and IDF1 over full-frame input on both datasets. Packing kept part of that gain at 2.2–2.8× the speed of per-ROI inference. **Full-frame input remained the fastest.** Per-sequence tables and ten limitations are in [docs/evaluation.md](docs/evaluation.md).

### Visualization results in CARLA

Each clip shows the moment of decision. Click a title for the full video.

| | |
|:---:|:---:|
| <img src="assets/demo/case1_blind_spot.gif" width="400"/> | <img src="assets/demo/case1_no_app.gif" width="400"/> |
| [**CASE 1 · Blind spot**](assets/videos/case1_blind_spot.mp4)<br>Caution → warning → danger; stops 2 m before the pedestrian | [**CASE 1 control · No app**](assets/videos/case1_no_app.mp4)<br>Same scene without the alert: collision |
| <img src="assets/demo/case2_signal_change.gif" width="400"/> | <img src="assets/demo/case3_red_light.gif" width="400"/> |
| [**CASE 2 · Signal change**](assets/videos/case2_signal_change.mp4)<br>Pedestrian runs just after red; warning, gradual stop | [**CASE 3 · Red-light runner**](assets/videos/case3_red_light.mp4)<br>Only the vehicle running the light is alerted |
| <img src="assets/demo/case4_speeding.gif" width="400"/> | <img src="assets/demo/case5_selective.gif" width="400"/> |
| [**CASE 4 · Speeding**](assets/videos/case4_speeding.mp4)<br>50 km/h approach; alerted earlier, stops in time | [**CASE 5 · Selective alert**](assets/videos/case5_selective.mp4)<br>Two vehicles in adjacent lanes; only the one still in conflict is alerted |
| <img src="assets/demo/case0_wait_only.gif" width="400"/> | |
| [**CASE 0 control · Loitering**](assets/videos/case0_wait_only.mp4)<br>Pedestrian only waits; no false alert | |

In these recordings the camera frames are analysed on a real Raspberry Pi 5 and the alert is shown on a real phone. Vehicle position and speed, signals, and the scripted motion come from the simulator.

## Installation

### 1. Laptop only (no CARLA, no Pi)

```shell
git clone https://github.com/hongzzang-Lab/BlindGuard-X.git
cd BlindGuard-X
pip3 install starlette "uvicorn[standard]" numpy pandas opencv-python scipy websockets
```

### 2. Raspberry Pi 5

```shell
bash tools/deploy_pi.sh <pi-user>@<pi-ip>     # on the laptop: copies the code to ~/realtest on the Pi
cd ~/realtest && bash pi/setup_pi.sh          # on the Pi: venv, ultralytics, ncnn, YOLO26n → NCNN 416
```

Addresses in this repository are placeholders; use your own. CARLA 0.9.16 is needed for the full simulation demo.

## Demo

Watch the full decision chain on a laptop with a fake CARLA and the real server code:

```shell
python3 tests/test_core.py
FAKE_RT=0.6 PYTHONPATH=tests/fakecarla python3 tests/run_fake_demo.py case1_blind_spot
# → vehicle A: caution → warning → danger, stops 2 m before the pedestrian · vehicle B: no alert
```

With CARLA and a Pi:

```shell
bash simdemo/start_sim_server.sh                                   # laptop
cd ~/realtest && bash simdemo/run_pi_sim.sh <laptop-ip>            # Pi
bash simdemo/run_demo.sh --pi <pi-ip> --variant case1_blind_spot --takes 1 --view chase --no-check   # laptop
```

Scenarios: `case1_blind_spot`, `case2_signal_change`, `case3_red_light`, `case4_speeding`, `case5_selective`, `case1_no_app`, `wait_only`. Step-by-step instructions are in the [setup guide](docs/setup.md).

## Evaluation

```shell
bash bench/send_to_pi.sh <pi-user>@<pi-ip>                         # laptop
cd ~/realtest && bash bench/run_on_pi.sh                           # Pi: MOT15 (downloaded by the script)
cd ~/realtest && bash bench/run_on_pi.sh --bench sompt22           # Pi: SOMPT22 (obtain the data yourself)
```

Datasets and model weights are not included. See [bench/README.md](bench/README.md) (Korean).

## Limitations

- **Not validated on public roads.** Simulation, a limited webcam test and public datasets only. This is a prototype, not a safety system.
- **ROI Packing is not faster than plain full-frame detection** (about 25 vs 30 FPS). It trades speed for accuracy.
- **Crowded scenes defeat packing**, and on one SOMPT22 sequence the ROI strategies were clearly worse than full-frame.
- **A larger-input full-frame detector was not measured**, so whether packing is the cheaper route to a given accuracy is open.
- **The Cam2 early cue worked in simulation but did not trigger in our real-camera test.** Its weights and threshold were set by hand.
- **CPA assumes constant velocity**, and vehicle position depends on phone GPS (or simulated GNSS).
- Night scenes were not detected reliably.

## Acknowledgement

This project builds on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [NCNN](https://github.com/Tencent/ncnn), the [ByteTrack](https://github.com/ifzhang/ByteTrack) association method, and the [CARLA](https://carla.org/) simulator, and is evaluated on [MOT15](https://motchallenge.net/) and [SOMPT22](https://sompt22.github.io/). ROI packing was introduced by earlier work such as [Pack and Detect](https://arxiv.org/abs/1809.01701); see [related work](docs/related-work.md). The structure of this README follows common practice in open-source tracking repositories such as ByteTrack and OC-SORT.

<details>
<summary><strong>한국어 요약</strong></summary>

사각지대 횡단보도에서 보행자 충돌 위험을 운전자에게 미리 알리는 도로변 엣지 AI 시스템. 고정 카메라 2대와 Raspberry Pi 5가 보행자를 검출·추적하고(Optical Flow + Track Memory로 ROI를 만들어 416×416 한 장에 Packing → YOLO26n-NCNN → ByteTrack), 서버가 차량별 CPA로 위험을 판정해 해당 차량의 폰 앱에만 주의·경고·긴급을 보낸다. CARLA 시연 7가지 상황과 공개 데이터셋 2종(MOT15, SOMPT22) 평가까지 진행했으며, 실제 도로 검증은 하지 않았다. 코드 주석과 일부 실행 문서, 영상 속 글자는 한국어다.

</details>
