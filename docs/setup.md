# Run It

Three ways to reproduce what the demonstration shows, depending on the equipment at hand.

| Level | You need | Time | What you see |
|---|---|---|---|
| [1. Laptop only](#1-laptop-only-5-minutes) | Python 3 | 5 min | The full decision chain (caution → warning → danger) and the monitor page |
| [2. CARLA + Raspberry Pi 5](#2-carla--raspberry-pi-5) | Ubuntu laptop, CARLA 0.9.16, Pi 5 | 1–2 h | The same scenes as the demo, with a real Pi analysing the frames |
| [3. Real cameras](#3-real-cameras) | Pi 5, two USB webcams, an Android phone | Half a day | Real people in front of the cameras, alerts on a phone |

> **Verified scope:** Level 1 was re-run from a clean checkout. Levels 2 and 3 have only been run on the team's own equipment (Ubuntu laptop, CARLA 0.9.16, Raspberry Pi 5); other environments may need adjustments.

Addresses in this repository are placeholders. Replace `<laptop-ip>`, `<pi-ip>` and `<pi-user>` with your own (`hostname -I` shows a machine's address). Some Korean run notes use `<노트북IP>`, `<PiIP>` and `<Pi계정>` for the same things.

---

## 1. Laptop only (5 minutes)

No CARLA and no Pi. A **fake CARLA** (`tests/fakecarla/`) drives the real server code. Pedestrian perception is replaced by ground-truth positions; the per-vehicle CPA decision and alert levels come from the actual server.

```bash
git clone https://github.com/hongzzang-Lab/BlindGuard-X.git
cd BlindGuard-X
pip install starlette "uvicorn[standard]" numpy pandas opencv-python scipy websockets
```

### 1-1. Check that the tests pass

```bash
python3 tests/test_core.py          # coordinates, Kalman filter, CPA → "ok" per line
python3 tests/test_sim.py           # simulation geometry, frame transport → "ok" per line
python3 tests/test_sim_server.py    # server flow: A warning → danger, B none → "ok test_sim_server"
```

### 1-2. Run a scenario

```bash
FAKE_RT=0.6 PYTHONPATH=tests/fakecarla python3 tests/run_fake_demo.py case1_blind_spot
```

After about a minute the terminal shows the alert levels for vehicle A and a result line (program output is in Korean):

```
[경고] 차량 A None→caution
[경고] 차량 A caution→warning
[경고] 차량 A warning→danger
━━ [결과] CASE 1 … │ A 최고 단계 긴급 │ 경고 받고 보행자 2 m 앞에서 정지 · 차량 B 경고 없음 │ ✔ 기대대로
```

That is: vehicle A reached *danger*, stopped 2 m before the pedestrian, vehicle B received no alert, as expected.

While it runs, open **http://127.0.0.1:9765/monitor** to see the monitor page. Camera panes stay empty because no Pi is connected.

`FAKE_RT` is the playback speed. The decision measures speed against the real clock, so **running much faster (for example 4) makes alerts arrive late and the take ends as a failure.** 0.6 is recommended.

### Other scenarios

Change the last argument.

| Name | Situation | Expected for vehicle A |
|---|---|---|
| `case1_blind_spot` | Pedestrian steps out from behind a parked truck | Caution → Warning → Danger, hard stop |
| `case2_signal_change` | Pedestrian runs just after the signal turns red | Warning, gradual stop |
| `case3_red_light` | Vehicle runs a red light during a normal crossing | Caution → Warning, stop |
| `case4_speeding` | Vehicle approaches at 50 km/h | Caution → Warning, stop |
| `case5_selective` | Two vehicles in adjacent lanes | Alert to the conflicting vehicle only |
| `case1_no_app` | Control: case 1 without the app | Collision |
| `wait_only` | Control: pedestrian only loiters | No alert |

---

## 2. CARLA + Raspberry Pi 5

Two virtual CARLA cameras are streamed to a **real Pi**, which runs detection, tracking and the Cam2 analysis; the laptop server decides. A phone is optional—the monitor page shows the alerts.

```
Laptop (CARLA + server) ──TCP frames──▶ Raspberry Pi 5 ──WebSocket results──▶ Laptop server ──▶ (phone app)
```

### Requirements

- Ubuntu laptop with CARLA 0.9.16 and its Python API (a Python where `import carla` works)
- Raspberry Pi 5 on the **same network** as the laptop
- Optional: an Android phone with the driver app. The full app is not in this repository; only the changed files are in `app_patch/`

### 2-1. Prepare the Pi (once)

From the laptop, copy the code to `~/realtest` on the Pi. The scripts assume this folder name.

```bash
bash tools/deploy_pi.sh <pi-user>@<pi-ip>
```

On the Pi, with internet access:

```bash
cd ~/realtest && bash pi/setup_pi.sh
```

This creates the virtual environment `~/bgx-venv`, installs `ultralytics` and `ncnn`, and exports YOLO26n to NCNN at 416 into `models/yolo26n_ncnn_model`. It ends with a check that prints the model speed.

### 2-2. Run (every time)

| # | Where | Command | Check |
|---|---|---|---|
| 1 | Laptop | Start CARLA with the `Town10HD_Opt` map | CARLA window |
| 2 | Laptop | `bash simdemo/run_demo.sh --inspect` (first time only) | In `simdemo/logs/sim_*/check_cam1.jpg` the red kerb line matches the road |
| 3 | Laptop | `bash simdemo/start_sim_server.sh` | `http://localhost:8000/monitor` opens |
| 4 | Pi | `cd ~/realtest && bash simdemo/run_pi_sim.sh <laptop-ip>` | The Pi waits for CARLA frames; the monitor shows the Pi as connected |
| 5 | Laptop | `bash simdemo/run_demo.sh --pi <pi-ip> --variant case1_blind_spot --takes 1 --view chase --no-check` | A result line ending in `✔ 기대대로` (as expected) |

- `run_demo.sh` must use a Python that has the `carla` module. It defaults to `python3`; override with `BLIND_PY=/path/to/python bash simdemo/run_demo.sh …`.
- `--no-check` skips the pre-flight check for the phone app and the recording views. Drop it once a phone is connected.
- Add `--clear` if other vehicles or pedestrians are in the scene.
- The script takes over CARLA's synchronous mode. Stop any other traffic script first.
- The monitor page shows the analysed Cam1 and Cam2 views, a map, the per-vehicle decision with its reason, the alert log and latencies.

### 2-3. With the phone app (optional)

If you have the app source, overwrite `lib/config.dart`, `lib/services/location_service.dart` and `lib/services/server_link.dart` with the files in `app_patch/patched/`, then build against your laptop:

```bash
flutter build apk --release --dart-define=SERVER_URL=ws://<laptop-ip>:8000/ws
flutter install
```

**The server address is yours to set.** The app configuration in this repository contains only a placeholder (`YOUR-SERVER.example.com`). On a shared network, passing `SERVER_URL` as above is enough. To use a cloud server, deploy `server/exp_server.py` under your own account and use that address.

### 2-4. Recording layout

`bash simdemo/studio.sh` opens the single-page layout used for the demo recordings (status strip, monitor and alert log). The OBS arrangement is described in [`simdemo/README.md`](../simdemo/README.md) (Korean).

### Common problems

| Symptom | Fix |
|---|---|
| Laptop cannot connect Cam1 to the Pi | Check that step 4 is running, the Pi address, and the firewall (`sudo ufw allow 8765:8767/tcp` on the Pi) |
| Pedestrian dots appear in the wrong place on the monitor | Check the grid in the `--inspect` images against the road |
| CARLA runs slowly | Use `-quality-level=Low`. Scenarios follow simulation time, so results are unchanged |
| The take ends with a safety stop (alert too late) | The alert did not arrive in time; check the Pi's processing rate and Cam1 detections on the monitor |

---

## 3. Real cameras

Two USB webcams on the Pi, with people playing the pedestrian and the vehicle (phone GPS). Floor calibration from four points, GPS anchoring and the trial sequence are long enough to live in separate notes:

- Full procedure: [`ko/field-test-notes.md`](ko/field-test-notes.md) (Korean)
- Capture only: [`../shoot/촬영방법.md`](../shoot/촬영방법.md) (Korean)

A person plays the vehicle, so run this **in a controlled space without real traffic**.

---

## Re-running the public-dataset evaluation

See [`../bench/README.md`](../bench/README.md) (Korean). Each dataset took about 15 minutes on the Pi 5.

[Back to README](../README.md) · [System flow](system-flow.md) · [Evaluation](evaluation.md)
