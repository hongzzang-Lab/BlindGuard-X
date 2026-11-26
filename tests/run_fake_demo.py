# tests/run_fake_demo.py — carla_demo.py 전체 흐름 시험 (가짜 CARLA + 진짜 서버 + 가짜 인식)
#   서버(exp_server, site_sim)를 같은 프로세스에서 uvicorn 으로 띄우고, 가짜 CARLA 의 틱마다
#   '완벽한 Pi 인식'(보행자 정답 위치·속도 → Cam1 장면, 연석 접근 → Cam2 EMR)을 서버 상태에 직접 넣는다.
#   → 차량 A 가 경고·긴급을 받고 감속하는지, B 는 경고가 없는지, take 기록이 남는지 확인.
#   PYTHONPATH=tests/fakecarla python3 tests/run_fake_demo.py [variant]

import os
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests", "fakecarla"))
sys.path.insert(0, ROOT)
os.environ["BGX_SITE"] = os.path.join(ROOT, "simdemo", "site_sim.json")
os.environ["BGX_LOGDIR"] = tempfile.mkdtemp(prefix="bgx_fake_demo_")
os.environ["BGX_CALIB_DIR"] = tempfile.mkdtemp(prefix="bgx_fake_calib_")

import carla  # noqa: E402  (가짜)
import uvicorn  # noqa: E402

import server.exp_server as SV  # noqa: E402

PORT = 8765 + 1000
variant = sys.argv[1] if len(sys.argv) > 1 else "main"


def serve():
    uvicorn.run(SV.app, host="127.0.0.1", port=PORT, log_level="warning")


threading.Thread(target=serve, daemon=True).start()
time.sleep(1.5)

# 가짜 Pi 가 '연결됨' 처럼 보이게 (보정 명령 전송 성공)
class FakeWS:
    async def send_text(self, t):
        pass


SV.S["edge_ws"] = FakeWS()
state = {"prev": None, "frame": 0}


def perception(world):
    """완벽한 인식: 보행자 정답 → 바닥 좌표 (carla_demo 가 계산한 바닥 좌표계 그대로)."""
    if "floor" not in GEOM:
        return
    walkers = [a for a in world.actors if a.type_id.startswith("walker") and a.is_alive]
    now = int(time.time() * 1000)
    state["frame"] += 1
    if world.frame % 2:
        return
    peds = []
    emr_state = False
    for w in walkers:
        l = w.get_location()
        fx, fy = GEOM["floor"].to_floor(l.x, l.y)
        prev = state["prev"]
        vx = vy = 0.0
        tw = time.time()          # 실제 Pi 처럼 '실제 시계'로 속도를 잰다
        if prev is not None:
            dt = max(tw - prev[2], 1e-3)
            vx, vy = (fx - prev[0]) / dt, (fy - prev[1]) / dt
        state["prev"] = (fx, fy, tw)
        if -3 < fx < 3 and fy < 8:
            peds.append({"track_id": 1, "x_m": fx, "y_m": fy, "vx": vx, "vy": vy, "v_ok": prev is not None})
        if 0.0 < fy < 1.7 and vy < -0.5:      # 연석 쪽으로 다가감 → EMR 높음
            emr_state = True
        if -2.5 < fy <= 0.0 and vy < -0.3:    # 실제 Cam2 처럼: 경계선을 넘어 차도 쪽 띠 안에 있는 동안 '경계 넘음'
            state["crossed"] = True
        else:
            state["crossed"] = False
    SV.S["scene"] = {"frame": state["frame"], "ts_cap": now - 80, "peds": peds, "fps": 10, "timing": {}, "mode": "packed",
                     "every": 1, "n_rois": 1, "pack_scale": 1, "fallback": False, "temp_c": 55, "offset_ms": 0,
                     "det_ran": True}
    SV.S["scene_recv"] = now
    SV.S["cam2"] = {"frame": state["frame"], "ts_cap": now - 60, "_recv": now, "warmup": False, "emr_state": emr_state,
                    "crossed": state.get("crossed", False), "emr": 0.8 if emr_state else 0.2, "present": True, "fps": 10, "timing": {}}


carla._HOOKS.append(perception)

sys.argv = ["carla_demo.py", "--server", "http://127.0.0.1:%d" % PORT, "--variant", variant, "--takes", "1",
            "--warmup", "0.5", "--realtime", os.environ.get("FAKE_RT", "8"), "--camera-config",
            "/mnt/user-data/uploads/BlindGuard-X/sim_carla/j189_camera_configs.json"] + os.environ.get("FAKE_ARGS", "").split()
sys.path.insert(0, os.path.join(ROOT, "simdemo"))
import carla_demo  # noqa: E402

GEOM = {}
_bg = carla_demo.build_geometry


def _keep(world, cfg):
    g = _bg(world, cfg)
    GEOM.update(g)
    return g


carla_demo.build_geometry = _keep

carla_demo.main()
time.sleep(0.5)
import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402

alerts = list(csv.DictReader(open(os.path.join(SV.LOGDIR, "alerts.csv"), encoding="utf-8")))
print("서버 경고 기록:", [(a["vehicle"], a["level"], a["reasons"]) for a in alerts])
summ = json.load(open(sorted(glob.glob(os.path.join(ROOT, "simdemo", "logs", "sim_*", "summary.json")))[-1]))
print("take 요약:", summ)
trials = list(csv.DictReader(open(os.path.join(SV.LOGDIR, "trials.csv"), encoding="utf-8")))
print("시험 기록:", [(t["trial_id"], t["exp"], t["label"], t["result_note"]) for t in trials])
