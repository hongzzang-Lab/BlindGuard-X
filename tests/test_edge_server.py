# tests/test_edge_server.py — 실제 pi/edge.py (카메라 대신 합성 영상, 탐지기 = 색 막대) ↔ 실제 서버 코드
# 메시지·명령(사진, 보정 적용, 모드 바꾸기, 녹화, Cam2 구역, 미리보기) 전체 확인. 약 30초.
#   python3 tests/test_edge_server.py
# 진짜 websockets 대신 tests/fakews 의 가짜 모듈(같은 프로세스 안 연결)을 쓴다.
# 이 폴더를 임시 폴더로 복사해서 돌리므로 realtest/calib, logs 는 건드리지 않는다.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # fork 전 numpy 스레드 풀 방지 (시험 하네스용)
import asyncio, shutil, sys, tempfile, threading, time

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="bgx_es_root_")
ROOT = os.path.join(TMP, "realtest")
shutil.copytree(SRC, ROOT, ignore=shutil.ignore_patterns("logs", "pi_logs", "*.avi", "__pycache__"))
for f in ("H_cam1.json", "cam2_geom.json", "gps_anchor.json", "H_cam1_server_copy.json"):
    p = os.path.join(ROOT, "calib", f)
    if os.path.exists(p):
        os.remove(p)
sys.path[:0] = [os.path.join(ROOT, "tests", "fakews"), ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "pi")]
os.environ["BGX_LOGDIR"] = os.path.join(TMP, "server_log")
os.environ["BGX_CALIB_DIR"] = os.path.join(TMP, "server_calib")
os.chdir(ROOT)

# 합성 영상 2개 — 별도 프로세스에서 만든다 (이 프로세스에서 OpenCV 스레드가 돌기 시작하면 fork 한 자식이 멈출 수 있음.
#  실제 pi/edge.py 도 카메라 프로세스를 띄우기 전에는 OpenCV 를 쓰지 않는다)
import subprocess  # noqa: E402
GEN = r"""
import sys, cv2
sys.path[:0] = [%r, %r]
import synth
TMP = %r
bg = synth.background()
people = [(lambda t: synth.crossing(t, t0=2.0, speed=1.3), (0, 0, 230)),
          (lambda t: synth.walker_x(t, t0=1.0, x0=-4, speed=2.5, y=-1.75, x_end=4), (230, 0, 0))]
w = cv2.VideoWriter(TMP + "/cam1.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 480))
for k in range(120):
    w.write(synth.scene_frame(bg, people, k / 15)[0])
w.release()
bg2 = synth.background(3)
w = cv2.VideoWriter(TMP + "/cam2.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 480))
for k in range(150):
    f = bg2.copy()
    if k >= 50:
        fy, fx = (420 if k < 100 else max(420 - 4 * (k - 100), 230)), 320
        cv2.rectangle(f, (fx - 18, fy - 110), (fx + 18, fy), (40, 40, 160), -1)
        for j in range(0, 110, 12):
            cv2.line(f, (fx - 18, fy - j), (fx + 18, fy - j - 6), (200, 200, 240), 2)
    w.write(f)
w.release()
""" % (ROOT, os.path.join(ROOT, "tests"), TMP)
subprocess.run([sys.executable, "-c", GEN], check=True)
H_IMG = [[60, 470], [580, 470], [400, 150], [240, 150], [320, 300], [150, 400]]
H_WORLD = [[-3.0, 2.0], [3.0, 2.0], [3.0, -5.0], [-3.0, -5.0]]

import server.exp_server as SV  # noqa: E402
from websockets.sync import client as fake  # noqa: E402  (tests/fakews)
import test_integration as TI  # noqa: E402

loop = asyncio.new_event_loop()
RESULT = {}


def start_all():
    """edge.py 가 카메라 프로세스를 fork 한 뒤(첫 연결 때) 서버 루프·조작 스레드를 켠다.
    fork 전에 다른 스레드가 돌고 있으면 자식 프로세스가 잠금에 걸릴 수 있어서."""
    threading.Thread(target=loop.run_forever, daemon=True).start()
    asyncio.run_coroutine_threadsafe(SV.ticker(), loop)
    threading.Thread(target=driver, daemon=True).start()


fake.BRIDGE.update(loop=loop, app=SV.app, start=start_all)


def call(coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(15)


def driver():
    time.sleep(4)
    RESULT["connected"] = call(TI.http("GET", "/api/status"))["edge"]["connected"]
    RESULT["snap1"] = call(TI.http("POST", "/api/snapshot", {"cam": "cam1"}))["ok"]
    RESULT["snap2"] = call(TI.http("POST", "/api/snapshot", {"cam": "cam2"}))["ok"]
    r = call(TI.http("POST", "/api/calib/cam1", {"img_pts": H_IMG[:4], "world_pts": H_WORLD, "apply": True}))
    RESULT["calib"] = r["sent"]
    RESULT["geom"] = call(TI.http("POST", "/api/calib/cam2", {"polygon": [[80, 300], [560, 300], [620, 470], [20, 470]],
                                                             "boundary": [[80, 300], [560, 300]]}))["sent"]
    modes = []
    for mode in ("full", "adaptive", "packed"):
        call(TI.http("POST", "/api/edge/cmd", {"cmd": "set_mode", "cam": "cam1", "mode": mode}))
        time.sleep(1.5)
        st = call(TI.http("GET", "/api/status"))
        modes.append((st["edge"]["mode"], st["edge"]["has_H"]))
    RESULT["modes"] = modes
    call(TI.http("POST", "/api/trial/start", {"exp": "V1", "label": "1명_횡단", "raw": True}))
    time.sleep(2.5)
    call(TI.http("POST", "/api/trial/stop", {"result_note": "ok"}))
    RESULT["acks"] = [(x.get("cmd"), x.get("ok")) for x in call(TI.http("GET", "/api/status"))["edge"]["log"]
                      if x.get("type") == "ack"]
    RESULT["preview"] = (len(SV.S["preview"].get("cam1", (b"",))[0]), len(SV.S["preview"].get("cam2", (b"",))[0]))


import multiprocessing as mp  # noqa: E402
import edge  # noqa: E402
sys.argv = ["edge.py", "--server", "ws://x:8000/edge", "--detector", "color", "--cam1", os.path.join(TMP, "cam1.avi"),
            "--cam2", os.path.join(TMP, "cam2.avi"), "--loop", "--minutes", "0.27", "--tag", "es"]
mp.set_start_method("fork", force=True)
edge.main()
time.sleep(1)
print(RESULT)
vids = []
for d, _, fs in os.walk(os.path.join(ROOT, "logs")):
    vids += [f for f in fs if f.endswith(".avi")]
print("녹화:", sorted(vids))
assert RESULT["connected"] and RESULT["snap1"] and RESULT["snap2"] and RESULT["calib"] and RESULT["geom"]
assert [m for m, _ in RESULT["modes"]] == ["full", "adaptive", "packed"] and all(h for _, h in RESULT["modes"])
assert ("record", True) in RESULT["acks"] and min(RESULT["preview"]) > 1000
assert any("T001_V1_cam1_raw" in v for v in vids) and any("T001_V1_cam2_overlay" in v for v in vids)
print("OK — 임시 폴더:", TMP)
