# tests/test_integration.py
#
# 실제 서버 코드(server/exp_server.py, Starlette ASGI 앱)를 네트워크 없이 직접 구동해서
#   폰 앱(WebSocket /ws) + Pi(WebSocket /edge) + 조작 페이지(HTTP API) 를 흉내 내고,
#   GPS 기준점 설정 → 보정 적용 → C1 충돌 시나리오 → 경고 전송·수신 확인 → 기록 파일 까지 확인한다.
# Pi 쪽 장면은 합성 영상을 실제 Cam1 파이프라인(PACKED 모드)에 넣어서 만든다.
#
#   python3 tests/test_integration.py   (약 30초)

import asyncio
import json
import math
import os
import random
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ["BGX_LOGDIR"] = tempfile.mkdtemp(prefix="bgx_it_")

import numpy as np  # noqa: E402

import synth  # noqa: E402
from bgx.cam1_pipeline import Cam1Pipeline  # noqa: E402
from bgx.detector import ColorBoxDetector  # noqa: E402
from bgx.engine import R_EARTH, rot  # noqa: E402

# 서버가 쓰는 보정 파일(GPS 기준점 등)은 임시 폴더로 → 현장에서 만든 realtest/calib 를 건드리지 않음
os.environ["BGX_CALIB_DIR"] = tempfile.mkdtemp(prefix="bgx_it_calib_")

import server.exp_server as SV  # noqa: E402

# ───────── 가짜 세계 ─────────
LAT0, LNG0, PHI = 37.5665, 126.9780, math.radians(25)   # 임의의 기준점(서울시청), +x 축이 동쪽에서 25° 돌아감
BIAS = (1.8, -1.2)
PHONE_CLOCK = -2000      # 폰 시계 = 서버 시계 - 2초


def floor_to_ll(x, y, noise=1.0):
    x += BIAS[0] + random.gauss(0, noise)
    y += BIAS[1] + random.gauss(0, noise)
    de, dn = rot(-PHI, x, y)
    return LAT0 + math.degrees(dn / R_EARTH), LNG0 + math.degrees(de / (R_EARTH * math.cos(math.radians(LAT0))))


def heading_of_floor_x():
    de, dn = rot(-PHI, 1.0, 0.0)
    return math.degrees(math.atan2(de, dn)) % 360


# ───────── ASGI 흉내 ─────────
class WSClient:
    def __init__(self, path):
        self.path = path
        self.inq = asyncio.Queue()
        self.out = []
        self.out_ev = asyncio.Event()

    async def run(self):
        scope = {"type": "websocket", "path": self.path, "raw_path": self.path.encode(), "headers": [],
                 "query_string": b"", "client": ("10.42.0.5", 5555), "server": ("test", 8000), "scheme": "ws",
                 "root_path": "", "subprotocols": [], "asgi": {"version": "3.0"}}
        await self.inq.put({"type": "websocket.connect"})

        async def receive():
            return await self.inq.get()

        async def send(m):
            if m["type"] == "websocket.send":
                self.out.append(json.loads(m["text"]))
                self.out_ev.set()

        self.task = asyncio.create_task(SV.app(scope, receive, send))

    async def send(self, obj):
        await self.inq.put({"type": "websocket.receive", "text": json.dumps(obj)})

    async def close(self):
        await self.inq.put({"type": "websocket.disconnect", "code": 1000})


async def http(method, path, body=None):
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 1), "server": ("t", 80),
             "scheme": "http", "root_path": "", "http_version": "1.1", "asgi": {"version": "3.0"}}
    data = json.dumps(body or {}).encode()
    sent = {"done": False}

    async def receive():
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": data, "more_body": False}
        await asyncio.sleep(3600)

    res = {"status": None, "body": b""}

    async def send(m):
        if m["type"] == "http.response.start":
            res["status"] = m["status"]
        elif m["type"] == "http.response.body":
            res["body"] += m.get("body", b"")

    await SV.app(scope, receive, send)
    try:
        return json.loads(res["body"])
    except ValueError:
        return res


def now_ms():
    return int(time.time() * 1000)


async def main():
    random.seed(7)
    ticker = asyncio.create_task(SV.ticker())
    phone = WSClient("/ws")
    edge = WSClient("/edge")
    await phone.run()
    await edge.run()
    await phone.send({"type": "hello", "v": 1, "device_id": "phone-A"})
    await edge.send({"type": "hello_edge", "edge_id": "cw-01", "cams": {}, "has_H": False})

    st = {"veh_x": None, "run": True}

    async def phone_loop():
        k = 0
        while st["run"]:
            pm = now_ms() + PHONE_CLOCK
            await phone.send({"type": "ping", "v": 1, "t": pm})
            if st["veh_x"] is not None:
                x, y = st["veh_x"], -1.75
                lat, lng = floor_to_ll(x, y)
                await phone.send({"type": "location", "v": 1, "device_id": "phone-A", "ts": pm, "fix_ts": pm - 300,
                                  "lat": lat, "lng": lng, "accuracy_m": 4.0, "speed_mps": st.get("spd", 0.0),
                                  "heading_raw_deg": heading_of_floor_x() if st.get("spd", 0) > 0.3 else -1.0,
                                  "speed_acc": 0.3, "heading_acc": 15})
            # 앱처럼: 받은 alert 마다 ack
            for m in phone.out[k:]:
                if m.get("type") == "alert":
                    await phone.send({"type": "alert_ack", "v": 1, "seq": m["seq"], "level": (m.get("risk") or {}).get("level"),
                                      "t": now_ms() + PHONE_CLOCK})
            k = len(phone.out)
            await asyncio.sleep(1.0)

    ptask = asyncio.create_task(phone_loop())
    await phone.send({"type": "drive", "v": 1, "device_id": "phone-A", "ts": now_ms(), "state": "start"})

    # ── 1) GPS 기준점: p0 (0,-1.75), p1 (-20,-1.75) 에 폰을 3초씩 ──
    for name, x in (("p0", 0.0), ("p1", -20.0)):
        st["veh_x"], st["spd"] = x, 0.0
        await http("POST", "/api/anchor/capture/start", {"name": name, "x": x, "y": -1.75})
        await asyncio.sleep(3.2)
        r = await http("POST", "/api/anchor/capture/stop")
        assert r["ok"], r
    r = await http("POST", "/api/anchor/compute")
    assert r["ok"], r
    print("기준점:", r["anchor"]["phi_deg"], "도 (정답 25), 거리차", r["anchor"]["detail"]["dist_err_m"])
    assert abs(r["anchor"]["phi_deg"] - 25) < 8

    # ── 2) 사진 요청 → Pi 가 답함 ──
    async def edge_answer_snapshot():
        while True:
            await edge.out_ev.wait()
            edge.out_ev.clear()
            for m in edge.out:
                if m.get("type") == "cmd" and m.get("cmd") == "snapshot" and not m.get("_done"):
                    m["_done"] = True
                    import base64

                    import cv2
                    ok, buf = cv2.imencode(".jpg", synth.background())
                    await edge.send({"type": "snapshot", "cam": m["cam"], "w": 640, "h": 480,
                                     "jpg": base64.b64encode(buf.tobytes()).decode()})
    snap_task = asyncio.create_task(edge_answer_snapshot())
    r = await http("POST", "/api/snapshot", {"cam": "cam1"})
    assert r["ok"] and r["w"] == 640, r

    # ── 3) 보정: 합성 H 의 대응점으로 계산·적용 → Pi 로 set_H 명령이 가야 함 ──
    img = [[60, 470], [580, 470], [400, 150], [240, 150], [320, 300], [150, 400]]
    world = synth.apply_h(synth.H_TRUE, img).tolist()
    r = await http("POST", "/api/calib/cam1", {"img_pts": img, "world_pts": world, "check_img": [[320, 250]],
                                              "check_world": synth.apply_h(synth.H_TRUE, [[320, 250]]).tolist(),
                                              "apply": True})
    assert r["ok"] and r["reproj_mean"] < 1e-3 and r["sent"], r
    assert any(m.get("cmd") == "set_H" for m in edge.out)

    # ── 4) C1 R1: 차량 A가 x=-15 에서 2.5 m/s, 보행자 B가 t=3.2s 에 횡단 ──
    r = await http("POST", "/api/trial/start", {"exp": "C1", "label": "빠른걸음", "scenario": "R1_상충",
                                                "countdown_s": 0})
    pipe = Cam1Pipeline(ColorBoxDetector(), synth.H_TRUE, (640, 480), {"mode": "packed"})
    bg = synth.background()
    t0 = time.time()
    frame = 0
    while time.time() - t0 < 9.0:
        t = time.time() - t0
        vx = -15 + 2.5 * t
        st["veh_x"], st["spd"] = vx, 2.5
        people = [(lambda tt: synth.crossing(tt, t0=3.2, y0=1.0, speed=1.3), (0, 0, 230)),
                  (lambda tt: (vx, -1.75) if -3.5 < vx < 3.5 else None, (230, 0, 0))]   # 차량 본인도 카메라에 찍힘
        im, truth = synth.scene_frame(bg, people, t)
        res = pipe.process(im, time.time())
        frame += 1
        cap = now_ms()
        await edge.send({"type": "scene", "cam": "cam1", "frame": frame, "ts_cap_local": cap, "ts_done_local": cap + 30,
                         "ts_cap": cap, "ts_done": cap + 30, "ts_send": cap + 31, "offset_ms": 0, "fps": 10.0,
                         "dropped": 0, "temp_c": 55.0, "rec": True, "mode": res["mode"], "det_ran": res["det_ran"],
                         "full_scan": res["full_scan"], "fallback": res["fallback"], "n_rois": res["n_rois"],
                         "roi_area": res["roi_area"], "pack_scale": res["pack_scale"],
                         "canvas_fill": res["canvas_fill"], "n_det": res["n_det"], "timing": res["timing"],
                         "peds": res["peds"], "every": 1, "has_H": True})
        if frame % 3 == 0:
            await edge.send({"type": "cam2", "cam": "cam2", "frame": frame, "ts_cap": cap, "ts_done": cap + 5,
                             "warmup": False, "present": True, "area": 3000, "signed_px": -50.0, "toward_px": 1.0,
                             "n_pts": 20, "persist": 0, "prox": 0.5, "tow": 0.5, "per": 0, "emr": 0.4,
                             "emr_state": False, "crossed": False, "events": ["enter"] if frame == 3 else [],
                             "timing": {"mog": 2, "lk": 2, "analysis": 0.1, "total": 4.1}, "fps": 15, "dropped": 0,
                             "rec": True})
        await asyncio.sleep(0.1)
    status = await http("GET", "/api/status")
    await http("POST", "/api/trial/stop", {"result_note": "test"})
    await asyncio.sleep(1.2)
    st["run"] = False

    alerts = [m for m in phone.out if m.get("type") == "alert" and m.get("risk")]
    levels = [m["risk"]["level"] for m in alerts]
    print("폰 A가 받은 경고 단계:", sorted(set(levels)), "개수", len(alerts))
    print("차량 판정 마지막:", status["vehicles"])
    assert alerts, "경고가 폰에 안 감"
    assert any(m.get("type") == "cmd" and m.get("cmd") == "record" for m in edge.out)
    await phone.close()
    await edge.close()
    for t in (ticker, ptask, snap_task):
        t.cancel()
    await asyncio.sleep(0.2)

    d = SV.LOGDIR
    for f in ("cam1_frames.csv", "cam1_tracks.csv", "gps.csv", "decisions.csv", "alerts.csv", "alert_acks.csv",
              "trials.csv", "cam2_frames.csv", "cam2_events.csv", "apps.csv", "operator.csv", "anchor_pts.json"):
        n = sum(1 for _ in open(os.path.join(d, f), encoding="utf-8"))
        print("  %-18s %d 줄" % (f, n))
        assert n >= 2, f
    import csv
    dec = list(csv.DictReader(open(os.path.join(d, "decisions.csv"), encoding="utf-8")))
    selfs = [r for r in dec if r["self_track"]]
    print("차량 본인 트랙으로 빠진 판정:", len(selfs), "예:", selfs[0]["self_track"] if selfs else None,
          "GPS-카메라 거리 중앙값 %.2f m" % np.median([float(r["self_dist"]) for r in selfs]) if selfs else "")
    acks = list(csv.DictReader(open(os.path.join(d, "alert_acks.csv"), encoding="utf-8")))
    print("ack 단방향 지연(합성):", [a["one_way_ms"] for a in acks[:5]])
    print("기록 폴더:", d)
    return d


if __name__ == "__main__":
    asyncio.run(main())
