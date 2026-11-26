# python3 tests/test_core.py   (pytest 로도 돌아감)
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from bgx.bytetrack import BYTETracker  # noqa: E402
from bgx.cam1_pipeline import build_canvas, plan_packing, unpack_dets  # noqa: E402
from bgx.engine import Anchor, ConflictEngine, GpsVehicle, enu, split_self  # noqa: E402
from bgx.geometry import VelocityEstimator, apply_h, homography_dlt, reproj_errors  # noqa: E402
from bgx.risk import AlertLevelHold, cpa, evaluate_vehicle, relative_angle_deg, side_from_angle  # noqa: E402


def test_cpa():
    t, d = cpa(10, 0, -5, 0)
    assert abs(t - 2) < 1e-9 and abs(d) < 1e-9
    assert cpa(10, 0, 5, 0)[0] < 0


def test_side():
    assert side_from_angle(relative_angle_deg(1, 0, 0, -1)) == "right"
    assert side_from_angle(relative_angle_deg(1, 0, 0, 1)) == "left"
    assert side_from_angle(relative_angle_deg(1, 0, 5, 0)) == "front"


def test_homography_roundtrip():
    src = [(60, 470), (580, 470), (400, 150), (240, 150), (320, 300)]
    Ht = homography_dlt(src[:4], [(-3, 2), (3, 2), (3, -5), (-3, -5)])
    dst = apply_h(Ht, src)
    H = homography_dlt(src, dst)
    assert reproj_errors(H, src, dst).max() < 1e-6
    w = H[2] @ np.array([320, 300, 1.0])
    assert w > 0


def test_velocity():
    e = VelocityEstimator()
    for i in range(11):
        out = e.update(1, i * 0.1, 0.0, 1.0 - 1.3 * i * 0.1)
    assert out[2] and abs(out[1] + 1.3) < 1e-6


def test_bytetrack_ids_survive_gap():
    tr = BYTETracker()
    ids = set()
    for k in range(40):
        x = 100 + 5 * k
        dets = [] if 15 <= k < 20 else [[x, 100, x + 30, 190, 0.9]]   # 5프레임 놓침
        out = tr.update(np.array(dets).reshape(-1, 5))
        ids |= {o[4] for o in out}
    assert ids == {1}, ids


def test_bytetrack_low_score_keeps_track():
    tr = BYTETracker()
    ids = set()
    for k in range(30):
        x = 100 + 4 * k
        sc = 0.15 if 10 <= k < 20 else 0.9    # 가려져서 점수가 낮아짐 → 2단계 연결
        out = tr.update(np.array([[x, 100, x + 30, 190, sc]]))
        if 10 <= k < 20:
            assert len(out) == 1
        ids |= {o[4] for o in out}
    assert ids == {1}


def test_packing_roundtrip():
    rois = [[10, 10, 110, 210], [400, 50, 460, 170], [300, 300, 340, 380]]
    plan = plan_packing(rois, 416, 4, 0.45, 1.5)
    assert plan is not None and plan["scale"] > 1.0
    # 각 타일 가운데에 박스 하나씩 → 되돌리면 ROI 가운데
    dets = []
    for t in plan["tiles"]:
        dets.append([t["x"] + t["w"] * .25, t["y"] + t["h"] * .25, t["x"] + t["w"] * .75, t["y"] + t["h"] * .75, .9])
    back = unpack_dets(np.array(dets), plan)
    for (x1, y1, x2, y2, _), r in zip(back, rois):
        assert abs((x1 + x2) / 2 - (r[0] + r[2]) / 2) < 1.5
        assert abs((y1 + y2) / 2 - (r[1] + r[3]) / 2) < 1.5
    canvas = build_canvas(np.zeros((480, 640, 3), np.uint8), plan, 416)
    assert canvas.shape == (416, 416, 3)


def test_anchor_two_points():
    lat0, lng0 = 37.5665, 126.9780
    # 바닥 +x 축 = 북동(45°) 방향, 20 m 떨어진 점
    b = math.radians(45)
    dn, de = 20 * math.cos(b), 20 * math.sin(b)
    lat1 = lat0 + math.degrees(dn / 6371008.8)
    lng1 = lng0 + math.degrees(de / (6371008.8 * math.cos(math.radians(lat0))))
    a = Anchor.from_two_points((lat0, lng0), (0, 0), (lat1, lng1), (20, 0))
    x, y = a.to_floor(lat1, lng1)
    assert abs(x - 20) < 1e-6 and abs(y) < 1e-6
    # 북서쪽(인도 쪽, +y 여야 함)으로 5 m
    b2 = math.radians(-45)
    lat2 = lat0 + math.degrees(5 * math.cos(b2) / 6371008.8)
    lng2 = lng0 + math.degrees(5 * math.sin(b2) / (6371008.8 * math.cos(math.radians(lat0))))
    x, y = a.to_floor(lat2, lng2)
    assert abs(x) < 1e-6 and abs(y - 5) < 1e-6
    vx, vy = a.heading_vec(45, 2.0)
    assert abs(vx - 2) < 1e-9 and abs(vy) < 1e-9
    a2 = Anchor.from_bearing((lat0, lng0), (0, 0), 45)
    assert abs(a2.to_floor(lat1, lng1)[0] - 20) < 1e-6


def test_gps_vehicle_and_self_exclusion():
    v = GpsVehicle("A", {"map_match": False})
    for i in range(5):
        v.update(i * 1.0, -20 + 2 * i, -1.5, speed=2.0, vel_doppler=(2.0, 0.0), acc=4)
    st = v.state(4.5)
    assert abs(st["x"] - (-12 + 1.0)) < 1e-9 and st["vsrc"] == "doppler"
    peds = [{"track_id": 7, "x_m": st["x"] + 1.0, "y_m": -1.0, "vx": 2.1, "vy": 0.0, "v_ok": True},
            {"track_id": 3, "x_m": 0.0, "y_m": 0.5, "vx": 0.0, "vy": -1.3, "v_ok": True}]
    used, me, how = split_self(peds, st)
    assert me["track_id"] == 7 and [p["track_id"] for p in used] == [3] and how == "auto"


def test_lane_kalman_map_match():
    import random
    random.seed(3)
    v = GpsVehicle("A", {"map_match": True, "lane_y": -1.75})
    for i in range(12):
        t = float(i)
        v.update(t, -30 + 2.5 * t + random.gauss(0, 2), -1.75 + random.gauss(0, 3), speed=2.5,
                 vel_doppler=(2.5, 0.0), acc=4)
    st = v.state(11.0)
    assert st["y"] == -1.75 and abs(st["vx"] - 2.5) < 0.3 and abs(st["x"] - (-2.5)) < 2.5, st


def test_engine_warns_and_holds():
    eng = ConflictEngine({"horizon_s": 3, "safe_radius_m": 2, "level_tcpa_s": {"danger": 1, "warning": 2,
                                                                                "caution": 3},
                          "compensate_latency": False}, 1.0)
    veh = {"id": "A", "x": -5, "y": -1.5, "vx": 2.5, "vy": 0}
    ped = {"track_id": 1, "x_m": 0, "y_m": 0.5, "vx": 0, "vy": -1.0, "v_ok": True, "ts_cap": 0}
    res, ch, prev, new = eng.step("A", 0, veh, [ped])
    assert res["status"] == "WARN" and ch and new is not None


def test_hold():
    h = AlertLevelHold(1.0)
    assert h.update(0, "caution") == (True, None, "caution")
    assert not h.update(500, None)[0]
    assert h.update(1200, None)[2] is None


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
