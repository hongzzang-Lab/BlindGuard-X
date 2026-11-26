# tests/test_sim.py — 시뮬 데모 계산 시험 (CARLA 없이)
#   bgx/fusion.py (팀 확정 규칙), simdemo/simgeom.py (좌표·카메라·Homography·차량 계획), bgx/netsource.py
#   python3 tests/test_sim.py

import math
import os
import socket
import struct
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "simdemo"))

import numpy as np  # noqa: E402

import simgeom as G  # noqa: E402
from bgx.fusion import FusionEngine, eta_to_crosswalk, signal_transition  # noqa: E402
from bgx.geometry import apply_h, homography_dlt  # noqa: E402
from bgx.risk import DEFAULT_RISK  # noqa: E402

RISK = dict(DEFAULT_RISK, level_tcpa_s={"danger": 1.5, "warning": 3.0, "caution": 3.0})
FCFG = {"near_band_y": [-7.0, 0.0], "waiting_xy": [0.0, 1.0]}
NOW = 1_000_000


def veh(x, y, vx, vy=0.0):
    return {"id": "A", "x": x, "y": y, "vx": vx, "vy": vy}


def cam2(emr_state, crossed=False, age_ms=100):
    return {"emr_state": emr_state, "crossed": crossed, "warmup": False, "emr": 0.8 if emr_state else 0.2,
            "ts_cap": NOW - age_ms}


def ped(x, y, vx, vy):
    return {"track_id": 7, "x_m": x, "y_m": y, "vx": vx, "vy": vy, "v_ok": True, "ts_cap": NOW}


def test_cpa_levels():
    fe = FusionEngine(RISK, FCFG, (-2, 2))
    # 보행자가 A 차로로 걸어 들어옴, A 가 정면에서 옴 → tCPA 2.5 s → 경고
    r = fe.judge(veh(20.8, -5.25, -8.33), [ped(0, -1.5, 0, -1.5)], None, None, NOW)
    assert r["cpa_level"] == "warning" and r["level"] == "warning", r
    r = fe.judge(veh(10.0, -5.25, -8.33), [ped(0, -3.5, 0, -1.5)], None, None, NOW)
    assert r["tcpa"] <= 1.5 and r["level"] == "danger", r
    # 차량 B: 먼 차로 반대 방향 → dCPA 큼 → 경고 안 함
    r = fe.judge(veh(-10, -12.25, 8.33), [ped(0, -1.0, 0, -1.5)], None, None, NOW)
    assert r["level"] is None and r["dcpa"] > 2.0, r
    print("ok test_cpa_levels")


def test_emr_caution_and_signal_boost():
    fe = FusionEngine(RISK, FCFG, (-2, 2))
    vA = veh(30.0, -5.25, -8.33)     # 횡단보도까지 (30-2)/8.33 = 3.4 s
    r = fe.judge(vA, [], cam2(True), {"ped": "GREEN"}, NOW)
    assert r["level"] == "caution" and r["reasons"] == ["EMR"], r
    r = fe.judge(vA, [], cam2(True), {"ped": "FLASH"}, NOW)
    assert r["level"] == "warning" and "SIGNAL" in r["reasons"], r
    r = fe.judge(vA, [], cam2(True), {"ped": "RED", "ped_red_since_ms": NOW - 2000}, NOW)
    assert r["level"] == "warning", r
    r = fe.judge(vA, [], cam2(True), {"ped": "RED", "ped_red_since_ms": NOW - 5000}, NOW)
    assert r["level"] == "caution", r                       # 전환 구간 지남
    # 멀리 있음 (ETA 8 s) → 없음
    assert fe.judge(veh(70.0, -5.25, -8.33), [], cam2(True), None, NOW)["level"] is None
    # 멀어지는 차 → 없음
    assert fe.judge(veh(-10.0, -5.25, -8.33), [], cam2(True), None, NOW)["level"] is None
    # 먼 차로 (B) → 없음
    assert fe.judge(veh(-30.0, -12.25, 8.33), [], cam2(True), {"ped": "FLASH"}, NOW)["level"] is None
    # Cam2 오래된 값 → 없음
    assert fe.judge(vA, [], cam2(True, age_ms=2500), None, NOW)["level"] is None
    # EMR 없음 → 없음
    assert fe.judge(vA, [], cam2(False), {"ped": "FLASH"}, NOW)["level"] is None
    # Cam2 경계 넘음 상태도 출발로 봄
    assert fe.judge(vA, [], cam2(False, crossed=True), None, NOW)["level"] == "caution"
    # EMR 경고 방향: 대기구역은 A(-x 진행) 기준 오른쪽 앞
    r = fe.judge(vA, [], cam2(True), None, NOW)
    assert r["side"] in ("front", "right"), r
    # CPA 가 더 높으면 CPA
    r = fe.judge(veh(10.0, -5.25, -8.33), [ped(0, -3.5, 0, -1.5)], cam2(True), {"ped": "FLASH"}, NOW)
    assert r["level"] == "danger" and r["reasons"][0] == "CPA", r
    print("ok test_emr_caution_and_signal_boost")


def test_hold_sequence():
    fe = FusionEngine(RISK, FCFG, (-2, 2))
    seq = []
    t = NOW
    for i in range(40):
        t += 100
        x = 40 - 8.33 * i * 0.1
        c2 = cam2(i >= 5) if i < 25 else cam2(False)
        c2["ts_cap"] = t - 50
        peds = [ped(0, 0.5 - 1.5 * max(0, (i - 12)) * 0.1, 0, -1.5)] if i >= 14 else []
        for p in peds:
            p["ts_cap"] = t
        _, ch, prev, new = fe.step("A", t, veh(x, -5.25, -8.33), peds, c2, {"ped": "GREEN"})
        if ch:
            seq.append(new)
    assert seq[:3] == ["caution", "warning", "danger"], seq
    print("ok test_hold_sequence", seq)


def test_eta_and_signal():
    assert eta_to_crosswalk(veh(12, -5, -2), (-2, 2)) == 5.0
    assert eta_to_crosswalk(veh(0, -5, -2), (-2, 2)) == 0.0
    assert eta_to_crosswalk(veh(12, -5, 2), (-2, 2)) is None
    assert signal_transition({"ped": "FLASH"}, NOW, 3)
    assert not signal_transition({"ped": "GREEN"}, NOW, 3)
    assert not signal_transition(None, NOW, 3)
    print("ok test_eta_and_signal")


def test_floor_frame():
    # J189 처럼 인도가 월드 +X 쪽: ey = (1, 0) → ex 는 월드 +Y (인도를 왼쪽에 두고 보는 방향)
    f = G.Floor((-41.0, 2.5, 0.0), (1.0, 0.0))
    assert np.allclose(f.ex, [0, 1])
    # CARLA: yaw 90(+Y 를 봄)의 왼쪽 = +X = 인도 ✓
    assert np.allclose(G.left_of_yaw(90), (1, 0), atol=1e-9)
    for p in [(3.0, -2.0), (-7.5, 1.2)]:
        w = f.to_world(*p)
        assert np.allclose(f.to_floor(w[0], w[1]), p)
    # 임의 방향
    f2 = G.Floor((10, 20, 1), (math.cos(0.7), math.sin(0.7)))
    w = f2.to_world(4, -3)
    assert np.allclose(f2.to_floor(w[0], w[1]), (4, -3))
    lx, ly = G.left_of_yaw(f2.yaw_of_floor_dir(1, 0))
    assert np.allclose((lx, ly), f2.ey, atol=1e-9)     # +x 를 볼 때 왼쪽이 인도
    print("ok test_floor_frame")


def test_camera_and_homography():
    f = G.Floor((-41.0, 2.5, 0.0), (1.0, 0.0))
    cx, cy, _ = f.to_world(0, 7.5)
    tx, ty, _ = f.to_world(0, 0.3)
    p, y = G.look_at((cx, cy, 5.5), (tx, ty, 0.0))
    w2c = np.linalg.inv(G.carla_matrix(cx, cy, 5.5, p, y, 0.0))
    u, v, d = G.project([(tx, ty, 0.0)], w2c, 640, 480, 70)
    assert abs(u[0] - 320) < 1e-6 and abs(v[0] - 240) < 1e-6 and d[0] > 0
    # 바닥에서 더 먼 점(차도 쪽)은 화면 위쪽, 인도 +x 쪽은 ... 왼/오른쪽이 일관
    u2, v2, _ = G.project([f.to_world(0, -3, 0)], w2c, 640, 480, 70)
    assert v2[0] < 240
    # Cam1 (팀 Exp 위치) 로 Homography → 바닥 좌표 복원
    cam = dict(x=-37.09, y=2.60, z=10.51, pitch=-53.70, yaw=-178.34, roll=0.0)
    w2c1 = np.linalg.inv(G.carla_matrix(cam["x"], cam["y"], cam["z"], cam["pitch"], cam["yaw"], cam["roll"]))
    img, flo = G.ground_homography(f, 0.0, w2c1, 1280, 720, 100, x_range=(-12, 12), y_range=(-15, 3))
    assert len(img) > 50, len(img)
    a, b = G.pick_spread(img, flo, 24)
    H = homography_dlt(a, b)
    err = np.linalg.norm(apply_h(H, img) - np.asarray(flo), axis=1)
    assert err.max() < 1e-3, err.max()
    # 대기구역 다각형 → 화면
    poly = G.convex_hull(G.project_floor_poly(f, 0.15, [[-3.5, 0], [3.5, 0], [3.5, 3], [-3.5, 3]], w2c, 640, 480, 70))
    assert len(poly) >= 4 and G.poly_area(poly) > 20000, G.poly_area(poly)
    print("ok test_camera_and_homography (Cam1 격자 %d점)" % len(img))


def test_vehicle_plan():
    v, a = 30 / 3.6, 2.5
    for dist, t_arr in ((72.0, 23.0), (8.0, 10.0)):
        tgo = G.plan_go_time(dist, v, a, t_arr)
        prof = G.SpeedProfile(0.0, tgo, v, a)
        t, dt = 0.0, 0.01
        while prof.s < dist:
            prof.step(t, dt)
            t += dt
        assert abs(t - t_arr) < 0.05, (dist, t, t_arr)
    prof = G.SpeedProfile(0, 0, v, a, decel=6.0)
    for i in range(500):
        prof.step(i * 0.01, 0.01)
    prof.brake()
    s0 = prof.s
    for i in range(500):
        prof.step(5 + i * 0.01, 0.01)
    assert prof.v == 0 and abs((prof.s - s0) - v * v / 12) < 0.2
    path = G.LanePath([(0, 0, 0), (0, 10, 0), (0, 20, 0)])
    assert path.at(15)[1] == 15 and abs(path.at(15)[3] - 90) < 1e-6
    print("ok test_vehicle_plan")


def test_walker_schedule():
    import json
    cfg = json.load(open(os.path.join(ROOT, "simdemo", "scenario_j189.json"), encoding="utf-8"))
    main = G.merge(cfg, cfg["variants"]["main"])
    segs, t_cross = G.walker_segments(main["pedestrian"], main["signal"], {"road_far_y": -14})
    ped = main["pedestrian"]
    assert abs(t_cross - (ped["approach_t"] + (ped["start_floor"][1] - ped["curb_y"]) / ped["approach_speed"])) < 1e-6
    ev = [s[4] for s in segs if s[4]]
    assert ev == ["ped_enter", "ped_wait", "ped_sway", "ped_approach", "ped_cross"], ev
    # 서성임은 대기 중에, 신호 FLASH(14) 뒤·RED(18) 전에 끝남
    assert 14.0 <= [s for s in segs if s[4] == "ped_sway"][0][0] and segs[3][1] < 18.0
    wo = G.merge(cfg, cfg["variants"]["wait_only"])
    segs2, tc2 = G.walker_segments(wo["pedestrian"], wo["signal"], {"road_far_y": -14})
    assert tc2 is None and "ped_cross" not in [s[4] for s in segs2]
    st = G.merge(cfg, cfg["variants"]["stages"])
    assert st["signal"]["ped_red_t"] == 11.0 and st["pedestrian"]["approach_t"] == 18.2
    print("ok test_walker_schedule")


def test_netsource():
    import cv2
    from bgx.netsource import NetFrameSource
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    src = NetFrameSource("tcp:127.0.0.1:%d" % port, 320, 240, 10)
    src.start()
    time.sleep(0.2)
    c = socket.create_connection(("127.0.0.1", port))
    for k in range(3):
        img = np.full((240, 320, 3), 40 * (k + 1), np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        payload = struct.pack("!dI", time.time(), 100 + k) + buf.tobytes()
        c.sendall(struct.pack("!I", len(payload)) + payload)
    time.sleep(0.3)
    f, t, n = src.get(0, timeout=1.0)
    assert f is not None and f.shape == (240, 320, 3) and n >= 1 and src.carla_frame == 102, (n, src.carla_frame)
    assert abs(int(f.mean()) - 120) < 5
    c.close()
    src.stop()
    print("ok test_netsource")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
