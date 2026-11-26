#!/usr/bin/env python3
# simdemo/carla_demo.py — BlindGuard-X CARLA 시뮬 데모 (Town10HD_Opt · J189)
#
# 포스터 최종본 그림 2 를 CARLA 에서 그대로 돌린다.
#   CARLA Cam1·Cam2 영상 ──TCP──▶ Raspberry Pi 5 (pi/edge.py: Cam1 YOLO26n·ByteTrack·Homography / Cam2 MOG2·LK·EMR)
#                                        │ 보행자 위치·속도, EMR
#   CARLA 차량 A·B: GNSS 위치 + 가상 OBU 속도 ─HTTP─▶ 노트북 서버 (server/exp_server.py, source=carla)
#   CARLA 보행자 신호 RED / GREEN / FLASH ─────HTTP─▶   차량별 CPA 위험 판정 (+ Cam2 조기 징후 · 신호)
#                                                         └─ WebSocket ─▶ 차량 A 폰 앱 (주의 / 경고 / 긴급)
#   서버가 차량 A 에 '긴급'을 보내면 A 가 감속한다 (경고를 받은 운전자 역할).
#
# 이 스크립트가 하는 일
#   1) J189 횡단보도에서 바닥 좌표(연석선·차로·대기구역)를 지도 정보로 자동 계산
#   2) Cam1(팀 Exp1~3 위치) · Cam2(대기구역 기둥) 가상 카메라를 만들고 영상을 Pi 로 보냄
#   3) 카메라 모델로 Cam1 Homography 와 Cam2 대기구역 다각형·경계선을 계산해서 서버 → Pi 로 적용 (손 보정 없음)
#   4) 시나리오 연출: 보행자(대기 → 서성임 → 연석 접근 → 횡단), 정차 트럭, 차량 A·B, 신호 전환
#   5) 차량 GNSS·OBU·신호를 10 Hz 로 서버에 보내고, 정답(GT)은 노트북 simdemo/logs/ 에만 저장 (판단에는 안 씀)
#
# 실행 (노트북, CARLA 서버가 Town10HD_Opt 로 떠 있는 상태)
#   PY=~/miniconda3/envs/blindguard-carla/bin/python
#   $PY simdemo/carla_demo.py --inspect                      # 1단계: 기하·카메라 확인 그림만 저장 (Pi·서버 없이)
#   $PY simdemo/carla_demo.py --pi <PiIP> --variant main --takes 3
#
# 주의: 이 스크립트가 CARLA 동기 모드(틱 주인)가 된다. 팀 Blind 교통 스크립트(blindguard_city_j189_v3.py)는 끄고 실행.
#       끝나면(Ctrl+C 포함) 만든 액터만 지우고 설정·신호등을 되돌린다.

import argparse
import csv
import http.client
import json
import math
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from bgx.engine import Anchor  # noqa: E402
from bgx.geometry import apply_h, homography_dlt, reproj_errors  # noqa: E402
import simgeom as G  # noqa: E402

try:
    import carla
except ImportError:
    raise SystemExit("CARLA 파이썬 API 가 없음. ~/miniconda3/envs/blindguard-carla/bin/python 으로 실행")

try:
    import cv2
except ImportError:
    raise SystemExit("OpenCV 없음: blindguard-carla 환경에서 pip install opencv-python")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ───────────────────────── 1. J189 기하 ─────────────────────────
ROAD_TYPES = None   # carla.LaneType 집합 (carla import 뒤 채움)


def crosswalk_polygons(cmap):
    pts = cmap.get_crosswalks()
    polys, cur = [], []
    for p in pts:
        if cur and abs(p.x - cur[0].x) < 1e-3 and abs(p.y - cur[0].y) < 1e-3:
            polys.append(cur)
            cur = []
        else:
            cur.append(p)
    if len(cur) >= 3:
        polys.append(cur)
    return polys


def lane_type_at(cmap, x, y, z):
    wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=z), project_to_road=False, lane_type=carla.LaneType.Any)
    return wp


def build_geometry(world, cfg):
    """J189 횡단보도에서 바닥 좌표계와 차로를 뽑는다."""
    cmap = world.get_map()
    gcfg = cfg["geometry"]
    hint = np.array(gcfg["crosswalk_hint"], float)
    polys = crosswalk_polygons(cmap)
    if not polys:
        raise SystemExit("지도에 횡단보도 정보가 없음 (get_crosswalks 비어 있음)")
    cands = []
    for poly in polys:
        a = np.array([[p.x, p.y] for p in poly])
        cands.append((float(np.linalg.norm(a.mean(0) - hint)), poly))
    cands.sort(key=lambda c: c[0])
    dist, poly = cands[0]
    log("[기하] 횡단보도 후보 %d개 중 힌트(%.1f, %.1f)에서 %.1f m 인 것을 씀" % (len(polys), hint[0], hint[1], dist))
    if dist > 15:
        log("[기하][경고] 가장 가까운 횡단보도도 15 m 넘게 떨어짐 → scenario_j189.json geometry.crosswalk_hint 확인")
    A = np.array([[p.x, p.y] for p in poly])
    zc = float(np.mean([p.z for p in poly]))
    c = A.mean(0)
    cov = np.cov((A - c).T)
    w, V = np.linalg.eigh(cov)
    u = V[:, int(np.argmax(w))]          # 긴 축 = 도로를 가로지르는 방향
    near = np.array(gcfg["near_side_hint"], float)
    if (near - c) @ u < 0:
        u = -u                           # u = 차도 → 대기구역(인도) 방향
    # 가로지르는 선을 따라 차선 종류를 훑어서 연석(양쪽)·차로를 찾는다
    samples = []
    for s in np.arange(-30.0, 15.0, 0.25):
        p = c + s * u
        wp = lane_type_at(cmap, float(p[0]), float(p[1]), zc)
        samples.append((float(s), wp))

    def is_road(wp):
        return wp is not None and wp.lane_type in ROAD_TYPES

    i0 = int(np.argmin([abs(s) for s, _ in samples]))
    if not is_road(samples[i0][1]):
        # 가운데가 차도가 아니면 가장 가까운 차도 표본에서 시작
        roads = [i for i, (s, wp) in enumerate(samples) if is_road(wp)]
        if not roads:
            raise SystemExit("횡단보도 선 위에서 차도를 못 찾음")
        i0 = min(roads, key=lambda i: abs(samples[i][0]))
    i = i0
    while i + 1 < len(samples) and is_road(samples[i + 1][1]):
        i += 1
    s_near = (samples[i][0] + samples[i + 1][0]) / 2 if i + 1 < len(samples) else samples[i][0]
    j = i0
    while j - 1 >= 0 and is_road(samples[j - 1][1]):
        j -= 1
    s_far = (samples[j][0] + samples[j - 1][0]) / 2 if j - 1 >= 0 else samples[j][0]
    # 연석 = 횡단보도 양 끝 (지도 횡단보도는 연석에서 연석까지 그려짐).
    # 교차로 옆 횡단보도는 차로 끝과 연석 사이에 '차로 아닌 아스팔트'가 있어서 차로만으로 찾으면 2~3 m 안쪽에 잡힘 (11/21 J189 확인)
    proj = (A - c) @ u
    s_lane_near, s_lane_far = s_near, s_far
    if gcfg.get("curb_from_crosswalk", True):
        s_near, s_far = float(proj.max()), float(proj.min())
        log("[기하] 연석 = 횡단보도 끝 (차로 끝보다 대기 쪽 %.1f m, 반대쪽 %.1f m 바깥)" %
            (s_near - s_lane_near, s_lane_far - s_far))
    # 인도 폭 (연석 너머로 Sidewalk 가 이어지는 길이)
    side_w, started = 0.0, False
    for s, wp in samples:
        if s <= s_near:
            continue
        if wp is not None and wp.lane_type == carla.LaneType.Sidewalk:
            side_w, started = s - s_near, True
        elif started:
            break
    Opt = c + s_near * u
    road_z = zc
    wp_mid = samples[i0][1]
    if wp_mid is not None:
        road_z = float(wp_mid.transform.location.z)
    floor = G.Floor((Opt[0], Opt[1], road_z), u)
    # 차로: 교차로 연결 차로까지 겹쳐 잡히므로, 같은 방향·1.2 m 안의 것을 한 차로로 묶고 교차로 밖 차로를 대표로
    raw = {}
    for s, wp in samples:
        if wp is None or wp.lane_type != carla.LaneType.Driving:
            continue
        key = (wp.road_id, wp.section_id, wp.lane_id)
        if key in raw:
            continue
        loc = wp.transform.location
        fx, fy = floor.to_floor(loc.x, loc.y)
        fwd = wp.transform.get_forward_vector()
        dfx, _ = floor.vec_to_floor(fwd.x, fwd.y)
        raw[key] = {"key": list(key), "y": round(fy, 2), "dir": 1 if dfx > 0 else -1, "width": round(wp.lane_width, 2),
                    "junction": bool(getattr(wp, "is_junction", False))}
    groups = []
    for L in sorted(raw.values(), key=lambda L: -L["y"]):
        g = next((g for g in groups if g[0]["dir"] == L["dir"] and abs(g[0]["y"] - L["y"]) < 1.2), None)
        if g is None:
            groups.append([L])
        else:
            g.append(L)
    lanes = []
    for g in groups:
        rep_ = sorted(g, key=lambda L: (L["junction"], abs(L["width"] - 3.5)))[0]
        lanes.append(dict(rep_, y=round(float(np.mean([L["y"] for L in g])), 2), n_merged=len(g)))
    log("[기하] 지도 차로 %d개 → 겹친 것 묶어서 %d차로" % (len(raw), len(lanes)))
    lanes = sorted(lanes, key=lambda L: -L["y"])
    for n, L in enumerate(lanes):
        L["name"] = "차로%d %s" % (n + 1, "→" if L["dir"] > 0 else "←")
    # 횡단보도 폭 (도로 방향으로)
    ex = floor.ex
    along = (A - c) @ ex
    half_w = max(1.0, (along.max() - along.min()) / 2)
    cx_off = float(((A.mean(0) - Opt) @ ex))
    geom = {
        "floor": floor, "road_z": road_z, "lanes": lanes, "s_near": s_near, "s_far": s_far,
        "road_far_y": round(s_far - s_near, 2), "sidewalk_w": round(side_w, 2),
        "crosswalk_x": [float(round(cx_off - half_w, 2)), float(round(cx_off + half_w, 2))], "crosswalk_poly_world": A.tolist(),
    }
    log("[기하] 원점(연석·횡단보도 가운데) 월드 (%.2f, %.2f, %.2f), 인도 방향 ey=(%.3f, %.3f)" %
        (Opt[0], Opt[1], road_z, u[0], u[1]))
    log("[기하] 차도 폭 %.1f m (y %.1f ~ 0), 인도 폭 약 %.1f m, 횡단보도 x %s" %
        (-(s_far - s_near), s_far - s_near, side_w, geom["crosswalk_x"]))
    for L in lanes:
        log("[기하]   %s  y=%.2f  폭 %.2f  (road %s lane %s)" % (L["name"], L["y"], L["width"], L["key"][0], L["key"][2]))
    if not lanes:
        raise SystemExit("주행 차로를 못 찾음")
    return geom


def choose_lanes(geom):
    lanes = geom["lanes"]
    near = [L for L in lanes if L["y"] < 0]
    d1 = near[0]["dir"]
    same = [L for L in near if L["dir"] == d1]
    opp = [L for L in lanes if L["dir"] != d1]
    truck = same[0]
    A = same[1] if len(same) >= 2 else same[0]
    B = opp[-1] if opp else lanes[-1]
    band_lo = min(L["y"] - L["width"] / 2 for L in same)
    note = []
    if A is truck:
        note.append("같은 방향 차로가 1개뿐 → 트럭 없이 진행")
    if not opp:
        note.append("반대 방향 차로 없음 → B 는 가장 먼 차로")
    return {"truck": truck if A is not truck else None, "A": A, "B": B, "B_near": opp[0] if opp else B,
            "B_same": same[0] if len(same) >= 2 else None,
            "near_band_y": [round(band_lo, 2), 0.0],
            "dir_A": A["dir"], "note": note}


def lane_path(cmap, floor, lane, road_z, back_m, fwd_m):
    """차로 lane 의 가운데선을 (진행 방향 순서로) back_m 뒤 ~ fwd_m 앞까지. s_cross = 보행자 선(x=0)의 s."""
    wx, wy, _ = floor.to_world(0.0, lane["y"], road_z)
    wp0 = cmap.get_waypoint(carla.Location(x=wx, y=wy, z=road_z), project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    def straightest(cands, yaw):
        # 교차로에서 갈림길이 여럿이면 지금 방향과 가장 비슷한(직진) 것
        return min(cands, key=lambda w: abs((w.transform.rotation.yaw - yaw + 180) % 360 - 180))

    back, wp = [], wp0
    d = 0.0
    while d < back_m:
        prev = wp.previous(1.0)
        if not prev:
            break
        wp = straightest(prev, wp.transform.rotation.yaw)
        back.append(wp)
        d += 1.0
    fwd, wp = [], wp0
    d = 0.0
    while d < fwd_m:
        nxt = wp.next(1.0)
        if not nxt:
            break
        wp = straightest(nxt, wp.transform.rotation.yaw)
        fwd.append(wp)
        d += 1.0
    seq = list(reversed(back)) + [wp0] + fwd
    pts = [(w.transform.location.x, w.transform.location.y, w.transform.location.z) for w in seq]
    path = G.LanePath(pts)
    return path, float(path.s[len(back)]), len(back) * 1.0


# ───────────────────────── 2. 카메라 ─────────────────────────
def saved_cam(path, name):
    c = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))["cameras"][name]
    t = c["transform"]
    return {"x": t["x"], "y": t["y"], "z": t["z"], "pitch": t["pitch"], "yaw": t["yaw"], "roll": t.get("roll", 0.0),
            "fov": float(c["fov"]), "width": int(c["width"]), "height": int(c["height"]), "src": "saved:" + name}


def auto_cam(floor, road_z, a):
    wx, wy, _ = floor.to_world(*a["floor_xy"])
    tx, ty, _ = floor.to_world(*a["look_at"])
    cz = road_z + a["z"]
    pitch, yaw = G.look_at((wx, wy, cz), (tx, ty, road_z))
    return {"x": wx, "y": wy, "z": cz, "pitch": pitch, "yaw": yaw, "roll": 0.0, "fov": float(a["fov"]),
            "width": int(a["width"]), "height": int(a["height"]), "src": "auto"}


def cam_w2c(cam):
    return np.linalg.inv(G.carla_matrix(cam["x"], cam["y"], cam["z"], cam["pitch"], cam["yaw"], cam["roll"]))


def cam1_homography(geom, cam):
    f = geom["floor"]
    img, flo = G.ground_homography(f, geom["road_z"], cam_w2c(cam), cam["width"], cam["height"], cam["fov"],
                                   x_range=(-14, 14), y_range=(geom["road_far_y"] - 1, 4), step=1.0)
    if len(img) < 8:
        raise RuntimeError("Cam1 화면에 바닥 격자점이 %d개뿐 → 카메라가 횡단보도를 못 봄" % len(img))
    img, flo = G.pick_spread(img, flo, 24)
    H = homography_dlt(img, flo)
    err = reproj_errors(H, img, flo)
    return img, flo, H, float(np.mean(err))


def cam2_geometry(geom, cam, cfg):
    f = geom["floor"]
    g = cfg["geometry"]
    cw = geom["crosswalk_x"]
    mx = g["waiting_margin_x_m"]
    depth = min(g["waiting_depth_m"], max(geom["sidewalk_w"], 1.5)) if geom["sidewalk_w"] > 0 else g["waiting_depth_m"]
    poly_f = [[cw[0] - mx, 0.0], [cw[1] + mx, 0.0], [cw[1] + mx, depth], [cw[0] - mx, depth]]
    w2c = cam_w2c(cam)
    W, Hh = cam["width"], cam["height"]
    zc = geom["road_z"] + 0.15

    def px(pts):
        u, v, _ = G.project([f.to_world(a, b, zc) for a, b in pts], w2c, W, Hh, cam["fov"])
        return [[float(np.clip(u[i], 0, W - 1)), float(np.clip(v[i], 0, Hh - 1))] for i in range(len(pts))]

    curb = g.get("curb_floor")
    path = None
    if curb and len(curb) >= 2:
        # 실제 연석(인도 끝)이 휘어 있음 (모퉁이). Cam2 화면에서 눈에 보이는 인도·차도 경계를 따라가도록
        # 대기구역 윗변과 경계선을 이 꺾은선으로 (바닥 좌표 → 카메라 모델로 투영하므로 카메라를 옮겨도 따라감)
        cs = sorted([float(a), float(b)] for a, b in curb)
        xs = np.linspace(cw[0] - mx, cw[1] + mx, 23)
        ys = np.interp(xs, [p[0] for p in cs], [p[1] for p in cs])
        dep = max(depth, float(ys.max()) + 1.0)
        edge = [[float(a), float(b)] for a, b in zip(xs, ys)]
        path = px(edge)
        poly_px = path + px([[cw[1] + mx, dep], [cw[0] - mx, dep]])
        boundary = [path[0], path[-1]]          # 예전 Pi 코드용 직선 (양 끝점)
    else:
        poly_px = G.convex_hull(G.project_floor_poly(f, zc, poly_f, w2c, W, Hh, cam["fov"]))
        boundary = px([[cw[0] - mx, 0.0], [cw[1] + mx, 0.0]])
    full = G.project([f.to_world(p[0], p[1], geom["road_z"]) for p in poly_f], w2c, W, Hh, cam["fov"])
    vis = G.poly_area(poly_px) / max(G.poly_area(list(zip(full[0], full[1]))), 1.0)
    # 화면에서 1 m 가 몇 px 인지 (EMR 의 근접·방향 기준을 맞추는 참고값)
    p0 = G.project([f.to_world(0, 0.0, geom["road_z"]), f.to_world(0, 1.0, geom["road_z"])], w2c, W, Hh, cam["fov"])
    px_per_m = float(math.hypot(p0[0][1] - p0[0][0], p0[1][1] - p0[1][0]))
    return {"poly_floor": poly_f, "polygon": [[round(a, 1), round(b, 1)] for a, b in poly_px],
            "boundary": [[round(a, 1), round(b, 1)] for a, b in boundary],
            "boundary_path": path and [[round(a, 1), round(b, 1)] for a, b in path], "visible_ratio": round(vis, 2),
            "px_per_m_depth": round(px_per_m, 1)}


def spawn_camera(world, cam, fps, cb):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    for k, v in (("image_size_x", cam["width"]), ("image_size_y", cam["height"]), ("fov", cam["fov"]),
                 ("sensor_tick", 1.0 / fps)):
        bp.set_attribute(k, str(v))
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "bgx_simdemo_cam")
    tf = carla.Transform(carla.Location(x=cam["x"], y=cam["y"], z=cam["z"]),
                         carla.Rotation(pitch=cam["pitch"], yaw=cam["yaw"], roll=cam["roll"]))
    a = world.spawn_actor(bp, tf)
    a.listen(cb)
    return a


def to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()


# ───────────────────────── 3. 네트워크 ─────────────────────────
class FrameSender(threading.Thread):
    """카메라 1대 → Pi 1개 포트. 최신 프레임만 (밀리면 버림). 끊기면 2초마다 다시 접속."""

    def __init__(self, name, host, port, quality=85):
        super().__init__(daemon=True)
        self.name, self.host, self.port, self.q = name, host, port, quality
        self.cv = threading.Condition()
        self.item = None
        self.running = True
        self.sent = self.dropped = 0
        self.connected = False

    def submit(self, bgr, frame_no):
        with self.cv:
            if self.item is not None:
                self.dropped += 1
            self.item = (bgr, frame_no, time.time())
            self.cv.notify()

    def run(self):
        sock = None
        while self.running:
            if sock is None:
                try:
                    sock = socket.create_connection((self.host, self.port), timeout=3)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.connected = True
                    log("[영상] %s → Pi %s:%d 연결" % (self.name, self.host, self.port))
                except OSError as e:
                    self.connected = False
                    if self.sent == 0:
                        log("[영상] %s → Pi %s:%d 접속 실패 (%s). Pi 에서 edge.py 가 떠 있는지 확인. 2초 뒤 다시" %
                            (self.name, self.host, self.port, e))
                    time.sleep(2)
                    continue
            with self.cv:
                self.cv.wait_for(lambda: self.item is not None or not self.running, timeout=1.0)
                item, self.item = self.item, None
            if item is None:
                continue
            bgr, fno, t = item
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.q])
            if not ok:
                continue
            payload = struct.pack("!dI", t, fno & 0xFFFFFFFF) + buf.tobytes()
            try:
                sock.sendall(struct.pack("!I", len(payload)) + payload)
                self.sent += 1
            except OSError as e:
                log("[영상] %s 전송 끊김 (%s) → 다시 접속" % (self.name, e))
                try:
                    sock.close()
                except OSError:
                    pass
                sock, self.connected = None, False

    def stop(self):
        self.running = False
        with self.cv:
            self.cv.notify_all()


class ServerLink(threading.Thread):
    """노트북 서버 (http://127.0.0.1:8000). 상태는 최신 것만, 사건은 모아서 다음 상태에 실어 보냄."""

    def __init__(self, url):
        super().__init__(daemon=True)
        u = urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.conn = None
        self.lock = threading.Condition()
        self.state = None
        self.events = []
        self.levels = {}
        self.level_age = {}       # 차량별: 지금 단계를 서버가 앱으로 보낸 뒤 지난 시간 (응답 시점 기준, 실제 초)
        self.resp_wall = time.time()
        self.resp = {}
        self.ok = False
        self.running = True

    def level_ages(self, scale):
        """차량별: 앱이 지금 단계를 받은 뒤 지난 시간 (시뮬 초). 운전자 반응 시간을 '폰 수신 시각'부터 세는 데 씀."""
        dw = max(0.0, time.time() - self.resp_wall)
        return {v: (a + dw) * scale for v, a in self.level_age.items()}

    def _post(self, path, obj, timeout=3.0):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        for attempt in range(2):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
                self.conn.request("POST", path, body, {"Content-Type": "application/json"})
                r = self.conn.getresponse()
                data = r.read()
                return json.loads(data or b"{}")
            except (OSError, http.client.HTTPException, ValueError):
                try:
                    if self.conn:
                        self.conn.close()
                except OSError:
                    pass
                self.conn = None
                if attempt:
                    raise
        return {}

    def post_now(self, path, obj, timeout=5.0):
        """동기 호출 (보정 적용·시작/끝). 별도 연결을 써서 상태 스레드와 안 섞임."""
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            c.request("POST", path, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                      {"Content-Type": "application/json"})
            return json.loads(c.getresponse().read() or b"{}")
        finally:
            c.close()

    def get_now(self, path, timeout=3.0):
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            c.request("GET", path)
            return json.loads(c.getresponse().read() or b"{}")
        finally:
            c.close()

    def push(self, state, events=()):
        with self.lock:
            self.state = state
            self.events.extend(events)
            self.lock.notify()

    def run(self):
        warned = False
        while self.running:
            with self.lock:
                self.lock.wait_for(lambda: self.state is not None or not self.running, timeout=1.0)
                st, self.state = self.state, None
                ev, self.events = self.events, []
            if st is None:
                continue
            st = dict(st, events=list(st.get("events", [])) + ev)
            try:
                r = self._post("/api/sim", st)
                sn, ls = r.get("server_now"), r.get("level_sent_ms") or {}
                self.level_age, self.resp_wall = (
                    {v: max(0.0, (sn - ms) / 1000.0) for v, ms in ls.items() if ms and sn}, time.time())
                self.levels = r.get("levels", {})
                self.resp = r
                self.ok = True
                warned = False
            except Exception as e:  # noqa: BLE001
                self.ok = False
                with self.lock:          # 사건은 잃지 않게 되돌려 놓음
                    self.events = st["events"] + self.events
                if not warned:
                    log("[서버] /api/sim 전송 실패 (%s). 서버(tools/start_sim_server.sh)가 떠 있는지 확인" % e)
                    warned = True
                time.sleep(0.5)

    def stop(self):
        self.running = False
        with self.lock:
            self.lock.notify_all()


# ───────────────────────── 4. 신호등 ─────────────────────────
class Signals:
    """보행자 신호(시나리오 값) + CARLA 차량 신호등(가능하면 같이 바꿈)."""

    def __init__(self, world, cmap, floor, lanes, road_z, dir_A):
        self.lights_A, self.lights_other, self.frozen = [], [], False
        self.ped, self.veh = "GREEN", "RED"
        try:
            got = {}
            for L in lanes:
                wx, wy, _ = floor.to_world(L["dir"] * 25.0 * -1, L["y"], road_z)   # 횡단보도 앞(상류) 25 m
                wp = cmap.get_waypoint(carla.Location(x=wx, y=wy, z=road_z), lane_type=carla.LaneType.Driving)
                for tl in world.get_traffic_lights_from_waypoint(wp, 60.0):
                    got.setdefault(tl.id, (tl, True))   # 이 도로(A·B 양방향) 신호 = 차량 신호. 교차 도로는 반대
            A_ids = {k for k, (tl, isA) in got.items() if isA}
            group = []
            for tl, _ in got.values():
                group = tl.get_group_traffic_lights()
                break
            for tl in group:
                (self.lights_A if tl.id in A_ids else self.lights_other).append(tl)
            if self.lights_A:
                for tl in group:
                    tl.freeze(True)
                self.frozen = True
            log("[신호] 이 도로(차량 A·B) 신호등 %d개, 교차 도로 %d개 제어" % (len(self.lights_A), len(self.lights_other)))
        except Exception as e:  # noqa: BLE001
            log("[신호][경고] CARLA 신호등 제어 못 함 (%s) → 보행자 신호는 서버·관제에만 반영" % e)

    def set(self, ped, veh):
        changed = (ped, veh) != (self.ped, self.veh)
        self.ped, self.veh = ped, veh
        if changed and self.frozen:
            st = carla.TrafficLightState.Green if veh == "GREEN" else carla.TrafficLightState.Red
            other = carla.TrafficLightState.Red if veh == "GREEN" else carla.TrafficLightState.Green
            try:
                for tl in self.lights_A:
                    tl.set_state(st)
                for tl in self.lights_other:
                    tl.set_state(other)
            except RuntimeError:
                pass
        return changed

    def release(self):
        if self.frozen:
            try:
                for tl in self.lights_A + self.lights_other:
                    tl.freeze(False)
            except RuntimeError:
                pass


# ───────────────────────── 4-1. 추적 카메라 (차량 A 뒤에서 따라가며 녹화) ─────────────────────────
LV_COLOR = {"caution": (0, 200, 255), "warning": (0, 120, 255), "danger": (40, 40, 230)}
LV_KO = {"caution": "주의", "warning": "경고", "danger": "긴급"}
LV_EN = {"caution": "CAUTION", "warning": "WARNING", "danger": "DANGER"}
RS_KO = {"CPA": "Cam1 CPA 충돌 예측", "EMR": "Cam2 조기 징후(EMR)", "SIGNAL": "보행자 신호 전환"}
RS_EN = {"CPA": "Cam1 CPA", "EMR": "Cam2 EMR", "SIGNAL": "Ped signal FLASH/RED"}
SIG_COLOR = {"GREEN": (80, 200, 60), "FLASH": (80, 200, 60), "RED": (40, 40, 230)}
FONT_PATHS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"]


class Hud:
    """영상 위 글자. 한글 글꼴(Noto CJK·나눔)이 있으면 한글, 없으면 영어."""

    def __init__(self):
        self.font = None
        try:
            from PIL import ImageFont
            for fp in FONT_PATHS:
                if os.path.exists(fp):
                    self.font = {sz: ImageFont.truetype(fp, sz) for sz in (22, 28, 40)}
                    break
        except Exception:  # noqa: BLE001
            self.font = None
        self.ko = self.font is not None

    def draw(self, img, h):
        W, H = img.shape[1], img.shape[0]
        lv = h.get("level_A")
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (W, 64), (20, 20, 20), -1)
        if lv:
            cv2.rectangle(ov, (W - 520, 74), (W - 16, 74 + 112), LV_COLOR[lv], -1)
        cv2.rectangle(ov, (16, H - 92), (560, H - 16), (20, 20, 20), -1)
        img[:] = cv2.addWeighted(ov, 0.72, img, 0.28, 0)
        # 신호등 아이콘
        sig = h.get("ped_sig", "GREEN")
        blink_off = sig == "FLASH" and int(h.get("t", 0) * 2) % 2 == 1
        cv2.circle(img, (W - 150, 32), 14, (60, 60, 60) if blink_off else SIG_COLOR[sig], -1)
        cv2.circle(img, (W - 40, 32), 14, SIG_COLOR.get(h.get("veh_sig", "RED"), (60, 60, 60)), -1)
        spd = h.get("speed_A", 0) * 3.6
        rs = h.get("reasons") or []
        if self.ko:
            lines = [((16, 16), "BlindGuard-X · CARLA Town10HD J189 · take %s (%s) · t=%.1fs" %
                      (h.get("take"), h.get("variant"), h.get("t", 0)), 22, (255, 255, 255)),
                     ((W - 330, 18), "보행 신호", 22, (255, 255, 255)), ((W - 128, 18), "차량", 22, (255, 255, 255)),
                     ((30, H - 86), "차량 A  %.0f km/h%s" % (spd, "  · 감속" if h.get("braking") else ""), 28, (255, 255, 255)),
                     ((30, H - 48), "차량 B  %s" % ("경고 안 함" if not h.get("level_B") else LV_KO[h["level_B"]]), 22,
                      (180, 255, 180))]
            if lv:
                lines += [((W - 500, 80), "차량 A 앱 경고: %s" % LV_KO[lv], 40, (255, 255, 255)),
                          ((W - 500, 140), " + ".join(RS_KO.get(r, r) for r in rs), 22, (255, 255, 255))]
            from PIL import Image as PImage, ImageDraw
            pim = PImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pim)
            for (x, y), txt, sz, col in lines:
                d.text((x, y), txt, font=self.font[sz], fill=col[::-1] if isinstance(col, tuple) else col)
            img[:] = cv2.cvtColor(np.asarray(pim), cv2.COLOR_RGB2BGR)
        else:
            def put(txt, org, sc, col, th=2):
                cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
                cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
            put("BlindGuard-X | CARLA Town10HD J189 | take %s (%s) | t=%.1fs" % (h.get("take"), h.get("variant"),
                                                                             h.get("t", 0)), (16, 40), 0.7, (255, 255, 255))
            put("PED", (W - 215, 40), 0.6, (255, 255, 255))
            put("CAR", (W - 105, 40), 0.6, (255, 255, 255))
            put("Vehicle A  %.0f km/h%s" % (spd, "  BRAKING" if h.get("braking") else ""), (30, H - 56), 0.8, (255, 255, 255))
            put("Vehicle B  %s" % ("no alert" if not h.get("level_B") else LV_EN[h["level_B"]]), (30, H - 26), 0.6,
                (180, 255, 180))
            if lv:
                put("APP ALERT (A): %s" % LV_EN[lv], (W - 505, 120), 1.0, (255, 255, 255), 2)
                put(" + ".join(RS_EN.get(r, r) for r in rs), (W - 505, 165), 0.6, (255, 255, 255))
        return img


class ChaseRecorder:
    """차량 A 에 붙인 카메라(뒤 back_m, 위 up_m, 인도 쪽으로 yaw_off) → mp4. 시뮬 시간 기준 fps 로 저장."""

    def __init__(self, world, vehicle, path, c, side_sign):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in (("image_size_x", c["width"]), ("image_size_y", c["height"]), ("fov", c["fov"]),
                     ("sensor_tick", 1.0 / c["fps"])):
            bp.set_attribute(k, str(v))
        tf = carla.Transform(carla.Location(x=-c["back_m"], y=c.get("side_m", 0.0) * side_sign, z=c["up_m"]),
                             carla.Rotation(pitch=c["pitch"], yaw=c["yaw_off"] * side_sign))
        self.cam = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.path = path
        self.fps = c["fps"]
        self.hud = Hud() if c.get("hud", True) else None
        self.state = {}
        self.on = False
        self.q = queue.Queue(maxsize=120)
        self.writer = None
        self.n = 0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        self.cam.listen(self._cb)

    def set(self, on, **state):
        self.on = on
        self.state = dict(state)

    def _cb(self, image):
        if not self.on:
            return
        try:
            self.q.put_nowait((to_bgr(image), dict(self.state)))
        except queue.Full:
            pass

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            img, st = item
            if self.hud is not None:
                img = self.hud.draw(img, st)
            if self.writer is None:
                h, w = img.shape[:2]
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.writer.write(img)
            self.n += 1

    def close(self):
        try:
            self.cam.stop()
            self.cam.destroy()
        except RuntimeError:
            pass
        self.q.put(None)
        self.th.join(timeout=30)
        if self.writer is not None:
            self.writer.release()
        return self.n


def chase_spectator(world, vehicle, c, side_sign):
    """CARLA 창(관전자)도 같은 추적 시점으로."""
    tf = vehicle.get_transform()
    yaw = math.radians(tf.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    rx, ry = -fy, fx    # 오른쪽 (CARLA: yaw+90)
    side = c.get("side_m", 0.0) * side_sign
    loc = carla.Location(x=tf.location.x - fx * c["back_m"] + rx * side, y=tf.location.y - fy * c["back_m"] + ry * side,
                         z=tf.location.z + c["up_m"])
    world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=c["pitch"],
                                                                            yaw=tf.rotation.yaw + c["yaw_off"] * side_sign)))


def overview_spectator(world, f, rz, dir_A, c):
    """도입 전경: 차량 A 가 출발하기 전까지 횡단보도·트럭·대기구역이 한눈에 보이는 높은 시점 (Cam1 기둥 근처 위쪽).
    보행자가 대기구역으로 걸어오는 모습이 여기서 보임 → A 출발 직전에 추적 시점으로 바뀜."""
    px, py = c.get("floor_xy", [8.0, 3.2])
    tx, ty = c.get("look_at", [-1.5, -5.5])
    wx, wy, _ = f.to_world(dir_A * px, py)
    lx, ly, _ = f.to_world(dir_A * tx, ty)
    z = rz + c.get("z", 9.5)
    p, y = G.look_at((wx, wy, z), (lx, ly, rz))
    world.get_spectator().set_transform(carla.Transform(carla.Location(x=wx, y=wy, z=z), carla.Rotation(pitch=p, yaw=y)))


def take_result(cfg, take, A, B_alerted, pfl, f):
    """화면(HUD)·터미널에 띄울 결과 한 줄. 아직 결과가 안 났으면 None.  → {"text", "ok", "end": stop|pass|collision}"""
    bmsg = " · 차량 B 도 경고(의도와 다름)" if B_alerted else " · 차량 B 경고 없음"
    if take.collided:
        return {"text": "충돌 · 시속 %.0f km — 앱이 없어 경고를 받지 못함" % take.collided[1], "ok": False, "end": "collision"}
    ax, _ = f.to_floor(*A["world"])
    dA = take.ctx["lanes_sel"]["dir_A"]
    gap = (pfl[0] - ax) * dA - take.lenA / 2          # 차 앞범퍼 ~ 보행자 (진행 방향)
    if A["v"] < 0.05 and (take.brake_t is not None or take.fallback_used):
        if take.fallback_used:
            return {"text": "안전장치로 비상 정지 (경고가 늦음)" + bmsg, "ok": False, "end": "stop"}
        return {"text": "경고 받고 보행자 %.0f m 앞에서 정지%s" % (max(gap, 0.0), bmsg), "ok": not B_alerted, "end": "stop"}
    if gap < -4.0 and take.t_cross is None:           # 보행자를 지나침 (건너지 않은 상황)
        if take.got_alert:
            return {"text": "건너지 않았는데 경고가 울림 (오경고)", "ok": False, "end": "pass"}
        return {"text": "보행자는 건너지 않음 → 경고 없이 통과 (오경고 없음)", "ok": True, "end": "pass"}
    return None


LEVEL_ORDER = {None: 0, "": 0, "caution": 1, "warning": 2, "danger": 3}
LEVEL_KO = {0: "경고 없음", 1: "주의", 2: "경고", 3: "긴급"}


def judge_take(cfg, res):
    """기대(scenario 의 check)와 비교 → (기호, 설명). ✔ 기대대로 / △ 단계만 다름 / ✘ 의도와 다름"""
    chk = cfg.get("check")
    if not chk:
        return "", ""
    bad, soft = [], []
    if res["end"] != chk.get("A_end"):
        bad.append("차량 A 결과가 기대(%s)와 다름(%s)" % (chk.get("A_end"), res["end"]))
    if res["B_alerted"] and not chk.get("B_alert"):
        bad.append("차량 B 에 경고가 감")
    if res["fallback"]:
        bad.append("안전장치 사용(경고 늦음)")
    if "A_max" in chk and LEVEL_ORDER.get(chk["A_max"], 0) != res["A_max"]:
        soft.append("A 최고 단계: 기대 %s, 실제 %s" % (LEVEL_KO[LEVEL_ORDER.get(chk["A_max"], 0)], LEVEL_KO[res["A_max"]]))
    if bad:
        return "✘", "의도와 다름 → 다시 촬영: " + ", ".join(bad + soft)
    if soft:
        return "△", "정지·선별은 맞음, " + ", ".join(soft)
    return "✔", "기대대로"


def preflight(link, args, no_app, world):
    """촬영 전 점검: Pi·카메라 분석·폰 앱·관제 화면·상태 띠가 살아 있고 최신인지.
    ✘ 가 있으면 고칠 때까지 기다렸다가(최대 3분) 모두 ✔ 가 되면 자동으로 시작. 기다리는 동안에도 CARLA 는 계속 돌림."""
    url = "/api/preflight?hud=%d&no_app=%d" % (0 if args.no_hud else 1, 1 if no_app else 0)
    shown, t0 = None, time.time()
    if no_app:
        # 대조(앱 없음) 촬영: 폰 화면 창(scrcpy, 창 제목 PHONE)을 닫아 OBS 의 폰 자리가 비게 함 → 촬영 화면의 '앱 없음' 글자가 보임.
        # (11/24: OBS 눈 아이콘을 손으로 끄는 방식은 세 번 찍는 동안 계속 폰 화면이 같이 찍혔음)
        import subprocess
        try:
            r = subprocess.run(["pkill", "-x", "scrcpy"], timeout=3)
            log("[점검] ※ 대조(앱 없음) 촬영: 폰 화면 창(PHONE)을 %s → OBS 폰 자리에 '앱 없음' 글자가 보여야 함"
                % ("닫았음" if r.returncode == 0 else "찾지 못함 (이미 닫혀 있음)"))
            log("[점검]    다른 상황을 다시 찍을 때는 폰 화면을 다시 켜기: scrcpy --window-title PHONE --max-size 1024")
        except Exception as e:  # noqa: BLE001
            log("[점검] ※ 폰 화면 창을 닫지 못함 (%s) → OBS 에서 PHONE 소스의 눈 아이콘을 끄세요" % e)
    else:
        log("[점검] ※ OBS 에서 PHONE 소스의 눈 아이콘이 켜져 있어야 함 (폰 화면이 보여야 함)")
    try:
        while True:
            try:
                pf = link.get_now(url)
            except Exception as e:  # noqa: BLE001
                log("[점검] 서버가 점검을 지원하지 않음 (%s) → 서버를 다시 켜야 함: bash simdemo/start_sim_server.sh" % e)
                return
            need = [c for c in pf["checks"] if c["need"]]
            key = tuple((c["name"], c["ok"]) for c in need)
            if key != shown:
                shown = key
                for c in need:
                    log("[점검] %s %s" % ("✔" if c["ok"] else "✘", c["name"] + ("" if c["ok"] else "  → " + c["msg"])))
                if all(c["ok"] for c in need):
                    log("[점검] 모두 정상 → 촬영 시작")
                    return
                log("[점검] ✘ 항목을 고치면 자동으로 시작함 (기다리는 중 · 멈추려면 Ctrl+C · 그냥 진행하려면 --no-check)")
            if time.time() - t0 > 180:
                raise SystemExit("[점검] 3분 동안 ✘ 가 남아 있어 멈춤. 고친 뒤 다시 실행")
            for _ in range(20):                      # 1초쯤 CARLA 를 돌리며 기다림 (영상이 Pi 로 계속 감)
                world.tick()
                time.sleep(0.05)
    except KeyboardInterrupt:
        raise SystemExit("[점검] 멈춤")


# ───────────────────────── 5. 시나리오 한 번 (take) ─────────────────────────
def pick_bp(lib, names, fallback_filter="vehicle.*"):
    for n in names:
        try:
            return lib.find(n)
        except (IndexError, RuntimeError):
            continue
    return lib.filter(fallback_filter)[0]


def spawn_static(world, bp, x, y, z, yaw, role):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "bgx_simdemo_" + role)
    for dz in (0.3, 0.6, 1.0, 1.5):
        a = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=z + dz), carla.Rotation(yaw=yaw)))
        if a is not None:
            return a
    raise RuntimeError("%s 를 (%.1f, %.1f) 에 못 놓음 (다른 차가 있는지 확인, --clear 옵션)" % (role, x, y))


class Take:
    def __init__(self, world, ctx, cfg, take_no, args):
        self.world, self.ctx, self.cfg, self.no, self.args = world, ctx, cfg, take_no, args
        self.actors = []
        self.gnss = {}
        self.events_sent = set()
        self.fallback_used = False

    def destroy(self):
        for a in reversed(self.actors):
            try:
                if a.is_alive:
                    if a.type_id.startswith("sensor"):
                        a.stop()
                    a.destroy()
            except RuntimeError:
                pass
        self.actors = []

    def setup(self):
        w, ctx, cfg = self.world, self.ctx, self.cfg
        lib = w.get_blueprint_library()
        f, rz = ctx["floor"], ctx["road_z"]
        lanes = ctx["lanes_sel"]
        dA = lanes["dir_A"]
        # 보행자 일정 → A·B 도착 계획
        self.segs, self.t_cross = G.walker_segments(cfg["pedestrian"], cfg["signal"], {"road_far_y": ctx["road_far_y"]})
        ped = cfg["pedestrian"]
        t_ref = self.t_cross
        if t_ref is None:   # 건너지 않는 변형: 건넜다면 내려섰을 시각 기준으로 차는 똑같이
            t_ref = (ped["approach_t"] or (ped["sway_t"] or 15) + 2.0) + 1.5
        # 차량 A: 보행자가 A 차로 가운데에 닿는 시각에 A 가 보행자 선(x = wait_x)을 지나도록
        vA = cfg["vehicle_A"]["speed_kmh"] / 3.6
        yA = lanes["A"]["y"]
        t_conf = t_ref + abs(yA - ped["curb_y"]) / ped["speed_cross"] + cfg["vehicle_A"]["conflict_offset_s"]
        self.pathA, sA_cross, backA = lane_path(ctx["cmap"], f, lanes["A"], rz, cfg["vehicle_A"]["approach_m"] + 10, 80)
        sA_ped = self.pathA.s_nearest(*f.to_world(ped["wait_x"], yA)[:2])
        approachA = min(cfg["vehicle_A"]["approach_m"], sA_ped - 1.0)
        # 차 '중심'이 보행자 선을 지날 때 보행자가 A 차로 가운데 → 정면 충돌 경로 (11/21: 앞범퍼 기준이면 0.3 s 일찍 지나가 dCPA 가 커짐)
        tgoA = G.plan_go_time(approachA, vA, cfg["vehicle_A"]["accel"], t_conf)
        self.profA = G.SpeedProfile(sA_ped - approachA, tgoA, vA, cfg["vehicle_A"]["accel"], cfg["vehicle_A"]["brake_decel"])
        self.sA_ped = sA_ped
        # 차량 B: 반대 방향 차로. 기본 = 먼 차로로 보행자가 차도에 내려선 뒤 pass_offset_s 에 지남.
        #   lane = "near" 면 반대 방향 중 가까운 차로, sync_with_A 면 A 와 거의 같은 순간(sync_offset_s)에 지남 (선별 경고 장면)
        cB = cfg["vehicle_B"]
        #   lane = "same" 면 A 와 같은 방향의 연석 쪽 차로(트럭 차로 → 트럭 끔). pass_before_approach_s 만큼
        #   보행자가 연석으로 다가가기 '전에' 지나감 → 보행자는 B 바로 뒤로 뛰어들고, 옆 차로 A 와 충돌 경로 (같은 방향 선별 경고)
        if cB.get("lane") == "same":
            if lanes.get("B_same") is None:
                raise SystemExit("같은 방향 차로가 2개가 아니라 CASE 5(같은 방향) 불가")
            LB = lanes["B_same"]
        else:
            LB = lanes["B_near"] if cB.get("lane") == "near" else lanes["B"]
        self.laneB = LB
        vB = cB["speed_kmh"] / 3.6
        self.pathB, sB_cross, _ = lane_path(ctx["cmap"], f, LB, rz, cB["approach_m"] + 10, 80)
        sB_ped = self.pathB.s_nearest(*f.to_world(ped["wait_x"], LB["y"])[:2])
        approachB = min(cB["approach_m"], sB_ped - 1.0)
        if cB.get("lane") == "same":
            tB = ped["approach_t"] - cB.get("pass_before_approach_s", 0.6)
        elif cB.get("sync_with_A"):
            tB = t_conf + cB.get("sync_offset_s", 0.0)
        else:
            tB = t_ref + cB["pass_offset_s"]
        tgoB = G.plan_go_time(approachB, vB, cB["accel"], tB)
        self.profB = G.SpeedProfile(sB_ped - approachB, tgoB, vB, cfg["vehicle_B"]["accel"], 6.0)
        if cB.get("stop_at_line"):
            # 정지선에서 빨간불을 지키고 섬: 차 중심이 보행자 선 stop_gap_m 앞에서 멈춤 (보행자는 그 앞을 건넘)
            self.profB.stop_at = sB_ped - cB.get("stop_gap_m", 6.0)
            self.profB.stop_decel = cB.get("stop_decel", 2.5)
        log("[take %d] 보행자 차도 진입 %.1fs, A 출발 %.1fs → 보행자 선 %.1fs (%.0f km/h, %.0f m), B(%s) 출발 %.1fs → %.1fs" %
            (self.no, t_ref, tgoA, t_conf, vA * 3.6, approachA, LB["name"], tgoB, tB))
        if tgoA < 0.5 or tgoB < 0.5:
            log("[take %d][경고] 차량 출발 시각이 너무 이름 → 보행자 일정(enter_t 등)을 늦추거나 approach_m 을 줄이기" % self.no)
        # 액터: 트럭
        self.truck = None
        if cfg["truck"]["enabled"] and lanes["truck"] is not None:
            bp = pick_bp(lib, cfg["truck"]["blueprints"])
            tmp_len = 8.0
            ext = None
            Lt = lanes["truck"]
            # 트럭 앞이 횡단보도 상류 가장자리에서 gap_m 떨어지게
            edge = ctx["crosswalk_x"][1] if dA < 0 else ctx["crosswalk_x"][0]
            fx = edge - dA * (cfg["truck"]["gap_m"] + tmp_len / 2)
            wx, wy, _ = f.to_world(fx, Lt["y"])
            yaw = f.yaw_of_floor_dir(dA, 0.0)
            self.truck = spawn_static(w, bp, wx, wy, rz, yaw, "truck")
            self.actors.append(self.truck)
            ext = self.truck.bounding_box.extent
            fx = edge - dA * (cfg["truck"]["gap_m"] + ext.x)
            wx, wy, _ = f.to_world(fx, Lt["y"])
            self.truck.set_simulate_physics(False)
            self.truck.set_transform(carla.Transform(carla.Location(x=wx, y=wy, z=rz + 0.05), carla.Rotation(yaw=yaw)))
            self.truck_floor = {"id": "truck", "label": "정차 차량", "x": round(fx, 2), "y": Lt["y"],
                                "len": round(2 * ext.x, 2), "wid": round(2 * ext.y, 2)}
        # 차량 A, B (물리 끔 → 경로를 정확히 따라감)
        self.veh = {}
        for vid, path, prof, bpn in (("A", self.pathA, self.profA, cfg["vehicle_A"]["blueprint"]),
                                     ("B", self.pathB, self.profB, cfg["vehicle_B"]["blueprint"])):
            x, y, z, yaw = path.at(prof.s)
            a = spawn_static(w, pick_bp(lib, [bpn, "vehicle.lincoln.mkz_2020"]), x, y, z, yaw, "veh" + vid)
            a.set_simulate_physics(False)
            self.actors.append(a)
            self.veh[vid] = a
            gbp = lib.find("sensor.other.gnss")
            gbp.set_attribute("sensor_tick", "0.1")
            g = w.spawn_actor(gbp, carla.Transform(carla.Location(z=1.5)), attach_to=a)
            self.actors.append(g)
            self.gnss[vid] = None
            g.listen(lambda m, vid=vid: self.gnss.__setitem__(vid, (m.latitude, m.longitude, m.timestamp)))
        self.lenA = 2 * self.veh["A"].bounding_box.extent.x
        # 보행자
        wbps = [b for b in lib.filter("walker.pedestrian.*")]
        wbp = wbps[min(3, len(wbps) - 1)]
        if wbp.has_attribute("is_invincible"):
            wbp.set_attribute("is_invincible", "true")
        sx, sy = ped["start_floor"]
        wx, wy, _ = f.to_world(sx, sy)
        self.walker = spawn_static(w, wbp, wx, wy, rz + 0.2, f.yaw_of_floor_dir(1, 0), "walker")
        self.actors.append(self.walker)
        self.brake_t = None
        self.resp_done = set()
        self.resp_pending = []     # (반응 시각, 단계)
        self.got_alert = False
        self.fallback_kind = None
        self.collided = None       # (t, km/h) 앱 없음 대조 변형에서 충돌 시각·속도
        self.done_t = None

    # ── 매 틱 ──
    def step(self, t, dt, levels, ages=None):
        cfg, f = self.cfg, self.ctx["floor"]
        events = []
        # 신호
        sg = cfg["signal"]
        ped_sig = "GREEN" if t < sg["ped_flash_t"] else ("FLASH" if t < sg["ped_red_t"] else "RED")
        veh_sig = "GREEN" if t >= sg["veh_green_t"] else "RED"
        prev_ped, prev_veh = self.ctx["signals"].ped, self.ctx["signals"].veh
        if self.ctx["signals"].set(ped_sig, veh_sig):
            if ped_sig != prev_ped:
                events.append({"FLASH": "ped_flash", "RED": "ped_red"}.get(ped_sig, "ped_green"))
            if veh_sig != prev_veh:
                events.append("veh_green" if veh_sig == "GREEN" else "veh_red")
        # 보행자
        cur = None
        for s in self.segs:
            if s[0] <= t < s[1] or (s[0] == s[1] and abs(t - s[0]) < dt / 2):
                cur = s
            if s[4] and s[0] <= t and s[4] not in self.events_sent:
                self.events_sent.add(s[4])
                events.append(s[4])
        if cur is not None and cur[3] > 0 and not self.collided:
            dx, dy = f.vec_to_world(*cur[2])
            self.walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(dx, dy, 0.0), speed=cur[3]))
        else:
            self.walker.apply_control(carla.WalkerControl(speed=0.0))
        wl = self.walker.get_location()
        pfx, pfy = f.to_floor(wl.x, wl.y)
        # 차량 A 운전자: 앱으로 받은 단계에 반응 (반응 시간 뒤). 경고 → 서서히 감속, 긴급 → 정지
        lvA = levels.get("A")
        resp = cfg["vehicle_A"].get("response", {})
        if lvA:
            self.got_alert = True
            if lvA in resp and lvA not in self.resp_done and isinstance(resp[lvA], dict):
                self.resp_done.add(lvA)
                # 반응 시간은 폰이 경고를 받은 때부터 (시뮬이 0.1초 주기로 단계를 읽느라 늦게 안 것은 운전자 탓이 아님)
                age = min(max((ages or {}).get("A", 0.0), 0.0), cfg["vehicle_A"]["reaction_s"])
                self.resp_pending.append((t + cfg["vehicle_A"]["reaction_s"] - age, lvA))
        for item in list(self.resp_pending):
            if t >= item[0]:
                self.resp_pending.remove(item)
                r = resp[item[1]]
                self.profA.slow_to(r["to_kmh"] / 3.6, r["decel"])
                if self.brake_t is None:
                    self.brake_t = t
                events.append("brake_A")
                log("[take %d] t=%.1fs 차량 A %s (앱 '%s' 수신 → %d km/h 까지)" %
                    (self.no, t, "정지" if r["to_kmh"] == 0 else "감속", item[1], r["to_kmh"]))
        # 안전장치: 보행자 앞에서 지금 감속으로는 못 멈추면 비상 정지 (기록에 남김)
        if cfg["vehicle_A"]["fallback_stop"] and self.fallback_kind is None and self.profA.v > 0.5:
            x, y, _, _ = self.pathA.at(self.profA.s)
            vfx, vfy = f.to_floor(x, y)
            dA = self.ctx["lanes_sel"]["dir_A"]
            front = vfx + dA * self.lenA / 2
            gap = (pfx - front) * dA
            in_lane = abs(pfy - self.ctx["lanes_sel"]["A"]["y"]) < 2.5
            cur_dec = self.profA.dec if (self.profA.cap is not None and self.profA.cap < 0.5) else 0.0
            can_stop = cur_dec > 0 and self.profA.v ** 2 / (2 * cur_dec) + 1.0 < gap
            if in_lane and 0 < gap <= self.profA.v ** 2 / (2 * 8.0) + 1.5 and not can_stop:
                self.profA.slow_to(0.0, 8.0)
                self.fallback_used = True
                self.fallback_kind = "late" if self.got_alert else "no_alert"
                events.append("fallback_stop_A")
                log("[take %d][주의] t=%.1fs 안전장치로 A 비상 정지 (%s)" %
                    (self.no, t, "경고는 받았지만 늦음" if self.got_alert else "경고 못 받음 = 시스템 실패"))
        # 앱 없음 대조: 운전자가 보행자를 못 보고 그대로 진행 → 차 앞이 보행자에 닿으면 '충돌'로 그 자리에 멈춤
        #   (CARLA 보행자는 넘어지지 않아서, 차가 보행자를 뚫고 지나가지 않게 닿는 순간 멈추고 화면에 표시)
        if cfg["vehicle_A"].get("collision_stop") and not self.collided and self.profA.v > 0.3:
            x, y, _, _ = self.pathA.at(self.profA.s)
            vfx, vfy = f.to_floor(x, y)
            dA = self.ctx["lanes_sel"]["dir_A"]
            gap = (pfx - (vfx + dA * self.lenA / 2)) * dA
            if abs(pfy - self.ctx["lanes_sel"]["A"]["y"]) < 1.3 and -1.0 < gap <= 0.35:
                kmh = self.profA.v * 3.6
                self.collided = (t, kmh)
                self.profA.v = 0.0
                self.profA.slow_to(0.0, 100.0)
                events.append("collision_A")
                log("[take %d][충돌] t=%.1fs 차량 A 가 보행자와 충돌 (%.0f km/h) · 앱 없음 대조" % (self.no, t, kmh))
                try:
                    wl2 = self.walker.get_location()
                    self.world.debug.draw_string(carla.Location(x=wl2.x, y=wl2.y, z=wl2.z + 2.2),
                                                 "COLLISION  %.0f km/h  (NO APP)" % kmh, draw_shadow=True,
                                                 color=carla.Color(255, 0, 0), life_time=12.0)
                    self.world.debug.draw_box(carla.BoundingBox(carla.Location(x=wl2.x, y=wl2.y, z=wl2.z),
                                                                carla.Vector3D(0.7, 0.7, 1.05)), carla.Rotation(),
                                              thickness=0.12, color=carla.Color(255, 0, 0), life_time=12.0)
                except Exception:
                    pass
        out = {}
        for vid, prof, path in (("A", self.profA, self.pathA), ("B", self.profB, self.pathB)):
            s, v = prof.step(t, dt)
            x, y, z, yaw = path.at(s)
            self.veh[vid].set_transform(carla.Transform(carla.Location(x=x, y=y, z=z + 0.05), carla.Rotation(yaw=yaw)))
            tx, ty = path.tangent(s)
            out[vid] = {"world": (x, y), "v": v, "dir": (tx, ty), "braking": prof.braking}
        return events, out, (pfx, pfy), ped_sig, veh_sig

    def finished(self, t):
        end = None
        last = max(s[1] for s in self.segs)
        end = last + self.cfg["end_after_cross_s"]
        if self.t_cross is None:
            end = max(end, self.profA.t_go + 12.0)
        return t >= end


# ───────────────────────── 6. inspect (확인 그림) ─────────────────────────
def draw_check(img, geom, cam, title, extra=None):
    f = geom["floor"]
    w2c = cam_w2c(cam)
    W, H = cam["width"], cam["height"]

    def P(pts):
        u, v, d = G.project(pts, w2c, W, H, cam["fov"])
        return [(int(a), int(b)) if dd > 0.3 else None for a, b, dd in zip(u, v, d)]

    rz = geom["road_z"]
    for fx in range(-14, 15, 2):            # 1 m 격자 (2 m 간격 선)
        pts = P([f.to_world(fx, y, rz) for y in np.linspace(geom["road_far_y"], 4, 30)])
        for a, b in zip(pts, pts[1:]):
            if a and b:
                cv2.line(img, a, b, (90, 90, 90), 1)
    for fy in range(int(geom["road_far_y"]), 5, 2):
        pts = P([f.to_world(x, fy, rz) for x in np.linspace(-14, 14, 40)])
        for a, b in zip(pts, pts[1:]):
            if a and b:
                cv2.line(img, a, b, (90, 90, 90), 1)
    for yy, col in ((0.0, (0, 0, 255)), (geom["road_far_y"], (0, 0, 180))):   # 연석선
        pts = P([f.to_world(x, yy, rz) for x in np.linspace(-14, 14, 40)])
        for a, b in zip(pts, pts[1:]):
            if a and b:
                cv2.line(img, a, b, col, 2)
    for L in geom["lanes"]:
        pts = P([f.to_world(x, L["y"], rz) for x in np.linspace(-14, 14, 40)])
        for a, b in zip(pts, pts[1:]):
            if a and b:
                cv2.line(img, a, b, (0, 200, 255), 1)
    cwp = P([(p[0], p[1], rz) for p in geom["crosswalk_poly_world"]])
    cwp = [p for p in cwp if p]
    if len(cwp) >= 3:
        cv2.polylines(img, [np.array(cwp, np.int32)], True, (255, 255, 0), 2)
    o = P([f.to_world(0, 0, rz), f.to_world(2, 0, rz), f.to_world(0, 2, rz)])
    if all(o):
        cv2.arrowedLine(img, o[0], o[1], (0, 0, 255), 2)
        cv2.arrowedLine(img, o[0], o[2], (0, 255, 0), 2)
        cv2.putText(img, "+x", o[1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "+y sidewalk", o[2], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    wp_ = geom.get("_waiting_poly")
    if wp_:
        pts = P([f.to_world(p[0], p[1], rz + 0.15) for p in wp_])
        pts = [p for p in pts if p]
        if len(pts) >= 3:
            cv2.polylines(img, [np.array(pts, np.int32)], True, (255, 0, 255), 2)
    for L in geom["lanes"]:
        q = P([f.to_world(geom["crosswalk_x"][1] + 3.0, L["y"], rz)])[0]
        if q and 0 <= q[0] < img.shape[1] and 0 <= q[1] < img.shape[0]:
            cv2.putText(img, ("%s%s" % (L.get("role", ""), "->" if L["dir"] > 0 else "<-")), q,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if extra:
        extra(img)
    cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


# ───────────────────────── main ─────────────────────────
def main():
    global ROAD_TYPES
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000, help="CARLA 포트 (팀 실험 기록에는 2000·2100 둘 다 있음)")
    ap.add_argument("--server", default="http://127.0.0.1:8000", help="노트북 서버 (exp_server, BGX_SITE=simdemo/site_sim.json)")
    ap.add_argument("--pi", help="Pi IP (예: <PiIP>). 없으면 영상 안 보냄")
    ap.add_argument("--cam1-port", type=int, default=8765)
    ap.add_argument("--cam2-port", type=int, default=8767)
    ap.add_argument("--config", default=str(HERE / "scenario_j189.json"))
    ap.add_argument("--variant", default="main", help="scenario_j189.json variants: main / stages / wait_only")
    ap.add_argument("--camera-config", default=str(Path.home() / "BlindGuard-X/sim_carla/j189_camera_configs.json"))
    ap.add_argument("--takes", type=int, default=1, help="같은 장면 반복 횟수 (촬영할 때 3~5)")
    ap.add_argument("--pause", type=float, default=6.0, help="take 사이 쉬는 시간(초)")
    ap.add_argument("--warmup", type=float, default=4.0, help="take 시작 전 빈 장면으로 Cam2 배경 학습 (초)")
    ap.add_argument("--inspect", action="store_true", help="기하·카메라 확인 그림만 저장하고 끝 (Pi·서버 필요 없음)")
    ap.add_argument("--clear", action="store_true", help="J189 주변(60 m)의 다른 차량·보행자를 지움 (팀 Blind 장면 액터 포함)")
    ap.add_argument("--view", default="approach", choices=["approach", "top", "chase", "none"],
                    help="CARLA 창(관전자) 시점. chase = 차량 A 뒤에서 따라감")
    ap.add_argument("--chase", action="store_true", help="차량 A 추적 카메라 영상을 take 마다 mp4 로 저장 (화면 녹화 필요 없음)")
    ap.add_argument("--chase-from", type=float, default=None, help="녹화 시작 시각(초). 기본 = 차량 A 출발 1초 전")
    ap.add_argument("--no-check", action="store_true", help="촬영 전 점검(Pi·앱·관제·상태 띠)을 건너뜀")
    ap.add_argument("--no-hud", action="store_true", help="상태 띠(/hud) 창 점검을 건너뜀 (띠 없이 촬영)")
    ap.add_argument("--no-intro", action="store_true", help="도입 전경 없이 처음부터 추적 시점")
    ap.add_argument("--no-brake", action="store_true", help="경고를 받아도 A 가 감속하지 않음 (안전장치만)")
    ap.add_argument("--realtime", type=float, default=1.0, help="1.0 = 실제 시간 속도로 진행")
    args = ap.parse_args()

    ROAD_TYPES = {carla.LaneType.Driving, carla.LaneType.Parking, carla.LaneType.Bidirectional,
                  carla.LaneType.Shoulder, carla.LaneType.Border, carla.LaneType.Restricted,
                  carla.LaneType.Median}
    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    variants = base.get("variants", {})
    if args.variant not in variants:
        raise SystemExit("variant '%s' 없음: %s" % (args.variant, list(variants)))
    cfg = G.merge(base, variants[args.variant])
    if args.no_brake:
        cfg["vehicle_A"]["response"] = {}

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    if "Town10" not in cmap.name:
        raise SystemExit("지금 맵은 %s. Town10HD_Opt 를 먼저 띄우기 (팀 Blind 런처로 CARLA 실행)" % cmap.name)
    log("[CARLA] %s 연결, 맵 %s" % (client.get_server_version(), cmap.name))

    geom = build_geometry(world, cfg)
    sel = choose_lanes(geom)
    for n in sel["note"]:
        log("[기하][주의]", n)
    f, rz = geom["floor"], geom["road_z"]
    for role, L in (("TRUCK", sel["truck"]), ("A", sel["A"]), ("B", sel["B"])):
        if L is not None:
            L["role"] = (L.get("role", "") + " " + role).strip()
    log("[차로] 트럭 %s / A %s / B %s / Cam2 주의 대상 y %s" % (
        sel["truck"] and sel["truck"]["name"], sel["A"]["name"], sel["B"]["name"], sel["near_band_y"]))

    camcfg = cfg["cameras"]
    cam1 = saved_cam(args.camera_config, "cam1") if camcfg["cam1"]["source"] == "saved" else auto_cam(f, rz, camcfg["cam1_auto"])
    cam2 = saved_cam(args.camera_config, "cam2") if camcfg["cam2"]["source"] == "saved" else auto_cam(f, rz, camcfg["cam2_auto"])
    try:
        c1_img, c1_flo, H1, H1_err = cam1_homography(geom, cam1)
    except RuntimeError as e:
        log("[Cam1][경고] 저장된 Cam1 로 Homography 실패 (%s) → 자동 위치로 바꿈" % e)
        cam1 = auto_cam(f, rz, camcfg["cam1_auto"])
        c1_img, c1_flo, H1, H1_err = cam1_homography(geom, cam1)
    c2 = cam2_geometry(geom, cam2, cfg)
    geom["_waiting_poly"] = c2["poly_floor"]
    if c2["visible_ratio"] < 0.6 and cam2["src"] != "auto":
        log("[Cam2][경고] 대기구역이 화면에 %.0f%%만 보임 → 자동 위치로 바꿈" % (100 * c2["visible_ratio"]))
        cam2 = auto_cam(f, rz, camcfg["cam2_auto"])
        c2 = cam2_geometry(geom, cam2, cfg)
    log("[Cam1] %s %dx%d fov %.0f, Homography 격자점 %d개 재투영 오차 평균 %.3f m" %
        (cam1["src"], cam1["width"], cam1["height"], cam1["fov"], len(c1_img), H1_err))
    log("[Cam2] %s %dx%d fov %.0f, 대기구역 보이는 비율 %.0f%%, 연석 근처 1 m ≈ %.0f px (깊이 방향)" %
        (cam2["src"], cam2["width"], cam2["height"], cam2["fov"], 100 * c2["visible_ratio"], c2["px_per_m_depth"]))

    # GNSS → 바닥 좌표 기준 (지도 투영 두 점, 정확)
    g0 = cmap.transform_to_geolocation(carla.Location(*f.to_world(0, 0, rz)))
    g1 = cmap.transform_to_geolocation(carla.Location(*f.to_world(30, 0, rz)))
    anchor = Anchor.from_two_points((g0.latitude, g0.longitude), (0.0, 0.0), (g1.latitude, g1.longitude), (30.0, 0.0))
    gchk = cmap.transform_to_geolocation(carla.Location(*f.to_world(10, -6, rz)))
    ex = anchor.to_floor(gchk.latitude, gchk.longitude)
    gnss_mirror = False
    if math.hypot(ex[0] - 10, ex[1] + 6) <= 0.5:
        log("[GNSS] 위경도 → 바닥 좌표 확인: (10, -6) → (%.2f, %.2f) OK" % ex)
    elif math.hypot(ex[0] - 10, -ex[1] + 6) <= 0.5:
        gnss_mirror = True     # CARLA 위경도는 월드 y 가 거울 방향 → y 부호만 뒤집으면 맞음 (11/21 확인)
        log("[GNSS] 위경도 좌표가 좌우 거울 방향 → y 부호 보정. (10, -6) → (%.2f, %.2f) OK" % (ex[0], -ex[1]))
    else:
        log("[GNSS][경고] 위경도 변환이 0.5 m 넘게 어긋남 %s → 차량 위치는 월드 좌표로 보냄" % (ex,))
        anchor = None

    geom_msg = {"curb_y": 0.0, "road_far_y": geom["road_far_y"], "crosswalk_x": geom["crosswalk_x"],
                "lane_y": sel["A"]["y"], "stop_x": None, "lanes": [{k: L[k] for k in ("y", "dir", "width", "name")}
                                                                 for L in geom["lanes"]],
                "near_band_y": sel["near_band_y"], "waiting_xy": [0.0, round(min(1.2, c2["poly_floor"][2][1] / 2), 2)],
                "waiting_poly": c2["poly_floor"], "source": "CARLA %s J189" % cmap.name,
                "cam1": {k: cam1[k] for k in ("x", "y", "z", "pitch", "yaw", "fov", "width", "height", "src")},
                "cam2": {k: cam2[k] for k in ("x", "y", "z", "pitch", "yaw", "fov", "width", "height", "src")},
                "floor_frame": f.to_dict(), "H1_mean_err_m": round(H1_err, 3)}
    outdir = HERE / "logs" / time.strftime("sim_%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "geometry.json").write_text(json.dumps(dict(geom_msg, cam2_px=c2, cam1_img_pts=c1_img,
                                                          cam1_floor_pts=c1_flo, H1=np.asarray(H1).tolist(),
                                                          variant=args.variant, config=cfg), ensure_ascii=False,
                                                     indent=1), encoding="utf-8")

    # ── 카메라 띄우기 ──
    actors_static = []
    latest = {}
    fps = camcfg["fps"]
    original = world.get_settings()
    sync_set = False

    def restore():
        """동기 모드를 꼭 풀고 끝냄. 안 풀리면 CARLA 가 멈춘 채로 남아 다음 실행이 'time-out' 으로 실패함."""
        for _ in range(3):
            try:
                st = world.get_settings()
                st.synchronous_mode = False
                st.fixed_delta_seconds = original.fixed_delta_seconds
                world.apply_settings(st)
                return
            except RuntimeError:
                try:
                    world.tick()
                except RuntimeError:
                    pass
        log("[주의] CARLA 동기 모드를 풀지 못함 → 다음 실행에서 time-out 이 나면 CARLA 를 다시 켜기")

    try:
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = 0.05
        world.apply_settings(s)
        sync_set = True
        if args.clear:
            O = carla.Location(*f.to_world(0, -5, rz))
            n = 0
            for a in world.get_actors():
                if (a.type_id.startswith("vehicle.") or a.type_id.startswith("walker.") or
                        a.type_id.startswith("controller.ai.walker")) and a.get_location().distance(O) < 60:
                    try:
                        if a.type_id.startswith("controller"):
                            a.stop()
                        a.destroy()
                        n += 1
                    except RuntimeError:
                        pass
            log("[정리] J189 주변 액터 %d개 지움" % n)
        cams = {}
        for name, cam in (("cam1", cam1), ("cam2", cam2)):
            cams[name] = spawn_camera(world, cam, fps, lambda im, name=name: latest.__setitem__(name, im))
            actors_static.append(cams[name])
        # 관전자 시점 (CARLA 창 = 촬영 화면)
        if args.view != "none":
            dA = sel["dir_A"]
            if args.view == "approach":
                vx, vy, _ = f.to_world(-dA * 32, sel["A"]["y"] - 1.0)
                tx, ty, _ = f.to_world(-dA * 2, sel["A"]["y"] + 2.0)
                pz = rz + 7.0
            else:
                vx, vy, _ = f.to_world(0, -6, rz)
                tx, ty, _ = f.to_world(0.01, -6, rz)
                pz = rz + 40.0
            p, y = G.look_at((vx, vy, pz), (tx, ty, rz))
            if args.view == "top":
                p, y = -89.0, f.yaw_of_floor_dir(1, 0)
            world.get_spectator().set_transform(carla.Transform(carla.Location(x=vx, y=vy, z=pz),
                                                                carla.Rotation(pitch=p, yaw=y)))
        for _ in range(6):
            world.tick()
        time.sleep(0.5)
        world.tick()

        if args.inspect:
            for name, cam, extra in (("cam1", cam1, None), ("cam2", cam2, c2)):
                im = latest.get(name)
                if im is None:
                    log("[확인] %s 영상 없음" % name)
                    continue
                img = to_bgr(im)

                def ex2(img, c2=extra):
                    cv2.polylines(img, [np.array(c2["polygon"], np.int32)], True, (255, 200, 0), 2)
                    cv2.polylines(img, [np.array(c2.get("boundary_path") or c2["boundary"], np.int32)], False,
                                  (0, 0, 255), 3)
                img = draw_check(img, geom, cam, "%s check (%s) grid 2m, red=curb, yellow=lanes" % (name, cam["src"]),
                                 ex2 if extra else None)
                p = outdir / ("check_%s.jpg" % name)
                cv2.imwrite(str(p), img)
                log("[확인] 저장:", p)
            spec = world.get_spectator().get_transform()
            log("[확인] 끝. 그림·geometry.json 폴더: %s" % outdir)
            log("[확인] 관전자 위치: (%.1f, %.1f, %.1f)" % (spec.location.x, spec.location.y, spec.location.z))
            return

        # ── 서버·Pi ──
        link = ServerLink(args.server)
        link.start()
        try:
            st = link.get_now("/api/status")
            log("[서버] 연결. 기록 폴더 %s, Pi %s" % (st.get("logdir"), "연결" if st["edge"]["connected"] else "없음"))
        except Exception as e:  # noqa: BLE001
            raise SystemExit("서버(%s)에 못 붙음: %s → bash simdemo/start_sim_server.sh 먼저" % (args.server, e))
        link.post_now("/api/sim", {"geom": geom_msg, "phase": "setup", "vehicles": [], "events": []})
        senders = {}
        if args.pi:
            senders["cam1"] = FrameSender("Cam1", args.pi, args.cam1_port, camcfg["jpeg_quality"])
            senders["cam2"] = FrameSender("Cam2", args.pi, args.cam2_port, camcfg["jpeg_quality"])
            for sd in senders.values():
                sd.start()
            for name, cam in (("cam1", cam1), ("cam2", cam2)):
                cams[name].stop()
                cams[name].listen(lambda im, name=name: (latest.__setitem__(name, im),
                                                         senders[name].submit(to_bgr(im), im.frame)))
        else:
            log("[영상][주의] --pi 가 없어 Pi 로 영상을 보내지 않음 (서버·앱 연결만 확인)")
        # 보정 적용 (서버 → Pi). Pi 가 붙을 때까지 기다림
        t_wait = time.time()
        applied = {"cam1": False, "cam2": False}
        while not all(applied.values()) and time.time() - t_wait < 40:
            world.tick()
            if not applied["cam1"]:
                r = link.post_now("/api/calib/cam1", {"img_pts": c1_img, "world_pts": c1_flo, "apply": True})
                applied["cam1"] = bool(r.get("sent"))
            if not applied["cam2"]:
                r = link.post_now("/api/calib/cam2", {"polygon": c2["polygon"], "boundary": c2["boundary"],
                                                      "boundary_path": c2.get("boundary_path")})
                applied["cam2"] = bool(r.get("sent"))
            if not all(applied.values()):
                if int(time.time() - t_wait) % 5 == 0:
                    log("[보정] Pi 연결 기다리는 중… (Pi: python3 pi/edge.py --site simdemo/site_sim.json)")
                time.sleep(1.0)
        log("[보정] Cam1 Homography %s, Cam2 대기구역 %s" % ("적용" if applied["cam1"] else "실패(Pi 없음)",
                                                       "적용" if applied["cam2"] else "실패(Pi 없음)"))
        if args.pi and not args.no_check:
            for _ in range(25):                       # 점검 전에 1초쯤 영상을 흘려 Pi 분석이 도는지 봄
                world.tick()
                time.sleep(0.05)
            preflight(link, args, bool(cfg["vehicle_A"].get("collision_stop")), world)
        signals = Signals(world, cmap, f, geom["lanes"], rz, sel["dir_A"])
        ctx = {"floor": f, "road_z": rz, "cmap": cmap, "lanes_sel": sel, "road_far_y": geom["road_far_y"],
               "crosswalk_x": geom["crosswalk_x"], "signals": signals}

        stop = {"v": False}
        signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
        summary = []
        for k in range(1, args.takes + 1):
            if stop["v"]:
                break
            signals.set("GREEN", "RED")
            take = Take(world, ctx, cfg, k, args)
            rec = None
            gt_f = open(outdir / ("take%02d_gt.csv" % k), "w", newline="", encoding="utf-8")
            gt = csv.writer(gt_f)
            gt.writerow(["t", "ped_signal", "veh_signal", "ped_gt_x", "ped_gt_y", "A_x", "A_y", "A_v", "A_braking",
                         "B_x", "B_y", "B_v", "level_A", "level_B", "events"])
            first_level = {}
            try:
                take.setup()
                # 빈 장면으로 Cam2 배경 학습
                link.post_now("/api/sim", {"cmd": "take_start", "take": k, "t": 0.0, "phase": "warmup",
                                           "meta": {"title": cfg.get("title", args.variant), "variant": args.variant,
                                                    "sub": cfg.get("hud_sub"),
                                                    "no_app": bool(cfg["vehicle_A"].get("collision_stop"))},
                                           "note": "variant=%s" % args.variant, "vehicles": [],
                                           "obstacles": [take.truck_floor] if take.truck else [],
                                           "signal": {"ped": "GREEN", "veh": "RED"}, "events": ["take_start"]})
                t0w = time.time()
                while time.time() - t0w < args.warmup and not stop["v"]:
                    world.tick()
                    time.sleep(0.05 / args.realtime)
                log("[take %d] 시작 (variant %s%s)" % (k, args.variant, (" · " + cfg["title"]) if cfg.get("title") else ""))
                if cfg.get("expect"):
                    log("[take %d] 기대 결과: %s" % (k, cfg["expect"]))
                ch = cfg.get("chase", {})
                side_sign = -sel["dir_A"]           # 보행자(인도) 쪽: A 가 -x 로 가면 오른쪽(+)
                rec = None
                if args.chase:
                    rec = ChaseRecorder(world, take.veh["A"], outdir / ("take%02d_chase.mp4" % k), ch, side_sign)
                rec_from = args.chase_from if args.chase_from is not None else max(0.0, take.profA.t_go - 1.0)
                t, dt = 0.0, 0.05
                vs_prev = None
                wall0 = time.time()
                pace = []           # (실제 시각, 시뮬 시각) 최근 1.5 초 → CARLA 가 실제 시간의 몇 배로 도는지
                time_scale = 1.0
                last_post = -1.0
                intro = ch.get("intro", {})
                intro_until = (take.profA.t_go - intro.get("until_before_go_s", 1.0)
                               if (args.view == "chase" and intro.get("enabled", True) and not args.no_intro) else -1.0)
                result, result_t, a_max = None, None, 0
                end_hold = cfg.get("end_after_result_s", 5.0)
                while not stop["v"] and not take.finished(t) and not (result_t is not None and t >= result_t + end_hold):
                    events, vs, pfl, ped_sig, veh_sig = take.step(t, dt, link.levels, link.level_ages(time_scale))
                    if rec is not None:
                        lvr = link.levels
                        rec.set(t >= rec_from, t=t, take=k, variant=cfg.get("title", args.variant), ped_sig=ped_sig, veh_sig=veh_sig,
                                level_A=lvr.get("A"), level_B=lvr.get("B"),
                                reasons=(link.resp.get("reasons") or {}).get("A"),
                                speed_A=vs_prev["A"]["v"] if vs_prev else 0.0,
                                braking=vs_prev["A"]["braking"] if vs_prev else False)
                    if args.view == "chase":
                        if t < intro_until:
                            overview_spectator(world, f, rz, sel["dir_A"], intro)
                        else:
                            chase_spectator(world, take.veh["A"], ch, side_sign)
                    sigmsg = {"ped": ped_sig, "veh": veh_sig}
                    if ped_sig == "RED":
                        sigmsg["ped_red_elapsed_s"] = round(t - cfg["signal"]["ped_red_t"], 2)
                    world.tick()
                    # 시간 기준: CARLA 가 실제보다 느리게 돌면 Pi 가 '실제 시계'로 잰 보행자 속도가 시뮬보다 느려 보임.
                    # 서버가 이 배율(time_scale)로 보행자 속도·지연을 시뮬 시간 기준으로 바꿔 판정 (tCPA·ETA = 시뮬 초).
                    # 11/21 실행: 0.63배로 돌아 dCPA 3 m 틀어지고, 실제 초 기준 긴급은 시뮬에서 0.9 s 전이라 늦음
                    nw = time.time()
                    pace.append((nw, t))
                    while len(pace) > 2 and nw - pace[0][0] > 1.5:
                        pace.pop(0)
                    if len(pace) >= 10 and nw - pace[0][0] > 0.5:
                        time_scale = max(0.15, min(1.5, (t - pace[0][1]) / (nw - pace[0][0])))
                    if t - last_post >= 0.099:
                        last_post = t
                        vlist = []
                        for vid, s in vs.items():
                            gx = take.gnss.get(vid)
                            if anchor is not None and gx is not None:
                                fx, fy = anchor.to_floor(gx[0], gx[1])
                                if gnss_mirror:
                                    fy = -fy
                            else:
                                fx, fy = f.to_floor(*s["world"])
                            vfx, vfy = f.vec_to_floor(s["dir"][0] * s["v"], s["dir"][1] * s["v"])   # 가상 OBU 속도 (시뮬 m/s)
                            vlist.append({"id": vid, "kind": "vehicle", "x": round(fx, 3), "y": round(fy, 3),
                                          "vx": round(vfx, 3), "vy": round(vfy, 3), "speed": round(s["v"], 3),
                                          "lat": gx and gx[0], "lng": gx and gx[1], "braking": s["braking"]})
                        link.push({"take": k, "t": round(t, 2), "phase": cfg.get("title", args.variant), "time_scale": round(time_scale, 3),
                                   "signal": sigmsg, "vehicles": vlist,
                                   "obstacles": [take.truck_floor] if take.truck else []}, events)
                        events = []
                    elif events:
                        link.push({"take": k, "t": round(t, 2), "phase": args.variant,
                                   "signal": sigmsg, "vehicles": []}, events)
                    lv = link.levels
                    for vid in ("A", "B"):
                        if lv.get(vid) and vid not in first_level:
                            first_level[vid] = (round(t, 2), lv.get(vid))
                    a_max = max(a_max, LEVEL_ORDER.get(lv.get("A"), 0))
                    if result is None:
                        result = take_result(cfg, take, vs["A"], "B" in first_level, pfl, f)
                        if result is not None:
                            result_t = t
                            log("[take %d] t=%.1fs 결과: %s" % (k, t, result["text"]))
                            link.post_now("/api/sim", {"take": k, "t": round(t, 2), "phase": cfg.get("title", args.variant),
                                                       "result": {"text": result["text"], "ok": result["ok"]},
                                                       "vehicles": [], "events": []})
                    vs_prev = vs
                    A, B = vs["A"], vs["B"]
                    ax, ay = f.to_floor(*A["world"])
                    bx, by = f.to_floor(*B["world"])
                    gt.writerow([round(t, 2), ped_sig, veh_sig, round(pfl[0], 3), round(pfl[1], 3), round(ax, 3),
                                 round(ay, 3), round(A["v"], 3), A["braking"], round(bx, 3), round(by, 3),
                                 round(B["v"], 3), lv.get("A"), lv.get("B"), "|".join(events)])
                    t += dt
                    lag = (wall0 + t / args.realtime) - time.time()
                    if lag > 0:
                        time.sleep(lag)
                link.post_now("/api/sim", {"cmd": "take_end", "take": k, "t": round(t, 2), "phase": "end",
                                           "vehicles": [], "events": ["take_end"],
                                           "note": ("fallback_" + take.fallback_kind) if take.fallback_used else "ok"})
                if rec is not None:
                    n = rec.close()
                    rec = None
                    log("[take %d] 추적 영상 저장: %s (%d 프레임, %.1f초)" %
                        (k, outdir / ("take%02d_chase.mp4" % k), n, n / ch["fps"]))
                log("[take %d] CARLA 속도 = 실제 시간의 %.2f배 (서버가 Pi 보행자 속도를 이 비율로 시뮬 시간 기준으로 바꿔 판정)" % (k, time_scale))
                res = {"take": k, "variant": args.variant, "first_alert": first_level, "time_scale": round(time_scale, 3),
                       "brake_t": take.brake_t, "fallback_stop": take.fallback_kind, "collision": take.collided, "t_cross": take.t_cross,
                       "pi_frames_sent": {n: sd.sent for n, sd in senders.items()},
                       "pi_frames_dropped": {n: sd.dropped for n, sd in senders.items()}}
                res.update(result=result and result["text"], end=(result or {}).get("end", "none"), A_max=a_max,
                           B_alerted="B" in first_level, fallback=bool(take.fallback_used))
                mark, why = judge_take(cfg, res)
                res.update(check=mark, check_note=why)
                summary.append(res)
                log("[take %d] 끝: 첫 경고 %s, A 감속 %s, 안전장치 %s" %
                    (k, first_level or "없음", take.brake_t, "사용(실패)" if take.fallback_used else "안 씀"))
                log("━━ [결과] %s │ A 최고 단계 %s │ %s │ %s %s" %
                    (cfg.get("title", args.variant), LEVEL_KO[a_max], (result or {}).get("text", "결과 없음"), mark, why))
            finally:
                gt_f.close()
                if args.chase and rec is not None:
                    rec.close()
                take.destroy()
                world.tick()
            if k < args.takes and not stop["v"]:
                log("다음 take 까지 %.0f초" % args.pause)
                t1 = time.time()
                while time.time() - t1 < args.pause and not stop["v"]:
                    world.tick()
                    time.sleep(0.05)
        (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        log("[끝] 기록: %s  (서버 기록은 realtest/logs/server_*)" % outdir)
        signals.release()
        for sd in senders.values():
            sd.stop()
        link.stop()
    finally:
        for a in actors_static:
            try:
                a.stop()
                a.destroy()
            except RuntimeError:
                pass
        if sync_set:
            restore()


if __name__ == "__main__":
    main()
