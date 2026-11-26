# simdemo/simgeom.py
#
# carla_demo.py 의 계산 부분 (CARLA 없이 시험할 수 있게 numpy 만 씀)
#   - CARLA 월드 좌표 ↔ 바닥 좌표 (realtest 약속: 연석선 y=0, 인도 +y, 차도 -y, '인도를 왼쪽에 두고 보는 방향'이 +x)
#   - CARLA 카메라 모델 (Transform 행렬, 투영) → Cam1 Homography (영상 → 바닥), Cam2 대기구역 다각형·경계선 (픽셀)
#   - 차로 경로 위 차량 움직임 (출발·가속·정속·감속)과 '정해진 시각에 정해진 곳 도착' 계획
#
# CARLA(UE4) 좌표: 왼손 좌표계. x 앞, y 오른쪽, z 위. yaw 는 위에서 볼 때 시계 방향(+x → +y).
#   방향 yaw 의 '왼쪽' = (sin yaw, -cos yaw).  → 바닥 +y(인도)가 왼쪽에 오도록 ex 를 정한다.

import math

import numpy as np


# ───────────── 바닥 좌표 ─────────────
class Floor:
    """O: 바닥 원점(연석선 위, 횡단보도 가운데)의 월드 좌표 (x, y, z).
    ey: 차도 → 인도 방향 단위 벡터 (월드 xy). ex 는 ey 가 왼쪽이 되는 방향."""

    def __init__(self, O, ey):
        self.O = np.array(O, float)
        ey = np.array(ey[:2], float)
        self.ey = ey / np.linalg.norm(ey)
        self.ex = np.array([-self.ey[1], self.ey[0]])   # left(ex) = (ex_y, -ex_x) = ey

    def to_floor(self, wx, wy):
        d = np.array([wx - self.O[0], wy - self.O[1]])
        return float(d @ self.ex), float(d @ self.ey)

    def to_world(self, fx, fy, z=None):
        p = self.O[:2] + fx * self.ex + fy * self.ey
        return float(p[0]), float(p[1]), float(self.O[2] if z is None else z)

    def vec_to_floor(self, vx, vy):
        v = np.array([vx, vy])
        return float(v @ self.ex), float(v @ self.ey)

    def vec_to_world(self, fx, fy):
        v = fx * self.ex + fy * self.ey
        return float(v[0]), float(v[1])

    def yaw_of_floor_dir(self, fx, fy):
        wx, wy = self.vec_to_world(fx, fy)
        return math.degrees(math.atan2(wy, wx))

    def to_dict(self):
        return {"O": self.O.tolist(), "ey": self.ey.tolist(), "ex": self.ex.tolist()}


def left_of_yaw(yaw_deg):
    y = math.radians(yaw_deg)
    return math.sin(y), -math.cos(y)


# ───────────── CARLA 카메라 ─────────────
def carla_matrix(x, y, z, pitch, yaw, roll):
    """carla.Transform.get_matrix() 와 같은 식 (LibCarla Transform::GetMatrix)."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
        [sp, -cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0]])


def focal(width, fov_deg):
    return width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def project(world_xyz, w2c, width, height, fov):
    """world_xyz (N,3) → (u, v, depth). CARLA 카메라: x 앞, y 오른쪽, z 위 → u = cx + f·y/x, v = cy − f·z/x."""
    P = np.c_[np.asarray(world_xyz, float), np.ones(len(world_xyz))]
    c = (np.asarray(w2c) @ P.T)
    depth = c[0]
    f = focal(width, fov)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2.0 + f * c[1] / depth
        v = height / 2.0 - f * c[2] / depth
    return u, v, depth


def look_at(cam_xyz, target_xyz):
    """카메라 위치에서 목표를 보는 (pitch, yaw) [도]. CARLA: pitch + 가 위."""
    dx, dy, dz = (np.asarray(target_xyz, float) - np.asarray(cam_xyz, float))
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return pitch, yaw


def ground_homography(floor, z_ground, w2c, width, height, fov, x_range=(-12, 12), y_range=(-16, 4), step=1.0,
                      margin=8):
    """바닥 격자점을 카메라로 투영해서 (영상 px → 바닥 m) Homography 의 대응점 목록을 만든다.
    반환: img_pts, floor_pts (화면 안에 보이는 점만)."""
    img, flo = [], []
    xs = np.arange(x_range[0], x_range[1] + 1e-6, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-6, step)
    pts_w, pts_f = [], []
    for fx in xs:
        for fy in ys:
            wx, wy, _ = floor.to_world(fx, fy)
            pts_w.append((wx, wy, z_ground))
            pts_f.append((fx, fy))
    u, v, d = project(pts_w, w2c, width, height, fov)
    for i in range(len(pts_w)):
        if d[i] > 0.5 and margin <= u[i] <= width - margin and margin <= v[i] <= height - margin:
            img.append([float(u[i]), float(v[i])])
            flo.append([float(pts_f[i][0]), float(pts_f[i][1])])
    return img, flo


def pick_spread(img_pts, floor_pts, n=24):
    """DLT 에 넣을 점을 화면 전체에 고르게 n 개 (격자 전부를 보내면 메시지가 커짐)."""
    if len(img_pts) <= n:
        return img_pts, floor_pts
    a = np.asarray(img_pts)
    chosen = [int(np.argmin(a[:, 0] + a[:, 1]))]
    dmin = np.linalg.norm(a - a[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(dmin))
        chosen.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(a - a[k], axis=1))
    return [img_pts[i] for i in chosen], [floor_pts[i] for i in chosen]


def project_floor_poly(floor, z, poly_floor, w2c, width, height, fov, samples=12):
    """바닥 다각형 → 화면 다각형 (변마다 촘촘히 투영해서 화면 밖은 잘라냄)."""
    pts = []
    n = len(poly_floor)
    for i in range(n):
        a, b = np.array(poly_floor[i], float), np.array(poly_floor[(i + 1) % n], float)
        for k in range(samples):
            p = a + (b - a) * k / samples
            pts.append(floor.to_world(p[0], p[1], z))
    u, v, d = project(pts, w2c, width, height, fov)
    out = []
    for i in range(len(pts)):
        if d[i] <= 0.3:
            continue
        out.append([float(np.clip(u[i], 0, width - 1)), float(np.clip(v[i], 0, height - 1))])
    return out


def convex_hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) <= 2:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def poly_area(p):
    p = np.asarray(p, float)
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))


# ───────────── 차로 경로 · 차량 움직임 ─────────────
class LanePath:
    """차로 중심선 위 점들 (진행 방향 순서). s = 누적 거리."""

    def __init__(self, pts_xyz):
        self.p = np.asarray(pts_xyz, float)
        seg = np.linalg.norm(np.diff(self.p[:, :2], axis=0), axis=1)
        self.s = np.r_[0.0, np.cumsum(seg)]
        self.length = float(self.s[-1])

    def at(self, s):
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.s) - 2))
        t = (s - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-9)
        a, b = self.p[i], self.p[i + 1]
        q = a + (b - a) * t
        yaw = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        return float(q[0]), float(q[1]), float(q[2]), yaw

    def tangent(self, s):
        x0, y0, _, _ = self.at(s - 0.5)
        x1, y1, _, _ = self.at(s + 0.5)
        d = math.hypot(x1 - x0, y1 - y0) or 1.0
        return (x1 - x0) / d, (y1 - y0) / d

    def s_nearest(self, wx, wy):
        d = np.hypot(self.p[:, 0] - wx, self.p[:, 1] - wy)
        return float(self.s[int(np.argmin(d))])


def plan_go_time(distance, v, accel, t_arrive):
    """정지 상태에서 t_go 에 출발 → accel 로 v 까지 → 정속. distance 를 t_arrive 에 지나도록 t_go 계산."""
    d_acc = v * v / (2 * accel)
    if distance <= d_acc:
        return t_arrive - math.sqrt(2 * distance / accel)
    return t_arrive - (v / accel) - (distance - d_acc) / v


class SpeedProfile:
    """매 틱 적분. go 시각 전엔 정지, 그 뒤 v_target 까지 가속.
    slow_to(v, decel): 운전자 반응 (경고 → 서서히 감속, 긴급 → 정지). 한 번 줄인 목표 속도는 다시 올리지 않음."""

    def __init__(self, s0, t_go, v_target, accel, decel=6.0):
        self.s, self.v = float(s0), 0.0
        self.t_go, self.vt, self.a, self.b = t_go, v_target, accel, decel
        self.cap = None          # 운전자가 정한 최고 속도 (None = 원래 목표)
        self.dec = decel
        self.stop_at = None      # 이 s 에서 서도록 (정지선에서 신호 지키기). None = 안 섬
        self.stop_decel = 2.5

    @property
    def braking(self):
        return self.cap is not None

    def slow_to(self, v, decel):
        if self.cap is None:
            self.cap, self.dec = v, decel
        else:
            self.cap, self.dec = min(self.cap, v), max(self.dec, decel)

    def brake(self, decel=None):
        self.slow_to(0.0, decel or self.b)

    def step(self, t, dt):
        if self.stop_at is not None and self.v > 0.05:
            d = self.stop_at - self.s
            if d <= self.v ** 2 / (2 * self.stop_decel) + self.v * dt:
                self.slow_to(0.0, self.v ** 2 / (2 * max(d, 0.05)))
        target = self.vt if t >= self.t_go else 0.0
        if self.cap is not None:
            target = min(target, self.cap)
        if self.v > target:
            self.v = max(target, self.v - self.dec * dt)
        elif t >= self.t_go:
            self.v = min(target, self.v + self.a * dt)
        self.s += self.v * dt
        return self.s, self.v


# ───────────── 보행자 일정 ─────────────
def walker_segments(ped, sig, floor_geom):
    """[(t0, t1, (fx, fy) 방향, 속력, 사건)] 바닥 좌표 기준 연출 일정과, 차도 진입 시각(없으면 None)."""
    segs = []
    x0, y0 = ped["start_floor"]
    wx = ped["wait_x"]
    t = ped["enter_t"]
    d = abs(wx - x0)
    dirx = 1.0 if wx >= x0 else -1.0
    segs.append((t, t + d / ped["speed_walk"], (dirx, 0.0), ped["speed_walk"], "ped_enter"))
    t_arrive = t + d / ped["speed_walk"]
    segs.append((t_arrive, t_arrive, (0.0, 0.0), 0.0, "ped_wait"))
    st = ped.get("sway_t")
    if st is not None:
        h = ped["sway_s"] / 2
        segs.append((st, st + h, (1.0, 0.0), ped["sway_speed"], "ped_sway"))
        segs.append((st + h, st + 2 * h, (-1.0, 0.0), ped["sway_speed"], None))
    t_cross = None
    at = ped.get("approach_t")
    if at is not None:
        da = max(y0 - ped["curb_y"], 0.0)
        t_curb = at + da / ped["approach_speed"]
        segs.append((at, t_curb, (0.0, -1.0), ped["approach_speed"], "ped_approach"))
        if ped.get("cross", True):
            dist = abs(floor_geom["road_far_y"]) + 1.0
            segs.append((t_curb, t_curb + dist / ped["speed_cross"], (0.0, -1.0), ped["speed_cross"], "ped_cross"))
            t_cross = t_curb
    return segs, t_cross


def merge(base, over):
    """설정 dict 깊은 합치기 (variants 용)."""
    out = dict(base)
    for k, v in (over or {}).items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out
