# bgx/engine.py
#
# 차량(폰 GPS) 위치·속도 추정 + 보행자와의 CPA 판정 묶음. 서버와 모의 실행이 같은 코드를 쓴다.
#
# GPS → 바닥 좌표
#   - 위도·경도를 기준점(바닥 좌표를 아는 점) 주변 평면(동-북, m)으로 펴고 회전·이동해서 바닥 좌표로.
#   - 기준점은 현장에서 '같은 폰'을 그 점에 30초 놓고 평균 → 폰 GPS의 느린 치우침(수 분 동안 같은 방향)도
#     같이 빼지는 효과가 있다. (팀의 10/20 정지 25시간 실측: 오차가 몇 분씩 같은 방향으로 이어짐)
#   - 방향 축은 두 번째 점(연석선을 따라 10~20 m 떨어진 점)으로 정한다. 멀수록 각도 오차가 작다.
#
# 차량 속도
#   - 폰이 주는 도플러 속도(speed)와 원시 진행 방향(heading)이 있으면 그것 (걷는 속도에서도 비교적 안정)
#   - 없으면 최근 4초 GPS 위치를 직선 맞춤
#
# 자기 자신 제외
#   - 차량 역할 팀원도 Cam1 에 '보행자'로 찍힌다. 그대로 두면 자기 자신과 충돌(tCPA≈0) 경고가 나온다.
#   - 차량 GPS 위치 근처(self_radius_m)에서 차량과 속도가 비슷한(self_dv_mps) 트랙 하나를 '차량 본인'으로 보고 뺀다.
#   - 조작 페이지에서 트랙을 직접 지정할 수도 있다 (지정하면 그 트랙만 뺀다).
#   - 이때 '차량 본인 트랙'과 GPS 위치의 거리 = GPS 오차 추정값 (카메라를 정답으로 본 것) → 같이 기록.

import math
from collections import deque

import numpy as np

from .risk import RANK, AlertLevelHold, evaluate_vehicle

R_EARTH = 6371008.8


def enu(lat, lng, lat0, lng0):
    de = math.radians(lng - lng0) * R_EARTH * math.cos(math.radians(lat0))
    dn = math.radians(lat - lat0) * R_EARTH
    return de, dn


def rot(a, x, y):
    c, s = math.cos(a), math.sin(a)
    return c * x - s * y, s * x + c * y


class Anchor:
    """바닥 = Rot(phi) · ENU(lat, lng; lat0, lng0) + (x0, y0) + bias"""

    def __init__(self, lat0, lng0, phi, x0=0.0, y0=0.0, source="", detail=None):
        self.lat0, self.lng0, self.phi = lat0, lng0, phi
        self.x0, self.y0 = x0, y0
        self.bias = (0.0, 0.0)
        self.source = source
        self.detail = detail or {}

    @classmethod
    def from_two_points(cls, p0, f0, p1, f1):
        """p0, p1: (lat, lng) 측정 평균. f0, f1: 그 점들의 바닥 좌표 (줄자로 잰 값)."""
        de, dn = enu(p1[0], p1[1], p0[0], p0[1])
        a_e = math.atan2(dn, de)
        a_f = math.atan2(f1[1] - f0[1], f1[0] - f0[0])
        gps_d = math.hypot(de, dn)
        true_d = math.hypot(f1[0] - f0[0], f1[1] - f0[1])
        return cls(p0[0], p0[1], a_f - a_e, f0[0], f0[1], "two_points",
                   {"gps_dist_m": round(gps_d, 2), "true_dist_m": round(true_d, 2),
                    "dist_err_m": round(gps_d - true_d, 2)})

    @classmethod
    def from_bearing(cls, p0, f0, bearing_deg):
        """+x 축이 북에서 시계 방향으로 bearing_deg (지도에서 잰 값)."""
        b = math.radians(bearing_deg)
        a_e = math.atan2(math.cos(b), math.sin(b))
        return cls(p0[0], p0[1], -a_e, f0[0], f0[1], "bearing", {"bearing_deg": bearing_deg})

    def to_floor(self, lat, lng, with_bias=True):
        de, dn = enu(lat, lng, self.lat0, self.lng0)
        x, y = rot(self.phi, de, dn)
        x, y = x + self.x0, y + self.y0
        if with_bias:
            x, y = x + self.bias[0], y + self.bias[1]
        return x, y

    def heading_vec(self, heading_deg, speed):
        """폰 진행 방향(북 0°, 시계) + 속력 → 바닥 좌표 속도 벡터."""
        h = math.radians(heading_deg)
        return rot(self.phi, speed * math.sin(h), speed * math.cos(h))

    def to_dict(self):
        return {"lat0": self.lat0, "lng0": self.lng0, "phi_deg": round(math.degrees(self.phi), 3),
                "x0": self.x0, "y0": self.y0, "bias": [round(self.bias[0], 2), round(self.bias[1], 2)],
                "source": self.source, "detail": self.detail}

    @classmethod
    def from_dict(cls, d):
        a = cls(d["lat0"], d["lng0"], math.radians(d["phi_deg"]), d.get("x0", 0), d.get("y0", 0),
                d.get("source", ""), d.get("detail"))
        a.bias = tuple(d.get("bias", (0, 0)))
        return a


def weighted_mean_fix(fixes, max_acc=20.0):
    """fixes: [(lat, lng, acc)] → (lat, lng, n, mean_acc). 정확도 나쁜 값은 버리고 1/acc² 가중."""
    good = [f for f in fixes if f[2] is None or f[2] <= max_acc]
    if not good:
        return None
    w = np.array([1.0 / max(f[2] or 10.0, 1.0) ** 2 for f in good])
    lat = float(np.sum(w * np.array([f[0] for f in good])) / w.sum())
    lng = float(np.sum(w * np.array([f[1] for f in good])) / w.sum())
    accs = [f[2] for f in good if f[2] is not None]
    return lat, lng, len(good), (float(np.mean(accs)) if accs else None)


class LaneKalman:
    """차로 방향(x) 1차원 등속 칼만 필터. 상태 [x, v].
    측정: GPS x (분산 = 정확도²), 도플러 속도 v (분산 0.3²).
    핸드오프 설계의 '칼만 필터 → 맵 매칭(접근로·정지선까지 거리·속도)'을 걷는 실험 규모로 옮긴 것."""

    def __init__(self, accel_sigma=1.0, v_sigma=0.3):
        self.x = None
        self.P = None
        self.t = None
        self.qa = accel_sigma ** 2
        self.rv = v_sigma ** 2

    def _predict(self, t):
        if self.t is None:
            return
        dt = max(0.0, t - self.t)
        F = np.array([[1, dt], [0, 1]])
        Q = self.qa * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, t, x_meas, x_var, v_meas=None):
        if self.x is None:
            self.x = np.array([x_meas, v_meas if v_meas is not None else 0.0])
            self.P = np.diag([x_var, 0.5 if v_meas is not None else 4.0])
            self.t = t
            return
        self._predict(t)
        for z, Hrow, r in ((x_meas, np.array([1.0, 0.0]), x_var),) + (
                ((v_meas, np.array([0.0, 1.0]), self.rv),) if v_meas is not None else ()):
            S = Hrow @ self.P @ Hrow + r
            K = self.P @ Hrow / S
            self.x = self.x + K * (z - Hrow @ self.x)
            self.P = self.P - np.outer(K, Hrow @ self.P)

    def at(self, t):
        dt = t - self.t
        return float(self.x[0] + self.x[1] * dt), float(self.x[1])


DEFAULT_GPS = {
    "doppler_min_speed": 0.5,  # 이보다 느리면 도플러 방향을 안 믿음
    "stop_speed": 0.3,         # 이보다 느리면 정지(속도 0)
    "window_s": 4.0,           # 위치 직선 맞춤 구간
    "stale_s": 3.0,            # 마지막 GPS가 이보다 오래되면 차량 위치 모름
    "max_extrap_s": 1.5,       # 마지막 GPS 이후 이 시간까지만 앞당김
    "max_acc_m": 25.0,         # 정확도가 이보다 나쁜 GPS는 무시
    "map_match": True,         # 차량은 차로 위를 달린다고 보고 y를 차로 중심선에 맞춤 + 차로 방향 칼만
    "lane_y": -1.75,           # 차로 중심선 (바닥 y)
    "lane_corridor_m": 10.0,   # GPS가 차로에서 이 거리 안일 때만 맵 매칭 (밖이면 원래 위치)
}


class GpsVehicle:
    def __init__(self, vid, cfg=None):
        self.id = vid
        self.cfg = dict(DEFAULT_GPS, **(cfg or {}))
        self.hist = deque(maxlen=30)   # (t_s, x, y)
        self.last = None               # dict
        self.kf = LaneKalman()

    def update(self, t_fix, x, y, speed=None, vel_doppler=None, acc=None):
        """t_fix: 서버 시계 기준 초. vel_doppler: (vx, vy) 바닥 좌표 or None."""
        c = self.cfg
        if acc is not None and acc > c["max_acc_m"]:
            return None
        if c["map_match"] and abs(y - c["lane_y"]) <= c["lane_corridor_m"]:
            v_meas = None
            if speed is not None and speed < c["stop_speed"]:
                v_meas = 0.0
            elif vel_doppler is not None and speed is not None and speed >= c["doppler_min_speed"]:
                v_meas = vel_doppler[0]
            elif speed is not None and self.kf.x is not None and abs(self.kf.x[1]) > 0.3:
                v_meas = math.copysign(speed, self.kf.x[1])
            self.kf.update(t_fix, x, max(acc or 5.0, 2.0) ** 2, v_meas)
            self.last = {"t": t_fix, "x": float(self.kf.x[0]), "y": c["lane_y"], "vx": float(self.kf.x[1]),
                         "vy": 0.0, "vsrc": "lane_kf", "acc": acc, "raw_x": x, "raw_y": y, "mm": True}
            return self.last
        self.hist.append((t_fix, x, y))
        while self.hist and t_fix - self.hist[0][0] > c["window_s"]:
            self.hist.popleft()
        src = None
        vx = vy = None
        if speed is not None and speed < c["stop_speed"]:
            vx, vy, src = 0.0, 0.0, "stopped"
        elif vel_doppler is not None and speed is not None and speed >= c["doppler_min_speed"]:
            vx, vy, src = vel_doppler[0], vel_doppler[1], "doppler"
        elif len(self.hist) >= 3:
            a = np.array(self.hist)
            tc = a[:, 0] - a[:, 0].mean()
            den = float((tc * tc).sum())
            if den > 1e-6:
                vx = float((tc * (a[:, 1] - a[:, 1].mean())).sum() / den)
                vy = float((tc * (a[:, 2] - a[:, 2].mean())).sum() / den)
                src = "track"
        self.last = {"t": t_fix, "x": x, "y": y, "vx": vx, "vy": vy, "vsrc": src, "acc": acc, "raw_x": x,
                     "raw_y": y, "mm": False}
        return self.last

    def state(self, now_s):
        L = self.last
        if L is None or L["vx"] is None:
            return None
        age = now_s - L["t"]
        if age > self.cfg["stale_s"] or age < -1.0:
            return None
        dt = min(max(age, 0.0), self.cfg["max_extrap_s"])
        return {"id": self.id, "x": L["x"] + L["vx"] * dt, "y": L["y"] + L["vy"] * dt, "vx": L["vx"],
                "vy": L["vy"], "age_s": round(age, 3), "vsrc": L["vsrc"], "acc": L["acc"], "mm": L["mm"]}


DEFAULT_EXCL = {"self_radius_m": 8.0, "self_dv_mps": 0.8, "self_min_speed": 0.4}


def split_self(peds, veh, designated=None, cfg=None):
    """peds 에서 '차량 본인' 트랙을 뺀다. 반환 (peds_used, self_ped or None, how)."""
    cfg = dict(DEFAULT_EXCL, **(cfg or {}))
    valid = [p for p in peds if p.get("x_m") is not None]
    if designated is not None:
        me = [p for p in valid if p["track_id"] == designated]
        return [p for p in valid if p["track_id"] != designated], (me[0] if me else None), "designated"
    best, bd = None, None
    vs = math.hypot(veh["vx"], veh["vy"])
    for p in valid:
        if not p.get("v_ok"):
            continue
        d = math.hypot(p["x_m"] - veh["x"], p["y_m"] - veh["y"])
        dv = math.hypot(p["vx"] - veh["vx"], p["vy"] - veh["vy"])
        sp = math.hypot(p["vx"], p["vy"])
        if d <= cfg["self_radius_m"] and dv <= cfg["self_dv_mps"] and (sp >= cfg["self_min_speed"] or
                                                                           vs < cfg["self_min_speed"]):
            if bd is None or d < bd:
                best, bd = p, d
    if best is None:
        return valid, None, None
    return [p for p in valid if p is not best], best, "auto"


class ConflictEngine:
    """차량 여러 대 각각의 경고 단계 유지 + 판정."""

    def __init__(self, risk_cfg, hold_s=1.0):
        self.risk_cfg = risk_cfg
        self.hold_s = hold_s
        self.holds = {}

    def hold(self, vid):
        if vid not in self.holds:
            self.holds[vid] = AlertLevelHold(self.hold_s)
        return self.holds[vid]

    def step(self, vid, now_ms, veh, peds, designated=None, exclude_self=True, excl_cfg=None):
        """veh None 이면 판정 없이 단계만 시간 경과로 갱신."""
        h = self.hold(vid)
        info = {"self_track": None, "self_dist": None, "self_how": None}
        if veh is None:
            res = {"status": "NO_VEHICLE", "level": None}
        else:
            used = [p for p in peds if p.get("x_m") is not None]
            if exclude_self:
                used, me, how = split_self(peds, veh, designated, excl_cfg)
                if me is not None:
                    info.update(self_track=me["track_id"], self_how=how,
                                self_dist=round(math.hypot(me["x_m"] - veh["x"], me["y_m"] - veh["y"]), 2))
            res = evaluate_vehicle(veh, used, now_ms, self.risk_cfg)
        changed, prev, new = h.update(now_ms, res.get("level"))
        res.update(info)
        return res, changed, prev, new


__all__ = ["Anchor", "GpsVehicle", "ConflictEngine", "split_self", "weighted_mean_fix", "enu", "RANK"]
