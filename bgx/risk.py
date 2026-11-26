# bgx/risk.py
#
# 차량별 CPA(최근접점) 위험 판정. 포스터 최종본 규칙 그대로:
#   - 3초 예측, 안전반경 2 m, 등속 가정
#   - dCPA <= 2 m 이고 0 <= tCPA <= 3 s 이면 경고
# 단계 나누기는 포스터에 없어서 [임시] 값: tCPA <= 1 s 긴급, <= 2 s 경고, <= 3 s 주의
#
# 순수 파이썬 (서버·시뮬·테스트가 같이 씀)

import math

LEVELS = ("caution", "warning", "danger")
RANK = {None: 0, "caution": 1, "warning": 2, "danger": 3}

DEFAULT_RISK = {
    "horizon_s": 3.0,
    "safe_radius_m": 2.0,
    "level_tcpa_s": {"danger": 1.0, "warning": 2.0, "caution": 3.0},
    "compensate_latency": True,  # 프레임 촬영 시각부터 지금까지 보행자를 등속으로 앞당겨 계산
    "max_compensate_s": 0.5,
    "ped_still_mps": 0.0,   # 보행자 속력이 이보다 작으면 '서 있음'(속도 0)으로 봄. 0 = 포스터 규칙 그대로(끔)
}


def cpa(px, py, vx, vy):
    """상대 위치 p(보행자 - 차량), 상대 속도 v(보행자 - 차량) → (tCPA, dCPA).
    상대 속도가 0이면 지금 거리가 계속 유지 → t = 0."""
    vv = vx * vx + vy * vy
    if vv < 1e-9:
        return 0.0, math.hypot(px, py)
    t = -(px * vx + py * vy) / vv
    return t, math.hypot(px + vx * t, py + vy * t)


def level_from_tcpa(t, cfg):
    th = cfg["level_tcpa_s"]
    if t <= th["danger"]:
        return "danger"
    if t <= th["warning"]:
        return "warning"
    return "caution"


def relative_angle_deg(hx, hy, rx, ry):
    """차량 진행 방향 (hx,hy) 기준으로 물체 방향 (rx,ry)의 각도.
    앱 약속: 정면 0°, 오른쪽 +90°, 왼쪽 -90° (바닥 좌표가 오른손 좌표계라는 전제)."""
    if hx == 0 and hy == 0:
        hx, hy = 1.0, 0.0
    cross = hx * ry - hy * rx  # 반시계(왼쪽)가 +
    dot = hx * rx + hy * ry
    return -math.degrees(math.atan2(cross, dot))


def side_from_angle(a):
    if -45 <= a <= 45:
        return "front"
    if 45 < a <= 135:
        return "right"
    if -135 <= a < -45:
        return "left"
    return "back"


def evaluate_vehicle(veh, peds, now_ms, cfg=None):
    """차량 1대 vs 보행자 여러 명.

    veh : {"id", "x", "y", "vx", "vy"}            (바닥 좌표, m, m/s)
    peds: [{"track_id","x_m","y_m","vx","vy","v_ok","ts_cap"}]   ts_cap: 서버 시계 ms
    반환: {"status": "WARN"|"SAFE"|"UNKNOWN", "level", "track_id", "tcpa", "dcpa",
           "dist", "angle_deg", "side"}
    """
    cfg = cfg or DEFAULT_RISK
    H, R = cfg["horizon_s"], cfg["safe_radius_m"]
    best = None
    n_valid = 0
    nearest = None  # 위험이 없을 때 레이더용 가장 가까운 보행자
    for p in peds:
        if not p.get("v_ok"):
            continue
        n_valid += 1
        px, py = p["x_m"], p["y_m"]
        pvx, pvy = p["vx"], p["vy"]
        if math.hypot(pvx, pvy) < cfg.get("ped_still_mps", 0.0):
            pvx = pvy = 0.0
        if cfg.get("compensate_latency") and p.get("ts_cap") is not None:
            dt = min(max((now_ms - p["ts_cap"]) / 1000.0, 0.0), cfg.get("max_compensate_s", 0.5))
            px += pvx * dt
            py += pvy * dt
        rx, ry = px - veh["x"], py - veh["y"]
        rvx, rvy = pvx - veh["vx"], pvy - veh["vy"]
        t, d = cpa(rx, ry, rvx, rvy)
        dist = math.hypot(rx, ry)
        ang = relative_angle_deg(veh["vx"], veh["vy"], rx, ry)
        cand = {"track_id": p["track_id"], "tcpa": t, "dcpa": d, "dist": dist,
                "angle_deg": ang, "side": side_from_angle(ang),
                "ped_xy": (round(px, 3), round(py, 3)), "ped_v": (round(pvx, 3), round(pvy, 3))}   # 판정에 쓴 보행자 (기록용)
        if nearest is None or dist < nearest["dist"]:
            nearest = cand
        if d <= R and 0.0 <= t <= H:
            cand["level"] = level_from_tcpa(t, cfg)
            if best is None or t < best["tcpa"]:
                best = cand
    if best is not None:
        return dict(best, status="WARN")
    if n_valid == 0:
        return {"status": "UNKNOWN", "level": None, "track_id": None, "tcpa": None,
                "dcpa": None, "dist": None, "angle_deg": None, "side": None}
    return dict(nearest, status="SAFE", level=None)


class VirtualVehicle:
    """시나리오의 가상 차량. 시각 t(출발 신호 기준 초)의 위치·속도를 돌려준다.
    t < 0 이거나 duration을 넘기면 None (판정 대상 아님)."""

    def __init__(self, d):
        self.id = d["id"]
        self.x0, self.y0 = d["start"]
        self.vx, self.vy = d["vel"]
        self.t0 = d.get("t0", 0.0)
        self.duration = d.get("duration_s", 12.0)
        self.app = d.get("app", False)

    def state(self, t):
        tt = t - self.t0
        if tt < 0 or tt > self.duration:
            return None
        return {"id": self.id, "x": self.x0 + self.vx * tt, "y": self.y0 + self.vy * tt,
                "vx": self.vx, "vy": self.vy}


class AlertLevelHold:
    """차량 1대의 경고 단계 관리.
    - 지금 단계 = 최근 hold_s 초 안의 판정 중 가장 높은 단계 (한 프레임 깜빡임으로 해제되지 않게)
    - 단계가 바뀐 때만 changed=True → 그때 앱으로 즉시 보냄
    - None → 단계가 생기면 새 사건(event)"""

    def __init__(self, hold_s=1.0):
        self.hold_ms = hold_s * 1000
        self.recent = []  # [(ms, level)]
        self.level = None
        self.event_no = 0
        self.event_id = None

    def update(self, now_ms, level):
        if level is not None:
            self.recent.append((now_ms, level))
        self.recent = [(t, l) for t, l in self.recent if now_ms - t <= self.hold_ms]
        new = None
        for _, l in self.recent:
            if RANK[l] > RANK[new]:
                new = l
        changed = new != self.level
        if changed and self.level is None and new is not None:
            self.event_no += 1
            self.event_id = "evt-%04d" % self.event_no
        prev, self.level = self.level, new
        return changed, prev, new
