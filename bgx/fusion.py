# bgx/fusion.py
#
# 차량별 CPA 위험 판정 + 보조 입력 융합 (포스터 최종본 그림 2)
#   입력: Cam1 보행자 위치·속도, 차량 GNSS + 가상 OBU 속도, Cam2 EMR Score(조기 징후 보조 입력),
#         보행자 신호 RED / GREEN / FLASH (CARLA)
#   출력: 차량 A 경고 / 차량 B 경고 안 함 → FastAPI → WebSocket → 해당 차량 앱 (주의 / 경고 / 긴급)
#
# 차량 1대마다 매 판정에서 세 가지를 본다.
#   ① Cam1 CPA (포스터 규칙 그대로): dCPA ≤ 2 m 이고 0 ≤ tCPA ≤ 3 s 이면 위험
#        tCPA ≤ 1.5 s → 긴급(danger), 그 밖 → 경고(warning)                       [팀 확정 11/21]
#   ② Cam2 출발 예측: EMR ≥ 기준이 연속(= emr_state) 이고, 그 차량이
#        - 보행자가 먼저 들어갈 쪽 차로(near_band_y)에 있고
#        - 횡단보도 쪽으로 오고 있으며 도착까지 emr_eta_s(5초) 안이면 → 주의(caution)   [팀 확정 11/21]
#   ③ 보행 신호 상태: 신호 전환 구간(보행 FLASH, 또는 보행 RED 가 된 지 red_grace_s 초 이내)에
#        ②가 나오면 주의 → 경고로 한 단계 올림                                       [팀 확정 11/21]
#      (②는 '아직 건너지 않은 사람'의 예측이라 전환 구간에서 위험이 더 크다는 포스터 문제 상황을 반영.
#       ①은 이미 3초 안 충돌 경로라 신호와 상관없이 같은 단계를 쓴다.)
#   최종 단계 = ①과 ②(③ 반영) 중 높은 것. 단계는 hold_s(1초) 동안 유지 (risk.AlertLevelHold).
#
# 순수 파이썬 (서버·시험이 같이 씀)

import math

from .risk import RANK, AlertLevelHold, evaluate_vehicle, relative_angle_deg, side_from_angle

DEFAULT_FUSION = {
    "emr_fresh_s": 1.0,            # Cam2 결과가 이보다 오래되면 없는 것으로
    "emr_eta_s": 5.0,              # 차량 → 횡단보도 도착 예상이 이 시간 안일 때만 Cam2 주의
    "near_band_y": [-7.0, 0.0],    # Cam2 주의를 받을 차로 범위 (바닥 y). 보통 대기구역 쪽 절반
    "min_speed_mps": 0.5,          # 이보다 느리면 '오고 있음'으로 보지 않음
    "use_crossed": True,           # Cam2 가 '경계 넘음'(cross)을 본 상태도 출발로 봄 (Cam1 이 잡기 전까지 공백 메우기)
    "signal_boost": True,
    "red_grace_s": 3.0,            # 보행 RED 가 된 뒤 이 시간까지는 '전환 구간'
    "waiting_xy": [0.0, 1.0],      # 대기구역 대표점 (Cam2 경고의 방향·거리 계산용)
}

UP = {None: None, "caution": "warning", "warning": "danger", "danger": "danger"}


def cpa_level(res, cfg_risk):
    """evaluate_vehicle 결과 → ① 단계. 팀 확정 규칙: tCPA ≤ danger_s 긴급, 그 밖 경고 (주의는 Cam2 몫)."""
    if res.get("status") != "WARN":
        return None
    return "danger" if res["tcpa"] <= cfg_risk["level_tcpa_s"]["danger"] else "warning"


def eta_to_crosswalk(veh, cw_x):
    """차량이 횡단보도 x 구간에 들어가기까지 시간(초). 멀어지거나 서 있으면 None, 이미 안이면 0."""
    x, vx = veh["x"], veh["vx"]
    x0, x1 = min(cw_x), max(cw_x)
    if x0 <= x <= x1:
        return 0.0
    if x > x1 and vx < 0:
        return (x - x1) / -vx
    if x < x0 and vx > 0:
        return (x0 - x) / vx
    return None


def signal_transition(sig, now_ms, red_grace_s):
    """보행 신호 전환 구간인가? sig = {"ped": "GREEN|FLASH|RED", "ped_red_since_ms": ms or None}"""
    if not sig:
        return False
    ped = sig.get("ped")
    if ped == "FLASH":
        return True
    if ped == "RED" and sig.get("ped_red_since_ms") is not None:
        return (now_ms - sig["ped_red_since_ms"]) <= red_grace_s * 1000
    return False


def emr_active(cam2, now_ms, cfg):
    """Cam2 출발 예측이 지금 살아 있는가 (+ 값)."""
    if not cam2 or cam2.get("warmup"):
        return False
    t = cam2.get("ts_cap") or cam2.get("_recv")
    if t is None or now_ms - t > cfg["emr_fresh_s"] * 1000:
        return False
    return bool(cam2.get("emr_state") or (cfg["use_crossed"] and cam2.get("crossed")))


class FusionEngine:
    """차량 여러 대 각각: ① CPA + ② Cam2 EMR + ③ 신호 → 최종 단계, 1초 유지."""

    def __init__(self, risk_cfg, fusion_cfg=None, crosswalk_x=(-1.5, 1.5), hold_s=1.0):
        self.risk_cfg = risk_cfg
        self.cfg = dict(DEFAULT_FUSION, **(fusion_cfg or {}))
        self.cw_x = tuple(crosswalk_x)
        self.hold_s = hold_s
        self.holds = {}

    def hold(self, vid):
        if vid not in self.holds:
            self.holds[vid] = AlertLevelHold(self.hold_s)
        return self.holds[vid]

    def judge(self, veh, peds, cam2, signal, now_ms):
        """한 차량 판정 (단계 유지 전). 반환 dict: CPA 결과 + level(최종) + reasons + 부가 정보."""
        c = self.cfg
        res = evaluate_vehicle(veh, peds, now_ms, self.risk_cfg) if peds else {
            "status": "UNKNOWN", "level": None, "track_id": None, "tcpa": None, "dcpa": None, "dist": None,
            "angle_deg": None, "side": None}
        l_cpa = cpa_level(res, self.risk_cfg)
        eta = eta_to_crosswalk(veh, self.cw_x)
        speed = math.hypot(veh["vx"], veh["vy"])
        band = c["near_band_y"]
        in_band = min(band) <= veh["y"] <= max(band)
        emr_on = emr_active(cam2, now_ms, c)
        trans = signal_transition(signal, now_ms, c["red_grace_s"])
        l_emr = None
        if emr_on and in_band and eta is not None and eta <= c["emr_eta_s"] and speed >= c["min_speed_mps"]:
            l_emr = "caution"
            if c["signal_boost"] and trans:
                l_emr = UP[l_emr]
        level = l_cpa if RANK[l_cpa] >= RANK[l_emr] else l_emr
        reasons = []
        if l_cpa:
            reasons.append("CPA")
        if l_emr:
            reasons.append("EMR")
            if l_emr != "caution":
                reasons.append("SIGNAL")
        out = dict(res, cpa_status=res.get("status"), cpa_level=l_cpa, emr_level=l_emr, level=level,
                   reasons=reasons, eta_s=None if eta is None else round(eta, 2), in_band=in_band, emr_on=emr_on,
                   signal_transition=trans, emr=(cam2 or {}).get("emr"))
        if level and not l_cpa:
            # Cam2 예측만으로 경고: 대상 = 대기구역 (Cam2 는 바닥 좌표가 없으므로 대표점)
            wx, wy = c["waiting_xy"]
            rx, ry = wx - veh["x"], wy - veh["y"]
            ang = relative_angle_deg(veh["vx"], veh["vy"], rx, ry)
            out.update(track_id=None, dist=math.hypot(rx, ry), angle_deg=ang, side=side_from_angle(ang),
                       target="cam2_waiting")
        out["status"] = "WARN" if level else (res.get("status") if res.get("status") != "WARN" else "SAFE")
        return out

    def step(self, vid, now_ms, veh, peds, cam2, signal):
        h = self.hold(vid)
        if veh is None:
            res = {"status": "NO_VEHICLE", "level": None, "reasons": []}
        else:
            res = self.judge(veh, peds, cam2, signal, now_ms)
        changed, prev, new = h.update(now_ms, res.get("level"))
        return res, changed, prev, new


__all__ = ["FusionEngine", "DEFAULT_FUSION", "eta_to_crosswalk", "signal_transition", "cpa_level"]
