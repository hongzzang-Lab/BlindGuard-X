# server/exp_server.py — BlindGuard-X 실물 테스트 서버 (노트북)
#
# 실행 (realtest 폴더에서):
#     uvicorn server.exp_server:app --host 0.0.0.0 --port 8000
#   또는 python3 server/exp_server.py
# 조작 페이지: http://<노트북IP>:8000/control   (노트북 자신은 http://localhost:8000/control)
#
# 연결
#   /ws    ← 운전자 앱 (메시지 형식 v1 그대로: hello·location·drive·ping → alert·pong)
#   /edge  ← Raspberry Pi (pi/edge.py: Cam1 장면·Cam2·미리보기·사진, ← 명령)
#
# 차량 위치 (조작 페이지에서 고름)
#   gps     : 앱이 1초마다 보내는 폰 GPS (실제 서비스와 같은 구조) → 기준점으로 바닥 좌표 → 맵 매칭 + 칼만
#   cam     : Cam1 이 찍은 트랙 하나를 차량으로 지정 (GPS 가 너무 나쁠 때 비교용)
#   virtual : scenarios.json 의 가상 차량 (예비)
# 판정: 차량별 CPA (포스터 규칙: dCPA ≤ 2 m 이고 0 ≤ tCPA ≤ 3 s → 경고, 단계는 tCPA 1/2/3 s [임시])
#       경고 단계가 바뀌는 즉시 그 차량에 연결된 앱으로 alert 전송, 앱은 받자마자 alert_ack 회신
#
# 기록: logs/server_<시각>/ 에 전부 CSV (+ 사진·보정 결과). 시각은 모두 이 노트북 시계 기준 epoch ms.

import asyncio
import base64
import contextlib
import csv
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bgx.engine import Anchor, ConflictEngine, GpsVehicle, weighted_mean_fix  # noqa: E402
from bgx.fusion import FusionEngine  # noqa: E402
from bgx.geometry import apply_h, homography_dlt, reproj_errors  # noqa: E402
from bgx.risk import DEFAULT_RISK, VirtualVehicle, relative_angle_deg, side_from_angle  # noqa: E402


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


SITE_FILE = os.environ.get("BGX_SITE") or os.path.join(ROOT, "site.json")   # 시뮬 데모: simdemo/site_sim.json
SITE = load_json(SITE_FILE)
RISK = dict(DEFAULT_RISK, **SITE.get("risk", {}))
FUSION_CFG = SITE.get("fusion", {})
HOLD_S = RISK.get("hold_s", 1.0)
GPS_CFG = dict(SITE.get("gps", {}), lane_y=SITE["floor"].get("lane_y", -1.75))
EXCL_CFG = SITE.get("self_exclusion", {})
SCENARIOS = {k: v for k, v in load_json(os.path.join(ROOT, "scenarios.json")).items() if not k.startswith("_")}
CALIB_DIR = os.environ.get("BGX_CALIB_DIR") or os.path.join(ROOT, "calib")   # 시험(tests)은 임시 폴더를 씀
ANCHOR_FILE = os.path.join(CALIB_DIR, "gps_anchor.json")
ZONE_ID = "mock-1"
APP_ORDER = ["A", "B"]


def now_ms():
    return int(time.time() * 1000)


def r3(v):
    return None if v is None else round(float(v), 3)


# ───────────── 기록 ─────────────
LOGDIR = os.environ.get("BGX_LOGDIR") or os.path.join(ROOT, "logs", "server_" + time.strftime("%m%d_%H%M%S"))
os.makedirs(os.path.join(LOGDIR, "snapshots"), exist_ok=True)


class CsvLog:
    def __init__(self, name, cols):
        self.f = open(os.path.join(LOGDIR, name), "w", newline="", encoding="utf-8", buffering=1)
        self.w = csv.writer(self.f)
        self.w.writerow(cols)

    def row(self, values):
        self.w.writerow(values)


L = {
    "frames": CsvLog("cam1_frames.csv", [
        "recv_ms", "trial_id", "frame", "ts_cap", "ts_done", "ts_send", "offset_ms", "mode", "every", "det_ran",
        "full_scan", "fallback", "n_rois", "roi_area", "pack_scale", "canvas_fill", "n_det", "n_peds", "t_flow",
        "t_roi", "t_pack", "t_infer", "t_track", "t_homo", "t_total", "fps", "dropped", "temp_c", "rec", "policy",
        "lk_frame", "yolo_reason", "t_lk"]),
    "tracks": CsvLog("cam1_tracks.csv", [
        "recv_ms", "trial_id", "frame", "ts_cap", "track_id", "x_m", "y_m", "vx", "vy", "v_ok", "conf", "u", "v",
        "x1", "y1", "x2", "y2", "extrapolated"]),
    "cam2": CsvLog("cam2_frames.csv", [
        "recv_ms", "trial_id", "frame", "ts_cap", "ts_done", "warmup", "present", "area", "signed_px", "toward_px",
        "n_pts", "persist", "prox", "tow", "per", "emr", "emr_state", "crossed", "events", "t_mog", "t_lk",
        "t_total", "fps", "dropped", "rec"]),
    "cam2ev": CsvLog("cam2_events.csv", ["ts_cap", "recv_ms", "trial_id", "frame", "event", "emr", "signed_px"]),
    "gps": CsvLog("gps.csv", [
        "recv_ms", "trial_id", "device_id", "vehicle", "ts_phone", "fix_ts", "t_fix_ms", "phone_offset_ms", "lat",
        "lng", "acc", "speed", "speed_acc", "heading_raw", "heading_acc", "raw_x", "raw_y", "nobias_x",
        "nobias_y", "dop_vx", "dop_vy", "est_x", "est_y", "est_vx", "est_vy", "vsrc", "map_matched", "capture"]),
    "dec": CsvLog("decisions.csv", [
        "now_ms", "trial_id", "source", "frame", "ts_cap", "vehicle", "veh_x", "veh_y", "veh_vx", "veh_vy",
        "veh_age_s", "status", "level", "track_id", "tcpa", "dcpa", "dist", "side", "hold_level", "self_track",
        "self_dist", "self_how", "n_peds"]),
    "alert": CsvLog("alerts.csv", [
        "sent_ms", "seq", "trial_id", "source", "vehicle", "prev", "level", "event_id", "frame", "ts_cap",
        "n_apps", "devices", "side", "tcpa", "dcpa", "frame_recv_ms", "cap_to_server_ms", "judge_to_sent_ms",
        "cap_to_sent_ms", "reasons", "trigger"]),
    "ack": CsvLog("alert_acks.csv", ["recv_ms", "device_id", "vehicle", "seq", "level", "t_phone",
                                     "phone_offset_ms", "sent_ms", "one_way_ms", "rtt_ms"]),
    "trial": CsvLog("trials.csv", [
        "trial_id", "exp", "label", "scenario", "source", "expected", "note", "start_ms", "go_ms", "stop_ms",
        "result_note", "marks", "cam1_mode", "designated"]),
    "app": CsvLog("apps.csv", ["ms", "event", "device_id", "vehicle", "detail"]),
    "edge": CsvLog("edge_events.csv", ["ms", "kind", "detail"]),
    "op": CsvLog("operator.csv", ["ms", "action", "detail"]),
    # ── CARLA 시뮬 데모 (source = carla) ──
    "fusion": CsvLog("fusion.csv", [
        "now_ms", "trial_id", "take", "sim_t", "frame", "ts_cap", "vehicle", "veh_x", "veh_y", "veh_vx", "veh_vy",
        "cpa_status", "cpa_level", "tcpa", "dcpa", "track_id", "emr", "emr_on", "eta_s", "in_band", "ped_signal",
        "signal_transition", "emr_level", "level", "reasons", "hold_level", "n_peds",
        "ped_x", "ped_y", "ped_vx", "ped_vy"]),
    "sim": CsvLog("sim_states.csv", [
        "recv_ms", "trial_id", "take", "sim_t", "phase", "ped_signal", "veh_signal", "vehicle", "x", "y", "vx", "vy",
        "lat", "lng", "speed", "braking"]),
}


def oplog(action, detail):
    L["op"].row([now_ms(), action, json.dumps(detail, ensure_ascii=False)[:2000]])


# ───────────── 상태 ─────────────
class AppSession:
    def __init__(self, ws):
        self.ws = ws
        self.device_id = None
        self.vehicle = None
        self.driving = False
        self.off_samples = deque(maxlen=60)   # (server_recv - phone_t) → 최솟값 = 폰 시계 차이 + 최소 지연
        self.gps = GpsVehicle("?", GPS_CFG)
        self.last_fix = None
        self.trail = deque(maxlen=90)
        self.state_msg = None
        self.last_ack_ms = None
        self.ping_rtt = None
        self.n_loc = 0

    @property
    def phone_offset(self):
        return min(self.off_samples) if self.off_samples else None

    async def send(self, msg):
        try:
            await self.ws.send_text(json.dumps(dict(msg, v=1), ensure_ascii=False))
            return True
        except Exception:  # noqa: BLE001
            return False


S = {
    "apps": [],
    "edge_ws": None, "edge_hello": None, "edge_log": deque(maxlen=30),
    "scene": None, "scene_recv": None, "cam2": None, "cam2_events": deque(maxlen=20),
    "preview": {}, "snapshot": {}, "snap_waiters": {},
    "source": "gps", "designated": None,
    "anchor": None, "anchor_pts": {}, "capture": None,
    "trial": None, "n_trial": 0,
    "engine": ConflictEngine(RISK, HOLD_S),
    "veh_last": {}, "alerts": deque(maxlen=30), "seq": 0,
    "sent_log": deque(maxlen=60),      # 통제실 화면: 앱으로 보낸 경고 메시지 원문 + 지연
    "events": deque(maxlen=80),        # 통제실 화면: 사건 로그 (위험 감지·해제, Cam2 이벤트, 연결)
    "cam1_mode": None, "h_detail": None,
    "cam2_set": False, "trials_done": deque(maxlen=12),
    # CARLA 시뮬 데모: 차량 = CARLA GNSS 위치 + 가상 OBU 속도, 신호 = CARLA 신호 단계
    "fusion": FusionEngine(RISK, FUSION_CFG, SITE["floor"].get("crosswalk_x", (-1.5, 1.5)), HOLD_S),
    "sim": {"vehicles": {}, "obstacles": [], "signal": None, "recv": None, "t": None, "phase": None, "take": None,
            "geom": None, "geom_rev": 0, "ped_prev": None, "ped_red_since_ms": None, "n": 0,
            "meta": {}, "ev": [], "result": None, "running": False},
    "heading": {},    # 차량별 마지막 진행 방향 (단위 벡터). 멈춘 뒤에도 앱 레이더의 '앞'이 유지되게
    "radar_ms": {},   # 차량별 마지막 레이더(보행자 위치) 갱신 전송 시각
    "pages": {},      # 화면 페이지가 마지막으로 서버를 부른 시각·버전 {"monitor": {"ms", "v"}, "hud": {...}} → 촬영 전 점검
}
PAGE_VER = {"monitor": "1124", "hud": "1124b", "studio": "1124c"}     # 페이지를 고치면 올림 → 새로고침 안 된 옛 화면을 촬영 전 점검이 잡아냄
_a = load_json(ANCHOR_FILE)
if _a:
    S["anchor"] = Anchor.from_dict(_a)


def trial_id():
    return S["trial"]["id"] if S["trial"] else None


def apps_for(vid):
    return [a for a in S["apps"] if a.vehicle == vid]


def assign_vehicle(sess):
    used = {a.vehicle for a in S["apps"] if a is not sess}
    sess.vehicle = next((v for v in APP_ORDER if v not in used), None)
    sess.gps.id = sess.vehicle or "?"


async def edge_cmd(cmd):
    ws = S["edge_ws"]
    if ws is None:
        return False
    try:
        await ws.send_text(json.dumps(dict(cmd, type="cmd"), ensure_ascii=False))
        L["edge"].row([now_ms(), "cmd", json.dumps({k: v for k, v in cmd.items() if k != "H"}, ensure_ascii=False)])
        return True
    except Exception:  # noqa: BLE001
        return False


# ───────────── 실물 Vision 테스트 (/vision 페이지) ─────────────
from server.vision import VisionTest  # noqa: E402
VT = VisionTest(LOGDIR, CALIB_DIR, edge_cmd, homography_dlt, reproj_errors, SITE.get("vision_test"), now_ms)


# ───────────── 경고 메시지 (앱 v1 형식) ─────────────
def zone_state(level):
    if level:
        return "alert"
    return "monitoring" if S["edge_ws"] is not None else "outside"


def hold_of(vid):
    """지금 쓰는 판정기의 경고 단계 유지 객체 (carla = 융합 판정기, 그 밖 = 기존 CPA 판정기)."""
    return S["fusion"].hold(vid) if S["source"] == "carla" else S["engine"].hold(vid)


REASON_KO = {"CPA": "CPA 충돌 예측", "EMR": "Cam2 조기 징후(EMR)", "SIGNAL": "보행자 신호 FLASH/RED 전환"}


def make_alert(vid, level, res):
    hold = hold_of(vid)
    objs = []
    if res and res.get("dist") is not None:
        objs.append({"track_id": res.get("track_id"), "kind": "pedestrian", "angle_deg": r3(res.get("angle_deg")),
                     "distance_m": r3(res.get("dist")), "target": bool(level)})
    risk = None
    if level:
        risk = {"event_id": hold.event_id, "level": level, "side": (res or {}).get("side") or "front",
                "object_kind": "pedestrian", "near_miss": level == "danger"}
        if res and res.get("reasons"):
            risk["reasons"] = res["reasons"]     # 앱은 모르는 키를 무시함. 관제·기록용
    S["seq"] += 1
    return {"type": "alert", "ts": now_ms(), "seq": S["seq"],
            "zone": {"id": ZONE_ID if S["edge_ws"] is not None else None, "state": zone_state(level),
                     "distance_m": None},
            "risk": risk, "objects": objs, "vehicle": vid, "trial_id": trial_id()}


def alert_targets(vid):
    if S["source"] == "carla" and S["sim"].get("running") and (S["sim"].get("meta") or {}).get("no_app") and vid == "A":
        return []         # '앱 없음' 대조 상황: 판정은 하지만 운전자 폰으로는 보내지 않음 (기록의 앱 수 = 0)
    return apps_for(vid)


def fix_heading(vid, st, res):
    """차량이 멈추면 속도 벡터로는 '앞'을 알 수 없음 → 마지막 진행 방향으로 보행자 방향(각도·좌우)을 다시 계산.
    (고치기 전: 정지한 차 앞을 지나는 보행자가 앱에서 '뒤'로 표시됐음)"""
    if st is None or not res:
        return
    sp = math.hypot(st["vx"], st["vy"])
    if sp > 0.5:
        S["heading"][vid] = (st["vx"] / sp, st["vy"] / sp)
    h = S["heading"].get(vid)
    if h is None or res.get("dist") is None:
        return
    if res.get("target") == "cam2_waiting":
        pxy = S["fusion"].cfg.get("waiting_xy")
    else:
        pxy = res.get("ped_xy")
    if not pxy:
        return
    ang = relative_angle_deg(h[0], h[1], pxy[0] - st["x"], pxy[1] - st["y"])
    res["angle_deg"], res["side"] = ang, side_from_angle(ang)


async def radar_update(vid, level, res, now):
    """경고 단계가 그대로인 동안에도 보행자 위치(차량 기준 방향·거리)를 앱 레이더에 갱신 (초당 5번까지).
    경고 기록(alerts.csv)에는 넣지 않음. seq 0 = 위치 갱신 (앱 회신은 지연 통계에서 제외)."""
    if now - S["radar_ms"].get(vid, 0) < 200:
        return
    S["radar_ms"][vid] = now
    objs = [{"track_id": res.get("track_id"), "kind": "pedestrian", "angle_deg": r3(res.get("angle_deg")),
             "distance_m": r3(res.get("dist")), "target": bool(level)}]
    for a in alert_targets(vid):
        if a.state_msg is None:
            continue
        a.state_msg = dict(a.state_msg, objects=objs)       # 1초마다 다시 보내는 상태에도 최신 위치
        await a.send(dict(a.state_msg, ts=now, seq=0))


async def push(vid, prev, new, res, frame, ts_cap, frecv=None, trigger=None):
    msg = make_alert(vid, new, res if res and res.get("status") in ("WARN", "SAFE") else None)
    targets = alert_targets(vid)
    for a in targets:
        a.state_msg = msg
        await a.send(msg)
    sent = now_ms()
    msg["_sent"] = sent
    if frecv is None:
        frecv = S["scene_recv"] if frame is not None else None
    lat_cs = (frecv - ts_cap) if (frecv and ts_cap) else None       # 촬영 → 서버 도착
    lat_js = (sent - frecv) if frecv else None                       # 서버 도착 → 판정 → 전송
    lat_ct = (sent - ts_cap) if ts_cap else None                     # 촬영 → 전송
    L["alert"].row([sent, msg["seq"], trial_id(), S["source"], vid, prev, new,
                    msg["risk"]["event_id"] if msg["risk"] else None, frame, ts_cap, len(targets),
                    "|".join(a.device_id or "?" for a in targets), (res or {}).get("side"),
                    r3((res or {}).get("tcpa")), r3((res or {}).get("dcpa")), frecv, lat_cs, lat_js, lat_ct,
                    "|".join((res or {}).get("reasons") or []), trigger])
    veh = (S["veh_last"].get(vid) or {}).get("veh")
    reasons = (res or {}).get("reasons") or []
    sim_t = None        # 보낸 순간의 시뮬 시각 (촬영 화면의 '+몇 초'. CARLA 가 실제보다 느리게 돌아도 맞도록 여기서 계산)
    if S["source"] == "carla" and S["sim"].get("t") is not None and S["sim"].get("recv"):
        sim_t = round(float(S["sim"]["t"]) + max(0.0, sent - S["sim"]["recv"]) / 1000.0 * sim_scale(), 2)
    S["sent_log"].append({
        "ms": sent, "seq": msg["seq"], "vehicle": vid, "prev": prev, "level": new, "n_apps": len(targets), "sim_t": sim_t,
        "side": (res or {}).get("side"), "track_id": (res or {}).get("track_id"), "tcpa": r3((res or {}).get("tcpa")),
        "dcpa": r3((res or {}).get("dcpa")), "dist": r3((res or {}).get("dist")), "reasons": reasons,
        "eta": (res or {}).get("eta_s"), "emr": (res or {}).get("emr"), "trigger": trigger,
        "veh": veh and {"x": r3(veh["x"]), "y": r3(veh["y"]), "v": r3(math.hypot(veh["vx"], veh["vy"]))},
        "lat": {"cap_server": lat_cs, "judge_sent": lat_js, "cap_sent": lat_ct, "server_phone": None},
        "payload": {k: v for k, v in msg.items() if not k.startswith("_")}})
    why = " + ".join(REASON_KO.get(r, r) for r in reasons)
    if new and "CPA" in reasons:
        detail = "보행자 #%s, tCPA %s s, dCPA %s m" % ((res or {}).get("track_id"), r3((res or {}).get("tcpa")),
                                                    r3((res or {}).get("dcpa")))
    elif new:
        detail = "대기구역 EMR %s, 횡단보도까지 %s s" % ((res or {}).get("emr"), (res or {}).get("eta_s"))
    S["events"].append({"ms": sent, "kind": "risk" if new else "clear", "vehicle": vid,
                        "text": ("차량 %s ← %s 전송 (%s%s, %s)" %
                                 (vid, {"caution": "주의", "warning": "경고", "danger": "긴급"}[new],
                                  {"front": "전방", "right": "우측", "back": "후방", "left": "좌측"}.get(
                                      (res or {}).get("side"), "-"), (" · " + why) if why else "", detail))
                        if new else ("차량 %s 경고 해제 전송" % vid)})
    print("[경고] 차량 %s %s→%s  앱 %d대  촬영→서버 %s ms · 판정→전송 %s ms · 촬영→전송 %s ms" %
          (vid, prev, new, len(targets), lat_cs, lat_js, lat_ct))
    S["alerts"].append({"ms": sent, "seq": msg["seq"], "vehicle": vid, "level": new, "prev": prev,
                        "n_apps": len(targets)})
    if S["trial"]:
        S["trial"]["alerts"].append({"ms": sent, "vehicle": vid, "level": new})


async def apps_clear(why):
    """에피소드 경계: 앱 레이더에 남은 보행자 표시(지난 에피소드의 마지막 위치)를 지움. 경고 기록에는 넣지 않음."""
    n = 0
    for a in list(S["apps"]):
        if not a.vehicle:
            continue
        a.state_msg = make_alert(a.vehicle, None, None)
        try:
            await a.send(a.state_msg)
            n += 1
        except Exception:  # noqa: BLE001
            pass
    oplog("apps_clear", {"why": why, "n_apps": n})


# ───────────── 차량 상태 ─────────────
def vehicles_now(now, scene):
    """[(vid, state or None)] — 지금 판정할 차량들."""
    src = S["source"]
    out = []
    if src == "gps":
        for a in S["apps"]:
            if a.vehicle:
                out.append((a.vehicle, a.gps.state(now / 1000.0)))
    elif src == "cam":
        st = None
        if scene is not None and S["designated"] is not None:
            for p in scene["peds"]:
                if p["track_id"] == S["designated"] and p.get("x_m") is not None and p.get("v_ok"):
                    st = {"id": "A", "x": p["x_m"], "y": p["y_m"], "vx": p["vx"], "vy": p["vy"], "age_s": 0.0,
                          "vsrc": "cam", "acc": None}
        out.append(("A", st))
    elif src == "virtual":
        tr = S["trial"]
        if tr and tr.get("virtual") and now >= tr["go_ms"]:
            t = (now - tr["go_ms"]) / 1000.0
            for v in tr["virtual"]:
                out.append((v.id, v.state(t)))
    elif src == "carla":
        for vid in sorted(S["sim"]["vehicles"]):
            v = S["sim"]["vehicles"][vid]
            age = (now - v["recv"]) / 1000.0
            if age > 1.0:            # CARLA 가 멈췄거나 끊김 → 위치 모름
                out.append((vid, None))
                continue
            dt = min(max(age, 0.0), 0.5) * sim_scale()     # 시뮬 시간으로 앞당김
            out.append((vid, {"id": vid, "x": v["x"] + v["vx"] * dt, "y": v["y"] + v["vy"] * dt, "vx": v["vx"],
                              "vy": v["vy"], "age_s": round(age, 3), "vsrc": "carla_gnss_obu", "acc": None}))
    return out


def sim_scale():
    """CARLA 가 실제 시간의 몇 배로 도는지 (carla_demo 가 잼). 판정을 시뮬 시간 기준으로 하려고 씀."""
    k = S["sim"].get("time_scale") or 1.0
    return min(max(float(k), 0.15), 1.5)


def fresh_scene(now):
    sc = S["scene"]
    if sc is None or not S["scene_recv"] or now - S["scene_recv"] > 1000:
        return None
    return sc


async def evaluate_fused(scene=None, trigger="sim"):
    """source = carla: Cam1 CPA + Cam2 EMR + 신호 융합 (bgx/fusion.py). Cam1 장면·Cam2·CARLA 상태가 올 때마다.
    trigger: 판정을 부른 입력 (scene / cam2 / sim / tick). 경고 지연은 그 입력의 촬영 시각 기준으로 잰다."""
    now = now_ms()
    scene = scene if scene is not None else fresh_scene(now)
    peds, frame, ts_cap = [], None, None
    ks = sim_scale()
    if scene is not None:
        frame, ts_cap = scene["frame"], scene["ts_cap"]
        # 판정은 시뮬 시간 기준: Pi 는 실제 시계로 속도를 잼 → 시뮬 m/s 로 (÷배율), 촬영 뒤 지난 시간도 시뮬 초로 (×배율)
        ts_sim = now - int((now - ts_cap) * ks)
        peds = [dict(p, ts_cap=ts_sim, vx=p["vx"] / ks, vy=p["vy"] / ks) for p in scene["peds"]
                if p.get("x_m") is not None]
    # 지연 기준: Cam1 장면이면 그 프레임, Cam2 면 그 프레임, CARLA 상태·시간 경과면 카메라 지연 없음
    lat_frame = lat_ts = lat_recv = None
    if trigger == "scene" and scene is not None:
        lat_frame, lat_ts, lat_recv = frame, ts_cap, S["scene_recv"]
    elif trigger == "cam2" and S["cam2"] is not None:
        lat_frame, lat_ts, lat_recv = S["cam2"].get("frame"), S["cam2"].get("ts_cap"), S["cam2"].get("_recv")
    cam2 = S["cam2"]
    sig = S["sim"]["signal"]
    if sig is not None:
        sig = dict(sig, ped_red_since_ms=S["sim"]["ped_red_since_ms"])
    for vid, st in vehicles_now(now, scene):
        res, changed, prev, new = S["fusion"].step(vid, now, st, peds, cam2, sig)
        fix_heading(vid, st, res)
        S["veh_last"][vid] = dict(res, veh=st, hold_level=new, ms=now)
        if st is not None:
            L["fusion"].row([now, trial_id(), S["sim"]["take"], S["sim"]["t"], frame, ts_cap, vid, r3(st["x"]),
                             r3(st["y"]), r3(st["vx"]), r3(st["vy"]), res.get("cpa_status"), res.get("cpa_level"),
                             r3(res.get("tcpa")), r3(res.get("dcpa")), res.get("track_id"), res.get("emr"),
                             res.get("emr_on"), res.get("eta_s"), res.get("in_band"), (sig or {}).get("ped"),
                             res.get("signal_transition"), res.get("emr_level"), res.get("level"),
                             "|".join(res.get("reasons") or []), new, len(peds),
                             *(res.get("ped_xy") or (None, None)), *(res.get("ped_v") or (None, None))])
        if changed:
            await push(vid, prev, new, res, lat_frame, lat_ts, frecv=lat_recv, trigger=trigger)
            S["radar_ms"][vid] = now
        elif (S["sim"].get("running") and vid in S["heading"] and res.get("dist") is not None
              and res.get("status") in ("WARN", "SAFE")):       # 출발 전(진행 방향 모름)에는 표시하지 않음
            await radar_update(vid, new, res, now)


async def evaluate(scene):
    if S["source"] == "carla":
        await evaluate_fused(scene, "scene" if scene is not None else "tick")
        return
    now = now_ms()
    peds = []
    frame = ts_cap = None
    if scene is not None:
        frame, ts_cap = scene["frame"], scene["ts_cap"]
        peds = [dict(p, ts_cap=ts_cap) for p in scene["peds"]]
    for vid, st in vehicles_now(now, scene):
        if scene is None:
            changed, prev, new = S["engine"].hold(vid).update(now, None)
            res = None
        else:
            res, changed, prev, new = S["engine"].step(
                vid, now, st, peds, designated=S["designated"],
                exclude_self=(S["source"] in ("gps", "cam")), excl_cfg=EXCL_CFG)
            S["veh_last"][vid] = dict(res, veh=st, hold_level=new, ms=now)
            L["dec"].row([now, trial_id(), S["source"], frame, ts_cap, vid,
                          r3(st and st["x"]), r3(st and st["y"]), r3(st and st["vx"]), r3(st and st["vy"]),
                          st and st.get("age_s"), res["status"], res.get("level"), res.get("track_id"),
                          r3(res.get("tcpa")), r3(res.get("dcpa")), r3(res.get("dist")), res.get("side"), new,
                          res.get("self_track"), res.get("self_dist"), res.get("self_how"), len(peds)])
        if changed:
            await push(vid, prev, new, res, frame, ts_cap)


async def ticker():
    while True:
        await asyncio.sleep(1.0)
        try:
            await evaluate(None)
            for a in list(S["apps"]):
                lvl = hold_of(a.vehicle).level if a.vehicle else None
                if a.state_msg is None or (not lvl and a.state_msg["zone"]["state"] != zone_state(None)):
                    a.state_msg = make_alert(a.vehicle, None, None)
                await a.send(dict(a.state_msg, ts=now_ms()))
        except Exception as e:  # noqa: BLE001
            print("[ticker] 오류:", repr(e))


# ───────────── WebSocket: 앱 ─────────────
async def ws_app(ws: WebSocket):
    await ws.accept()
    s = AppSession(ws)
    S["apps"].append(s)
    assign_vehicle(s)
    L["app"].row([now_ms(), "connect", None, s.vehicle, ws.client.host if ws.client else None])
    print("[앱] 접속 → 차량 %s" % s.vehicle)
    S["events"].append({"ms": now_ms(), "kind": "conn", "text": "폰 접속 → 차량 %s" % s.vehicle})
    try:
        while True:
            raw = await ws.receive_text()
            recv = now_ms()
            m = json.loads(raw)
            t = m.get("type")
            if t == "hello":
                s.device_id = m.get("device_id")
                for other in list(S["apps"]):   # 같은 폰이 다시 붙으면 예전 연결의 차량 이름을 물려받고 예전 연결은 닫음
                    if other is not s and other.device_id == s.device_id:
                        if other.vehicle:
                            s.vehicle, other.vehicle = other.vehicle, None
                            s.gps.id = s.vehicle
                        S["apps"].remove(other)
                        with contextlib.suppress(Exception):
                            await other.ws.close()
                print("[앱] 인사 %s → 차량 %s" % (s.device_id, s.vehicle))
                L["app"].row([recv, "hello", s.device_id, s.vehicle, None])
                s.state_msg = make_alert(s.vehicle, None, None)
                await s.send(s.state_msg)
            elif t == "ping":
                if isinstance(m.get("t"), (int, float)):
                    s.off_samples.append(recv - m["t"])
                await s.send({"type": "pong", "t": m.get("t")})
            elif t == "drive":
                s.driving = m.get("state") == "start"
                L["app"].row([recv, "drive_" + str(m.get("state")), s.device_id, s.vehicle, None])
            elif t == "location":
                handle_location(s, m, recv)
            elif t == "alert_ack" and not m.get("seq"):
                pass                       # 레이더 위치 갱신(seq 0)에 대한 회신 → 지연 통계에 넣지 않음
            elif t == "alert_ack":
                off = s.phone_offset
                sent = next((a["ms"] for a in reversed(S["alerts"]) if a["seq"] == m.get("seq")), None)
                tp = m.get("t")
                one_way = (tp + off - sent) if (tp is not None and off is not None and sent) else None
                s.last_ack_ms = one_way
                for e in S["sent_log"]:
                    if e["seq"] == m.get("seq") and e["lat"]["server_phone"] is None:
                        e["lat"]["server_phone"] = one_way
                        e["lat"]["rtt"] = (recv - sent) if sent else None
                        print("[수신] 폰 %s seq %s  서버→폰 %s ms" % (s.vehicle, m.get("seq"), one_way))
                L["ack"].row([recv, s.device_id, s.vehicle, m.get("seq"), m.get("level"), tp, off, sent,
                              one_way, (recv - sent) if sent else None])
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        print("[앱] 오류:", repr(e))
    finally:
        if s in S["apps"]:
            S["apps"].remove(s)
        L["app"].row([now_ms(), "disconnect", s.device_id, s.vehicle, None])
        print("[앱] 끊김 (차량 %s)" % s.vehicle)
        # 비어 버린 앞 순서 차량 이름(A 등)을 남은 앱에게 넘김 — 재접속 순서가 꼬여도 폰 A가 A로 남게
        for v in APP_ORDER:
            if not any(a.vehicle == v for a in S["apps"]):
                later = [a for a in S["apps"] if a.vehicle and APP_ORDER.index(a.vehicle) > APP_ORDER.index(v)]
                cand = next((a for a in S["apps"] if a.vehicle is None), None) or (later[0] if later else None)
                if cand is not None:
                    print("[앱] 차량 %s → %s 로 이름 변경" % (cand.vehicle, v))
                    cand.vehicle = v
                    cand.gps.id = v
                    L["app"].row([now_ms(), "rename", cand.device_id, v, None])


def handle_location(s, m, recv):
    s.n_loc += 1
    lat, lng = m.get("lat"), m.get("lng")
    if lat is None or lng is None:
        return
    acc = m.get("accuracy_m")
    off = s.phone_offset
    fix_ts = m.get("fix_ts") or m.get("ts")
    t_fix = recv
    if fix_ts is not None and off is not None:
        cand = fix_ts + off
        if -1500 <= recv - cand <= 5000:   # 말이 안 되는 값이면 받은 시각
            t_fix = cand
    cap = S["capture"]
    capturing = False
    if cap and cap["device"] in (None, s.device_id):
        cap["device"] = s.device_id
        cap["fixes"].append((lat, lng, acc))
        capturing = True
    raw = nob = est = (None, None)
    vel_d = None
    vx = vy = None
    vsrc = None
    mm = None
    A = S["anchor"]
    if A is not None:
        raw = A.to_floor(lat, lng)
        nob = A.to_floor(lat, lng, with_bias=False)
        hd = m.get("heading_raw_deg")
        hacc = m.get("heading_acc")
        spd = m.get("speed_mps")
        # 안드로이드는 방향을 모를 때 0(=북)을 주기도 함 → 방향 정확도가 0 초과 45° 이하일 때만 믿음
        hd_ok = hd is not None and hd >= 0 and spd is not None and (hacc is None or 0 < hacc <= 45)
        vel_d = A.heading_vec(hd, spd) if hd_ok else None
        st = s.gps.update(t_fix / 1000.0, raw[0], raw[1], spd, vel_d, acc)
        if st is not None:
            est = (st["x"], st["y"])
            vx, vy, vsrc, mm = st["vx"], st["vy"], st["vsrc"], st["mm"]
            s.trail.append({"ms": recv, "raw": [r3(raw[0]), r3(raw[1])], "est": [r3(est[0]), r3(est[1])],
                            "acc": acc})
    s.last_fix = {"ms": recv, "lat": lat, "lng": lng, "acc": acc, "speed": m.get("speed_mps"),
                  "heading_raw": m.get("heading_raw_deg"), "raw": raw, "est": est}
    L["gps"].row([recv, trial_id(), s.device_id, s.vehicle, m.get("ts"), m.get("fix_ts"), t_fix, off, lat, lng,
                  acc, m.get("speed_mps"), m.get("speed_acc"), m.get("heading_raw_deg"), m.get("heading_acc"),
                  r3(raw[0]), r3(raw[1]), r3(nob[0]), r3(nob[1]), r3(vel_d and vel_d[0]), r3(vel_d and vel_d[1]),
                  r3(est[0]), r3(est[1]), r3(vx), r3(vy), vsrc, mm, capturing])


# ───────────── WebSocket: Pi ─────────────
async def ws_edge(ws: WebSocket):
    await ws.accept()
    if S["edge_ws"] is not None:
        with contextlib.suppress(Exception):
            await S["edge_ws"].close()
    S["edge_ws"] = ws
    L["edge"].row([now_ms(), "connect", ws.client.host if ws.client else None])
    print("[Pi] 접속")
    S["events"].append({"ms": now_ms(), "kind": "conn", "text": "Pi 접속"})
    try:
        while True:
            raw = await ws.receive_text()
            recv = now_ms()
            m = json.loads(raw)
            t = m.get("type")
            if t == "sync":
                await ws.send_text(json.dumps({"type": "sync", "t0": m["t0"], "t1": now_ms()}))
                continue
            tid = trial_id()
            if t == "hello_edge":
                S["edge_hello"] = m
                L["edge"].row([recv, "hello", json.dumps(m, ensure_ascii=False)[:1500]])
                if S["cam1_mode"]:   # 서버가 기억하는 모드로 맞춤 (Pi 재시작 대비)
                    await edge_cmd({"cmd": "set_mode", "cam": "cam1", "mode": S["cam1_mode"]})
            elif t == "scene":
                S["scene"], S["scene_recv"] = m, recv
                yr = S.setdefault("_yr", deque(maxlen=50))
                yr.append(1 if m.get("det_ran") else 0)
                S["yolo_rate"] = round(100 * sum(yr) / len(yr))
                S["cam1_mode"] = S["cam1_mode"] or m.get("mode")
                tm = m["timing"]
                L["frames"].row([recv, tid, m["frame"], m["ts_cap"], m["ts_done"], m["ts_send"], m["offset_ms"],
                                 m["mode"], m["every"], m["det_ran"], m["full_scan"], m["fallback"], m["n_rois"],
                                 m["roi_area"], m["pack_scale"], m["canvas_fill"], m["n_det"], len(m["peds"]),
                                 tm["flow"], tm["roi"], tm["pack"], tm["infer"], tm["track"], tm["homo"],
                                 tm["total"], m["fps"], m["dropped"], m["temp_c"], m.get("rec"), m.get("policy"),
                                 m.get("lk_frame"), m.get("yolo_reason"), tm.get("lk")])
                for p in m["peds"]:
                    b = p.get("bbox") or [None] * 4
                    L["tracks"].row([recv, tid, m["frame"], m["ts_cap"], p["track_id"], p["x_m"], p["y_m"], p["vx"],
                                     p["vy"], p["v_ok"], p.get("conf"), p.get("u"), p.get("v"), b[0], b[1], b[2],
                                     b[3], p.get("extrapolated", False)])
                await VT.on_scene(m, recv)
                await evaluate(m)
            elif t == "cam2":
                m["_recv"] = recv
                S["cam2"] = m
                await VT.on_cam2(m, recv)
                tm = m["timing"]
                L["cam2"].row([recv, tid, m["frame"], m["ts_cap"], m["ts_done"], m["warmup"], m["present"],
                               m["area"], m["signed_px"], m["toward_px"], m["n_pts"], m["persist"], m["prox"],
                               m["tow"], m["per"], m["emr"], m["emr_state"], m["crossed"], "|".join(m["events"]),
                               tm["mog"], tm["lk"], tm["total"], m["fps"], m["dropped"], m.get("rec")])
                for e in m["events"]:
                    L["cam2ev"].row([m["ts_cap"], recv, tid, m["frame"], e, m["emr"], m["signed_px"]])
                    S["cam2_events"].append({"ms": recv, "event": e, "emr": m["emr"]})
                    if e in ("emr_high", "cross", "enter"):
                        S["events"].append({"ms": recv, "kind": "cam2", "text": {
                            "emr_high": "Cam2 출발 예측 (EMR %.2f ≥ 기준)" % m["emr"],
                            "cross": "Cam2 경계선 넘음 (차도 진입)", "enter": "Cam2 대기구역에 사람 들어옴"}[e]})
                if S["source"] == "carla":      # Cam2 출발 예측이 바로 경고에 반영되게
                    await evaluate_fused(None, "cam2")
            elif t == "preview":
                if m.get("jpg"):
                    S["preview"][m["cam"]] = (base64.b64decode(m["jpg"]), recv)
            elif t == "snapshot":
                if m.get("jpg"):
                    data = base64.b64decode(m["jpg"])
                    S["snapshot"][m["cam"]] = {"jpg": data, "w": m["w"], "h": m["h"], "ms": recv}
                    with open(os.path.join(LOGDIR, "snapshots", "%s_%d.jpg" % (m["cam"], recv)), "wb") as f:
                        f.write(data)
                    fut = S["snap_waiters"].pop(m["cam"], None)
                    if fut and not fut.done():
                        fut.set_result(True)
            elif t in ("ack", "log", "worker_ready"):
                line = {k: v for k, v in m.items() if k not in ("edge_id",)}
                S["edge_log"].append(dict(line, ms=recv))
                L["edge"].row([recv, t, json.dumps(line, ensure_ascii=False)[:1500]])
                if t == "ack" and m.get("cmd") == "set_mode" and m.get("ok"):
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        print("[Pi] 오류:", repr(e))
    finally:
        if S["edge_ws"] is ws:
            S["edge_ws"] = None
        L["edge"].row([now_ms(), "disconnect", ""])
        print("[Pi] 끊김")


# ───────────── HTTP ─────────────
async def body(req):
    try:
        return await req.json()
    except Exception:  # noqa: BLE001
        return {}


async def page_root(req):
    return RedirectResponse("/control")


async def page_control(req):
    return FileResponse(os.path.join(ROOT, "server", "control.html"), media_type="text/html")


def floor_now():
    """관제 지도용 현장 기하. CARLA 데모는 carla_demo.py 가 J189 에서 뽑아 보낸 값으로 덮어씀."""
    g = S["sim"]["geom"]
    return dict(SITE["floor"], **(g or {}))


async def api_config(req):
    return JSONResponse({"floor": floor_now(), "risk": RISK, "gps": GPS_CFG, "fusion": S["fusion"].cfg,
                         "geom_rev": S["sim"]["geom_rev"],
                         "scenarios": {k: {kk: vv for kk, vv in v.items()} for k, v in SCENARIOS.items()},
                         "logdir": LOGDIR, "cam1_modes": ["full", "fixed", "adaptive", "packed"]})


def ped_view(p):
    return {k: p.get(k) for k in ("track_id", "x_m", "y_m", "vx", "vy", "v_ok", "conf")}


async def api_status(req):
    now = now_ms()
    sc = S["scene"]
    fresh = sc is not None and S["scene_recv"] and now - S["scene_recv"] < 1500
    c2 = S["cam2"]
    apps = []
    for a in S["apps"]:
        lf = a.last_fix
        apps.append({"device_id": a.device_id, "vehicle": a.vehicle, "driving": a.driving, "n_loc": a.n_loc,
                     "phone_offset": a.phone_offset, "ack_ms": a.last_ack_ms,
                     "fix": None if lf is None else {"age_s": round((now - lf["ms"]) / 1000, 1), "acc": lf["acc"],
                                                      "speed": lf["speed"], "heading_raw": lf["heading_raw"],
                                                      "raw": [r3(v) for v in lf["raw"]],
                                                      "est": [r3(v) for v in lf["est"]]},
                     "trail": list(a.trail)[-40:]})
    vehs = []
    for vid, st in vehicles_now(now, sc if fresh else None):
        last = S["veh_last"].get(vid, {})
        vehs.append({"id": vid, "state": None if st is None else {k: (r3(v) if isinstance(v, float) else v)
                                                                   for k, v in st.items()},
                     "level": hold_of(vid).level, "status": last.get("status"),
                     "tcpa": r3(last.get("tcpa")), "dcpa": r3(last.get("dcpa")), "self_track": last.get("self_track"),
                     "self_dist": last.get("self_dist"), "track_id": last.get("track_id"),
                     "reasons": last.get("reasons"), "cpa_level": last.get("cpa_level"),
                     "cpa_status": last.get("cpa_status"), "emr_level": last.get("emr_level"),
                     "eta_s": last.get("eta_s"), "in_band": last.get("in_band"),
                     "kind": (S["sim"]["vehicles"].get(vid) or {}).get("kind"),
                     "braking": (S["sim"]["vehicles"].get(vid) or {}).get("braking"),
                     "speed_kmh": (S["sim"]["vehicles"].get(vid) or {}).get("speed_kmh"),
                     "gnss": (S["sim"]["vehicles"].get(vid) or {}).get("gnss")})
    tr = S["trial"]
    cap = S["capture"]
    return JSONResponse({
        "now": now, "logdir": LOGDIR, "source": S["source"], "designated": S["designated"],
        "edge": {"connected": S["edge_ws"] is not None, "age_ms": None if not S["scene_recv"] else now - S["scene_recv"],
                 "hello": S["edge_hello"] and {k: S["edge_hello"].get(k) for k in ("cams", "has_H", "detector",
                                                                                    "logdir")},
                 "fps": sc and sc["fps"], "timing": sc and sc["timing"], "mode": sc and sc["mode"],
                 "every": sc and sc["every"], "n_rois": sc and sc["n_rois"], "pack_scale": sc and sc["pack_scale"],
                 "policy": sc and sc.get("policy"), "lk_frame": sc and sc.get("lk_frame"),
                 "yolo_reason": sc and sc.get("yolo_reason"), "yolo_rate": S.get("yolo_rate"),
                 "fallback": sc and sc["fallback"], "temp_c": sc and sc["temp_c"], "offset_ms": sc and sc["offset_ms"],
                 "has_H": sc and sc.get("has_H"),
                 "pipe_ms": sc and (S["scene_recv"] - sc["ts_cap"]), "peds": [ped_view(p) for p in sc["peds"]]
                 if fresh else [], "log": list(S["edge_log"])[-8:]},
        "cam2": c2 and {k: c2.get(k) for k in ("emr", "present", "signed_px", "toward_px", "emr_state", "crossed",
                                               "warmup", "fps", "timing")},
        "cam2_events": list(S["cam2_events"])[-8:],
        "apps": apps, "vehicles": vehs, "alerts": list(S["alerts"])[-12:],
        "anchor": S["anchor"].to_dict() if S["anchor"] else None,
        "anchor_pts": S["anchor_pts"],
        "capture": cap and {"name": cap["name"], "n": len(cap["fixes"]), "elapsed": round((now - cap["start"]) / 1000, 1),
                            "x": cap["x"], "y": cap["y"]},
        "trial": tr and {"id": tr["id"], "exp": tr["exp"], "label": tr["label"], "scenario": tr["scenario"],
                         "t": round((now - tr["go_ms"]) / 1000, 2), "go_ms": tr["go_ms"], "alerts": tr["alerts"][-10:]},
        "n_trials": S["n_trial"], "trials_done": list(S["trials_done"]),
        "cam2_set": bool(S["cam2_set"] or (S["edge_hello"] or {}).get("cam2_geom")),
        "sim": {"connected": S["sim"]["recv"] is not None and now - S["sim"]["recv"] < 1500,
                "age_ms": None if S["sim"]["recv"] is None else now - S["sim"]["recv"], "t": S["sim"]["t"],
                "phase": S["sim"]["phase"], "take": S["sim"]["take"], "geom_rev": S["sim"]["geom_rev"],
                "signal": S["sim"]["signal"] and dict(S["sim"]["signal"], ped_red_since_ms=S["sim"]["ped_red_since_ms"]),
                "obstacles": S["sim"]["obstacles"], "walker_gt": None, "time_scale": S["sim"].get("time_scale")},
    })


async def page_monitor(req):
    return FileResponse(os.path.join(ROOT, "server", "monitor.html"), media_type="text/html")


def page_seen(req, name):
    try:
        paint = float(req.query_params.get("paint")) if req.query_params.get("paint") is not None else None
    except ValueError:
        paint = None
    now, v = now_ms(), req.query_params.get("v")
    if v != PAGE_VER.get(name):
        # 옛 버전 창이 아직 서버를 부르고 있음. 최신 창이 따로 떠 있으면 그 기록을 덮지 않고 '옛 창이 남아 있음'만 적어 둠
        # (11/24: 최신 창과 옛 창이 같이 떠 있어 점검은 통과했는데 OBS 는 옛 창을 찍고 있었음)
        S.setdefault("pages_old", {})[name] = now
        cur = S["pages"].get(name)
        if cur is not None and cur.get("v") == PAGE_VER.get(name) and now - cur["ms"] < 3000:
            return
    S["pages"][name] = {"ms": now, "v": v, "paint": paint}


async def api_hud(req):
    """촬영용 상태 띠(/hud, /studio): 상황 이름 · 보행/차량 신호 · 차량 A·B 속도와 경고 단계 · 충돌 · 결과 한 줄."""
    page_seen(req, "studio" if req.query_params.get("page") == "studio" else "hud")
    now = now_ms()
    sim = S["sim"]
    carla = S["source"] == "carla"
    veh = {}
    for vid in ("A", "B"):
        v = sim["vehicles"].get(vid)
        last = S["veh_last"].get(vid) or {}
        veh[vid] = {"present": v is not None and now - v["recv"] < 1500,
                    "speed_kmh": v and v.get("speed_kmh"), "braking": bool(v and v.get("braking")),
                    "level": (hold_of(vid).level or None) if carla else None, "reasons": last.get("reasons") or [],
                    "app": any(a.vehicle == vid for a in S["apps"])}
    sig = sim["signal"] or {}
    col = next((e for e in sim["ev"] if e["e"] == "collision_A"), None)
    return JSONResponse({
        "now": now, "connected": sim["recv"] is not None and now - sim["recv"] < 1500, "running": sim["running"],
        "take": sim["take"], "take_ms": sim.get("take_ms"), "t": sim["t"], "title": sim["meta"].get("title") or (sim["phase"] if sim["running"] else None),
        "meta": sim["meta"],
        "signal": {"ped": sig.get("ped"), "veh": sig.get("veh"),
                   "ped_red_s": None if sim["ped_red_since_ms"] is None else round((now - sim["ped_red_since_ms"]) / 1000, 1)},
        "veh": veh, "collision": col and {"age_s": round((now - col["ms"]) / 1000, 1)},
        "result": sim["result"] and dict(sim["result"], age_s=round((now - sim["result"]["ms"]) / 1000, 1)),
        "edge": S["edge_ws"] is not None, "ver": PAGE_VER["hud"]})


async def page_studio(req):
    return FileResponse(os.path.join(ROOT, "server", "studio.html"), media_type="text/html")


async def page_hud(req):
    return FileResponse(os.path.join(ROOT, "server", "hud.html"), media_type="text/html")


async def api_preflight(req):
    """촬영 전 점검 (carla_demo.py 가 take 전에 부름). 항목마다 ok / 설명."""
    now = now_ms()
    out = []

    def add(name, ok, msg, need=True):
        out.append({"name": name, "ok": bool(ok), "msg": msg, "need": need})

    add("Pi 연결", S["edge_ws"] is not None, "Pi 에서 bash simdemo/run_pi_sim.sh <노트북IP> 실행")
    sc_age = None if not S["scene_recv"] else now - S["scene_recv"]
    c2 = S["cam2"]
    c2_age = None if not c2 else now - c2.get("_recv", 0)
    add("Cam1 분석", sc_age is not None and sc_age < 3000, "Pi 가 Cam1 결과를 보내지 않음 (CARLA 영상이 Pi 로 가는지 확인)")
    add("Cam2 분석", c2_age is not None and c2_age < 3000, "Pi 가 Cam2 결과를 보내지 않음")
    appA = [a for a in S["apps"] if a.vehicle == "A"]
    add("폰 앱 (차량 A)", bool(appA), "폰 앱이 서버에 연결되지 않음 → 앱을 켜고 서버 주소 확인", need=req.query_params.get("no_app") != "1")
    add("폰 주행 모드", any(a.driving for a in appA), "앱에서 '주행 시작'을 누르기", need=req.query_params.get("no_app") != "1")
    stu = S["pages"].get("studio")
    if stu is not None and now - stu["ms"] < 120000:      # 촬영 화면(/studio)을 쓰는 중 → 그 창 하나만 봄
        pages = (("studio", "촬영 화면", "bash simdemo/studio.sh 로 다시 열기 (열려 있으면 그 창에서 Ctrl+Shift+R)"),)
    else:
        pages = (("monitor", "관제 화면", "관제 창에서 Ctrl+Shift+R (새로고침)"),
                 ("hud", "상태 띠(HUD)", "bash simdemo/open_hud.sh 로 띠 창을 열거나, 열려 있으면 Ctrl+Shift+R"))
    for name, label, fix in pages:
        pg = S["pages"].get(name)
        alive = pg is not None and now - pg["ms"] < 3000
        add(label + " 갱신", alive, (label + " 이 멈춰 있거나 닫힘 → " + fix), need=name != "hud" or req.query_params.get("hud") == "1")
        if alive:
            add(label + " 최신", pg.get("v") == PAGE_VER[name], label + " 이 옛 버전 → " + fix, need=True)
            old = (S.get("pages_old") or {}).get(name)
            add(label + " 창 하나만", not (old and now - old < 3000),
                label + " 창이 2개 떠 있고 그중 하나가 옛 버전 (OBS 가 옛 창을 찍으면 영상에 옛 화면이 나옴) → "
                + label + " 창을 모두 닫고 하나만 다시 열기, OBS 소스에서 창을 다시 선택", need=True)
            add(label + " 보임", pg.get("paint") is None or pg["paint"] < 1500,
                label + " 창이 최소화됐거나 다른 탭에 가려져 그려지지 않음 (녹화하면 멈춘 화면이 찍힘) → 창을 다시 띄우기", need=True)
    return JSONResponse({"ok": all(c["ok"] for c in out if c["need"]), "checks": out})


async def api_monitor(req):
    """통제실 화면용: 상태 + 앱 전송 로그(원문·지연) + 사건 로그 + 최근 지연 통계."""
    if req.query_params.get("page") is None:      # 촬영 화면(/studio)이 자료만 가져갈 때는 관제 창으로 치지 않음
        page_seen(req, "monitor")
    st = json.loads((await api_status(req)).body)
    sent = list(S["sent_log"])
    def med(key):
        v = sorted(x["lat"][key] for x in sent if x["lat"].get(key) is not None)
        return v[len(v) // 2] if v else None
    sc = S["scene"]
    st["sent_log"] = sent[-25:]
    st["events"] = list(S["events"])[-40:]
    st["lat_stats"] = {"n": len(sent), "cap_server": med("cap_server"), "judge_sent": med("judge_sent"),
                       "cap_sent": med("cap_sent"), "server_phone": med("server_phone"),
                       "live_cap_server": sc and S["scene_recv"] and (S["scene_recv"] - sc["ts_cap"])}
    gps = []
    for a in S["apps"]:
        lf = a.last_fix
        if lf:
            gps.append({"vehicle": a.vehicle, "lat": lf["lat"], "lng": lf["lng"], "acc": lf["acc"],
                        "speed": lf["speed"], "heading": lf["heading_raw"], "age_s": round((now_ms() - lf["ms"]) / 1000, 1)})
    st["gps_raw"] = gps
    return JSONResponse(st)


async def api_preview(req):
    cam = req.path_params["cam"]
    p = S["preview"].get(cam)
    if not p:
        return Response(status_code=404)
    return Response(p[0], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


async def api_snapshot_get(req):
    cam = req.path_params["cam"]
    p = S["snapshot"].get(cam)
    if not p:
        return Response(status_code=404)
    return Response(p["jpg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


async def api_snapshot_take(req):
    b = await body(req)
    cam = b.get("cam", "cam1")
    fut = asyncio.get_running_loop().create_future()
    S["snap_waiters"][cam] = fut
    if not await edge_cmd({"cmd": "snapshot", "cam": cam}):
        return JSONResponse({"ok": False, "error": "Pi 연결 안 됨"})
    try:
        await asyncio.wait_for(fut, 6.0)
    except asyncio.TimeoutError:
        return JSONResponse({"ok": False, "error": "Pi 응답 없음 (6초)"})
    sn = S["snapshot"][cam]
    return JSONResponse({"ok": True, "w": sn["w"], "h": sn["h"], "ms": sn["ms"]})


async def api_edge_cmd(req):
    b = await body(req)
    if b.get("cmd") == "set_mode":
        S["cam1_mode"] = b.get("mode")
    oplog("edge_cmd", b)
    return JSONResponse({"ok": await edge_cmd(b)})


async def api_calib_cam1(req):
    b = await body(req)
    try:
        src, dst = b["img_pts"], b["world_pts"]
        H = homography_dlt(src, dst)
        err = reproj_errors(H, src, dst).tolist()
        chk = None
        if b.get("check_img"):
            chk = reproj_errors(H, b["check_img"], b["check_world"]).tolist()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)})
    out = {"ok": True, "H": H.tolist(), "Hinv": np.linalg.inv(H).tolist(),
           "reproj": [round(e, 3) for e in err], "reproj_mean": round(sum(err) / len(err), 3),
           "check": chk and [round(e, 3) for e in chk], "check_mean": chk and round(sum(chk) / len(chk), 3)}
    if b.get("apply"):
        detail = {"img_pts": src, "world_pts": dst, "check_img": b.get("check_img"),
                  "check_world": b.get("check_world"), "reproj_mean": out["reproj_mean"],
                  "check_mean": out["check_mean"], "check": out["check"]}
        out["sent"] = await edge_cmd({"cmd": "set_H", "cam": "cam1", "H": H.tolist(), "detail": detail})
        save_json(os.path.join(LOGDIR, "calib_cam1_%d.json" % now_ms()), dict(detail, H=H.tolist()))
        save_json(os.path.join(CALIB_DIR, "H_cam1_server_copy.json"), dict(detail, H=H.tolist()))
        S["h_detail"] = detail
        oplog("calib_cam1_apply", detail)
    return JSONResponse(out)


async def api_calib_cam2(req):
    b = await body(req)
    if len(b.get("polygon", [])) < 3 or len(b.get("boundary", [])) != 2:
        return JSONResponse({"ok": False, "error": "다각형 3점 이상 + 경계선 2점 필요"})
    cmd = {"cmd": "set_geom", "cam": "cam2", "polygon": b["polygon"], "boundary": b["boundary"]}
    if len(b.get("boundary_path") or []) >= 2:     # 휘어진 연석: 경계선 꺾은선 (예전 Pi 코드는 이 키를 모르고 boundary 직선만 씀)
        cmd["boundary_path"] = b["boundary_path"]
    sent = await edge_cmd(cmd)
    save_json(os.path.join(LOGDIR, "calib_cam2_%d.json" % now_ms()), b)
    if sent:
        S["cam2_set"] = True
    oplog("calib_cam2_apply", b)
    return JSONResponse({"ok": True, "sent": sent})


async def api_source(req):
    b = await body(req)
    if b.get("source") in ("gps", "cam", "virtual", "carla"):
        S["source"] = b["source"]
        S["engine"] = ConflictEngine(RISK, HOLD_S)
        reset_fusion()
        S["veh_last"] = {}
        oplog("source", b)
    return JSONResponse({"source": S["source"]})


# ── CARLA 시뮬 데모 (simdemo/carla_demo.py 가 10 Hz 로 보냄) ──
def reset_fusion():
    g = S["sim"]["geom"] or {}
    fcfg = dict(FUSION_CFG)
    for k in ("near_band_y", "waiting_xy"):
        if g.get(k) is not None:
            fcfg[k] = g[k]
    S["fusion"] = FusionEngine(RISK, fcfg, floor_now().get("crosswalk_x", (-1.5, 1.5)), HOLD_S)


SIM_EVENT_KO = {"take_start": "시뮬 시작", "ped_enter": "보행자 대기구역으로 이동", "ped_wait": "보행자 대기",
                "ped_sway": "보행자 서성임 (출발 아님)", "ped_approach": "보행자 연석 쪽으로 다가감",
                "ped_cross": "보행자 차도 진입", "ped_flash": "보행 신호 깜빡임 (FLASH)",
                "ped_red": "보행 신호 빨강", "veh_green": "차량 신호 초록", "veh_red": "차량 신호 빨강",
                "brake_A": "차량 A 감속·정지 (앱 경고 수신)", "fallback_stop_A": "안전장치 비상 정지 (기록)",
                "collision_A": "차량 A 가 보행자와 충돌 (앱 없음 대조)", "take_end": "시뮬 끝"}


async def api_sim(req):
    """CARLA 상태 수신.
    {"take", "t", "phase", "signal": {"ped": "GREEN|FLASH|RED", "veh": "RED|GREEN"},
     "vehicles": [{"id","kind","x","y","vx","vy","lat","lng","speed","braking"}],   ← 바닥 좌표(GNSS 변환) + OBU 속도
     "obstacles": [{"id","x","y","len","wid","heading"}], "events": ["ped_cross", ...],
     "geom": {...} (처음 한 번), "cmd": "take_start" | "take_end"}"""
    b = await body(req)
    now = now_ms()
    sim = S["sim"]
    first = sim["recv"] is None
    sim["recv"] = now
    sim["n"] += 1
    if b.get("geom"):
        sim["geom"] = b["geom"]
        sim["geom_rev"] += 1
        save_json(os.path.join(LOGDIR, "sim_geom_%d.json" % now), b["geom"])
        oplog("sim_geom", b["geom"])
        reset_fusion()
    if S["source"] != "carla":
        S["source"] = "carla"
        reset_fusion()
        S["veh_last"] = {}
        oplog("source", {"source": "carla", "auto": True})
        S["events"].append({"ms": now, "kind": "conn", "text": "CARLA 연결 → 차량 위치 = CARLA GNSS + 가상 OBU"})
    elif first:
        S["events"].append({"ms": now, "kind": "conn", "text": "CARLA 연결"})
    cmd = b.get("cmd")
    if cmd == "take_start":
        reset_fusion()
        S["veh_last"] = {}
        S["heading"], S["radar_ms"] = {}, {}
        sim["vehicles"] = {}
        sim["ped_prev"], sim["ped_red_since_ms"] = None, None
        sim["take"] = b.get("take")
        sim["meta"], sim["ev"], sim["result"], sim["running"] = b.get("meta") or {}, [], None, True
        sim["take_ms"] = now
        if S["trial"]:
            await _trial_stop("자동 종료(새 시뮬 시작)")
        await _trial_start({"exp": "SIM", "label": "take%s" % b.get("take"), "note": b.get("note", ""),
                            "scenario": None, "raw": bool(b.get("raw"))})
        await edge_cmd({"cmd": "reset_bg", "cam": "cam2"})
        await apps_clear("take_start")
    elif cmd == "take_end":
        sim["running"] = False
        await _trial_stop(b.get("note", "시뮬 끝"))
        reset_fusion()                 # 남은 경고 단계도 같이 내림 → 앱이 평상시 화면으로
        S["veh_last"] = {}
        await apps_clear("take_end")
    if b.get("result"):          # 촬영 화면(HUD)에 띄울 결과 한 줄 {"text", "ok"}
        sim["result"] = dict(b["result"], ms=now)
        oplog("sim_result", b["result"])
    sim["t"], sim["phase"] = b.get("t"), b.get("phase")
    if b.get("time_scale") is not None:
        sim["time_scale"] = b["time_scale"]
    if b.get("take") is not None:
        sim["take"] = b["take"]
    sig = b.get("signal")
    if sig:
        if sig.get("ped") == "RED" and sig.get("ped_red_elapsed_s") is not None:
            sim["ped_red_since_ms"] = now - int(float(sig["ped_red_elapsed_s"]) * 1000)   # 시뮬 시간 기준
        elif sig.get("ped") == "RED" and sim["ped_prev"] != "RED":
            sim["ped_red_since_ms"] = now
        if sig.get("ped") != "RED":
            sim["ped_red_since_ms"] = None
        sim["ped_prev"] = sig.get("ped")
        sim["signal"] = sig
    if "obstacles" in b:
        sim["obstacles"] = b["obstacles"]
    for v in b.get("vehicles", []):
        vid = v["id"]
        sim["vehicles"][vid] = {"x": float(v["x"]), "y": float(v["y"]), "vx": float(v["vx"]), "vy": float(v["vy"]),
                                "recv": now, "kind": v.get("kind"), "braking": v.get("braking"),
                                "speed_kmh": None if v.get("speed") is None else round(float(v["speed"]) * 3.6, 1),
                                "gnss": v.get("lat") is not None and [v.get("lat"), v.get("lng")]}
        L["sim"].row([now, trial_id(), sim["take"], sim["t"], sim["phase"], (sig or {}).get("ped"),
                      (sig or {}).get("veh"), vid, r3(v["x"]), r3(v["y"]), r3(v["vx"]), r3(v["vy"]), v.get("lat"),
                      v.get("lng"), r3(v.get("speed")), v.get("braking")])
    for e in b.get("events", []):
        sim["ev"] = (sim["ev"] + [{"ms": now, "e": e, "t": b.get("t")}])[-30:]
        S["events"].append({"ms": now, "kind": "sim", "text": "[시뮬] " + SIM_EVENT_KO.get(e, e)})
        if S["trial"]:
            S["trial"]["marks"].append({"ms": now, "note": "sim:" + e})
    await evaluate_fused(None, "sim")
    levels = {vid: hold_of(vid).level for vid in sim["vehicles"]}
    reasons = {vid: (S["veh_last"].get(vid) or {}).get("reasons") for vid in sim["vehicles"]}
    # 지금 단계를 앱으로 보낸 시각. 시뮬 속 운전자는 0.1초마다 이 응답으로 단계를 읽는데, 반응 시간은
    # '폰이 경고를 받은 때'부터 세야 하므로 (읽은 때부터 세면 읽는 주기만큼 운전자가 늦게 반응한 것처럼 됨)
    sent_ms = {}
    for vid, lv in levels.items():
        if lv:
            sent_ms[vid] = next((a["ms"] for a in reversed(S["alerts"]) if a["vehicle"] == vid and a["level"] == lv), None)
    return JSONResponse({"ok": True, "levels": levels, "reasons": reasons, "server_now": now, "level_sent_ms": sent_ms,
                         "edge": S["edge_ws"] is not None, "apps": [a.vehicle for a in S["apps"]]})


async def api_designate(req):
    b = await body(req)
    tid = b.get("track_id")
    S["designated"] = int(tid) if tid not in (None, "") else None
    oplog("designate", {"track_id": S["designated"]})
    return JSONResponse({"designated": S["designated"]})


async def api_assign(req):
    b = await body(req)
    for a in S["apps"]:
        if a.device_id == b.get("device_id"):
            a.vehicle = b.get("vehicle") or None
            a.gps.id = a.vehicle or "?"
            oplog("assign", b)
            return JSONResponse({"ok": True})
    return JSONResponse({"ok": False})


# ── 앱 경고 시험 (카메라 없이 폰 경고 수신만 확인) ──
async def api_test_alert(req):
    b = await body(req)
    vid = b.get("vehicle") or "A"
    if not apps_for(vid):
        return JSONResponse({"ok": False, "error": "차량 %s 앱이 연결돼 있지 않음" % vid})
    oplog("test_alert", {"vehicle": vid})
    for a in apps_for(vid):
        a.last_ack_ms = None

    async def run():
        fake = {"status": "WARN", "side": "front", "track_id": 0, "dist": 5.0, "angle_deg": 0.0,
                "tcpa": None, "dcpa": None}
        seq = [(None, "caution"), ("caution", "warning"), ("warning", "danger"), ("danger", None)]
        for prev, new in seq:
            await push(vid, prev, new, fake if new else None, None, None)
            await asyncio.sleep(2.0)
    asyncio.get_running_loop().create_task(run())
    return JSONResponse({"ok": True})


# ── GPS 기준점 ──
async def api_capture_start(req):
    b = await body(req)
    dev_a = next((a.device_id for a in S["apps"] if a.vehicle == "A" and a.device_id), None)
    S["capture"] = {"name": b.get("name", "p0"), "x": float(b.get("x", 0)), "y": float(b.get("y", 0)),
                    "device": dev_a, "fixes": [], "start": now_ms()}   # 차량 A 폰의 GPS만 모음
    oplog("capture_start", b)
    return JSONResponse({"ok": True})


async def api_capture_stop(req):
    cap = S["capture"]
    S["capture"] = None
    if not cap:
        return JSONResponse({"ok": False, "error": "기록 중 아님"})
    r = weighted_mean_fix(cap["fixes"], GPS_CFG.get("max_acc_m", 25))
    if r is None:
        return JSONResponse({"ok": False, "error": "쓸 만한 GPS 값이 없음 (%d개)" % len(cap["fixes"])})
    lat, lng, n, acc = r
    lats = [f[0] for f in cap["fixes"]]
    lngs = [f[1] for f in cap["fixes"]]
    from bgx.engine import enu
    spread = [math.hypot(*enu(la, lo, lat, lng)) for la, lo in zip(lats, lngs)]
    pt = {"lat": lat, "lng": lng, "n": n, "mean_acc": acc, "x": cap["x"], "y": cap["y"],
          "spread_p50": round(sorted(spread)[len(spread) // 2], 2), "device": cap["device"],
          "seconds": round((now_ms() - cap["start"]) / 1000, 1)}
    S["anchor_pts"][cap["name"]] = pt
    oplog("capture_stop", dict(pt, name=cap["name"]))
    save_json(os.path.join(LOGDIR, "anchor_pts.json"), S["anchor_pts"])
    msg = {"ok": True, "point": pt}
    if cap["name"] == "zero" and S["anchor"] is not None:
        # 알고 있는 바닥 점에서의 GPS 평균 → 치우침(bias) 보정값
        x0, y0 = S["anchor"].to_floor(lat, lng, with_bias=False)
        S["anchor"].bias = (cap["x"] - x0, cap["y"] - y0)
        save_json(ANCHOR_FILE, S["anchor"].to_dict())
        save_json(os.path.join(LOGDIR, "anchor_%d.json" % now_ms()), S["anchor"].to_dict())
        msg["bias"] = [round(v, 2) for v in S["anchor"].bias]
    return JSONResponse(msg)


async def api_anchor_compute(req):
    p0, p1 = S["anchor_pts"].get("p0"), S["anchor_pts"].get("p1")
    if not p0 or not p1:
        return JSONResponse({"ok": False, "error": "기준점 p0, p1 을 먼저 기록"})
    A = Anchor.from_two_points((p0["lat"], p0["lng"]), (p0["x"], p0["y"]), (p1["lat"], p1["lng"]),
                               (p1["x"], p1["y"]))
    S["anchor"] = A
    save_json(ANCHOR_FILE, A.to_dict())
    save_json(os.path.join(LOGDIR, "anchor_%d.json" % now_ms()), A.to_dict())
    oplog("anchor_compute", A.to_dict())
    return JSONResponse({"ok": True, "anchor": A.to_dict()})


async def api_anchor_manual(req):
    b = await body(req)
    try:
        A = Anchor.from_bearing((float(b["lat"]), float(b["lng"])), (float(b.get("x", 0)), float(b.get("y", 0))),
                                float(b["bearing"]))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)})
    S["anchor"] = A
    save_json(ANCHOR_FILE, A.to_dict())
    oplog("anchor_manual", A.to_dict())
    return JSONResponse({"ok": True, "anchor": A.to_dict()})


async def api_anchor_clear_bias(req):
    if S["anchor"]:
        S["anchor"].bias = (0.0, 0.0)
        save_json(ANCHOR_FILE, S["anchor"].to_dict())
        oplog("anchor_clear_bias", {})
    return JSONResponse({"ok": True})


# ── 시험 ──
async def api_trial_start(req):
    return JSONResponse(await _trial_start(await body(req)))


async def _trial_start(b):
    if S["trial"]:
        await _trial_stop("자동 종료(새 시험 시작)")
    S["n_trial"] += 1
    sc_name = b.get("scenario") or None
    sc = SCENARIOS.get(sc_name) if sc_name else None
    start = now_ms()
    go = start + int(float(b.get("countdown_s", 0)) * 1000)
    tid = "T%03d" % S["n_trial"]
    tr = {"id": tid, "exp": b.get("exp", ""), "label": b.get("label", ""), "scenario": sc_name,
          "expected": sc.get("expected") if sc else None, "note": b.get("note", ""), "start_ms": start,
          "go_ms": go, "alerts": [], "marks": [],
          "virtual": [VirtualVehicle(v) for v in sc["vehicles"]] if (sc and sc.get("source") == "virtual") else None}
    if sc and sc.get("source") in ("gps", "virtual") and S["source"] != sc["source"] and \
            not (sc["source"] == "gps" and S["source"] == "cam"):
        S["source"] = sc["source"]
        S["engine"] = ConflictEngine(RISK, HOLD_S)
    S["trial"] = tr
    name = "%s_%s" % (tid, tr["exp"])
    await edge_cmd({"cmd": "record", "on": True, "name": name, "raw": bool(b.get("raw"))})
    oplog("trial_start", {k: v for k, v in tr.items() if k != "virtual"})
    print("[시험] %s 시작 %s %s %s" % (tid, tr["exp"], tr["label"], sc_name or ""))
    return {"trial_id": tid, "start_ms": start, "go_ms": go, "server_now": now_ms()}


async def _trial_stop(result_note=""):
    tr = S["trial"]
    if not tr:
        return None
    S["trial"] = None
    stop = now_ms()
    await edge_cmd({"cmd": "record", "on": False})
    L["trial"].row([tr["id"], tr["exp"], tr["label"], tr["scenario"], S["source"],
                    json.dumps(tr["expected"], ensure_ascii=False), tr["note"], tr["start_ms"], tr["go_ms"], stop,
                    result_note, json.dumps(tr["marks"], ensure_ascii=False), S["cam1_mode"], S["designated"]])
    oplog("trial_stop", {"id": tr["id"], "result_note": result_note})
    lv = [a["level"] for a in tr["alerts"] if a["vehicle"] == "A" and a["level"]]
    order = {"caution": 1, "warning": 2, "danger": 3}
    S["trials_done"].append({"id": tr["id"], "exp": tr["exp"], "label": tr["label"], "scenario": tr["scenario"],
                             "sec": round((stop - tr["go_ms"]) / 1000, 1), "expected": tr["expected"],
                             "alert_A": max(lv, key=lambda x: order[x]) if lv else None})
    print("[시험] %s 종료" % tr["id"])
    return {"trial_id": tr["id"], "alerts": tr["alerts"]}


async def api_trial_stop(req):
    b = await body(req)
    return JSONResponse(await _trial_stop(b.get("result_note", "")) or {"error": "진행 중인 시험 없음"})


async def api_trial_mark(req):
    b = await body(req)
    if S["trial"]:
        S["trial"]["marks"].append({"ms": now_ms(), "note": b.get("note", "")})
        return JSONResponse({"ok": True})
    oplog("mark_no_trial", b)
    return JSONResponse({"ok": False})


async def page_vision(req):
    return FileResponse(os.path.join(ROOT, "server", "vision.html"), media_type="text/html")


async def api_vision_status(req):
    st = VT.status(S["edge_ws"] is not None)
    st["emr_thr"] = SITE.get("cam2", {}).get("pipeline", {}).get("emr_thr", 0.7)
    st["cam1_mode"] = (S["scene"] or {}).get("mode")
    st["model"] = (S["edge_hello"] or {}).get("model")
    return JSONResponse(st)


async def api_vision(req):
    b = await body(req)
    act = req.path_params["act"]
    if act == "start":
        out = await VT.start(b)
    elif act == "stop":
        out = await VT.stop(b)
    elif act == "calib_add":
        out = VT.calib_add(b)
    elif act == "calib_del":
        out = VT.calib_del(b)
    elif act == "calib_apply":
        out = await VT.calib_apply()
    elif act == "calib_clear":
        out = await VT.calib_clear()
    elif act == "demo_speed":
        out = await VT.demo_speed(b)
    elif act == "reset_bg":
        out = {"ok": await edge_cmd({"cmd": "reset_bg", "cam": "cam2"})}
    else:
        out = {"ok": False, "error": "모르는 동작"}
    oplog("vision_" + act, {"req": b, "ok": out.get("ok"), "error": out.get("error")})
    return JSONResponse(out)


VIDEO_DIR = os.environ.get("BGX_VIDEO_DIR") or os.path.join(ROOT, "video")   # /vision 화면 녹화가 저장되는 곳


def _remux(path):
    """브라우저가 보낸 조각 영상을 보통 영상 파일로 다시 묶음 (길이 정보·탐색). ffmpeg 가 없거나 실패하면 원본 그대로."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return False
    tmp = path + ".tmp." + path.rsplit(".", 1)[-1]
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c", "copy", "-movflags", "+faststart", tmp],
                       check=True, timeout=120)
        if os.path.getsize(tmp) > 1000:
            os.replace(tmp, path)
            return True
    except Exception:  # noqa: BLE001
        pass
    with contextlib.suppress(OSError):
        os.remove(tmp)
    return False


async def api_vision_rec(req):
    """/vision 페이지가 테스트 동안 녹화한 화면을 1초 조각으로 보냄 → video/<이름> 에 이어 붙임."""
    import re
    name = re.sub(r"[^0-9A-Za-z_.\-]", "_", req.path_params["name"])[:120]
    if name.rsplit(".", 1)[-1] not in ("mp4", "webm"):
        return JSONResponse({"ok": False, "error": "mp4 또는 webm 만"})
    os.makedirs(VIDEO_DIR, exist_ok=True)
    path = os.path.join(VIDEO_DIR, name)
    if req.query_params.get("done"):
        if not os.path.exists(path):
            return JSONResponse({"ok": False, "error": "받은 영상이 없음"})
        fixed = await asyncio.to_thread(_remux, path)
        oplog("vision_screen_saved", {"file": path, "bytes": os.path.getsize(path), "remux": fixed})
        return JSONResponse({"ok": True, "file": os.path.relpath(path, ROOT), "mb": round(os.path.getsize(path) / 1e6, 1)})
    data = await req.body()
    with open(path, "wb" if req.query_params.get("seq") == "0" else "ab") as f:
        f.write(data)
    return JSONResponse({"ok": True})


@contextlib.asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(ticker())
    save_json(os.path.join(LOGDIR, "session.json"), {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "site": SITE,
                                                     "site_file": SITE_FILE,
                                                     "scenarios": SCENARIOS})
    print("기록 폴더:", LOGDIR)
    print("조작 페이지: http://localhost:8000/control  (다른 기기에서는 http://<노트북IP>:8000/control)")
    print("실물 Vision 테스트: http://localhost:8000/vision")
    print("시뮬 촬영 화면: http://localhost:8000/studio  (bash simdemo/studio.sh)")
    try:
        yield
    finally:
        task.cancel()


app = Starlette(routes=[
    Route("/", page_root),
    Route("/control", page_control),
    Route("/monitor", page_monitor),
    Route("/hud", page_hud),
    Route("/studio", page_studio),
    Route("/api/hud", api_hud),
    Route("/api/preflight", api_preflight),
    Route("/vision", page_vision),
    Route("/api/vision/status", api_vision_status),
    Route("/api/vision/rec/{name}", api_vision_rec, methods=["POST"]),
    Route("/api/vision/{act}", api_vision, methods=["POST"]),
    Route("/api/monitor", api_monitor),
    Route("/api/config", api_config),
    Route("/api/status", api_status),
    Route("/api/preview/{cam}", api_preview),
    Route("/api/snapshot/{cam}", api_snapshot_get),
    Route("/api/snapshot", api_snapshot_take, methods=["POST"]),
    Route("/api/edge/cmd", api_edge_cmd, methods=["POST"]),
    Route("/api/calib/cam1", api_calib_cam1, methods=["POST"]),
    Route("/api/calib/cam2", api_calib_cam2, methods=["POST"]),
    Route("/api/source", api_source, methods=["POST"]),
    Route("/api/designate", api_designate, methods=["POST"]),
    Route("/api/assign", api_assign, methods=["POST"]),
    Route("/api/test_alert", api_test_alert, methods=["POST"]),
    Route("/api/anchor/capture/start", api_capture_start, methods=["POST"]),
    Route("/api/anchor/capture/stop", api_capture_stop, methods=["POST"]),
    Route("/api/anchor/compute", api_anchor_compute, methods=["POST"]),
    Route("/api/anchor/manual", api_anchor_manual, methods=["POST"]),
    Route("/api/anchor/clear_bias", api_anchor_clear_bias, methods=["POST"]),
    Route("/api/trial/start", api_trial_start, methods=["POST"]),
    Route("/api/trial/stop", api_trial_stop, methods=["POST"]),
    Route("/api/trial/mark", api_trial_mark, methods=["POST"]),
    Route("/api/sim", api_sim, methods=["POST"]),
    WebSocketRoute("/ws", ws_app),
    WebSocketRoute("/edge", ws_edge),
], lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=32 * 1024 * 1024)
