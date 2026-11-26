# server/vision.py — 실물 Vision 테스트 (웹 페이지 /vision 의 서버 쪽)
#
# 목적: USB 웹캠 2대 실물에서 "Pi 한 대가 Cam1·Cam2 를 동시에 처리하는 상태"로
#       Cam1(검출·추적) / Cam2(출발 징후) 를 시험하고, 화면을 그대로 데모 영상으로 쓴다.
#
#   - 동시 처리 확인: 두 카메라 결과가 모두 최근 2초 안에 들어오고 FPS 가 기준 이상인 상태가 3초 이어져야
#                     테스트를 시작할 수 있음. 테스트 도중 2초 넘게 끊기면 그 테스트는 '무효'.
#   - 테스트: 시작하면 Pi 가 원본·분석 영상을 녹화하고, 끝내면 그 구간의 수치를 한 줄로 정리해 저장.
#   - 거리 보정: 사람이 선 자리를 "앞 몇 m, 오른쪽 몇 m" 로 적어 4곳 이상 모으면 Cam1 바닥 보정(H)을 만듦.
#                바닥 좌표: x = 오른쪽(m), y = -앞(m), 원점 = Cam1 삼각대 바로 아래.
#
# exp_server.py 가 Pi 메시지를 받을 때 on_scene / on_cam2 를 불러 준다.
import csv
import json
import math
import os
import time
from collections import deque

import numpy as np

SCENARIOS = {
    "cam1": [
        {"key": "stand", "name": "서 있기 (거리별)", "guide": "정한 거리에 15초 서 있기", "note": "거리 m"},
        {"key": "walk_side", "name": "가로로 걷기", "guide": "화면을 좌↔우로 보통 걸음"},
        {"key": "walk_depth", "name": "멀어졌다 돌아오기", "guide": "가까이에서 멀리 갔다가 돌아오기"},
        {"key": "fast", "name": "빠르게", "guide": "빠른 걸음·뛰기로 가로지르기"},
        {"key": "two", "name": "두 명 엇갈림", "guide": "두 사람이 서로 엇갈려 지나가기 (ID 유지)"},
        {"key": "free", "name": "자유", "guide": "자유롭게"},
    ],
    "cam2": [
        {"key": "depart", "name": "출발", "guide": "구역에 3초 서 있다가 경계선 쪽으로 나가기", "expect": "alert"},
        {"key": "dash", "name": "급출발", "guide": "서 있다가 갑자기 뛰어나가기", "expect": "alert"},
        {"key": "loiter", "name": "서성이기만", "guide": "구역 안에서 20초 서성이기 (안 건넘)", "expect": "quiet"},
        {"key": "pass", "name": "지나가기", "guide": "경계선과 나란히 지나가기", "expect": "quiet"},
        {"key": "turn_back", "name": "갔다가 돌아섬", "guide": "경계선 앞까지 갔다가 돌아오기", "expect": "no_cross"},
        {"key": "free", "name": "자유", "guide": "자유롭게"},
    ],
}
CSV_COLS = ["id", "cam", "scenario", "note", "start", "sec", "valid", "invalid_why", "verdict", "frames", "fps",
            "ms_p50", "ms_p95", "yolo_ms_p50", "yolo_pct", "lat_p50", "det_pct", "ids", "id_switch", "track_longest_s",
            "dist_min", "dist_max", "speed_med", "present_pct", "emr_max", "emr_high", "cross", "back", "lead_s",
            "other_fps", "other_ms_p50", "temp_max", "video", "screen"]


def _pct(a, q):
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def _med(a, nd=1):
    return round(float(np.median(a)), nd) if len(a) else None


class Live:
    """카메라 하나의 최근 5초 창: 실제 FPS·처리 시간."""

    def __init__(self):
        self.q = deque()          # (recv_ms, ts_cap_ms, t_total_ms)
        self.temp = None

    def add(self, recv, ts_cap, t_total):
        """돌려주는 값: 바로 앞 프레임과의 촬영 시각 간격(ms). 처리가 멈췄다 돌아오면 이 값이 커짐."""
        gap = ts_cap - self.q[-1][1] if self.q else 0
        self.q.append((recv, ts_cap, t_total))
        self._trim(recv)
        return gap

    def _trim(self, recv):
        while self.q and recv - self.q[0][0] > 5000:
            self.q.popleft()

    def stats(self, now):
        q = [x for x in self.q if now - x[0] <= 3000]
        if len(q) < 2:
            return {"fps": 0.0, "ms": None, "age_ms": (now - self.q[-1][0]) if self.q else None, "n": len(q)}
        span = (q[-1][1] - q[0][1]) / 1000.0
        fps = (len(q) - 1) / span if span > 0 else 0.0
        return {"fps": round(fps, 1), "ms": _med([x[2] for x in q]), "age_ms": now - q[-1][0], "n": len(q)}


class VisionTest:
    def __init__(self, logdir, calib_dir, edge_cmd, homography_dlt, reproj_errors, cfg=None, now_ms=None):
        self.logdir, self.calib_dir, self.edge_cmd = logdir, calib_dir, edge_cmd
        self.dlt, self.reproj = homography_dlt, reproj_errors
        self.cfg = dict({"min_fps": 5.0, "fresh_ms": 2000, "ready_hold_ms": 3000, "invalid_gap_ms": 2000}, **(cfg or {}))
        self.now = now_ms or (lambda: int(time.time() * 1000))
        self.live = {"cam1": Live(), "cam2": Live()}
        self.fps_hist = deque(maxlen=240)        # (ms, cam1 fps, cam2 fps) 0.5초마다 → 화면의 동시 처리 그래프
        self.both_since = None
        self.last_tick = None
        self.test = None
        self.n_test = 0
        self.results = []
        self.feet = deque()                      # 최근 3초 가장 큰 사람의 발 (recv, u, v, 박스높이)
        self.peds = []                           # 화면용 최근 Cam1 사람들
        self.trail = deque(maxlen=400)           # 바닥 지도 궤적 (ms, id, x, y)
        self.cam2_last = None
        self.emr_hist = deque(maxlen=400)        # (ms, emr, state, crossed)
        self.has_H = False
        self.calib_pts = []
        self.calib_info = self._load_calib_info()
        self.want_stop = False
        self.csv_path = os.path.join(logdir, "vision_tests.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(CSV_COLS)

    def _load_calib_info(self):
        try:
            with open(os.path.join(self.calib_dir, "H_cam1_server_copy.json"), encoding="utf-8") as f:
                d = json.load(f)
            return {"camera_frame": bool(d.get("spots_fwd_right")), "n": len(d.get("img_pts", [])),
                    "reproj_mean": d.get("reproj_mean"), "saved": d.get("saved")}
        except (OSError, ValueError):
            return None

    # ── 동시 처리 확인 ──
    def _tick(self, now):
        s1, s2 = self.live["cam1"].stats(now), self.live["cam2"].stats(now)
        ok = {}
        for k, s in (("cam1", s1), ("cam2", s2)):
            ok[k] = s["age_ms"] is not None and s["age_ms"] <= self.cfg["fresh_ms"] and s["fps"] >= self.cfg["min_fps"]
        both = ok["cam1"] and ok["cam2"]
        if both:
            self.both_since = self.both_since or now
        else:
            self.both_since = None
        if not self.fps_hist or now - self.fps_hist[-1][0] >= 500:
            self.fps_hist.append((now, s1["fps"] if ok["cam1"] or s1["n"] else 0.0, s2["fps"] if ok["cam2"] or s2["n"] else 0.0))
        t = self.test
        if t is not None:
            if not both and self.last_tick is not None:
                t["bad_ms"] += min(now - self.last_tick, 1000)
                t["bad_run"] += min(now - self.last_tick, 1000)
                if t["bad_run"] > self.cfg["invalid_gap_ms"] and not t["invalid"]:
                    t["invalid"] = "%s 처리가 %.0f초 넘게 끊김" % (
                        "Cam1·Cam2" if not ok["cam1"] and not ok["cam2"] else ("Cam1" if not ok["cam1"] else "Cam2"),
                        self.cfg["invalid_gap_ms"] / 1000)
            elif both:
                t["bad_run"] = 0
            if t.get("auto_s") and now - t["start_ms"] >= t["auto_s"] * 1000:
                self.want_stop = True
        self.last_tick = now
        ready = both and now - self.both_since >= self.cfg["ready_hold_ms"]
        return s1, s2, ok, both, ready

    def _gap(self, name, gap_ms):
        """테스트 중 한 카메라의 프레임 간격이 기준(2초)을 넘으면 = 그동안 동시 처리가 아니었음 → 무효."""
        t = self.test
        if t is not None and not t["invalid"] and gap_ms > self.cfg["invalid_gap_ms"]:
            t["invalid"] = "%s 처리가 %.1f초 끊김" % (name, gap_ms / 1000.0)

    # ── Pi 메시지 ──
    async def on_scene(self, m, recv):
        tm = m.get("timing", {})
        self._gap("Cam1", self.live["cam1"].add(recv, m["ts_cap"], tm.get("total") or 0.0))
        self.live["cam1"].temp = m.get("temp_c")
        self.has_H = bool(m.get("has_H"))
        peds = m.get("peds", [])
        prim = None
        for p in peds:
            b = p.get("bbox")
            if b and p.get("u") is not None and (prim is None or b[3] - b[1] > prim["bbox"][3] - prim["bbox"][1]):
                prim = p
        if prim is not None:
            self.feet.append((recv, float(prim["u"]), float(prim["v"]), float(prim["bbox"][3] - prim["bbox"][1])))
        while self.feet and recv - self.feet[0][0] > 3000:
            self.feet.popleft()
        self.peds = [{"id": p["track_id"], "x": p.get("x_m"), "y": p.get("y_m"),
                      "v": p["spd"] if p.get("spd") is not None else (
                          round(math.hypot(p["vx"], p["vy"]), 2) if p.get("v_ok") else None),
                      "v_src": p.get("spd_src"),
                      "main": p is prim} for p in peds]
        for p in peds:
            if p.get("x_m") is not None:
                self.trail.append((recv, p["track_id"], p["x_m"], p["y_m"]))
        t = self.test
        if t is not None:
            if t["cam"] == "cam1":
                t["n"] += 1
                t["ms"].append(tm.get("total") or 0.0)
                t["lat"].append(m["ts_done"] - m["ts_cap"])
                if m.get("det_ran"):
                    t["n_yolo"] += 1
                    t["infer"].append(tm.get("infer") or 0.0)
                if m.get("temp_c") is not None:
                    t["temp"] = max(t["temp"] or 0, m["temp_c"])
                if peds:
                    t["n_ped"] += 1
                pid = prim["track_id"] if prim is not None else None
                if pid is not None:
                    t["ids"].add(pid)
                    if t["last_id"] is not None and pid != t["last_id"]:
                        t["switch"] += 1
                    if pid != t["run_id"]:
                        t["run_id"], t["run_t0"] = pid, m["ts_cap"]
                    t["longest"] = max(t["longest"], (m["ts_cap"] - t["run_t0"]) / 1000.0)
                    t["last_id"] = pid
                    if prim.get("x_m") is not None:
                        d = math.hypot(prim["x_m"], prim["y_m"])
                        if d < 80:
                            t["dist"].append(d)
                    sp = prim.get("spd") if prim.get("spd_src") != "demo" else None   # 예시값은 결과에 넣지 않음
                    if sp is None and prim.get("v_ok") and prim.get("spd_src") != "demo":
                        sp = math.hypot(prim["vx"], prim["vy"])
                    if sp is not None and 0.3 < sp < 12:
                        t["speed"].append(sp)
                for p in peds:
                    t["ids_all"].add(p["track_id"])
            else:
                t["o_n"] += 1
                t["o_ms"].append(tm.get("total") or 0.0)
                t["o_t"] = (t["o_t"][0] or m["ts_cap"], m["ts_cap"])
        self._tick(recv)
        if self.want_stop:
            await self.stop({"auto": True})

    async def on_cam2(self, m, recv):
        tm = m.get("timing", {})
        self._gap("Cam2", self.live["cam2"].add(recv, m["ts_cap"], tm.get("total") or 0.0))
        self.cam2_last = {"emr": m.get("emr"), "state": bool(m.get("emr_state")), "crossed": bool(m.get("crossed")),
                          "present": bool(m.get("present")), "warmup": bool(m.get("warmup")),
                          "prox": m.get("prox"), "tow": m.get("tow"), "per": m.get("per")}
        self.emr_hist.append((recv, m.get("emr") or 0.0, bool(m.get("emr_state")), bool(m.get("crossed"))))
        t = self.test
        if t is not None:
            if t["cam"] == "cam2":
                t["n"] += 1
                t["ms"].append(tm.get("total") or 0.0)
                t["lat"].append(m["ts_done"] - m["ts_cap"])
                if not m.get("warmup"):
                    t["n_live"] += 1
                    t["n_present"] += 1 if m.get("present") else 0
                    t["emr_max"] = max(t["emr_max"], m.get("emr") or 0.0)
                for e in m.get("events", []):
                    t["ev"][e] = t["ev"].get(e, 0) + 1
                    t["ev_log"].append({"t": round((m["ts_cap"] - t["start_ms"]) / 1000.0, 2), "e": e})
                    if e == "emr_high":
                        t["last_high"] = m["ts_cap"]
                    if e == "cross" and t["last_high"] is not None:
                        t["lead"].append((m["ts_cap"] - t["last_high"]) / 1000.0)
                        t["last_high"] = None
            else:
                t["o_n"] += 1
                t["o_ms"].append(tm.get("total") or 0.0)
                t["o_t"] = (t["o_t"][0] or m["ts_cap"], m["ts_cap"])
        self._tick(recv)
        if self.want_stop:
            await self.stop({"auto": True})

    # ── 테스트 ──
    async def start(self, b):
        now = self.now()
        _, _, ok, both, ready = self._tick(now)
        cam = b.get("cam")
        if cam not in ("cam1", "cam2"):
            return {"ok": False, "error": "cam 은 cam1 또는 cam2"}
        if self.test is not None:
            return {"ok": False, "error": "이미 테스트 중 (%s)" % self.test["id"]}
        if not ready:
            miss = [k.replace("cam", "Cam") for k in ("cam1", "cam2") if not ok[k]]
            return {"ok": False, "error": ("%s 가 처리되고 있지 않음" % "·".join(miss)) if miss else
                    "동시 처리 확인 중 (3초 유지되면 시작 가능)"}
        sc = next((s for s in SCENARIOS[cam] if s["key"] == b.get("scenario")), SCENARIOS[cam][-1])
        self.n_test += 1
        tid = "V%03d" % self.n_test
        name = "%s_%s_%s" % (tid, cam, sc["key"])
        screen = "%s_%s" % (time.strftime("%m%d_%H%M%S", time.localtime(now / 1000.0)), name)   # 화면 녹화 파일 이름 (video/)
        self.test = {"id": tid, "cam": cam, "sc": sc, "note": str(b.get("note") or "")[:80], "start_ms": now,
                     "auto_s": float(b["seconds"]) if b.get("seconds") else None, "video": name,
                     "screen": screen if b.get("screen") else None,
                     "bad_ms": 0, "bad_run": 0, "invalid": None, "n": 0, "ms": [], "lat": [], "temp": None,
                     "o_n": 0, "o_ms": [], "o_t": (None, None),
                     # cam1
                     "n_ped": 0, "n_yolo": 0, "infer": [], "ids": set(), "ids_all": set(), "last_id": None, "switch": 0,
                     "run_id": None, "run_t0": now, "longest": 0.0, "dist": [], "speed": [],
                     # cam2
                     "n_live": 0, "n_present": 0, "emr_max": 0.0, "ev": {}, "ev_log": [], "last_high": None, "lead": []}
        self.want_stop = False
        self.trail.clear()
        rec = await self.edge_cmd({"cmd": "record", "on": True, "name": name, "raw": True})
        self.test["rec"] = bool(rec)
        return {"ok": True, "id": tid, "recording": bool(rec), "screen": screen}

    def _summary(self, t, now):
        sec = (now - t["start_ms"]) / 1000.0
        r = {"id": t["id"], "cam": t["cam"], "scenario": t["sc"]["name"], "note": t["note"],
             "start": time.strftime("%H:%M:%S", time.localtime(t["start_ms"] / 1000.0)), "sec": round(sec, 1),
             "valid": t["invalid"] is None, "invalid_why": t["invalid"], "frames": t["n"],
             "fps": round(t["n"] / sec, 1) if sec > 0 else None, "ms_p50": _pct(t["ms"], 50), "ms_p95": _pct(t["ms"], 95),
             "lat_p50": _pct(t["lat"], 50), "temp_max": t["temp"], "video": t["video"] if t.get("rec") else None, "screen": t.get("screen"),
             "other_ms_p50": _pct(t["o_ms"], 50)}
        a, z = t["o_t"]
        r["other_fps"] = round((t["o_n"] - 1) / ((z - a) / 1000.0), 1) if a and z and z > a else None
        verdict = None
        if t["cam"] == "cam1":
            r.update({"yolo_ms_p50": _pct(t["infer"], 50),
                      "yolo_pct": round(100.0 * t["n_yolo"] / t["n"]) if t["n"] else None,
                      "det_pct": round(100.0 * t["n_ped"] / t["n"], 1) if t["n"] else None,
                      "ids": len(t["ids_all"]), "id_switch": t["switch"], "track_longest_s": round(t["longest"], 1),
                      "dist_min": round(min(t["dist"]), 1) if t["dist"] else None,
                      "dist_max": round(max(t["dist"]), 1) if t["dist"] else None, "speed_med": _med(t["speed"], 2)})
        else:
            ev = t["ev"]
            r.update({"present_pct": round(100.0 * t["n_present"] / t["n_live"], 1) if t["n_live"] else None,
                      "emr_max": round(t["emr_max"], 2), "emr_high": ev.get("emr_high", 0), "cross": ev.get("cross", 0),
                      "back": ev.get("back", 0), "lead_s": _med(t["lead"], 2), "events": t["ev_log"][-20:]})
            exp = t["sc"].get("expect")
            alert = ev.get("emr_high", 0) > 0 or ev.get("cross", 0) > 0
            if exp == "alert":
                verdict = "감지 성공" if alert else "감지 못함"
            elif exp == "quiet":
                verdict = "오경보" if alert else "오경보 없음"
            elif exp == "no_cross":
                verdict = "넘음으로 잘못 판단" if ev.get("cross", 0) else "넘음 없음"
        r["verdict"] = verdict
        if not r["valid"]:
            r["verdict"] = "무효"
        return r

    async def stop(self, b=None):
        t = self.test
        if t is None:
            return {"ok": False, "error": "진행 중인 테스트 없음"}
        self.test = None
        self.want_stop = False
        await self.edge_cmd({"cmd": "record", "on": False})
        r = self._summary(t, self.now())
        if (b or {}).get("discard"):
            return {"ok": True, "discarded": True}
        self.results.append(r)
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([r.get(k) for k in CSV_COLS])
        with open(os.path.join(self.logdir, "vision_tests.json"), "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=1)
        return {"ok": True, "result": r}

    # ── 거리 보정 ──
    def calib_add(self, b):
        try:
            fwd, right = float(b["fwd"]), float(b.get("right") or 0.0)
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "앞 거리(m)를 숫자로 적어 주세요"}
        if fwd <= 0:
            return {"ok": False, "error": "앞 거리는 0보다 커야 함"}
        now = self.now()
        pts = [x for x in self.feet if now - x[0] <= 1500]
        if len(pts) < 5:
            return {"ok": False, "error": "Cam1 에 사람이 잡히지 않음 (자리에 선 뒤 박스가 뜨면 누르기)"}
        u, v, h = np.median([p[1] for p in pts]), np.median([p[2] for p in pts]), np.median([p[3] for p in pts])
        spread = max(math.hypot(p[1] - u, p[2] - v) for p in pts)
        if spread > max(10.0, 0.12 * h):
            return {"ok": False, "error": "움직이는 중 → 가만히 선 뒤 다시 누르기"}
        self.calib_pts.append({"fwd": fwd, "right": right, "u": round(float(u), 1), "v": round(float(v), 1)})
        return {"ok": True, "n": len(self.calib_pts)}

    def calib_del(self, b):
        i = b.get("index")
        if i is None:
            self.calib_pts = []
        elif 0 <= int(i) < len(self.calib_pts):
            self.calib_pts.pop(int(i))
        return {"ok": True, "n": len(self.calib_pts)}

    def _calib_fit(self):
        P = self.calib_pts
        if len(P) < 4:
            return None, "자리가 4곳 이상 필요 (지금 %d곳)" % len(P)
        world = np.array([[p["right"], -p["fwd"]] for p in P], float)
        img = np.array([[p["u"], p["v"]] for p in P], float)
        for arr, why in ((world, "자리들이 거의 한 줄 위에 있음 → 좌우로 벌린 자리를 추가"),
                         (img, "화면에서 발 위치가 거의 한 줄 → 다른 자리를 추가")):
            sv = np.linalg.svd(arr - arr.mean(0), compute_uv=False)
            if sv[1] < 0.12 * sv[0]:
                return None, why
        H = self.dlt(img, world)
        err = self.reproj(H, img, world)
        return (H, img, world, err), None

    async def demo_speed(self, b):
        """속도 예시값 표시 켜기/끄기 (측정값 아님. 화면에 (demo)/(예시값) 표시가 붙고 결과 기록에는 안 들어감)."""
        on = bool(b.get("on"))
        lo, hi = (float(b.get("min", 2.2)), float(b.get("max", 3.0))) if on else (0.0, 0.0)
        sent = await self.edge_cmd({"cmd": "set_cfg", "cam": "cam1", "cfg": {"demo_speed_min": lo, "demo_speed_max": hi}})
        return {"ok": bool(sent), "error": None if sent else "Pi 연결 안 됨", "on": on}

    async def calib_clear(self):
        """적용된 바닥 보정을 지움 → 위치·거리는 표시하지 않고, 속도는 사람 키 기준 추정으로 바뀜."""
        sent = await self.edge_cmd({"cmd": "set_H", "cam": "cam1", "H": None})
        if not sent:
            return {"ok": False, "error": "Pi 연결 안 됨"}
        p = os.path.join(self.calib_dir, "H_cam1_server_copy.json")
        if os.path.exists(p):
            os.replace(p, p + ".removed")
        self.calib_info = None
        self.trail.clear()
        return {"ok": True}

    async def calib_apply(self):
        fit, why = self._calib_fit()
        if fit is None:
            return {"ok": False, "error": why}
        H, img, world, err = fit
        detail = {"img_pts": img.tolist(), "world_pts": world.tolist(),
                  "spots_fwd_right": [[p["fwd"], p["right"]] for p in self.calib_pts],
                  "reproj_mean": round(float(err.mean()), 3), "reproj_max": round(float(err.max()), 3),
                  "frame": "x=오른쪽 m, y=-앞 m, 원점=Cam1 삼각대 아래", "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        sent = await self.edge_cmd({"cmd": "set_H", "cam": "cam1", "H": H.tolist(), "detail": detail})
        if not sent:
            return {"ok": False, "error": "Pi 연결 안 됨 → 보정을 보내지 못함"}
        obj = dict(detail, H=H.tolist())
        os.makedirs(self.calib_dir, exist_ok=True)
        for p in (os.path.join(self.calib_dir, "H_cam1_server_copy.json"),
                  os.path.join(self.logdir, "calib_cam1_distance_%d.json" % self.now())):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=1)
        self.calib_info = {"camera_frame": True, "n": len(img), "reproj_mean": detail["reproj_mean"],
                           "saved": detail["saved"]}
        self.trail.clear()
        return {"ok": True, "reproj": [round(float(e), 2) for e in err], "reproj_mean": detail["reproj_mean"]}

    # ── 화면 ──
    def status(self, edge_connected):
        now = self.now()
        s1, s2, ok, both, ready = self._tick(now)
        t = self.test
        cur = None
        if t is not None:
            cur = self._summary(t, now)
            cur["guide"] = t["sc"]["guide"]
            cur["auto_s"] = t["auto_s"]
            cur["recording"] = t.get("rec")
        fit, why = self._calib_fit()
        return {
            "now": now, "edge": edge_connected, "ok": ok, "both": both, "ready": ready,
            "ready_in_ms": 0 if ready else (max(0, self.cfg["ready_hold_ms"] - (now - self.both_since)) if both else None),
            "min_fps": self.cfg["min_fps"],
            "cam1": dict(s1, temp=self.live["cam1"].temp), "cam2": s2,
            "fps_hist": [[h[0] - now, h[1], h[2]] for h in self.fps_hist],
            "peds": self.peds if s1["age_ms"] is not None and s1["age_ms"] < 1500 else [],
            "trail": [[x[0] - now, x[1], x[2], x[3]] for x in self.trail if now - x[0] <= 8000],
            "cam2_now": self.cam2_last if s2["age_ms"] is not None and s2["age_ms"] < 1500 else None,
            "emr_hist": [[h[0] - now, h[1], h[2], h[3]] for h in self.emr_hist if now - h[0] <= 20000],
            "has_H": self.has_H, "calib": self.calib_info, "calib_pts": self.calib_pts,
            "calib_fit": None if fit is None else {"reproj": [round(float(e), 2) for e in fit[3]],
                                                   "mean": round(float(fit[3].mean()), 2)},
            "calib_why": why, "test": cur, "results": self.results[-30:], "scenarios": SCENARIOS,
        }
