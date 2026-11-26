#!/usr/bin/env python3
# pi/edge.py  —  BlindGuard-X Raspberry Pi 5 실환경 엣지 프로그램 (v2)
#
# USB 웹캠 2대를 Pi에 직접 연결:
#   Cam1 (횡단보도)  : Optical Flow + Track Memory → Adaptive ROI + ROI Packing → YOLO26n-NCNN → ByteTrack → Homography
#   Cam2 (대기구역)  : Polygon ROI → MOG2 → Sparse LK → 경계 접근 분석 → EMR (기록·영상)
# 결과를 노트북 서버(server/exp_server.py)로 보내고, Pi에도 CSV로 남긴다.
# 서버(조작 페이지)에서 오는 명령: 사진 찍기, Homography 적용, Cam1 모드 바꾸기, Cam2 구역 적용, 배경 다시 학습, 녹화 시작/끝
#
# 실행 구조 (지연을 줄이려고)
#   - Cam1, Cam2 를 별도 프로세스로. CPU 코어 0·1·2 = Cam1, 코어 3 = Cam2
#   - 카메라마다 '최신 프레임만' 보관 → 처리가 밀리면 옛 프레임은 버림 (dropped 로 셈)
#   - 오버레이 영상 녹화는 별도 스레드에서 인코딩
#   - 메인 프로세스: 서버 통신·시계 맞추기·기록
#
# 예)
#   python3 pi/edge.py                                   # site.json 대로 (서버 주소, 카메라 장치)
#   python3 pi/edge.py --server ws://10.42.0.1:8000/edge
#   python3 pi/edge.py --offline --minutes 1             # 서버 없이 Pi 기록만
#   python3 pi/edge.py --cam1 logs/x/T003_cam1_raw.avi --cam2 none --offline   # 녹화 영상을 같은 파이프라인으로 다시 분석
#
# 필요 패키지: ultralytics, opencv-python, numpy, websockets   (pi/setup_pi.sh)

import argparse
import base64
import csv
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 시뮬 데모는 BGX_CALIB_DIR=calib_sim 으로 실행 → 공원 실물 테스트 보정 파일(calib/)을 덮어쓰지 않음
CALIB_DIR = os.environ.get("BGX_CALIB_DIR") or os.path.join(ROOT, "calib")
if not os.path.isabs(CALIB_DIR):
    CALIB_DIR = os.path.join(ROOT, CALIB_DIR)
CALIB_H = os.path.join(CALIB_DIR, "H_cam1.json")
CALIB_C2 = os.path.join(CALIB_DIR, "cam2_geom.json")


def now_ms():
    return int(time.time() * 1000)


def cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000.0, 1)
    except OSError:
        return None


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


# ───────────── 영상 입력: 카메라 또는 영상 파일, '최신 프레임만' ─────────────
def resolve_device(spec):
    """카메라 지정 → /dev/videoN.
    "name:USB 2.0 Camera" 처럼 이름(일부)으로 쓰면 꽂는 순서·재부팅으로 번호가 바뀌어도 같은 카메라를 찾음.
    /dev/v4l/by-id/... 같은 링크도 실제 장치로 바꿔 줌. 그 밖에는 그대로."""
    if not spec or not isinstance(spec, str):
        return spec
    if spec.startswith("name:"):
        want = spec[5:].strip().lower()
        base = os.environ.get("BGX_V4L_SYS", "/sys/class/video4linux")
        found = []
        try:
            for d in os.listdir(base):
                if not d.startswith("video"):
                    continue
                try:
                    name = open(os.path.join(base, d, "name")).read().strip().lower()
                    idx = int(open(os.path.join(base, d, "index")).read().strip() or 0)
                except (OSError, ValueError):
                    continue
                if want in name and idx == 0:
                    found.append(int(d[5:]))
        except OSError:
            pass
        if not found:
            raise RuntimeError("이름이 '%s' 인 카메라를 못 찾음 (v4l2-ctl --list-devices 로 이름 확인)" % spec[5:])
        return "/dev/video%d" % min(found)
    if spec.startswith("/dev/") and os.path.islink(spec):
        return os.path.realpath(spec)
    return spec


def apply_v4l2(dev, ctrls):
    """카메라 설정(v4l2-ctl)을 적용. 예: 어두울 때 FPS를 스스로 낮추는 exposure_dynamic_framerate 끄기."""
    import subprocess
    out = []
    for k, v in (ctrls or {}).items():
        if k.startswith("_"):
            continue
        try:
            r = subprocess.run(["v4l2-ctl", "-d", dev, "-c", "%s=%s" % (k, v)], capture_output=True, text=True, timeout=3)
            out.append("%s=%s%s" % (k, v, "" if r.returncode == 0 else " (실패: %s)" % r.stderr.strip()[:80]))
        except Exception as e:  # noqa: BLE001
            out.append("%s=%s (실패: %s)" % (k, v, e))
    return out


class FrameSource(threading.Thread):
    def __init__(self, src, w, h, fps, loop=False, v4l2=None):
        super().__init__(daemon=True)
        import cv2
        self.cv2 = cv2
        self.is_file = os.path.isfile(src)
        if self.is_file:
            self.cap = cv2.VideoCapture(src)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        else:
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.fps = fps
            self.v4l2_result = apply_v4l2(src, v4l2)
        if not self.cap.isOpened():
            raise RuntimeError("영상을 열 수 없음: %s (cam_check.py 로 장치 번호 확인)" % src)
        self.loop = loop
        self.cond = threading.Condition()
        self.frame, self.t, self.n = None, 0.0, 0
        self.running, self.eof = True, False

    def run(self):
        period = 1.0 / self.fps if self.is_file else 0
        nxt = time.time()
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                if self.is_file and self.loop:
                    self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.is_file:
                    self.eof = True
                    with self.cond:
                        self.cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            if period:  # 파일은 실제 속도로 재생
                nxt += period
                time.sleep(max(0.0, nxt - time.time()))
            t = time.time()
            with self.cond:
                self.frame, self.t, self.n = f, t, self.n + 1
                self.cond.notify_all()

    def get(self, last_n, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.n != last_n or self.eof, timeout=timeout)
            if self.n == last_n:
                return None, None, last_n
            return self.frame, self.t, self.n

    def stop(self):
        self.running = False
        self.cap.release()


# ───────────── 녹화: 별도 스레드에서 인코딩 ─────────────
class Recorder(threading.Thread):
    def __init__(self, path, fps):
        super().__init__(daemon=True)
        import cv2
        self.cv2 = cv2
        self.path, self.fps = path, fps
        self.q = queue.Queue(maxsize=60)
        self.writer = None
        self.ts = open(path + ".ts.csv", "w")
        self.ts.write("i,ts_cap_local\n")
        self.i = 0
        self.dropped = 0
        self.start()

    def write(self, img, t_cap):
        try:
            self.q.put_nowait((img, t_cap))
        except queue.Full:
            self.dropped += 1

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            img, t_cap = item
            if self.writer is None:
                h, w = img.shape[:2]
                self.writer = self.cv2.VideoWriter(self.path, self.cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
            self.writer.write(img)
            self.ts.write("%d,%d\n" % (self.i, int(t_cap * 1000)))
            self.i += 1
        if self.writer is not None:
            self.writer.release()
        self.ts.close()

    def close(self):
        self.q.put(None)
        self.join(timeout=10)


def put_latest(q, msg):
    try:
        q.put_nowait(msg)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass


def jpeg_b64(cv2, img, width=None, quality=75):
    if width and img.shape[1] > width:
        h = int(img.shape[0] * width / img.shape[1])
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def default_cam2_geom(w, h):
    poly = [[int(w * .15), int(h * .55)], [int(w * .85), int(h * .55)], [int(w * .95), int(h * .98)],
            [int(w * .05), int(h * .98)]]
    return {"polygon": poly, "boundary": [poly[0], poly[1]]}


# ───────────── 카메라 프로세스 (Cam1 / Cam2 공통 틀) ─────────────
def worker(kind, site, args, out_q, cmd_q, stop_ev, cores):
    """프로세스가 예외로 죽으면 그 이유를 서버·화면에 남긴다."""
    import traceback
    try:
        _worker(kind, site, args, out_q, cmd_q, stop_ev, cores)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        try:
            out_q.put({"type": "log", "level": "error", "cam": kind, "text": "프로세스 오류: %r" % e}, timeout=2)
        except Exception:  # noqa: BLE001
            pass


def _worker(kind, site, args, out_q, cmd_q, stop_ev, cores):
    try:
        os.sched_setaffinity(0, cores)
    except (AttributeError, OSError):
        pass
    os.environ["OMP_NUM_THREADS"] = str(len(cores))
    import cv2
    import numpy as np
    cv2.setNumThreads(len(cores))

    c = site[kind]
    src_arg = args.cam1 if kind == "cam1" else args.cam2
    src = src_arg or c["device"]
    try:
        spec = src
        if str(src).startswith("tcp:"):   # CARLA 가상 카메라 (simdemo/carla_demo.py 가 보냄)
            from bgx.netsource import NetFrameSource
            source = NetFrameSource(src, c["width"], c["height"], c.get("fps", 10))
        else:
            src = resolve_device(src)
            if src != spec:
                out_q.put({"type": "log", "level": "info", "cam": kind, "text": "카메라 %s → %s" % (spec, src)})
            source = FrameSource(src, c["width"], c["height"], c.get("fps", 15), loop=args.loop, v4l2=c.get("v4l2"))
        if getattr(source, "v4l2_result", None):
            out_q.put({"type": "log", "level": "info", "cam": kind, "text": "카메라 설정: " + ", ".join(source.v4l2_result)})
    except Exception as e:  # noqa: BLE001
        out_q.put({"type": "log", "level": "error", "cam": kind, "text": str(e)})
        return
    source.start()
    W, H = c["width"], c["height"]
    if source.is_file:
        W = int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if kind == "cam1":
        from bgx.cam1_pipeline import Cam1Pipeline
        from bgx.detector import make_detector
        try:
            det = make_detector(args.detector, os.path.join(ROOT, c["model"]), c.get("imgsz", 416),
                                c.get("conf", 0.1))
        except Exception as e:  # noqa: BLE001
            out_q.put({"type": "log", "level": "error", "cam": kind,
                       "text": "탐지 모델을 불러오지 못함: %s (setup_pi.sh 확인)" % e})
            return
        hj = load_json(CALIB_H)
        cfg = dict(c.get("pipeline", {}))
        cfg["mode"] = args.mode or c.get("mode", "packed")
        if args.every:
            cfg["detect_every"] = args.every
        if args.policy:
            cfg["det_policy"] = args.policy
        if c.get("fixed_roi"):
            cfg["fixed_roi"] = c["fixed_roi"]
        pipe = Cam1Pipeline(det, hj["H"] if hj else None, (W, H), cfg)
        if hj is None:
            out_q.put({"type": "log", "level": "warn", "cam": kind,
                       "text": "Homography 없음 → 조작 페이지 '보정' 탭에서 Cam1 보정 필요 (바닥 좌표 없이 추적만 함)"})
    else:
        from bgx.cam2_pipeline import Cam2Analyzer
        g = load_json(CALIB_C2) or (c if c.get("polygon") else None) or default_cam2_geom(W, H)
        pipe = Cam2Analyzer(g["polygon"], g.get("boundary_path") or g["boundary"], (W, H), c.get("pipeline", {}))

    out_q.put({"type": "worker_ready", "cam": kind, "w": W, "h": H, "src": src,
               "mode": pipe.cfg.get("mode") if kind == "cam1" else None})
    last_n = 0
    fps_win = []
    last_prev = 0.0
    rec = rec_raw = None
    rec_name = ""
    temp, temp_t = cpu_temp_c(), 0.0
    frame = None
    while not stop_ev.is_set():
        # ── 명령 처리 ──
        while True:
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                break
            name = cmd.get("cmd")
            ok, msg = True, ""
            try:
                if name == "snapshot":
                    if frame is not None:
                        out_q.put({"type": "snapshot", "cam": kind, "w": frame.shape[1], "h": frame.shape[0],
                                   "jpg": jpeg_b64(cv2, frame, None, 92), "req": cmd.get("req")}, timeout=2)
                    else:
                        ok, msg = False, "아직 프레임 없음"
                elif name == "set_H" and kind == "cam1":
                    pipe.set_H(cmd["H"])
                elif name == "set_mode" and kind == "cam1":
                    pipe.set_mode(cmd["mode"])
                elif name == "set_every" and kind == "cam1":
                    pipe.set_every(cmd["every"])
                elif name == "set_cfg" and kind == "cam1":
                    msg = "적용: %s" % pipe.set_cfg(cmd.get("cfg"))
                elif name == "set_geom" and kind == "cam2":
                    pipe.set_geometry(cmd["polygon"], cmd.get("boundary_path") or cmd["boundary"])   # 꺾은선이 오면 그것을
                    pipe.reset_bg()
                elif name == "reset_bg" and kind == "cam2":
                    pipe.reset_bg()
                elif name == "record":
                    if rec is not None:
                        rec.close()
                        rec = None
                    if rec_raw is not None:
                        rec_raw.close()
                        rec_raw = None
                    if cmd.get("on"):
                        rec_name = cmd.get("name") or time.strftime("rec_%H%M%S")
                        d = os.path.join(args.logdir, "video")
                        os.makedirs(d, exist_ok=True)
                        rfps = args.rec_fps or (10.0 if kind == "cam1" else float(c.get("fps", 15)))
                        rec = Recorder(os.path.join(d, "%s_%s_overlay.avi" % (rec_name, kind)), rfps)
                        if cmd.get("raw", args.raw):
                            rec_raw = Recorder(os.path.join(d, "%s_%s_raw.avi" % (rec_name, kind)), rfps)
                    else:
                        rec_name = ""
                else:
                    continue  # 이 카메라와 상관없는 명령
            except Exception as e:  # noqa: BLE001
                ok, msg = False, str(e)
            out_q.put({"type": "ack", "cam": kind, "cmd": name, "ok": ok, "msg": msg, "req": cmd.get("req")})

        f, t_cap, n = source.get(last_n)
        if f is None:
            if source.eof:
                out_q.put({"type": "log", "level": "info", "cam": kind, "text": "영상 파일 끝"})
                break
            continue
        frame = f
        dropped = max(n - last_n - 1, 0) if last_n else 0
        last_n = n
        r = pipe.process(frame, t_cap) if kind == "cam1" else pipe.process(frame)
        fps_win.append(time.time())
        fps_win = [x for x in fps_win if fps_win[-1] - x <= 2.0]
        fps = (len(fps_win) - 1) / (fps_win[-1] - fps_win[0]) if len(fps_win) > 1 else 0.0
        if time.time() - temp_t > 1.0:
            temp, temp_t = cpu_temp_c(), time.time()
        base = {"cam": kind, "frame": n, "ts_cap_local": int(t_cap * 1000), "ts_done_local": now_ms(),
                "fps": round(fps, 2), "dropped": dropped, "temp_c": temp, "rec": rec is not None}
        if kind == "cam1":
            msg = dict(base, type="scene", mode=r["mode"], det_ran=r["det_ran"], full_scan=r["full_scan"],
                       fallback=r["fallback"], n_rois=r["n_rois"], roi_area=r["roi_area"],
                       pack_scale=r["pack_scale"], canvas_fill=r["canvas_fill"], n_det=r["n_det"],
                       timing=r["timing"], peds=r["peds"], every=pipe.cfg["detect_every"],
                       has_H=pipe.H is not None, policy=r.get("policy"), lk_frame=r.get("lk_frame"),
                       yolo_reason=r.get("yolo_reason"))
        else:
            rr = {k: v for k, v in r.items() if not k.startswith("_")}
            msg = dict(base, type="cam2", **rr)
        put_latest(out_q, msg)

        need_prev = args.preview_hz > 0 and time.time() - last_prev >= 1.0 / args.preview_hz
        if rec is not None or need_prev:
            extra = "%.1f fps %s" % (fps, ("REC " + rec_name) if rec is not None else "")
            vis = pipe.draw(frame, r, extra)
            if rec is not None:
                rec.write(vis, t_cap)
            if rec_raw is not None:
                rec_raw.write(frame, t_cap)
            if need_prev:
                last_prev = time.time()
                put_latest(out_q, {"type": "preview", "cam": kind, "jpg": jpeg_b64(cv2, vis, 640, 70)})
    for rr in (rec, rec_raw):
        if rr is not None:
            rr.close()
    source.stop()


# ───────────── 서버 연결 + 시계 맞추기 ─────────────
class Uplink:
    """서버로 보내는 WebSocket. 끊기면 2초 뒤 다시 연결.
    서버와 Pi 시계 차이(offset)를 왕복 시간의 절반 방식으로 계속 잰다 → 모든 시각을 '서버 시계'로 바꿔 보냄."""

    def __init__(self, url, hello, on_cmd):
        self.url, self.hello, self.on_cmd = url, hello, on_cmd
        self.ws = None
        self.offset_ms = 0
        self.rtt_ms = None
        self.samples = []
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _send_raw(self, ws, obj):
        with self.lock:
            ws.send(json.dumps(obj, ensure_ascii=False))

    def _loop(self):
        from websockets.sync.client import connect
        while True:
            try:
                ws = connect(self.url, open_timeout=3, max_size=None)
                self._send_raw(ws, dict(self.hello(), type="hello_edge"))
                self.ws = ws
                print("[서버] 연결됨:", self.url)
                last_sync = 0
                while True:
                    if time.time() - last_sync > 2.0:
                        self._send_raw(ws, {"type": "sync", "t0": now_ms()})
                        last_sync = time.time()
                    try:
                        raw = ws.recv(timeout=0.5)
                    except TimeoutError:
                        continue
                    m = json.loads(raw)
                    if m.get("type") == "sync":
                        t3 = now_ms()
                        rtt = t3 - m["t0"]
                        off = m["t1"] - (m["t0"] + t3) / 2.0
                        self.samples = (self.samples + [(rtt, off)])[-15:]
                        best = min(self.samples)
                        self.rtt_ms, self.offset_ms = best[0], int(best[1])
                    elif m.get("type") == "cmd":
                        self.on_cmd(m)
            except Exception as e:  # noqa: BLE001
                if self.ws is not None:
                    print("[서버] 끊김:", e)
                self.ws = None
                time.sleep(2)

    def send(self, msg):
        ws = self.ws
        if ws is None:
            return False
        try:
            self._send_raw(ws, msg)
            return True
        except Exception:  # noqa: BLE001
            return False


class Logs:
    C1 = ["frame", "ts_cap_local", "ts_done_local", "ts_send_local", "offset_ms", "mode", "every", "det_ran",
          "full_scan", "fallback", "n_rois", "roi_area", "pack_scale", "canvas_fill", "n_det", "n_peds", "t_flow",
          "t_roi", "t_pack", "t_infer", "t_track", "t_homo", "t_total", "fps", "dropped", "temp_c", "rec", "sent",
          "policy", "lk_frame", "yolo_reason", "t_lk"]
    TR = ["frame", "ts_cap_local", "track_id", "x_m", "y_m", "vx", "vy", "v_ok", "conf", "u", "v", "x1", "y1", "x2",
          "y2"]
    C2 = ["frame", "ts_cap_local", "ts_done_local", "offset_ms", "warmup", "present", "area", "signed_px",
          "toward_px", "n_pts", "persist", "prox", "tow", "per", "emr", "emr_state", "crossed", "events", "t_mog",
          "t_lk", "t_total", "fps", "dropped", "rec"]

    def __init__(self, d):
        self.files = []

        def mk(name, cols):
            f = open(os.path.join(d, name), "w", newline="", encoding="utf-8", buffering=1)
            self.files.append(f)
            w = csv.writer(f)
            w.writerow(cols)
            return w
        self.c1 = mk("cam1_frames.csv", self.C1)
        self.tr = mk("cam1_tracks.csv", self.TR)
        self.c2 = mk("cam2_frames.csv", self.C2)
        self.ev = mk("events.csv", ["ts_local", "kind", "detail"])

    def scene(self, m, sent):
        t = m["timing"]
        self.c1.writerow([m["frame"], m["ts_cap_local"], m["ts_done_local"], m["ts_send"] - m["offset_ms"],
                          m["offset_ms"], m["mode"], m["every"], m["det_ran"], m["full_scan"], m["fallback"],
                          m["n_rois"], m["roi_area"], m["pack_scale"], m["canvas_fill"], m["n_det"], len(m["peds"]),
                          t["flow"], t["roi"], t["pack"], t["infer"], t["track"], t["homo"], t["total"], m["fps"],
                          m["dropped"], m["temp_c"], m["rec"], sent, m.get("policy"), m.get("lk_frame"),
                          m.get("yolo_reason"), t.get("lk")])
        for p in m["peds"]:
            b = p["bbox"]
            self.tr.writerow([m["frame"], m["ts_cap_local"], p["track_id"], p["x_m"], p["y_m"], p["vx"], p["vy"],
                              p["v_ok"], p["conf"], p["u"], p["v"], b[0], b[1], b[2], b[3]])

    def cam2(self, m):
        t = m["timing"]
        self.c2.writerow([m["frame"], m["ts_cap_local"], m["ts_done_local"], m["offset_ms"], m["warmup"],
                          m["present"], m["area"], m["signed_px"], m["toward_px"], m["n_pts"], m["persist"],
                          m["prox"], m["tow"], m["per"], m["emr"], m["emr_state"], m["crossed"],
                          "|".join(m["events"]), t["mog"], t["lk"], t["total"], m["fps"], m["dropped"], m["rec"]])
        for e in m["events"]:
            self.event("cam2_" + e, "frame=%d" % m["frame"])

    def event(self, kind, detail=""):
        self.ev.writerow([now_ms(), kind, detail])

    def close(self):
        for f in self.files:
            f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.path.join(ROOT, "site.json"))
    ap.add_argument("--server", help="ws://노트북IP:8000/edge (site.json 값을 덮어씀)")
    ap.add_argument("--offline", action="store_true", help="서버로 안 보내고 Pi에만 기록")
    ap.add_argument("--cam1", help="Cam1 입력 (장치 /dev/videoN 또는 영상 파일)")
    ap.add_argument("--cam2", help="Cam2 입력 ('none' 이면 끔)")
    ap.add_argument("--no-cam2", action="store_true")
    ap.add_argument("--detector", default="ultralytics", choices=["ultralytics", "color"],
                    help="color = 시험용 (합성 영상의 색 막대를 사람으로 봄)")
    ap.add_argument("--mode", choices=["full", "fixed", "adaptive", "packed"])
    ap.add_argument("--every", type=int)
    ap.add_argument("--policy", choices=["event", "every"], help="YOLO 실행: event=필요할 때만(LK 로 이어감), every=매 프레임")
    ap.add_argument("--preview-hz", type=float, default=5.0, help="관제·조작 페이지 카메라 화면 갱신 (초당, 0 = 끔)")
    ap.add_argument("--record-all", action="store_true", help="시험과 상관없이 처음부터 오버레이 녹화")
    ap.add_argument("--raw", action="store_true", help="녹화할 때 원본 영상도 같이 (재분석용, 저장 공간 2배)")
    ap.add_argument("--rec-fps", type=float, default=0, help="녹화 영상 재생 FPS (0 = Cam1 10, Cam2 카메라 FPS)")
    ap.add_argument("--loop", action="store_true", help="영상 파일 입력을 반복")
    ap.add_argument("--dump-msgs", help="보낸 메시지를 jsonl 로 저장 (시험·재생용)")
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.cam2 == "none":
        args.no_cam2 = True

    site = load_json(args.site)
    stamp = time.strftime("%m%d_%H%M%S")
    args.logdir = os.path.join(ROOT, "logs", "pi_%s%s" % (stamp, "_" + args.tag if args.tag else ""))
    os.makedirs(args.logdir, exist_ok=True)
    save_json(os.path.join(args.logdir, "args.json"), vars(args))
    logs = Logs(args.logdir)
    dump = open(args.dump_msgs, "w", encoding="utf-8") if args.dump_msgs else None

    out_q = mp.Queue(maxsize=300)
    stop_ev = mp.Event()
    cmd_qs = {"cam1": mp.Queue(), "cam2": mp.Queue()}
    procs = {"cam1": mp.Process(target=worker, args=("cam1", site, args, out_q, cmd_qs["cam1"], stop_ev, {0, 1, 2}),
                                daemon=True)}
    if not args.no_cam2:
        procs["cam2"] = mp.Process(target=worker, args=("cam2", site, args, out_q, cmd_qs["cam2"], stop_ev, {3}),
                                   daemon=True)
    for p in procs.values():
        p.start()

    state = {"ready": {}, "mode": None}

    def hello():
        return {"edge_id": site.get("edge_id", "cw-01"), "cams": state["ready"], "has_H": os.path.exists(CALIB_H),
                "cam2_geom": load_json(CALIB_C2), "detector": args.detector, "logdir": args.logdir,
                "model": site["cam1"].get("model")}

    def on_cmd(m):
        """서버 명령 → 저장할 것은 저장하고 해당 카메라 프로세스로 전달."""
        name = m.get("cmd")
        logs.event("cmd_" + str(name), json.dumps({k: v for k, v in m.items() if k not in ("H",)},
                                                  ensure_ascii=False)[:300])
        if name == "set_H" and m.get("H") is None:      # 보정 지우기
            if os.path.exists(CALIB_H):
                os.replace(CALIB_H, CALIB_H + ".removed")
        elif name == "set_H":
            save_json(CALIB_H, {"H": m["H"], "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "from": "server",
                                "detail": m.get("detail")})
        if name == "set_geom":
            save_json(CALIB_C2, {"polygon": m["polygon"], "boundary": m["boundary"],
                                 "boundary_path": m.get("boundary_path"),
                                 "saved": time.strftime("%Y-%m-%d %H:%M:%S")})
        targets = [m["cam"]] if m.get("cam") in cmd_qs else list(cmd_qs)
        for k in targets:
            if k in procs:
                cmd_qs[k].put(m)

    up = None if args.offline else Uplink(args.server or site["server_url"], hello, on_cmd)
    if args.record_all:
        on_cmd({"cmd": "record", "on": True, "name": "all"})

    print("기록 폴더:", args.logdir)
    t_end = time.time() + args.minutes * 60 if args.minutes else None
    last_print = 0
    try:
        while t_end is None or time.time() < t_end:
            try:
                m = out_q.get(timeout=0.5)
            except queue.Empty:
                if all(not p.is_alive() for p in procs.values()):
                    print("카메라 프로세스가 모두 끝남")
                    break
                continue
            typ = m["type"]
            off = up.offset_ms if up else 0
            m["edge_id"] = site.get("edge_id", "cw-01")
            if "ts_cap_local" in m:
                m["offset_ms"] = off
                m["ts_cap"] = m["ts_cap_local"] + off
                m["ts_done"] = m["ts_done_local"] + off
                m["ts_send"] = now_ms() + off
            if typ == "worker_ready":
                state["ready"][m["cam"]] = {"w": m["w"], "h": m["h"], "src": m["src"], "mode": m.get("mode")}
                logs.event("ready_" + m["cam"], json.dumps(state["ready"][m["cam"]], ensure_ascii=False))
                print("[%s] 준비됨 %dx%d (%s)" % (m["cam"], m["w"], m["h"], m["src"]))
            if typ in ("log",):
                print("[%s] %s: %s" % (m.get("cam"), m.get("level"), m.get("text")))
                logs.event("log_" + str(m.get("level")), "%s %s" % (m.get("cam"), m.get("text")))
            if typ == "ack":
                logs.event("ack", json.dumps(m, ensure_ascii=False))
            sent = up.send(m) if up else False
            if dump is not None and typ not in ("preview", "snapshot"):
                dump.write(json.dumps(m, ensure_ascii=False) + "\n")
            if typ == "scene":
                logs.scene(m, sent)
                if time.time() - last_print > 2:
                    last_print = time.time()
                    tm = m["timing"]
                    print("Cam1 %-8s %-5s %4.1f FPS  %s  전체 %s ms  사람 %d  %s°C  서버 %s  시계차 %s ms" %
                          (m["mode"], m.get("policy") or "", m["fps"],
                           ("LK %s ms" % tm.get("lk")) if m.get("lk_frame") else ("YOLO %s ms" % tm["infer"]),
                           tm["total"], len(m["peds"]), m["temp_c"], "연결" if (up and up.ws) else "-", off))
            elif typ == "cam2":
                logs.cam2(m)
    except KeyboardInterrupt:
        pass
    finally:
        stop_ev.set()
        for p in procs.values():
            p.join(timeout=5)
        logs.close()
        if dump is not None:
            dump.close()
        print("종료. 기록:", args.logdir)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
