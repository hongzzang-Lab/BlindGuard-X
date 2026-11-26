# bgx/cam1_pipeline.py
#
# Cam1 · 횡단보도 분석 (포스터 최종본 구조 그대로)
#
#   프레임 ─▶ ① Optical Flow (움직임 영역) ─┐
#            ② Track Memory (기존 트랙 예측) ─┴▶ ③ Adaptive ROI (합치기·여유) ─▶ ④ ROI Packing (416×416 한 장)
#          ─▶ ⑤ YOLO26n-NCNN (1회 추론) ─▶ 원본 좌표로 되돌리기 ─▶ ⑥ ByteTrack ─▶ ⑦ Homography (바닥 위치·속도·방향)
#
# 비교용 모드 (실행 중 바꿀 수 있음)
#   full     : 프레임 전체를 YOLO에 (416으로 줄여서)              ← 기준선
#   fixed    : 미리 정한 고정 ROI만 잘라서 YOLO에                ← 포스터 표의 FIXED
#   adaptive : ①②③으로 만든 ROI들을 하나로 감싼 영역을 잘라서 YOLO에 ← ADAPTIVE
#   crops    : ①②③ ROI를 하나씩 따로 잘라 YOLO 를 ROI 수만큼      ← 포스터 표의 OURS (Packing 전 단계, 비교 실험용)
#   packed   : ①②③ ROI들을 ④로 한 캔버스에 모아 YOLO 1회        ← OURS_PACKED (포스터 최종 구조)
#
# 단계별 시간(ms)을 모두 재서 돌려준다 → "포스터의 각 단계가 실제로 돌고, 얼마나 걸리는지"의 근거.

import math
import time
from collections import deque

import cv2
import numpy as np

from .bytetrack import BYTETracker, iou_matrix
from .geometry import VelocityEstimator, foot_point, orient_H

MODES = ("full", "fixed", "adaptive", "crops", "packed")
ROI_MODES = ("adaptive", "crops", "packed")     # ①②③ 으로 ROI 를 만드는 방식들

DEFAULT_CFG = {
    "mode": "packed",
    "person_height_m": 1.7,    # 바닥 보정(H)이 없을 때 속도 추정에 쓰는 사람 키. 박스 높이(px)가 이 키라고 보고 px → m 로 바꿈
    "demo_speed_min": 0.0,     # 둘 다 0 이면 끔. 켜면 실제 속도 대신 이 범위의 '예시값'을 "(demo)" 표시와 함께 보여 줌
    "demo_speed_max": 0.0,     #   (화면 구성을 먼저 잡아 두는 용도. 측정값이 아니므로 결과 기록에는 넣지 않음)
    "draw_curb": False,        # 분석 화면에 연석선(y=0, 빨간 선) 그리기. 실물 테스트는 끔, CARLA 시뮬 데모(site_sim.json)만 켬
    "canvas": 416,             # YOLO 입력 크기 (NCNN 내보낼 때와 같아야 함)
    "fixed_roi": None,         # [x, y, w, h], None 이면 프레임 전체
    "detect_every": 1,
    # 조건부 추론 (adaptive/packed 에서만): YOLO 를 매 프레임 돌리지 않고, 사이 프레임은 Sparse LK 로 트랙을 따라감
    "det_policy": "event",     # "every" = 매 프레임 YOLO (포스터 실측 방식) / "event" = 필요할 때만 YOLO
    "max_skip": 3,             # event: 최소 N 프레임에 한 번은 YOLO (트랙 확인·크기 보정)
    "lk_min_pts": 5,           # 트랙 상자 안에서 따라간 점이 이보다 적으면 'LK 실패' → 다음 프레임 YOLO
    "new_motion_cover": 0.5,   # 움직임 상자가 기존 트랙에 이 비율 미만 덮이면 '새 움직임' → YOLO
    "lk_scale": 1.0,           # LK 계산 해상도 배율 (0.5 는 먼 사람에서 점이 부족했음)
    # ① Optical Flow
    "flow_width": 160,         # 이 폭으로 줄여서 Farneback (Pi에서 수 ms)
    "flow_thr": 0.8,           # 줄인 영상 기준 프레임당 이동량(px) 이상이면 움직임 (실내 11/21: 0.6은 잡음까지 잡음)
    "flow_min_area": 20,       # 줄인 영상 기준 최소 면적(px)
    "flow_persist": 2,         # 같은 자리에서 연속 N 프레임 움직여야 인정 (조명 깜빡임·센서 잡음 제거)
    "flow_blur": 3,            # 흐름 계산 전 흐림(홀수, 0=끔) — 센서 잡음 억제
    # ② Track Memory
    "memory_s": 1.0,           # 트랙이 사라진 뒤에도 이 시간 동안 예측 위치를 ROI로 유지
    # ③ Adaptive ROI
    "roi_margin": 0.3,         # 박스 크기 대비 여유
    "roi_min_wh": [56, 96],    # 너무 작은 ROI 방지 (원본 px)
    "full_scan_every": 15,     # N 프레임마다 전체 화면 한 번 (가만히 서 있는 새 사람 찾기)
    "adaptive_max_area": 0.7,  # 합친 ROI가 프레임의 이 비율을 넘으면 그냥 전체
    # ④ ROI Packing
    "pack_gap": 4,
    "pack_scale_min": 0.45,
    "pack_scale_max": 1.5,     # 작은(먼) 사람은 확대해서 넣음 → 원거리 검출 개선이 Packing 의 목적
    # 후처리
    "nms_iou": 0.6,
}


def _clip_box(b, W, H):
    x1, y1, x2, y2 = b
    return [max(0, min(W, x1)), max(0, min(H, y1)), max(0, min(W, x2)), max(0, min(H, y2))]


def _expand(b, margin, min_wh, W, H):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    w2 = max(w * (1 + 2 * margin), min_wh[0])
    h2 = max(h * (1 + 2 * margin), min_wh[1])
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return _clip_box([cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2], W, H)


def merge_boxes(boxes):
    """겹치거나 맞닿은 박스를 합쳐서 더 이상 겹치지 않을 때까지."""
    boxes = [list(map(float, b)) for b in boxes]
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                b = boxes[i]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return boxes


def shelf_pack(sizes, C, gap):
    """sizes: [(w, h)] → [(x, y)] 또는 None (안 들어가면). 높이 큰 순서로 선반 쌓기."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    pos = [None] * len(sizes)
    x = y = shelf_h = 0
    for i in order:
        w, h = sizes[i]
        if w > C or h > C:
            return None
        if x + w > C:
            x, y = 0, y + shelf_h + gap
            shelf_h = 0
        if y + h > C:
            return None
        pos[i] = (x, y)
        x += w + gap
        shelf_h = max(shelf_h, h)
    return pos


def plan_packing(rois, C, gap, s_min, s_max):
    """모든 ROI에 같은 배율 s를 쓰고, 들어가는 가장 큰 s를 찾는다 (이분 탐색)."""
    if not rois:
        return None
    sizes0 = [(b[2] - b[0], b[3] - b[1]) for b in rois]

    def fits(s):
        return shelf_pack([(max(1, int(w * s)), max(1, int(h * s))) for w, h in sizes0], C, gap)

    if fits(s_min) is None:
        return None
    lo, hi = s_min, s_max
    if fits(hi) is not None:
        lo = hi
    else:
        for _ in range(14):
            mid = (lo + hi) / 2
            if fits(mid) is not None:
                lo = mid
            else:
                hi = mid
    s = lo
    sizes = [(max(1, int(w * s)), max(1, int(h * s))) for w, h in sizes0]
    pos = shelf_pack(sizes, C, gap)
    tiles = [{"roi": [int(r[0]), int(r[1]), int(r[2]), int(r[3])], "x": p[0], "y": p[1], "w": sz[0], "h": sz[1]}
             for r, p, sz in zip(rois, pos, sizes)]
    return {"scale": s, "tiles": tiles}


def build_canvas(frame, plan, C):
    canvas = np.full((C, C, 3), 114, np.uint8)
    for t in plan["tiles"]:
        x1, y1, x2, y2 = t["roi"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        canvas[t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"]] = cv2.resize(
            crop, (t["w"], t["h"]), interpolation=cv2.INTER_LINEAR)
    return canvas


def unpack_dets(dets, plan):
    """캔버스 좌표 탐지 → 원본 프레임 좌표. 중심이 어느 타일에도 없으면 버림."""
    out = []
    for x1, y1, x2, y2, sc in dets:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for t in plan["tiles"]:
            if t["x"] <= cx < t["x"] + t["w"] and t["y"] <= cy < t["y"] + t["h"]:
                rx1, ry1, rx2, ry2 = t["roi"]
                sx = (rx2 - rx1) / t["w"]
                sy = (ry2 - ry1) / t["h"]
                bx1 = rx1 + (max(x1, t["x"]) - t["x"]) * sx
                by1 = ry1 + (max(y1, t["y"]) - t["y"]) * sy
                bx2 = rx1 + (min(x2, t["x"] + t["w"]) - t["x"]) * sx
                by2 = ry1 + (min(y2, t["y"] + t["h"]) - t["y"]) * sy
                out.append([bx1, by1, bx2, by2, sc])
                break
    return np.array(out, float) if out else np.zeros((0, 5))


def nms(dets, thr):
    if len(dets) < 2:
        return dets
    order = np.argsort(-dets[:, 4])
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(dets[i:i + 1, :4], dets[order[1:], :4])[0]
        order = order[1:][ious < thr]
    return dets[keep]


class Cam1Pipeline:
    def __init__(self, detector, H, frame_size, cfg=None):
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.det = detector
        self.W, self.Hh = frame_size
        self.set_H(H)
        self.tracker = BYTETracker()
        self.vel = VelocityEstimator()
        self.vel_px = VelocityEstimator()   # 화면(px) 속도: 바닥 보정이 없을 때 키 기준 속도 추정에 씀
        self.box_h = {}
        self.demo_spd = {}
        self.demo_rng = np.random.default_rng()
        self.prev_small = None
        self.k = 0
        self.last_tracks = []      # [(tid, box, t)]
        self.track_motion = {}     # tid -> deque[(t, cx, cy, w, h)]  픽셀 이동 (Track Memory 예측용)
        self.last_seen = {}        # tid -> (t, box)
        self.last_peds = []
        self.last_t = None
        self.cur_tracks = []       # 지금 트랙 상자 (YOLO 또는 LK 로 갱신)
        self.since_det = 0         # 마지막 YOLO 이후 LK 로만 넘긴 프레임 수
        self.lk_failed = False
        self.prev_lk = None        # LK 용 이전 회색 영상

    # ── 실행 중 바꾸기 ──
    def set_H(self, H):
        self.H = None if H is None else orient_H(H, (self.W / 2, self.Hh * 0.9))
        self.Hinv = None if H is None else np.linalg.inv(self.H)

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        if mode != self.cfg["mode"]:
            self.cfg["mode"] = mode
            self.tracker.reset()   # 모드끼리 공정하게 비교하려고 추적 상태 초기화
            self.vel = VelocityEstimator()
            self.last_tracks, self.track_motion, self.last_seen = [], {}, {}
            self.prev_small = None
            self._flow_hist = []
            self.cur_tracks, self.since_det, self.lk_failed = [], 0, False
            self.k = 0          # 바꾼 직후 첫 프레임은 전체 화면 스캔 → 바로 다시 찾음

    def set_every(self, n):
        self.cfg["detect_every"] = max(1, int(n))

    TUNABLE = ("flow_thr", "flow_min_area", "flow_persist", "flow_blur", "full_scan_every", "memory_s", "roi_margin",
               "pack_scale_max", "det_policy", "max_skip", "lk_min_pts", "new_motion_cover", "demo_speed_min",
               "demo_speed_max", "person_height_m")

    def set_cfg(self, d):
        """현장에서 조절 (조작 페이지 ②탭). 바꾼 값만 돌려줌."""
        out = {}
        for k, v in (d or {}).items():
            if k in self.TUNABLE:
                self.cfg[k] = type(DEFAULT_CFG[k])(v)
                out[k] = self.cfg[k]
        return out

    # ── ① Optical Flow ──
    def _motion_boxes(self, frame):
        fw = self.cfg["flow_width"]
        s = fw / self.W
        small = cv2.cvtColor(cv2.resize(frame, (fw, int(self.Hh * s)), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY)
        kb = int(self.cfg.get("flow_blur", 0))
        if kb >= 3:
            small = cv2.GaussianBlur(small, (kb | 1, kb | 1), 0)
        boxes = []
        if self.prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_small, small, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            raw = (mag > self.cfg["flow_thr"]).astype(np.uint8)
            raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # 점 잡음 제거
            # 연속성: 직전 N-1 프레임에도 근처가 움직였던 곳만 남김
            hist = getattr(self, "_flow_hist", [])
            n_need = max(1, int(self.cfg.get("flow_persist", 1)))
            m = raw.copy()
            if n_need > 1:
                for h in hist[-(n_need - 1):]:
                    m &= cv2.dilate(h, np.ones((5, 5), np.uint8))
            self._flow_hist = (hist + [raw])[-4:]
            m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)
            n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, n):
                x, y, w, h, a = st[i]
                if a >= self.cfg["flow_min_area"]:
                    boxes.append([x / s, y / s, (x + w) / s, (y + h) / s])
        self.prev_small = small
        return boxes

    # ── ② Track Memory ──
    def _memory_boxes(self, t):
        out = []
        mem = self.cfg["memory_s"]
        for tid, (ts, box) in list(self.last_seen.items()):
            age = t - ts
            if age > mem:
                del self.last_seen[tid]
                self.track_motion.pop(tid, None)
                continue
            q = self.track_motion.get(tid)
            vx = vy = 0.0
            if q and len(q) >= 2 and q[-1][0] > q[0][0]:
                dt = q[-1][0] - q[0][0]
                vx, vy = (q[-1][1] - q[0][1]) / dt, (q[-1][2] - q[0][2]) / dt
            dx, dy = vx * age, vy * age
            out.append([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy])
        return out

    def _remember(self, tracks, t):
        for x1, y1, x2, y2, tid, _ in tracks:
            self.last_seen[tid] = (t, [x1, y1, x2, y2])
            q = self.track_motion.setdefault(tid, deque(maxlen=6))
            q.append((t, (x1 + x2) / 2, (y1 + y2) / 2))

    # ── 조건부 추론 ──
    def _need_yolo(self, motion):
        """YOLO 가 필요한 이유 (없으면 None → 이번 프레임은 LK 로 따라감)."""
        c = self.cfg
        if self.since_det + 1 >= c["max_skip"]:
            return "주기"
        if self.lk_failed:
            return "LK실패"
        if not self.cur_tracks:
            return "트랙없음"
        W, H = self.W, self.Hh
        tb = [_expand(tr[:4], 0.3, (0, 0), W, H) for tr in self.cur_tracks]
        for mb in motion:
            area = max((mb[2] - mb[0]) * (mb[3] - mb[1]), 1.0)
            cov = 0.0
            for b in tb:
                iw = min(mb[2], b[2]) - max(mb[0], b[0])
                ih = min(mb[3], b[3]) - max(mb[1], b[1])
                if iw > 0 and ih > 0:
                    cov = max(cov, iw * ih / area)
            if cov < c["new_motion_cover"]:
                return "새움직임"
        return None

    def _lk_propagate(self, gray, t):
        """기존 트랙 상자 안의 Shi-Tomasi 점을 Sparse LK 로 따라가서 상자를 옮김."""
        c = self.cfg
        ls = c["lk_scale"]
        out, failed = [], False
        gh, gw = gray.shape[:2]
        for x1, y1, x2, y2, tid, sc in self.cur_tracks:
            bw, bh = x2 - x1, y2 - y1
            # 점을 고를 안쪽 상자 (가장자리 배경 제외)
            ix1, iy1 = (x1 + 0.15 * bw) * ls, (y1 + 0.1 * bh) * ls
            ix2, iy2 = (x2 - 0.15 * bw) * ls, (y2 - 0.1 * bh) * ls
            # 계산은 상자 주변만 잘라서 (전체 영상에 돌리면 Pi 에서 느림)
            pad = int(max(24, 0.5 * max(bw, bh) * ls))
            cx1, cy1 = max(0, int(ix1) - pad), max(0, int(iy1) - pad)
            cx2, cy2 = min(gw, int(ix2) + pad), min(gh, int(iy2) + pad)
            dx = dy = None
            if ix2 - ix1 >= 3 and iy2 - iy1 >= 3 and cx2 - cx1 > 8 and cy2 - cy1 > 8:
                a = self.prev_lk[cy1:cy2, cx1:cx2]
                b = gray[cy1:cy2, cx1:cx2]
                mask = np.zeros_like(a)
                mask[max(0, int(iy1) - cy1):max(0, int(iy2) - cy1), max(0, int(ix1) - cx1):max(0, int(ix2) - cx1)] = 255
                p0 = cv2.goodFeaturesToTrack(a, maxCorners=30, qualityLevel=0.01, minDistance=3, mask=mask)
                if p0 is not None and len(p0) >= c["lk_min_pts"]:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(15, 15), maxLevel=2)
                    ok = st.reshape(-1) == 1
                    if ok.sum() >= c["lk_min_pts"]:
                        d = (p1.reshape(-1, 2)[ok] - p0.reshape(-1, 2)[ok]) / ls
                        dx, dy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
            if dx is None:   # 점을 못 따라감 → 직전 픽셀 속도로 앞당기고 다음 프레임 YOLO
                failed = True
                q = self.track_motion.get(tid)
                dx = dy = 0.0
                if q and len(q) >= 2 and q[-1][0] > q[0][0]:
                    dtq = q[-1][0] - q[0][0]
                    step = t - q[-1][0]
                    dx = (q[-1][1] - q[0][1]) / dtq * step
                    dy = (q[-1][2] - q[0][2]) / dtq * step
            out.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy, tid, sc))
        self.lk_failed = failed
        return out

    # ── 본체 ──
    def process(self, frame, t):
        c = self.cfg
        mode = c["mode"]
        W, H = self.W, self.Hh
        tm = {"flow": None, "roi": None, "pack": None, "infer": None, "track": None, "homo": None}
        t_all = time.perf_counter()
        info = {"mode": mode, "full_scan": False, "fallback": None, "rois": [], "motion": [], "memory": [],
                "n_rois": 0, "roi_area": None, "pack_scale": None, "canvas_fill": None, "_canvas": None,
                "det_ran": False, "n_det": None, "policy": c.get("det_policy", "every"), "lk_frame": False,
                "yolo_reason": None}
        tm["lk"] = None
        run_det = (self.k % c["detect_every"] == 0)
        self.k += 1
        event = mode in ROI_MODES and c.get("det_policy") == "event"
        lk_gray = None
        if event:
            ls = c["lk_scale"]
            lk_gray = cv2.cvtColor(cv2.resize(frame, (int(self.W * ls), int(self.Hh * ls)),
                                              interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        lk_tracks = None

        dets = None
        if mode in ROI_MODES:
            t0 = time.perf_counter()
            motion = self._motion_boxes(frame)
            tm["flow"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            memory = self._memory_boxes(t)
            cands = [_expand(b, c["roi_margin"], c["roi_min_wh"], W, H) for b in motion + memory]
            cands = [b for b in cands if b[2] - b[0] > 4 and b[3] - b[1] > 4]
            rois = merge_boxes(cands)
            tm["roi"] = (time.perf_counter() - t0) * 1000
            info.update(motion=motion, memory=memory, rois=rois, n_rois=len(rois),
                        roi_area=round(sum((b[2] - b[0]) * (b[3] - b[1]) for b in rois) / (W * H), 4))
            full_scan = (self.k - 1) % c["full_scan_every"] == 0
            info["full_scan"] = full_scan
            if event and run_det and not full_scan and self.prev_lk is not None:
                reason = self._need_yolo(motion)
                info["yolo_reason"] = reason
                if reason is None:
                    t0 = time.perf_counter()
                    lk_tracks = self._lk_propagate(lk_gray, t)
                    tm["lk"] = (time.perf_counter() - t0) * 1000
                    info["lk_frame"] = True
                    run_det = False
            elif event and full_scan:
                info["yolo_reason"] = "전체스캔"
            if run_det:
                if full_scan:
                    dets, tm["infer"] = self.det.detect(frame)
                elif not rois:
                    dets = np.zeros((0, 5))  # 움직임도 기억할 트랙도 없음 → 추론 생략 (Adaptive 의 절약)
                    tm["infer"] = 0.0
                else:
                    ux1 = min(b[0] for b in rois)
                    uy1 = min(b[1] for b in rois)
                    ux2 = max(b[2] for b in rois)
                    uy2 = max(b[3] for b in rois)
                    union_area = (ux2 - ux1) * (uy2 - uy1) / (W * H)
                    plan = None
                    if mode == "crops":
                        # ROI 마다 따로 추론 (사람을 크게 보지만 YOLO 를 ROI 수만큼 돌림 → Packing 이 줄이려는 비용)
                        parts, ms = [], 0.0
                        for b in rois:
                            x1, y1, x2, y2 = (int(b[0]), int(b[1]), int(math.ceil(b[2])), int(math.ceil(b[3])))
                            raw, m1 = self.det.detect(frame[y1:y2, x1:x2])
                            ms += m1
                            if len(raw):
                                raw = raw.copy()
                                raw[:, [0, 2]] += x1
                                raw[:, [1, 3]] += y1
                                parts.append(raw)
                        dets = np.vstack(parts) if parts else np.zeros((0, 5))
                        tm["infer"] = ms
                        info["n_infer"] = len(rois)
                    if mode == "packed":
                        t0 = time.perf_counter()
                        plan = plan_packing(rois, c["canvas"], c["pack_gap"], c["pack_scale_min"],
                                            c["pack_scale_max"])
                        if plan is not None:
                            canvas = build_canvas(frame, plan, c["canvas"])
                            info["_canvas"] = canvas
                            info["pack_scale"] = round(plan["scale"], 3)
                            info["canvas_fill"] = round(sum(tl["w"] * tl["h"] for tl in plan["tiles"])
                                                        / c["canvas"] ** 2, 3)
                        else:
                            info["fallback"] = "pack_overflow"
                        tm["pack"] = (time.perf_counter() - t0) * 1000
                        if plan is not None:
                            raw, tm["infer"] = self.det.detect(canvas)
                            dets = unpack_dets(raw, plan)
                    if dets is None:  # adaptive, 또는 packing 이 안 들어갈 때
                        if union_area > c["adaptive_max_area"]:
                            info["fallback"] = (info["fallback"] or "") + "+union_large"
                            dets, tm["infer"] = self.det.detect(frame)
                        else:
                            x1, y1, x2, y2 = (int(ux1), int(uy1), int(math.ceil(ux2)), int(math.ceil(uy2)))
                            raw, tm["infer"] = self.det.detect(frame[y1:y2, x1:x2])
                            dets = raw.copy()
                            if len(dets):
                                dets[:, [0, 2]] += x1
                                dets[:, [1, 3]] += y1
        else:
            if run_det:
                if mode == "fixed" and c["fixed_roi"]:
                    x, y, w, h = c["fixed_roi"]
                    raw, tm["infer"] = self.det.detect(frame[y:y + h, x:x + w])
                    dets = raw.copy()
                    if len(dets):
                        dets[:, [0, 2]] += x
                        dets[:, [1, 3]] += y
                    info["rois"] = [[x, y, x + w, y + h]]
                else:
                    dets, tm["infer"] = self.det.detect(frame)

        tracks = None
        if dets is not None:
            dets = nms(dets, c["nms_iou"])
            info["det_ran"] = True
            info["n_det"] = int(len(dets))
            t0 = time.perf_counter()
            tracks = self.tracker.update(dets, steps=self.since_det + 1)
            tm["track"] = (time.perf_counter() - t0) * 1000
            self.since_det = 0
            self.lk_failed = False
        elif lk_tracks is not None:
            tracks = lk_tracks
            self.since_det += 1
        if lk_gray is not None:
            self.prev_lk = lk_gray
        if tracks is not None:
            self.cur_tracks = tracks
            self._remember(tracks, t)
            t0 = time.perf_counter()
            peds = []
            ids = set()
            for x1, y1, x2, y2, tid, sc in tracks:
                u, v = foot_point(x1, y1, x2, y2)
                p = {"track_id": int(tid), "conf": round(float(sc), 3), "u": round(u, 1), "v": round(v, 1),
                     "bbox": [int(x1), int(y1), int(x2), int(y2)], "x_m": None, "y_m": None,
                     "vx": 0.0, "vy": 0.0, "v_ok": False}
                if self.H is not None:
                    q = self.H @ np.array([u, v, 1.0])
                    if q[2] > 0:
                        x, y = q[0] / q[2], q[1] / q[2]
                        vx, vy, ok = self.vel.update(tid, t, x, y)
                        p.update(x_m=round(float(x), 3), y_m=round(float(y), 3), vx=round(vx, 3),
                                 vy=round(vy, 3), v_ok=ok)
                        if ok:
                            p.update(spd=round(math.hypot(vx, vy), 2), spd_src="homography")
                else:
                    # 바닥 보정이 없을 때: 박스 높이를 사람 키(기본 1.7 m)로 보고 화면 이동(px/s)을 m/s 로 바꾼 추정값.
                    # 화면을 가로지르는 움직임에서 잘 맞고, 카메라 쪽으로 오가는 움직임은 작게 나옴. 박스가 화면 위·아래에
                    # 걸려 잘리면(키가 다 안 보임) 내지 않음.
                    pu, pv, okp = self.vel_px.update(tid, t, u, v)
                    bh = float(y2 - y1)
                    self.box_h[tid] = bh if tid not in self.box_h else 0.8 * self.box_h[tid] + 0.2 * bh
                    if okp and self.box_h[tid] > 25 and y1 > 2 and y2 < self.Hh - 3:
                        sp = math.hypot(pu, pv) * float(c.get("person_height_m", 1.7)) / self.box_h[tid]
                        p.update(spd=round(sp if sp >= 0.15 else 0.0, 2), spd_src="height")
                if c.get("demo_speed_max", 0) > 0:
                    # 예시값: 범위 안에서 천천히 바뀌는 임의의 수. 측정값이 아님 → spd_src="demo", 화면에 (demo) 표시
                    lo, hi = float(c["demo_speed_min"]), float(c["demo_speed_max"])
                    cur, tgt = self.demo_spd.get(tid, (None, None))
                    if cur is None:
                        cur = tgt = float(self.demo_rng.uniform(lo, hi))
                    if self.demo_rng.random() < 0.08:
                        tgt = float(self.demo_rng.uniform(lo, hi))
                    cur = min(hi, max(lo, cur + 0.15 * (tgt - cur)))
                    self.demo_spd[tid] = (cur, tgt)
                    p.update(spd=round(cur, 2), spd_src="demo")
                ids.add(tid)
                peds.append(p)
            self.vel.prune(ids, t)
            self.vel_px.prune(ids, t)
            for k in [k for k in self.box_h if k not in ids and k not in self.vel_px.hist]:
                del self.box_h[k]
            for k in [k for k in self.demo_spd if k not in ids]:
                del self.demo_spd[k]
            tm["homo"] = (time.perf_counter() - t0) * 1000
            self.last_peds, self.last_t = peds, t
        else:
            # detect_every > 1 로 건너뛴 프레임: 직전 결과를 속도로 앞당김
            dt = t - self.last_t if self.last_t is not None else 0.0
            peds = []
            for p in self.last_peds:
                q = dict(p, extrapolated=True)
                if p["x_m"] is not None:
                    q["x_m"] = round(p["x_m"] + p["vx"] * dt, 3)
                    q["y_m"] = round(p["y_m"] + p["vy"] * dt, 3)
                peds.append(q)
        total = (time.perf_counter() - t_all) * 1000
        info["timing"] = {k: (None if v is None else round(v, 2)) for k, v in tm.items()}
        info["timing"]["total"] = round(total, 2)
        info["peds"] = peds
        return info

    # ── 오버레이 ──
    def draw(self, frame, r, extra_text=""):
        out = frame.copy()
        for b in r["motion"]:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 0), 1)
        for b in r["memory"]:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 1)
        for b in r["rois"]:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 255), 2)
        if self.Hinv is not None and self.cfg.get("draw_curb"):
            pts = []
            for x in np.linspace(-15, 15, 61):
                q = self.Hinv @ np.array([x, 0.0, 1.0])
                if q[2] > 0:
                    pts.append((int(q[0] / q[2]), int(q[1] / q[2])))
            for a, b in zip(pts, pts[1:]):
                if -2000 < a[0] < 4000 and -2000 < a[1] < 4000 and -2000 < b[0] < 4000 and -2000 < b[1] < 4000:
                    cv2.line(out, a, b, (0, 0, 255), 2)
        for p in r["peds"]:
            x1, y1, x2, y2 = p["bbox"]
            col = (0, 200, 0) if (p["v_ok"] or p.get("spd") is not None) else (160, 160, 160)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.circle(out, (int(p["u"]), int(p["v"])), 4, (0, 0, 255), -1)
            txt = "#%d" % p["track_id"]
            if p.get("spd_src") == "demo":
                txt += " %.1fm/s (demo)" % p["spd"]
            elif p["x_m"] is not None:
                txt += " (%.1f,%.1f) %.1fm/s" % (p["x_m"], p["y_m"], math.hypot(p["vx"], p["vy"]))
            elif p.get("spd") is not None:
                txt += " ~%.1fm/s" % p["spd"]
            cv2.putText(out, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tmg = r["timing"]
        step = "LK" if r.get("lk_frame") else ("YOLO" if r["det_ran"] else "-")
        why = {"주기": "period", "LK실패": "lk-fail", "트랙없음": "no-track", "새움직임": "new-motion",
               "전체스캔": "full-scan"}.get(r.get("yolo_reason"), "")
        line1 = "Cam1 %s %s %s%s  n_roi=%d%s" % (r["mode"].upper(), r.get("policy", ""), step,
                                               (" (" + why + ")") if (why and step == "YOLO") else "",
                                               r["n_rois"], (" [" + r["fallback"] + "]") if r["fallback"] else "")
        line2 = "flow %s roi %s pack %s yolo %s lk %s track %s homo %s = %s ms" % tuple(
            "-" if tmg.get(k) is None else "%.0f" % tmg[k] for k in ("flow", "roi", "pack", "infer", "lk", "track",
                                                                       "homo", "total"))
        for i, s in enumerate([line1, line2, extra_text]):
            if s:
                cv2.putText(out, s, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, s, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv = r.get("_canvas")
        if cv is not None:
            sz = min(160, out.shape[0] // 3)
            inset = cv2.resize(cv, (sz, sz))
            out[4:4 + sz, out.shape[1] - sz - 4:out.shape[1] - 4] = inset
            cv2.rectangle(out, (out.shape[1] - sz - 5, 3), (out.shape[1] - 3, 5 + sz), (0, 220, 255), 1)
            cv2.putText(out, "packed x%.2f" % (r["pack_scale"] or 0), (out.shape[1] - sz, sz + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
        return out
