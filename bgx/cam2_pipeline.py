# bgx/cam2_pipeline.py
#
# Cam2 · 인도 대기구역 움직임 분석 (포스터 최종본 구조)
#
#   프레임 ─▶ ① Polygon ROI (대기구역 + 경계 너머 띠) ─▶ ② MOG2 (전경) ─▶ ③ Sparse LK (Shi-Tomasi 점 → 흐름)
#          ─▶ ④ 경계 접근 분석 (발 위치 ↔ 경계선 부호 거리, 차도 방향 움직임, 지속성) ─▶ ⑤ EMR Score [임시 식]
#
# 화면 약속 (고정 캠이므로 화면에서 한 번 정함)
#   polygon  : 대기구역 꼭짓점 (원본 px)
#   boundary : 차도-인도 경계선 (원본 px). 양 끝 2점(직선) 또는 여러 점(꺾은선 — 모퉁이처럼 연석이 휘어 있을 때).
#              보통 polygon 의 한 변
#   경계 '너머' 쪽(차도 쪽)은 polygon 중심의 반대편으로 자동 결정
#
# 판정 이벤트 (프레임 단위로 기록)
#   enter / leave : 구역에 사람(전경 덩어리)이 생김 / 사라짐
#   emr_high      : EMR ≥ emr_thr 가 emr_frames 프레임 연속 (출발 '예측')
#   cross         : 발 위치가 경계선 너머로 cross_frames 프레임 연속 (실제 '진입' = 정답)
#   → 선행 시간 = cross 시각 - emr_high 시각  (+ 이면 미리 알아챔)
#
# ⚠ EMR 식(가중치·기준)은 팀이 확정하지 않았다 → [임시] 값. 위험 판정(CPA)에는 아직 합치지 않음.

import time

import cv2
import numpy as np

DEFAULT_CFG = {
    "scale": 0.5,            # 처리 해상도 배율 (속도)
    "strip_px": 140,         # 경계 너머 띠 두께 (원본 px) — 진입을 보려면 경계 바깥도 봐야 함
    "warmup_frames": 40,     # 시작 후 배경 학습 (사람 없이!)
    "lr_idle": 0.005,        # 사람 없을 때 배경 학습률
    "lr_busy": 0.0,          # 사람 있을 때 0 → 오래 서 있어도 배경으로 흡수되지 않음
    "shadows": True,         # 야외 그림자 구분 (그림자는 전경에서 뺌)
    "var_threshold": 25,
    "min_blob_px": 600,      # 원본 px 기준 최소 사람 덩어리 면적
    "near_px": 150,          # 경계까지 이 거리 안이면 근접도가 올라감
    "toward_px": 2.0,        # 차도 방향 흐름이 프레임당 이만큼(원본 px)이면 방향성 1
    "persist_n": 5,
    "w_prox": 0.4, "w_tow": 0.4, "w_per": 0.2,
    "emr_thr": 0.7, "emr_frames": 3,
    "cross_frames": 3, "back_frames": 5,
    "lk_max_pts": 80,
}


class Cam2Analyzer:
    def __init__(self, polygon, boundary, frame_size, cfg=None):
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.W, self.H = frame_size
        self.set_geometry(polygon, boundary)
        self.reset_bg()

    # ── 설정 ──
    def set_geometry(self, polygon, boundary):
        c = self.cfg
        s = c["scale"]
        self.sw, self.sh = int(self.W * s), int(self.H * s)
        self.polygon = [list(map(float, p)) for p in polygon]
        self.boundary = [list(map(float, p)) for p in boundary]
        B = np.array(self.boundary, float)                    # (K, 2), K ≥ 2  (2점이면 예전과 똑같이 직선)
        seg = B[1:] - B[:-1]
        seg_n = np.c_[-seg[:, 1], seg[:, 0]] / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9)
        cen = np.array(self.polygon).mean(0)
        mid = (B[1:] + B[:-1]) / 2
        if np.sum(((cen - mid) * seg_n).sum(1) > 0) * 2 > len(seg_n):      # polygon 중심 반대쪽이 차도
            seg_n = -seg_n
        vn = np.zeros_like(B)                                 # 꼭짓점 법선 = 이웃 선분 법선의 평균
        vn[:-1] += seg_n
        vn[1:] += seg_n
        vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)
        n = seg_n.mean(0)
        n = n / (np.linalg.norm(n) + 1e-9)
        self.B, self.seg, self.seg_n = B, seg, seg_n
        self.b0, self.b1, self.n = B[0], B[-1], n
        strip = list(B) + list((B + vn * c["strip_px"])[::-1])   # 경계선을 차도 쪽으로 strip_px 만큼 민 띠
        self.strip = [list(map(float, p)) for p in strip]
        self.mask_poly = np.zeros((self.sh, self.sw), np.uint8)
        cv2.fillPoly(self.mask_poly, [(np.array(self.polygon) * s).astype(np.int32)], 255)
        self.mask_all = self.mask_poly.copy()
        cv2.fillPoly(self.mask_all, [(np.array(strip) * s).astype(np.int32)], 255)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def reset_bg(self):
        c = self.cfg
        self.mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=c["var_threshold"],
                                                      detectShadows=c["shadows"])
        self.k = 0
        self.prev_gray = None
        self.prev_blob = None
        self.persist = 0
        self.emr_run = 0
        self.cross_run = 0
        self.back_run = 0
        self.present = False
        self.emr_state = False
        self.crossed = False

    def _nearest(self, p):
        """경계선(꺾은선)에서 p 에 가장 가까운 곳 → (부호 거리, 그 선분 번호). 양 끝 선분은 바깥으로 길게 이어진 것으로 봄."""
        p = np.asarray(p, float)
        best = None
        last = len(self.seg) - 1
        for i in range(last + 1):
            a, d = self.B[i], self.seg[i]
            t = float((p - a) @ d) / (float(d @ d) + 1e-9)
            if i > 0:
                t = max(t, 0.0)
            if i < last:
                t = min(t, 1.0)
            q = a + t * d
            dist = float(np.linalg.norm(p - q))
            if best is None or dist < best[0]:
                best = (dist, i, q)
        dist, i, q = best
        return (dist if (p - q) @ self.seg_n[i] >= 0 else -dist), i

    def signed_dist(self, p):
        """원본 px. + 면 경계 너머(차도 쪽), - 면 대기구역 쪽."""
        return self._nearest(p)[0]

    def normal_at(self, p):
        """p 근처 경계선의 차도 방향 단위 벡터 (휘어진 경계선이면 위치마다 다름)."""
        return self.seg_n[self._nearest(p)[1]]

    # ── 본체 ──
    def process(self, frame):
        c = self.cfg
        s = c["scale"]
        t0 = time.perf_counter()
        small = cv2.resize(frame, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.k += 1
        warm = self.k <= c["warmup_frames"]
        lr = -1 if warm else (c["lr_busy"] if self.present else c["lr_idle"])
        fgm = self.mog.apply(small, learningRate=lr)
        fg = ((fgm == 255).astype(np.uint8) * 255)          # 127 = 그림자 → 제외
        fg = cv2.bitwise_and(fg, self.mask_all)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        t_mog = time.perf_counter()

        blob = None
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
        if n > 1 and not warm:
            i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            if st[i, cv2.CC_STAT_AREA] >= c["min_blob_px"] * s * s:
                x, y, w, h, a = st[i]
                ys, xs = np.nonzero(lab == i)
                bottom = ys >= ys.max() - max(2, int(0.08 * h))
                feet = (float(xs[bottom].mean()) / s, float(ys.max()) / s)
                blob = {"bbox": [x / s, y / s, (x + w) / s, (y + h) / s], "area": int(a / (s * s)),
                        "feet": feet, "mask": (lab == i).astype(np.uint8) * 255}

        # ③ Sparse LK: 이전 프레임 덩어리 안의 Shi-Tomasi 점 → 이번 프레임
        toward, n_pts, flows = 0.0, 0, None
        if blob is not None and self.prev_gray is not None and self.prev_blob is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=c["lk_max_pts"], qualityLevel=0.01,
                                         minDistance=4, mask=self.prev_blob)
            if p0 is not None:
                p1, stt, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, winSize=(15, 15),
                                                      maxLevel=2)
                ok = stt.reshape(-1) == 1
                if ok.any():
                    a = p0.reshape(-1, 2)[ok]
                    b = p1.reshape(-1, 2)[ok]
                    f = (b - a) / s
                    proj = f @ self.normal_at(blob["feet"])
                    toward = float(np.median(proj))
                    n_pts = int(ok.sum())
                    flows = (a / s, f)
        t_lk = time.perf_counter()

        # ④ 경계 접근 분석
        events = []
        sd = None
        if blob is not None:
            sd = self.signed_dist(blob["feet"])
        present = blob is not None
        if present and not self.present:
            events.append("enter")
        if not present and self.present:
            events.append("leave")
        self.present = present
        self.persist = self.persist + 1 if (present and toward > c["toward_px"] * 0.5) else 0
        if present:
            prox = 1.0 if sd >= 0 else float(np.clip(1 + sd / c["near_px"], 0, 1))
            tow = float(np.clip(toward / c["toward_px"], 0, 1))
            per = float(min(self.persist / c["persist_n"], 1.0))
        else:
            prox = tow = per = 0.0
        emr = c["w_prox"] * prox + c["w_tow"] * tow + c["w_per"] * per
        # 이벤트 (연속 프레임 조건)
        self.emr_run = self.emr_run + 1 if emr >= c["emr_thr"] else 0
        if self.emr_run >= c["emr_frames"] and not self.emr_state:
            self.emr_state = True
            events.append("emr_high")
        if self.emr_run == 0 and self.emr_state:
            self.emr_state = False
            events.append("emr_low")
        self.cross_run = self.cross_run + 1 if (sd is not None and sd > 0) else 0
        self.back_run = self.back_run + 1 if (sd is None or sd <= 0) else 0
        if self.cross_run >= c["cross_frames"] and not self.crossed:
            self.crossed = True
            events.append("cross")
        if self.crossed and self.back_run >= c["back_frames"]:
            self.crossed = False
            events.append("back")
        t_end = time.perf_counter()

        self.prev_gray = gray
        self.prev_blob = blob["mask"] if blob is not None else None
        return {
            "warmup": warm, "present": present, "area": blob["area"] if blob else 0,
            "feet": [round(v, 1) for v in blob["feet"]] if blob else None,
            "bbox": [round(v, 1) for v in blob["bbox"]] if blob else None,
            "signed_px": None if sd is None else round(sd, 1), "toward_px": round(toward, 3), "n_pts": n_pts,
            "persist": self.persist, "prox": round(prox, 3), "tow": round(tow, 3), "per": round(per, 3),
            "emr": round(emr, 3), "emr_state": self.emr_state, "crossed": self.crossed, "events": events,
            "timing": {"mog": round((t_mog - t0) * 1000, 2), "lk": round((t_lk - t_mog) * 1000, 2),
                       "analysis": round((t_end - t_lk) * 1000, 2), "total": round((t_end - t0) * 1000, 2)},
            "_flows": flows,
        }

    def draw(self, frame, r, extra_text=""):
        out = frame.copy()
        ov = out.copy()
        cv2.fillPoly(ov, [np.array(self.strip, np.int32)], (60, 60, 200))
        out = cv2.addWeighted(ov, 0.25, out, 0.75, 0)
        cv2.polylines(out, [np.array(self.polygon, np.int32)], True, (255, 200, 0), 2)
        cv2.polylines(out, [self.B.astype(np.int32)], False, (0, 0, 255), 3, cv2.LINE_AA)
        im = (len(self.seg) - 1) // 2
        mid = (self.B[im] + self.B[im + 1]) / 2 if len(self.seg) % 2 else self.B[im + 1]
        cv2.arrowedLine(out, tuple(mid.astype(int)), tuple((mid + self.seg_n[im] * 40).astype(int)), (0, 0, 255), 2)
        if r["bbox"]:
            x1, y1, x2, y2 = map(int, r["bbox"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
            fx, fy = map(int, r["feet"])
            cv2.circle(out, (fx, fy), 6, (0, 0, 255) if r["crossed"] else (0, 255, 0), -1)
        fl = r.get("_flows")
        if fl is not None:
            for (x, y), (dx, dy) in zip(*fl):
                cv2.arrowedLine(out, (int(x), int(y)), (int(x + dx * 4), int(y + dy * 4)), (0, 255, 0), 1,
                                tipLength=0.4)
        bar = int(260 * r["emr"])
        cv2.rectangle(out, (8, 8), (268, 28), (50, 50, 50), -1)
        cv2.rectangle(out, (8, 8), (8 + bar, 28), (0, 0, 255) if r["emr_state"] else (0, 200, 255), -1)
        thr_x = 8 + int(260 * self.cfg["emr_thr"])
        cv2.line(out, (thr_x, 4), (thr_x, 32), (255, 255, 255), 1, cv2.LINE_AA)
        state = "BG LEARNING (keep area empty)" if r["warmup"] else (
            "CROSSED" if r["crossed"] else ("EMR HIGH" if r["emr_state"] else ("PERSON" if r["present"] else "EMPTY")))
        lines = ["EMR(tmp) %.2f  %s" % (r["emr"], state),
                 "dist %s px  toward %.1f px/f  pts %d" % (r["signed_px"], r["toward_px"], r["n_pts"]),
                 "mog %.0f lk %.0f = %.0f ms  %s" % (r["timing"]["mog"], r["timing"]["lk"], r["timing"]["total"],
                                                    extra_text)]
        for i, s in enumerate(lines):
            cv2.putText(out, s, (8, 48 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, s, (8, 48 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if r["events"]:
            cv2.putText(out, " ".join(r["events"]).upper(), (8, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
        return out
