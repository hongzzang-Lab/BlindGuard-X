# bgx/geometry.py
#
# 영상 좌표(픽셀) ↔ 바닥 좌표(m), Homography 계산(numpy만 사용), 보행자 속도 추정.
#
# 바닥 좌표 약속 (핸드오프 문서 05_interfaces.md와 같음)
#   - 연석선(차도-인도 경계) = X축 (y = 0)
#   - 인도(대기 쪽) = +y, 차도 = -y
#   - 위에서 내려다볼 때 x는 오른쪽, y는 위쪽(인도 쪽)  → 오른손 좌표계
#     = 연석선 위에 서서 '인도를 왼쪽'에 두고 바라보는 방향이 +x
#   - 단위 m

import json
import math
from collections import deque

import numpy as np


# ───────────── Homography ─────────────
def _normalize(pts):
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    d = np.sqrt(((pts - c) ** 2).sum(1)).mean()
    s = math.sqrt(2) / d if d > 0 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.c_[pts, np.ones(len(pts))] @ T.T
    return ph, T


def homography_dlt(src, dst):
    """정규화 DLT. src(픽셀) → dst(바닥 m). 4점 이상. OpenCV 없이 numpy만 씀 (노트북 서버용)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("점이 4개 이상이고 개수가 같아야 함")
    a, Ta = _normalize(src)
    b, Tb = _normalize(dst)
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta
    H = H / H[2, 2]
    return orient_H(H, src.mean(0))


def orient_H(H, ref_px):
    """Homography 는 부호(-1배)가 달라도 같은 변환이지만, 우리는 w>0 을 '카메라 앞'으로 쓰므로
    기준 픽셀(보정점 중심 또는 화면 아래 가운데)에서 w>0 이 되게 부호를 맞춘다."""
    H = np.asarray(H, float)
    w = H[2] @ np.array([ref_px[0], ref_px[1], 1.0])
    return -H if w < 0 else H


def apply_h(H, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


def reproj_errors(H, src, dst):
    return np.linalg.norm(apply_h(H, src) - np.asarray(dst, float), axis=1)


def load_H(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["H"], dtype=float)


def img_to_floor(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def foot_point(x1, y1, x2, y2):
    """박스 아래 가운데 = 발 위치 (픽셀)."""
    return (x1 + x2) / 2.0, y2


# ───────────── 속도 ─────────────
class VelocityEstimator:
    """트랙별 최근 window_s 초의 바닥 위치로 직선 맞춤(최소제곱) 속도.
    한 프레임 차분보다 흔들림이 훨씬 작다 (발 위치가 프레임마다 몇 cm씩 튐).
    점이 min_pts개 이상이고 시간 폭이 min_span_s 이상일 때만 유효(ok=True)."""

    def __init__(self, window_s=1.0, min_pts=5, min_span_s=0.3):
        self.window_s = window_s
        self.min_pts = min_pts
        self.min_span_s = min_span_s
        self.hist = {}

    def update(self, tid, t, x, y):
        q = self.hist.setdefault(tid, deque())
        q.append((t, x, y))
        while q and t - q[0][0] > self.window_s:
            q.popleft()
        if len(q) < self.min_pts or (q[-1][0] - q[0][0]) < self.min_span_s:
            return 0.0, 0.0, False
        a = np.array(q)
        tc = a[:, 0] - a[:, 0].mean()
        den = float((tc * tc).sum())
        if den <= 1e-9:
            return 0.0, 0.0, False
        vx = float((tc * (a[:, 1] - a[:, 1].mean())).sum() / den)
        vy = float((tc * (a[:, 2] - a[:, 2].mean())).sum() / den)
        return vx, vy, True

    def prune(self, active_ids, t, keep_s=2.0):
        for tid in list(self.hist):
            q = self.hist[tid]
            if tid not in active_ids and (not q or t - q[-1][0] > keep_s):
                del self.hist[tid]
