# tests/synth.py
#
# 합성 장면 생성기 (장비 없이 파이프라인을 검증하려고).
# 알려진 Homography 로 바닥 경로를 영상에 투영하고, 사람 대신 색 막대를 그린다.
# 색 막대는 bgx.detector.ColorBoxDetector 가 '사람'으로 잡는다.

import math

import cv2
import numpy as np

from bgx.geometry import apply_h, homography_dlt

IMG_PTS = [(60, 470), (580, 470), (400, 150), (240, 150)]
FLOOR_PTS = [(-3, 2), (3, 2), (3, -5), (-3, -5)]
H_TRUE = homography_dlt(IMG_PTS, FLOOR_PTS)   # 영상 → 바닥
G_TRUE = np.linalg.inv(H_TRUE)                  # 바닥 → 영상


def background(seed=0, W=640, H=480):
    rng = np.random.default_rng(seed)
    bg = (rng.random((H // 8, W // 8)) * 50 + 90).astype(np.uint8)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)


def draw_person(img, x, y, color):
    """바닥 (x, y)에 서 있는 사람(색 막대). 발 = 막대 아래 가운데."""
    u, v = apply_h(G_TRUE, [(x, y)])[0]
    l, r = apply_h(G_TRUE, [(x - 0.25, y), (x + 0.25, y)])
    w = max(4, abs(r[0] - l[0]))
    h = max(8, w * 3.4)
    cv2.rectangle(img, (int(u - w / 2), int(v - h)), (int(u + w / 2), int(v)), color, -1)
    return u, v


def scene_frame(bg, people, t):
    """people: [(fn(t) -> (x, y) or None, color)]"""
    img = bg.copy()
    noise = np.random.default_rng(int(t * 1000)).integers(-3, 4, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    truth = []
    for i, (fn, col) in enumerate(people):
        p = fn(t)
        if p is not None:
            draw_person(img, p[0], p[1], col)
            truth.append((i, p))
    return img, truth


def crossing(t, t0=1.0, y0=1.5, speed=1.3, x=0.0, y_end=-4.5):
    if t < t0:
        return (x, y0)
    return (x, max(y0 - speed * (t - t0), y_end))


def walker_x(t, t0=0.0, x0=-3.0, speed=1.0, y=-3.5, x_end=3.0):
    if t < t0:
        return None
    x = x0 + speed * (t - t0)
    return (x, y) if x <= x_end else None
