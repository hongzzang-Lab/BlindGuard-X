# bgx/detector.py
#
# 탐지기. 약속: detect(img) → (dets, infer_ms)
#   dets: np.ndarray (N, 5) [x1, y1, x2, y2, score]  (img 좌표, 사람 클래스만)
#
# - UltralyticsDetector: YOLO26n-NCNN (Pi 실전용). ultralytics 의 predict 만 쓴다 (추적은 bgx.bytetrack).
# - ColorBoxDetector  : 시험용. 채도가 높은 색 덩어리를 '사람'으로 본다 (합성 영상으로 파이프라인을 검증할 때).

import time

import numpy as np


class UltralyticsDetector:
    def __init__(self, model_path, imgsz=416, conf=0.1):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="detect")
        self.imgsz = imgsz
        self.conf = conf  # ByteTrack 2단계(낮은 신뢰도 연결)를 쓰려고 0.1까지 받는다
        warm = np.full((imgsz, imgsz, 3), 114, np.uint8)
        for _ in range(2):
            self.model.predict(warm, imgsz=imgsz, conf=conf, classes=[0], verbose=False)

    def detect(self, img):
        t = time.perf_counter()
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False)[0]
        ms = (time.perf_counter() - t) * 1000
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 5)), ms
        xyxy = b.xyxy.cpu().numpy().astype(float)
        conf = b.conf.cpu().numpy().astype(float).reshape(-1, 1)
        return np.hstack([xyxy, conf]), ms


class ColorBoxDetector:
    """합성 영상 시험용: HSV 채도가 높은 덩어리 → 사람 박스 (점수 0.9).
    면적이 작은 덩어리는 점수를 낮게 줘서 ByteTrack 2단계도 흉내 낸다."""

    def __init__(self, min_area=60, delay_ms=0.0):
        import cv2
        self.cv2 = cv2
        self.min_area = min_area
        self.delay_ms = delay_ms

    def detect(self, img):
        cv2 = self.cv2
        t = time.perf_counter()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 1] > 150) & (hsv[..., 2] > 80)).astype(np.uint8) * 255
        n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
        out = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < self.min_area:
                continue
            score = 0.9 if a >= self.min_area * 3 else 0.18
            out.append([x, y, x + w, y + h, score])
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)
        return (np.array(out, float) if out else np.zeros((0, 5))), (time.perf_counter() - t) * 1000


def make_detector(kind, model_path=None, imgsz=416, conf=0.1):
    if kind == "color":
        return ColorBoxDetector()
    return UltralyticsDetector(model_path, imgsz, conf)
