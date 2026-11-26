#!/usr/bin/env python3
# pi/cam_check.py
#
# [E0 준비] USB 웹캠 2대가 어느 장치 번호인지, 원하는 해상도·FPS가 실제로 나오는지 확인한다.
#
#   python3 pi/cam_check.py                    # /dev/video0~9 중 열리는 것 목록 + 실측 FPS
#   python3 pi/cam_check.py --snap /dev/video0 # 한 장 저장 (calib/snap_video0.jpg) → 보정·구도 확인용
#
# 웹캠 1대가 /dev/video0, /dev/video1 처럼 장치 2개를 만드는 경우가 많다 (하나는 메타데이터).
# 실제로 영상이 나오는 번호를 site.json 의 cam1.device / cam2.device 에 적는다.
# 두 대를 같은 USB 컨트롤러에 꽂아 MJPG가 아니면 대역폭 부족으로 하나가 안 열릴 수 있음 → MJPG 강제.

import argparse
import glob
import os
import time

import cv2


def open_cam(dev, w=640, h=480, fps=30):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def measure(dev, secs=3.0):
    cap = open_cam(dev)
    if not cap.isOpened():
        return None
    ok, f = cap.read()
    if not ok:
        cap.release()
        return None
    n, t0 = 0, time.time()
    while time.time() - t0 < secs:
        ok, f = cap.read()
        if ok:
            n += 1
    cap.release()
    return f.shape[1], f.shape[0], n / secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", help="이 장치에서 한 장 저장")
    args = ap.parse_args()
    if args.snap:
        cap = open_cam(args.snap)
        for _ in range(10):  # 자동 노출이 자리 잡게 몇 장 버림
            cap.read()
        ok, f = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("열 수 없음: " + args.snap)
        os.makedirs("calib", exist_ok=True)
        out = "calib/snap_%s.jpg" % os.path.basename(args.snap)
        cv2.imwrite(out, f)
        print("저장:", out, f.shape)
        return
    for dev in sorted(glob.glob("/dev/video*")):
        r = measure(dev)
        if r:
            print("%-12s 영상 OK  %dx%d  실측 %.1f FPS" % (dev, r[0], r[1], r[2]))
        else:
            print("%-12s (영상 아님/열리지 않음)" % dev)


if __name__ == "__main__":
    main()
