#!/usr/bin/env python3
# pi/check_pi.py — 실험 전 Pi 점검 (2분)
#
#   python3 pi/check_pi.py                       # 패키지·카메라·모델 속도
#   python3 pi/check_pi.py --server 10.42.0.1    # + 노트북 서버 접속 확인
#
# 확인 항목
#   1) 패키지: opencv, numpy, ultralytics, ncnn, websockets
#   2) 카메라: /dev/video* 중 영상이 나오는 것, 실측 FPS (site.json 의 cam1/cam2 장치가 맞는지)
#   3) 모델: models/yolo26n_ncnn_model 로 416×416 추론 20회 → 중앙값 ms (시뮬 실측 26~28 ms 와 비교)
#   4) CPU 온도
#   5) (선택) 서버 접속

import argparse
import glob
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OK, BAD = "  [OK] ", "  [!!] "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", help="노트북 IP (예: 10.42.0.1)")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    site = json.load(open(os.path.join(ROOT, "site.json"), encoding="utf-8"))
    problems = 0

    print("1) 패키지")
    for mod in ["cv2", "numpy", "ultralytics", "ncnn", "websockets", "scipy"]:
        try:
            m = __import__(mod)
            print(OK + "%s %s" % (mod, getattr(m, "__version__", "")))
        except Exception as e:  # noqa: BLE001
            opt = mod == "scipy"
            print((OK if opt else BAD) + "%s 없음 (%s)%s" % (mod, e, " — 없어도 됨(추적 매칭을 탐욕법으로)" if opt else ""))
            problems += 0 if opt else 1

    print("2) 카메라")
    import cv2
    found = []
    for dev in sorted(glob.glob("/dev/video*"), key=lambda d: int(d.replace("/dev/video", "") or 0)):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        ok, f = cap.read() if cap.isOpened() else (False, None)
        if ok:
            n, t0 = 0, time.time()
            while time.time() - t0 < 2:
                n += cap.read()[0]
            found.append(dev)
            print(OK + "%s 영상 %dx%d, %.1f FPS" % (dev, f.shape[1], f.shape[0], n / 2))
        cap.release()
    for k in ("cam1", "cam2"):
        spec = site[k]["device"]
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from edge import resolve_device
            d = resolve_device(spec)
        except Exception as e:  # noqa: BLE001
            print(BAD + "site.json %s = %s : %s" % (k, spec, e))
            problems += 1
            continue
        if d in found:
            print(OK + "site.json %s = %s (영상 나옴)" % (k, d))
        else:
            print(BAD + "site.json %s = %s 인데 영상이 안 나옴 → 위 목록에서 골라 site.json 수정" % (k, d))
            problems += 1

    if not args.skip_model:
        print("3) 모델 속도 (416×416)")
        mp = os.path.join(ROOT, site["cam1"]["model"])
        if not os.path.exists(mp):
            print(BAD + "모델 없음: %s → bash pi/setup_pi.sh" % mp)
            problems += 1
        else:
            try:
                import numpy as np
                from bgx.detector import UltralyticsDetector
                det = UltralyticsDetector(mp, site["cam1"].get("imgsz", 416), 0.1)
                img = (np.random.default_rng(0).random((416, 416, 3)) * 255).astype("uint8")
                ts = [det.detect(img)[1] for _ in range(20)]
                med = statistics.median(ts)
                print((OK if med < 60 else BAD) + "추론 중앙값 %.1f ms (시뮬 때 26~28 ms)" % med)
            except Exception as e:  # noqa: BLE001
                print(BAD + "모델 실행 실패: %s" % e)
                problems += 1

    print("4) CPU 온도")
    try:
        t = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
        print((OK if t < 70 else BAD) + "%.1f °C %s" % (t, "" if t < 70 else "(쿨러·그늘 필요)"))
    except OSError:
        print("  (온도 읽기 불가)")

    if args.server:
        print("5) 서버")
        import urllib.request
        try:
            with urllib.request.urlopen("http://%s:8000/api/status" % args.server, timeout=3) as r:
                print(OK + "서버 응답 %s" % r.status)
        except Exception as e:  # noqa: BLE001
            print(BAD + "서버 접속 실패: %s (같은 Wi-Fi? 노트북 방화벽 8000?)" % e)
            problems += 1

    print("\n결과:", "모두 OK" if problems == 0 else "%d개 문제 — 위 [!!] 확인" % problems)


if __name__ == "__main__":
    main()
