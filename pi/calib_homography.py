#!/usr/bin/env python3
# pi/calib_homography.py
#
# [E0] Cam1 바닥 Homography 보정 + 검증.
# 화면이 있는 컴퓨터에서 실행한다 (Pi에 모니터가 없으면 snap 사진을 노트북으로 옮겨서 노트북에서).
#
# 1) 바닥에 테이프로 점을 표시하고 줄자로 바닥 좌표(m)를 잰다 (최소 4점, 6~8점 권장, 넓게 퍼지게).
#    좌표 약속: 연석선 위가 y=0, 인도 쪽 +y, 차도 쪽 -y, 횡단보도 가운데가 x=0.
#    calib/points_world.txt 에 클릭할 순서대로 "x y" 한 줄씩 적는다.
# 2) 사진 찍기:  python3 pi/cam_check.py --snap /dev/video0
# 3) 보정:      python3 pi/calib_homography.py --image calib/snap_video0.jpg --world calib/points_world.txt
#    → 창에서 points_world.txt 순서대로 테이프 점을 클릭, 다 찍으면 아무 키 → calib/H_cam1.json 저장
#    → calib/check_grid.jpg (바닥 0.5 m 격자를 영상에 겹친 그림) 로 눈으로 확인
# 4) 검증 (보정에 안 쓴 테이프 점 3~5개):
#    python3 pi/calib_homography.py --verify --image calib/snap_video0.jpg --world calib/points_check.txt
#    → 점마다 오차(m)와 평균 오차 출력. 이 숫자가 "위치 추정 오차" 실측값의 기준이 된다.

import argparse
import json
import os

import cv2
import numpy as np


def read_world(path):
    pts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                x, y = map(float, line.replace(",", " ").split()[:2])
                pts.append((x, y))
    return pts


def click_points(img, n, title):
    pts = []
    disp = img.copy()

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(disp, str(len(pts)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(title, disp)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, disp)
    print("%d개 점을 순서대로 클릭한 뒤 아무 키나 누르세요 (r = 다시)" % n)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord("r"):
            pts.clear()
            disp[:] = img
            cv2.imshow(title, disp)
            continue
        if len(pts) == n:
            break
        print("아직 %d/%d 점" % (len(pts), n))
    cv2.destroyAllWindows()
    return pts


def draw_grid(img, H, xr=(-3, 3), yr=(-5, 3), step=0.5):
    Hi = np.linalg.inv(H)
    out = img.copy()

    def to_img(x, y):
        p = Hi @ np.array([x, y, 1.0])
        return int(p[0] / p[2]), int(p[1] / p[2])

    xs = np.arange(xr[0], xr[1] + 1e-6, step)
    ys = np.arange(yr[0], yr[1] + 1e-6, step)
    for x in xs:
        c = (0, 255, 255) if abs(x) < 1e-6 else (0, 160, 0)
        cv2.line(out, to_img(x, yr[0]), to_img(x, yr[1]), c, 1)
    for y in ys:
        c = (0, 0, 255) if abs(y) < 1e-6 else (0, 160, 0)  # 빨간 선 = 연석선 y=0
        cv2.line(out, to_img(xr[0], y), to_img(xr[1], y), c, 2 if abs(y) < 1e-6 else 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--world", required=True, help="바닥 좌표 목록 파일 (클릭 순서대로 'x y')")
    ap.add_argument("--out", default="calib/H_cam1.json")
    ap.add_argument("--verify", action="store_true", help="저장된 H로 오차만 계산")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("사진을 못 읽음: " + args.image)
    world = read_world(args.world)
    if not args.verify and len(world) < 4:
        raise SystemExit("보정에는 최소 4점 필요")

    px = click_points(img, len(world), "click: " + os.path.basename(args.world))
    src = np.array(px, dtype=np.float64)
    dst = np.array(world, dtype=np.float64)

    if args.verify:
        with open(args.out, encoding="utf-8") as f:
            H = np.array(json.load(f)["H"])
    else:
        H, _ = cv2.findHomography(src, dst, 0)

    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1)
    for (wx, wy), (px_, py_), e in zip(world, proj, err):
        print("  실제 (%.2f, %.2f)  추정 (%.2f, %.2f)  오차 %.3f m" % (wx, wy, px_, py_, e))
    print("%s 평균 오차 %.3f m, 최대 %.3f m" % ("검증" if args.verify else "보정점 재투영", err.mean(), err.max()))

    if not args.verify:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"H": H.tolist(), "image_points": px, "world_points": world,
                       "image": args.image, "image_size": [img.shape[1], img.shape[0]],
                       "reproj_mean_m": float(err.mean())}, f, ensure_ascii=False, indent=1)
        print("저장:", args.out)
        g = draw_grid(img, H)
        cv2.imwrite("calib/check_grid.jpg", g)
        print("격자 확인 그림: calib/check_grid.jpg (빨간 선 = 연석선, 노란 선 = x=0)")


if __name__ == "__main__":
    main()
