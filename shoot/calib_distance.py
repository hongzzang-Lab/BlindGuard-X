#!/usr/bin/env python3
# shoot/calib_distance.py — 줄자 없이 "대략 몇 m" 로 Cam1 바닥 보정(H) 만들기
#
# 1) 보정 촬영:  bash shoot/shoot.sh calib 보정
#    한 사람이 정한 자리마다 3초 이상 가만히 서 있다가 다음 자리로 걸어감 (4곳 이상, 한 줄로만 서지 않기)
# 2) 선 자리를 순서대로 "앞,오른쪽" (m) 으로 적어서 실행. 기준은 Cam1 삼각대 바로 아래 바닥.
#    오른쪽은 카메라가 보는 방향 기준이고 왼쪽이면 음수.
#
#    python3 shoot/calib_distance.py --spots "3,-1.5  3,1.5  6,-2  6,2  9,0"
#
#    → 가장 최근 calib 촬영 기록에서 '서 있던 구간'을 찾아 발 위치(픽셀)와 짝지어 H 를 계산하고
#      calib/H_cam1.json 에 저장 (이전 파일은 .bak 으로 남김). 다음 촬영부터 바로 쓰임.
#
# 바닥 좌표: x = 오른쪽(m), y = -앞(m)  → 카메라 바로 아래가 (0,0), 멀어질수록 y 가 음수 (기존 약속: 차도 = -y)
import argparse
import csv
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bgx.geometry import apply_h, homography_dlt, reproj_errors  # noqa: E402

CALIB_DIR = os.environ.get("BGX_CALIB_DIR") or os.path.join(ROOT, "calib")


def read_feet(logdir):
    """프레임마다 가장 큰 사람(박스 높이)의 발 위치 → [(t초, u, v, 박스높이)]"""
    best = {}
    with open(os.path.join(logdir, "cam1_tracks.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                fr, t = int(r["frame"]), float(r["ts_cap_local"]) / 1000.0
                u, v = float(r["u"]), float(r["v"])
                h = float(r["y2"]) - float(r["y1"])
            except (ValueError, KeyError):
                continue
            if fr not in best or h > best[fr][3]:
                best[fr] = (t, u, v, h)
    rows = [best[k] for k in sorted(best)]
    t0 = rows[0][0] if rows else 0.0
    return [(t - t0, u, v, h) for t, u, v, h in rows]


def find_stands(rows, min_s=2.0, rad_frac=0.10, rad_min=8.0, gap_s=1.0):
    """가만히 서 있던 구간 찾기: 발 위치가 (박스 높이의 10%) 안에서 min_s 초 이상 머문 구간."""
    out, i, n = [], 0, len(rows)
    while i < n:
        j, us, vs = i, [rows[i][1]], [rows[i][2]]
        while j + 1 < n:
            t, u, v, h = rows[j + 1]
            R = max(rad_min, rad_frac * h)
            if t - rows[j][0] > gap_s or math.hypot(u - np.mean(us), v - np.mean(vs)) > R:
                break
            us.append(u)
            vs.append(v)
            j += 1
        if rows[j][0] - rows[i][0] >= min_s:
            out.append({"t0": rows[i][0], "t1": rows[j][0], "u": float(np.median(us)), "v": float(np.median(vs)),
                        "n": j - i + 1})
            i = j + 1
        else:
            i += 1
    return out


def parse_spots(s):
    pts = []
    for tok in s.replace(";", " ").split():
        a, b = tok.split(",")
        pts.append((float(a), float(b)))
    return pts


def collinear(pts):
    p = np.asarray(pts, float)
    p = p - p.mean(0)
    sv = np.linalg.svd(p, compute_uv=False)
    return sv[1] < 0.15 * sv[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", required=True, help='선 자리, 순서대로 "앞,오른쪽" m (예: "3,-1.5 3,1.5 6,-2 6,2")')
    ap.add_argument("--log", help="보정 촬영 기록 폴더 (없으면 가장 최근 logs/pi_*_calib_*)")
    ap.add_argument("--pick", help="찾은 구간 중 쓸 번호 (예: 1,2,4,5). 구간 수와 자리 수가 다를 때")
    ap.add_argument("--min-s", type=float, default=2.0, help="서 있던 것으로 볼 최소 시간(초)")
    ap.add_argument("--dry", action="store_true", help="계산만 하고 저장 안 함")
    a = ap.parse_args()

    logdir = a.log
    if not logdir:
        c = sorted(glob.glob(os.path.join(ROOT, "logs", "pi_*_calib_*")))
        if not c:
            sys.exit("보정 촬영 기록이 없음 → 먼저: bash shoot/shoot.sh calib 보정")
        logdir = c[-1]
    print("보정 촬영 기록:", logdir)
    spots = parse_spots(a.spots)
    rows = read_feet(logdir)
    if not rows:
        sys.exit("이 기록에서 사람이 한 번도 검출되지 않음 (cam1_tracks.csv 가 비어 있음)")
    stands = find_stands(rows, a.min_s)
    print("\n서 있던 구간 %d곳 (자리 %d곳 입력)" % (len(stands), len(spots)))
    for k, s in enumerate(stands, 1):
        print("  %d번: %5.1f ~ %5.1f초  발 위치 픽셀 (%.0f, %.0f)" % (k, s["t0"], s["t1"], s["u"], s["v"]))
    if a.pick:
        stands = [stands[int(k) - 1] for k in a.pick.split(",")]
    if len(stands) != len(spots):
        sys.exit("\n구간 수와 자리 수가 다름. 위 목록(시간)을 보고 쓸 번호를 --pick 1,2,4 처럼 골라 다시 실행.\n"
                 "구간이 너무 적으면 --min-s 1.5, 걷다 멈춘 곳까지 잡히면 --min-s 3")
    if len(spots) < 4:
        sys.exit("자리가 4곳 이상 필요함")
    world = [(right, -fwd) for fwd, right in spots]
    if collinear(world):
        sys.exit("자리들이 거의 한 줄 위에 있음 → 좌우로 벌린 자리를 넣어야 함 (예: 앞 3 m 에서 왼쪽·오른쪽 한 번씩)")
    img = [(s["u"], s["v"]) for s in stands]
    H = homography_dlt(img, world)
    err = reproj_errors(H, img, world)
    print("\n자리별 맞춤 오차 (말한 위치와 H 로 계산한 위치의 차이)")
    for (fwd, right), e, w in zip(spots, err, apply_h(H, img)):
        print("  앞 %4.1f m, 오른쪽 %4.1f m → 계산값 앞 %4.1f, 오른쪽 %4.1f  (차이 %.2f m)" % (fwd, right, -w[1], w[0], e))
    print("  평균 %.2f m, 최대 %.2f m%s" % (err.mean(), err.max(), "" if len(spots) > 4 else
                                        "  (4곳이면 항상 0 에 가까움. 5곳 이상이어야 오차를 알 수 있음)"))
    if len(spots) > 4 and err.max() > 1.0:
        print("  [주의] 1 m 넘게 어긋난 자리가 있음. 자리 순서나 좌우 부호(왼쪽은 음수)를 확인.")
    if a.dry:
        return
    os.makedirs(CALIB_DIR, exist_ok=True)
    dst = os.path.join(CALIB_DIR, "H_cam1.json")
    if os.path.exists(dst):
        bak = dst + ".bak_" + time.strftime("%m%d_%H%M%S")
        shutil.copy(dst, bak)
        print("\n이전 보정 보관:", bak)
    obj = {"H": H.tolist(), "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "from": "calib_distance",
           "detail": {"img_pts": img, "world_pts": world, "spots_fwd_right": spots, "reproj_mean": float(err.mean()),
                      "reproj_max": float(err.max()), "log": os.path.basename(logdir),
                      "frame": "x=오른쪽 m, y=-앞 m, 원점=Cam1 삼각대 아래"}}
    for p in (dst, os.path.join(logdir, "H_cam1_from_this.json")):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
    print("저장:", dst, "\n다음 촬영부터 이 보정이 쓰임. 이미 찍은 영상도 report.py 가 이 보정으로 거리·속도를 다시 계산함.")


if __name__ == "__main__":
    main()
