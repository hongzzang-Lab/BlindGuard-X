#!/usr/bin/env python3
# tools/simulate_gps.py
#
# 공원 C1 시나리오를 실험 전에 모의 실행: "폰 GPS 오차가 이 정도면 경고가 몇 % 나올까?"
# 서버와 같은 코드(bgx.engine: GPS 차량·맵 매칭·칼만·자기 제외·CPA·단계 유지)를 10 Hz로 돌린다.
#
#   python3 tools/simulate_gps.py              # 표 출력
#   python3 tools/simulate_gps.py --runs 100
#
# 모델 (가정)
#   - GPS 1 Hz, 측정→도착 0.5 s 늦음, 흰잡음 σ_w, 남은 치우침 σ_b (기준점을 같은 폰으로 바로 전에 잡았을 때 남는 양)
#   - 도플러 속력 잡음 0.2 m/s, 방향 잡음 15° (0.5 m/s 이상일 때)
#   - 보행자(카메라) 위치 잡음 5 cm, 10 Hz
#   - R1: A가 x=-30 에서 출발, A가 '출발 신호 콘'(x = -2.5 s × 속력)을 지날 때 B가 0.5 s 반응 후 1.3 m/s 로 횡단
#         A는 x=-2 정지선에서 멈춤 (안전) → 그 전에 경고가 와야 성공
#   - R2: B가 A 통과(x=+5) 뒤 횡단 / R3: B 대기선에서 서성임 / R4: A가 x=+5에서 멀어지고 B 횡단

import argparse
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bgx.engine import ConflictEngine, GpsVehicle  # noqa: E402
from bgx.geometry import VelocityEstimator  # noqa: E402
from bgx.risk import DEFAULT_RISK  # noqa: E402

LANE_Y = -1.75
WAIT_Y = 1.0
STOP_X = -2.0


def run_once(scn, v_car, sig_w, sig_b, map_match, rng, still=0.0):
    bias = (rng.gauss(0, sig_b), rng.gauss(0, sig_b))
    gv = GpsVehicle("A", {"map_match": map_match, "lane_y": LANE_Y})
    eng = ConflictEngine(dict(DEFAULT_RISK, compensate_latency=True, ped_still_mps=still), 1.0)
    vel = VelocityEstimator()
    dt = 0.1
    t = 0.0
    if scn == "R4":
        ax0, sign = 5.0, 1.0
    else:
        ax0, sign = -30.0, 1.0
    cue_x = -2.5 * v_car
    b_start = None
    first = None
    a_stop_t = None
    pending = []   # GPS 지연 큐
    next_fix = rng.uniform(0, 1.0)
    while t < 30:
        # 차량 실제 위치
        ax = ax0 + sign * v_car * t
        moving = True
        if scn in ("R1",) and ax >= STOP_X:
            if a_stop_t is None:
                a_stop_t = t
            ax = STOP_X
            moving = False
        # 보행자 B
        if scn == "R1" and b_start is None and ax >= cue_x:
            b_start = t + 0.5
        if scn == "R2" and b_start is None and ax >= 5.0:
            b_start = t + 0.5
        if scn == "R4" and b_start is None and t >= 2.0:
            b_start = t
        if scn == "R3":
            bx, by = 0.3 * math.sin(1.1 * t), WAIT_Y + 0.25 * math.sin(1.9 * t + 1)
        elif b_start is not None and t >= b_start:
            bx, by = 0.0, max(WAIT_Y - 1.3 * (t - b_start), -6.0)
        else:
            bx, by = 0.0, WAIT_Y
        # GPS 측정 (1 Hz, 0.5 s 지연)
        if t >= next_fix:
            next_fix += 1.0
            spd = (v_car if moving else 0.0) + rng.gauss(0, 0.2)
            hd_err = math.radians(rng.gauss(0, 15))
            vd = (spd * math.cos(hd_err) * sign, spd * math.sin(hd_err)) if spd >= 0.5 else None
            pending.append((t + 0.5, t, ax + bias[0] + rng.gauss(0, sig_w), LANE_Y + bias[1] + rng.gauss(0, sig_w),
                            max(spd, 0.0), vd))
        while pending and pending[0][0] <= t:
            _, tf, gx, gy, spd, vd = pending.pop(0)
            gv.update(tf, gx, gy, spd, vd, max(3.0, sig_w * 2))
        # 카메라 (B만; A 본인도 화면에 있으면 자기 제외가 처리 — 여기선 B만 넣음)
        cx, cy = bx + rng.gauss(0, 0.05), by + rng.gauss(0, 0.05)
        pvx, pvy, ok = vel.update(1, t, cx, cy)
        peds = [{"track_id": 1, "x_m": cx, "y_m": cy, "vx": pvx, "vy": pvy, "v_ok": ok, "ts_cap": int(t * 1000)}]
        st = gv.state(t)
        res, ch, prev, new = eng.step("A", int(t * 1000), st, peds, exclude_self=False)
        if new and first is None:
            first = t
        t = round(t + dt, 3)
        if scn == "R1" and a_stop_t is not None and t > a_stop_t + 4:
            break
        if scn != "R1" and t > 22:
            break
    if scn == "R1":
        ok = first is not None and (a_stop_t is None or first <= a_stop_t)
        margin = (a_stop_t - first) if (first is not None and a_stop_t is not None) else None
        return ok, margin
    return first is not None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--still", type=float, default=0.0, help="보행자 정지 판정 속력 (site.json risk.ped_still_mps)")
    args = ap.parse_args()
    rng = random.Random(1)
    noise = [("좋음 σw1 σb1", 1.0, 1.0), ("보통 σw2 σb2", 2.0, 2.0), ("나쁨 σw3 σb4", 3.0, 4.0)]
    speeds = [("빠른걸음 1.8", 1.8), ("뛰기 2.8", 2.8), ("자전거 4.5", 4.5)]
    print("R1 = 경고가 정지선 전에 '나온' 비율 (높을수록 좋음),  R2·R3·R4 = 경고가 나온 비율 (낮을수록 좋음)")
    print("맵매칭+칼만(서버 기본) / 끔 비교. 각 칸 %d회\n" % args.runs)
    print("보행자 정지 판정 속력 ped_still_mps = %.2f m/s\n" % args.still)
    for mm in (True, False):
        print("== 맵 매칭 %s ==" % ("켬 (기본)" if mm else "끔"))
        print("%-14s %-12s %8s %10s %8s %8s %8s" % ("GPS", "차량 속도", "R1", "R1 여유s", "R2", "R3", "R4"))
        for nname, sw, sb in noise:
            for sname, v in speeds:
                row = []
                margins = []
                for scn in ("R1", "R2", "R3", "R4"):
                    hits = 0
                    for _ in range(args.runs):
                        ok, mg = run_once(scn, v, sw, sb, mm, rng, args.still)
                        hits += ok
                        if mg is not None and ok:
                            margins.append(mg)
                    row.append(hits / args.runs)
                mg = sorted(margins)[len(margins) // 2] if margins else float("nan")
                print("%-14s %-12s %7.0f%% %9.1f %7.0f%% %7.0f%% %7.0f%%" % (nname, sname, row[0] * 100, mg,
                                                                           row[1] * 100, row[2] * 100, row[3] * 100))
        print()


if __name__ == "__main__":
    main()
