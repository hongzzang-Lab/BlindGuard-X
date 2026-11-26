#!/usr/bin/env python3
# analysis/replay.py — C1 충돌 경고를 '기록된 입력'으로 다시 판정해서 설정별로 비교
#
# 현장에서는 서버 기본 설정 하나로만 경고가 나간다. 이 스크립트는 같은 기록(카메라 트랙 + 폰 GPS)을
# 설정만 바꿔 다시 돌려서 "맵 매칭이 없었다면?", "치우침 보정을 안 했다면?" 같은 비교표를 만든다.
# (서버와 같은 bgx.engine 코드 사용 → 서버 기본 설정 결과는 실제 현장 결과와 거의 같아야 정상)
#
#   python3 analysis/replay.py                    # 가장 최근 logs/server_*
#   python3 analysis/replay.py logs/server_1122_101500

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bgx.engine import ConflictEngine, GpsVehicle  # noqa: E402
from bgx.risk import DEFAULT_RISK  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "analysis"))
from analyze import md, read  # noqa: E402


def variants(site):
    risk = dict(DEFAULT_RISK, **site.get("risk", {}))
    gps = dict(site.get("gps", {}), lane_y=site["floor"].get("lane_y", -1.75))
    return {
        "서버 기본": (risk, gps, "raw", True),
        "맵매칭·칼만 끔": (risk, dict(gps, map_match=False), "raw", True),
        "치우침 보정 없음": (risk, gps, "nobias", True),
        "보행자 정지판정 0.5": (dict(risk, ped_still_mps=0.5), gps, "raw", True),
        "안전반경 2.5 m": (dict(risk, safe_radius_m=2.5), gps, "raw", True),
        "자기 제외 끔": (risk, gps, "raw", False),
    }


def replay_trial(t, F, TR, G, device, risk, gps_cfg, coord, excl, excl_cfg):
    f = F[F.trial_id == t.trial_id].sort_values("recv_ms")
    if len(f) == 0:
        return None
    tr = TR[TR.trial_id == t.trial_id]
    by_frame = {k: v for k, v in tr.groupby(["frame", "ts_cap"])}
    g = G[(G.device_id == device) & (G.recv_ms >= t.start_ms - 15000) & (G.recv_ms <= t.stop_ms)].sort_values("recv_ms")
    xc, yc = ("raw_x", "raw_y") if coord == "raw" else ("nobias_x", "nobias_y")
    g = g[g[xc].notna()]
    gv = GpsVehicle("A", gps_cfg)
    eng = ConflictEngine(risk, risk.get("hold_s", 1.0))
    gi = 0
    rows_g = list(g.itertuples())
    first = None
    peak = None
    order = {None: 0, "caution": 1, "warning": 2, "danger": 3}
    for fr in f.itertuples():
        while gi < len(rows_g) and rows_g[gi].recv_ms <= fr.recv_ms:
            r = rows_g[gi]
            vd = None if pd.isna(r.dop_vx) else (r.dop_vx, r.dop_vy)
            gv.update(r.t_fix_ms / 1000.0, getattr(r, xc), getattr(r, yc), None if pd.isna(r.speed) else r.speed,
                      vd, None if pd.isna(r.acc) else r.acc)
            gi += 1
        peds = []
        sub = by_frame.get((fr.frame, fr.ts_cap))
        if sub is not None:
            for p in sub.itertuples():
                if pd.isna(p.x_m):
                    continue
                peds.append({"track_id": int(p.track_id), "x_m": p.x_m, "y_m": p.y_m, "vx": p.vx, "vy": p.vy,
                             "v_ok": bool(p.v_ok), "ts_cap": fr.ts_cap})
        st = gv.state(fr.recv_ms / 1000.0)
        res, ch, prev, new = eng.step("A", int(fr.recv_ms), st, peds, exclude_self=excl, excl_cfg=excl_cfg)
        if new and first is None and fr.recv_ms >= t.go_ms:
            first = fr.recv_ms
        if order[new] > order[peak] and fr.recv_ms >= t.go_ms:
            peak = new
    return first, peak


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(os.path.join(ROOT, "logs", "server_*")), key=os.path.getmtime)
    site = json.load(open(os.path.join(ROOT, "site.json"), encoding="utf-8"))
    T, F, TR, G = read(d, "trials.csv"), read(d, "cam1_frames.csv"), read(d, "cam1_tracks.csv"), read(d, "gps.csv")
    T = T[(T.exp == "C1") & (T.source == "gps")] if len(T) else T
    if len(T) == 0 or len(G) == 0:
        sys.exit("GPS 차량 C1 시험이 없음")
    devA = G[G.vehicle == "A"].device_id.mode()
    if len(devA) == 0:
        sys.exit("차량 A 폰 기록 없음")
    devA = devA.iloc[0]
    rows = []
    for name, (risk, gcfg, coord, excl) in variants(site).items():
        for _, t in T.iterrows():
            exp = json.loads(t.expected).get("A") if isinstance(t.expected, str) else None
            r = replay_trial(t, F, TR, G, devA, risk, gcfg, coord, excl, site.get("self_exclusion", {}))
            if r is None:
                continue
            first, peak = r
            got = first is not None
            rows.append({"설정": name, "trial": t.trial_id, "scenario": t.scenario, "기대": exp, "경고": got,
                         "최고": peak, "첫경고 t(s)": round((first - t.go_ms) / 1000, 2) if got else None})
    R = pd.DataFrame(rows)
    out = os.path.join(d, "analysis")
    os.makedirs(out, exist_ok=True)
    R.to_csv(os.path.join(out, "C1_replay_variants.csv"), index=False)
    summ = []
    for name, g in R.groupby("설정", sort=False):
        pos, neg = g[g["기대"] == True], g[g["기대"] == False]  # noqa: E712
        summ.append({"설정": name,
                     "성공률": "%d/%d" % (pos["경고"].sum(), len(pos)) if len(pos) else "-",
                     "오경고율": "%d/%d" % (neg["경고"].sum(), len(neg)) if len(neg) else "-",
                     "첫경고 P50(s)": round(float(np.nanmedian(pos["첫경고 t(s)"].astype(float))), 2)
                     if len(pos) and pos["경고"].any() else None})
    S = pd.DataFrame(summ)
    txt = "## C1 다시 판정 (같은 기록, 설정만 바꿈)\n\n" + md(S) + "\n### 시험별\n" + md(R)
    with open(os.path.join(out, "C1_replay.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print("저장:", out)


if __name__ == "__main__":
    main()
