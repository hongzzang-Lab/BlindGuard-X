#!/usr/bin/env python3
# analysis/pi_modes.py — 같은 원본 영상을 Pi에서 모드별로 다시 돌린 결과 비교 (V1 공정 비교)
#
# 현장 V1 시험에서 '원본 영상도 녹화'를 켜 두면 Pi에 <시험>_cam1_raw.avi 가 생긴다.
# 집에 와서(또는 현장 쉬는 시간에) Pi에서 같은 영상을 4가지 모드로 다시 돌린다:
#
#   bash tools/replay_modes_pi.sh logs/pi_XXXX/video/T005_V1_cam1_raw.avi
#     → logs/pi_*_replay_<모드>/ 폴더 4개
#   python3 analysis/pi_modes.py logs/pi_*_replay_*          (Pi 또는 노트북에서; 폴더 여러 개)
#
# 같은 장면이므로 모드 차이(속도·검출·ID)가 공정하게 비교된다. 처리 시간은 Pi에서 돈 실측값.

import json
import os
import sys

import numpy as np
import pandas as pd


def p(s, q):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return round(float(np.percentile(s, q)), 2) if len(s) else None


def main():
    dirs = [d for d in sys.argv[1:] if os.path.isdir(d)]
    if not dirs:
        sys.exit("사용법: python3 analysis/pi_modes.py logs/pi_*_replay_*")
    rows = []
    for d in dirs:
        f = pd.read_csv(os.path.join(d, "cam1_frames.csv"))
        tr = pd.read_csv(os.path.join(d, "cam1_tracks.csv"))
        a = json.load(open(os.path.join(d, "args.json")))
        if len(f) < 3:
            continue
        span = (f.ts_cap_local.max() - f.ts_cap_local.min()) / 1000
        det = f[f.det_ran.astype(str) == "True"]
        rows.append({
            "폴더": os.path.basename(d), "모드": f["mode"].mode().iloc[0],
            "YOLO실행": (f["policy"].dropna().mode().iloc[0] if "policy" in f.columns and f["policy"].notna().any() else "every"),
            "YOLO실행%": round(100 * float((f.det_ran.astype(str) == "True").mean()), 1),
            "입력": os.path.basename(a.get("cam1") or ""),
            "프레임": len(f), "FPS": round((len(f) - 1) / span, 2) if span > 0 else None,
            "YOLO P50": p(f.t_infer, 50), "YOLO P95": p(f.t_infer, 95), "Flow P50": p(f.t_flow, 50),
            "Pack P50": p(f.t_pack, 50), "LK P50": p(f.t_lk, 50) if "t_lk" in f.columns else None,
            "전체 P50": p(f.t_total, 50), "전체 P95": p(f.t_total, 95),
            "추론생략%": round(100 * float((det.t_infer == 0).mean()), 1) if len(det) else None,
            "검출률": round(float((det.n_peds >= 1).mean()), 3) if len(det) else None,
            "사람-프레임 수": len(tr), "고유 ID": int(tr.track_id.nunique()) if len(tr) else 0,
        })
    if not rows:
        sys.exit("결과가 있는 폴더가 없음 (cam1_frames.csv 가 비었음 — 모델 오류였는지 events.csv 확인)")
    R = pd.DataFrame(rows).sort_values(["입력", "모드", "YOLO실행"])
    print(R.to_string(index=False))
    out = os.path.join(os.path.dirname(os.path.abspath(dirs[0])), "pi_modes_summary.csv")
    R.to_csv(out, index=False)
    print("\n저장:", out)


if __name__ == "__main__":
    main()
