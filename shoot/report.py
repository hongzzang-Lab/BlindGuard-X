#!/usr/bin/env python3
# shoot/report.py — 촬영 기록을 표로 정리 (Pi 또는 노트북에서 실행)
#
#   python3 shoot/report.py              # logs/ (Pi) 와 pi_logs/ (노트북으로 가져온 것) 의 촬영 기록 전부
#   python3 shoot/report.py 폴더1 폴더2  # 고른 기록만
#
# 세션 cam1 기록 → Cam1 표 (검출률·ID·거리·속도·FPS·지연),  세션 cam2 기록 → Cam2 표 (EMR·경계 넘음·FPS·지연)
# 각 기록에는 '함께 켜 둔 다른 카메라'의 FPS 도 같이 적음 (두 대 동시 분석 조건이었다는 근거).
# 거리·속도는 지금의 calib/H_cam1.json 으로 발 위치(픽셀)에서 다시 계산 → 보정을 나중에 해도 됨.
# 결과: 화면 출력 + shoot_report.csv / shoot_report.md (기록 폴더 옆)
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bgx.geometry import apply_h  # noqa: E402


def rows_of(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return None


def col(rows, k):
    return np.array([x for x in (num(r, k) for r in rows) if x is not None], float)


def pct(a, q):
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "1.0")


def load_H():
    for p in (os.path.join(os.environ.get("BGX_CALIB_DIR") or os.path.join(ROOT, "calib"), "H_cam1.json"), os.path.join(ROOT, "pi_logs", "calib_from_pi", "H_cam1.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return np.array(json.load(f)["H"], float), p
    return None, None


def cam1_stats(d, H):
    fr, tr = rows_of(os.path.join(d, "cam1_frames.csv")), rows_of(os.path.join(d, "cam1_tracks.csv"))
    if not fr:
        return {}
    ts = col(fr, "ts_cap_local") / 1000.0
    dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    npeds = col(fr, "n_peds")
    out = {"c1_frames": len(fr), "c1_sec": round(dur, 1),
           "c1_fps": round((len(fr) - 1) / dur, 1) if dur > 0 else None,
           "c1_ms_p50": pct(col(fr, "t_total"), 50), "c1_ms_p95": pct(col(fr, "t_total"), 95),
           "c1_yolo_ms_p50": pct(col([r for r in fr if truthy(r.get("det_ran"))], "t_infer"), 50),
           "c1_det_pct": round(100.0 * float((npeds >= 1).mean()), 1) if len(npeds) else None,
           "temp_max": float(col(fr, "temp_c").max()) if len(col(fr, "temp_c")) else None}
    drop = col(fr, "dropped")
    out["c1_dropped"] = int(drop.max()) if len(drop) else 0
    # 가장 큰 사람 하나를 따라가며 ID 가 몇 번 바뀌었는지, 거리·속도
    best = {}
    for r in tr:
        f, h = num(r, "frame"), (num(r, "y2") or 0) - (num(r, "y1") or 0)
        if f is None or num(r, "u") is None:
            continue
        if f not in best or h > best[f][0]:
            best[f] = (h, r)
    seq = [best[k][1] for k in sorted(best)]
    ids = [s["track_id"] for s in seq]
    out["c1_ids"] = len(set(ids))
    out["c1_id_switch"] = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
    if H is not None and seq:
        px = np.array([[num(s, "u"), num(s, "v")] for s in seq])
        w = apply_h(H, px)
        t = np.array([num(s, "ts_cap_local") for s in seq]) / 1000.0
        dist = np.hypot(w[:, 0], w[:, 1])
        ok = np.isfinite(dist) & (dist < 60)
        if ok.any():
            out["c1_dist_min"], out["c1_dist_max"] = round(float(dist[ok].min()), 1), round(float(dist[ok].max()), 1)
            sp = []
            j = 0
            for i in range(len(t)):                       # 약 1초 간격 이동 거리 → 속도
                while j < len(t) and t[j] - t[i] < 1.0:
                    j += 1
                if j < len(t) and t[j] - t[i] < 1.5 and ok[i] and ok[j]:
                    sp.append(float(np.hypot(*(w[j] - w[i])) / (t[j] - t[i])))
            mv = [s for s in sp if s > 0.3]
            out["c1_speed_med"] = round(float(np.median(mv)), 2) if mv else None
    return out


def cam2_stats(d):
    fr = rows_of(os.path.join(d, "cam2_frames.csv"))
    if not fr:
        return {}
    ts = col(fr, "ts_cap_local") / 1000.0
    dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    live = [r for r in fr if not truthy(r.get("warmup"))]
    ev = {}
    for r in fr:
        for e in (r.get("events") or "").split("|"):
            if e:
                ev[e] = ev.get(e, 0) + 1
    emr = col(live, "emr")
    return {"c2_frames": len(fr), "c2_fps": round((len(fr) - 1) / dur, 1) if dur > 0 else None,
            "c2_ms_p50": pct(col(fr, "t_total"), 50), "c2_ms_p95": pct(col(fr, "t_total"), 95),
            "c2_present_pct": round(100.0 * np.mean([truthy(r.get("present")) for r in live]), 1) if live else None,
            "c2_emr_max": round(float(emr.max()), 2) if len(emr) else None,
            "c2_emr_high": ev.get("emr_high", 0), "c2_cross": ev.get("cross", 0), "c2_back": ev.get("back", 0),
            "c2_enter": ev.get("enter", 0)}


def md_table(rows, cols):
    if not rows:
        return "(기록 없음)\n"
    s = "| " + " | ".join(h for h, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for _, k in cols) + " |\n"
    return s


def main():
    dirs = sys.argv[1:]
    if not dirs:
        for base in ("logs", "pi_logs"):
            dirs += sorted(glob.glob(os.path.join(ROOT, base, "pi_*_cam[12]_*")))
    if not dirs:
        sys.exit("촬영 기록이 없음 (logs/pi_*_cam1_* 또는 pi_logs/...). 노트북이면 먼저 bash tools/sync_pi.sh 사용자@Pi주소")
    H, hp = load_H()
    print("Cam1 보정:", hp or "없음 (거리·속도 칸은 비움)")
    allr = []
    for d in dirs:
        m = re.search(r"pi_(\d{4}_\d{6})_(cam1|cam2)_(.+)$", os.path.basename(os.path.normpath(d)))
        r = {"folder": os.path.basename(os.path.normpath(d)), "session": m.group(2) if m else "?",
             "name": m.group(3) if m else os.path.basename(d), "when": m.group(1) if m else ""}
        r.update(cam1_stats(d, H))
        r.update(cam2_stats(d))
        allr.append(r)
    c1 = [("이름", "name"), ("길이 s", "c1_sec"), ("검출률 %", "c1_det_pct"), ("ID 수", "c1_ids"),
          ("ID 바뀜", "c1_id_switch"), ("거리 최소 m", "c1_dist_min"), ("거리 최대 m", "c1_dist_max"),
          ("걷는 속도 m/s", "c1_speed_med"), ("Cam1 FPS", "c1_fps"), ("처리 P50 ms", "c1_ms_p50"),
          ("처리 P95 ms", "c1_ms_p95"), ("YOLO P50 ms", "c1_yolo_ms_p50"), ("같이 켠 Cam2 FPS", "c2_fps"),
          ("최고 온도 °C", "temp_max")]
    c2 = [("이름", "name"), ("사람 있음 %", "c2_present_pct"), ("EMR 최고", "c2_emr_max"),
          ("출발 징후 횟수", "c2_emr_high"), ("경계 넘음", "c2_cross"), ("되돌아옴", "c2_back"), ("Cam2 FPS", "c2_fps"),
          ("처리 P50 ms", "c2_ms_p50"), ("처리 P95 ms", "c2_ms_p95"), ("같이 켠 Cam1 FPS", "c1_fps"),
          ("최고 온도 °C", "temp_max")]
    md = "# 실물 Vision 촬영 결과\n\n## 세션 1 · Cam1 (횡단보도)\n\n" + md_table([r for r in allr if r["session"] == "cam1"], c1)
    md += "\n## 세션 2 · Cam2 (대기구역)\n\n" + md_table([r for r in allr if r["session"] == "cam2"], c2)
    md += ("\n- 검출률 = 사람이 한 명 이상 잡힌 프레임 비율. 사람이 화면에 계속 있었던 촬영에서만 의미가 있음\n"
           "- ID 바뀜 = 가장 크게 보이는 사람의 추적 번호가 바뀐 횟수 (한 명만 걸었다면 0 이 정상)\n"
           "- 거리 = Cam1 삼각대 아래 바닥에서 사람 발까지. '대략 몇 m' 로 만든 보정이라 참고값\n"
           "- 출발 징후 횟수 = EMR 이 기준을 넘은 횟수, 경계 넘음 = 대기구역 경계선을 차도 쪽으로 넘은 횟수\n")
    print("\n" + md)
    outdir = os.path.dirname(os.path.normpath(dirs[0]))
    keys = sorted({k for r in allr for k in r})
    with open(os.path.join(outdir, "shoot_report.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(allr)
    with open(os.path.join(outdir, "shoot_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("저장:", os.path.join(outdir, "shoot_report.csv"), "/ shoot_report.md")


if __name__ == "__main__":
    main()
