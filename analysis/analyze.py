#!/usr/bin/env python3
# analysis/analyze.py — 서버 기록 폴더 하나 → 실험별 지표 표 (report.md + CSV)
#
#   python3 analysis/analyze.py                        # 가장 최근 logs/server_* 폴더
#   python3 analysis/analyze.py logs/server_1122_101500
#   python3 analysis/analyze.py logs/server_1122_101500 --pi pi_logs/pi_1122_101210   # Pi 녹화 영상 목록도 붙임
#
# 지표 정의 (README '지표 정의'와 같음)
#   V1 Cam1 비전   : 모드별 FPS, 단계별 시간 P50/P95 (Flow·ROI·Pack·YOLO·Track·Homo·전체), 전체 화면 스캔 비율,
#                    추론 생략 비율, ROI 수·면적·Packing 배율, 검출률, 고유 ID 수, 촬영→서버 지연
#   H0 바닥 보정   : 점 위에 선 사람의 추정 위치 오차 P50/P95, 검출률
#   V2 Cam2        : depart(출발) 시험에서 emr_high 가 cross(경계 넘음)보다 먼저 나온 비율과 선행 시간,
#                    wait/sway 시험에서 emr_high 가 나온 비율(오탐), 분석 시간
#   G0 GPS 정지    : 점 위에 둔 폰의 GPS 바닥 위치 오차 (원본 / 맵 매칭)
#   C1 충돌 경고   : 위험 감지 성공률 = 경고 나와야 하는 시험 중 나온 비율, 오경고율 = 나오면 안 되는 시험 중 나온 비율,
#                    선행 시간 = 보행자 차도 진입(연석선 y<0) - 첫 경고 전송,
#                    정지선까지 = 첫 경고 순간 차량 위치·속도로 계산한 정지선(floor.stop_x)까지 남은 시간,
#                    전달 지연 = 촬영→서버판정→전송(서버) + 서버→폰(앱 회신, 폰 시계 보정)

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p50(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    return round(float(s.median()), 2) if len(s) else None


def p95(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    return round(float(np.percentile(s, 95)), 2) if len(s) else None


def xy(label):
    m = re.search(r"x\s*=\s*(-?[\d.]+)\s*,?\s*y\s*=\s*(-?[\d.]+)", str(label))
    return (float(m.group(1)), float(m.group(2))) if m else None


def nperson(label):
    m = re.search(r"(\d+)\s*명", str(label))
    return int(m.group(1)) if m else 1


def read(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in df.columns:
        if df[c].dtype == object and set(df[c].dropna().astype(str).unique()) <= {"True", "False"}:
            df[c] = df[c].astype(str) == "True"
    return df


def md(df, max_rows=200):
    if df is None or len(df) == 0:
        return "_(데이터 없음)_\n"
    df = df.head(max_rows)
    cols = list(df.columns)
    out = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        vals = []
        for v in r:
            if isinstance(v, float) and math.isnan(v):
                vals.append("")
            else:
                vals.append(str(v))
        out += "| " + " | ".join(vals) + " |\n"
    return out


def in_trial(df, t, col="ts_cap"):
    if len(df) == 0:
        return df
    if "trial_id" in df.columns:
        return df[df.trial_id == t.trial_id]
    return df[(df[col] >= t.start_ms) & (df[col] <= t.stop_ms)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", nargs="?")
    ap.add_argument("--pi", help="Pi 기록 폴더 (녹화 영상 목록용)")
    args = ap.parse_args()
    d = args.logdir or max(glob.glob(os.path.join(ROOT, "logs", "server_*")), key=os.path.getmtime)
    site = json.load(open(os.path.join(ROOT, "site.json"), encoding="utf-8"))
    curb = site["floor"]["curb_y"]
    stop_x = site["floor"].get("stop_x", site["floor"]["crosswalk_x"][0] - 0.5)
    cam_pos = site["floor"].get("cam1_pos")

    T = read(d, "trials.csv")
    F = read(d, "cam1_frames.csv")
    TR = read(d, "cam1_tracks.csv")
    C2 = read(d, "cam2_frames.csv")
    C2E = read(d, "cam2_events.csv")
    G = read(d, "gps.csv")
    D = read(d, "decisions.csv")
    A = read(d, "alerts.csv")
    K = read(d, "alert_acks.csv")
    if len(T) == 0:
        sys.exit("trials.csv 가 비어 있음 — 조작 페이지에서 시험을 시작·종료했는지 확인: " + d)
    out = os.path.join(d, "analysis")
    os.makedirs(out, exist_ok=True)
    rep = ["# BlindGuard-X 실물 테스트 결과", "", "기록 폴더: `%s`  ·  시험 %d개" % (d, len(T)), ""]

    # ── 시험 목록 ──
    tl = T[["trial_id", "exp", "label", "scenario", "source", "cam1_mode", "note", "result_note"]].copy()
    tl["길이(s)"] = ((T.stop_ms - T.start_ms) / 1000).round(1)
    rep += ["## 시험 목록", md(tl)]

    # ── V1 Cam1 비전 파이프라인 ──
    rows = []
    for _, t in T.iterrows():
        f = in_trial(F, t)
        if len(f) < 5:
            continue
        span = (f.ts_cap.max() - f.ts_cap.min()) / 1000
        tr = in_trial(TR, t)
        det = f[f.det_ran]
        n = nperson(t.label)
        pol = f["policy"].dropna().mode() if "policy" in f.columns else pd.Series([], dtype=object)
        r = {"trial": t.trial_id, "exp": t.exp, "label": t.label, "mode": f["mode"].mode().iloc[0],
             "YOLO실행": pol.iloc[0] if len(pol) else "every",
             "YOLO실행%": round(100 * float(f.det_ran.mean()), 1),
             "every": int(f.every.mode().iloc[0]), "frames": len(f),
             "FPS": round((len(f) - 1) / span, 2) if span > 0 else None}
        if "t_lk" not in f.columns:
            f = f.assign(t_lk=np.nan)
        for k, c in (("Flow", "t_flow"), ("ROI", "t_roi"), ("Pack", "t_pack"), ("YOLO", "t_infer"), ("LK", "t_lk"),
                     ("Track", "t_track"), ("Homo", "t_homo"), ("전체", "t_total")):
            r[k + " P50"] = p50(f[c])
            r[k + " P95"] = p95(f[c])
        r.update({
            "전체스캔%": round(100 * float(f.full_scan.mean()), 1),
            "추론생략%": round(100 * float((det.t_infer == 0).mean()), 1) if len(det) else None,
            "ROI수": p50(f.n_rois), "ROI면적%": round(100 * float(f.roi_area.mean()), 1) if f.roi_area.notna().any() else None,
            "Pack배율": p50(f.pack_scale), "fallback": int(f.fallback.notna().sum()),
            "검출률": round(float((det.n_peds >= 1).mean()), 3) if len(det) else None,
            "고유ID": int(tr.track_id.nunique()) if len(tr) else 0, "인원(라벨)": n,
            "촬영→서버 P50": p50(f.recv_ms - f.ts_cap), "촬영→서버 P95": p95(f.recv_ms - f.ts_cap),
            "버린프레임": int(f.dropped.sum()), "최고온도": f.temp_c.max(),
        })
        rows.append(r)
    v1 = pd.DataFrame(rows)
    if len(v1):
        v1.to_csv(os.path.join(out, "V1_cam1_by_trial.csv"), index=False)
        by = v1.groupby(["mode", "YOLO실행"]).agg(
            시험수=("trial", "count"), FPS=("FPS", "median"), YOLO실행률=("YOLO실행%", "median"),
            YOLO_P50=("YOLO P50", "median"), LK_P50=("LK P50", "median"),
            Flow_P50=("Flow P50", "median"), Pack_P50=("Pack P50", "median"), 전체_P50=("전체 P50", "median"),
            전체_P95=("전체 P95", "median"), 추론생략=("추론생략%", "median"), 검출률=("검출률", "median"),
            촬영서버_P50=("촬영→서버 P50", "median")).reset_index()
        by.to_csv(os.path.join(out, "V1_cam1_by_mode.csv"), index=False)
        rep += ["## V1 · Cam1 비전 파이프라인 (모드별 중앙값)", "포스터 구조 = PACKED. 시뮬 실측: YOLO 26.2~28.0 ms, 9.3~9.8 FPS", "",
                md(by), "", "### 시험별", md(v1[v1.exp == "V1"] if (v1.exp == "V1").any() else v1)]

    # ── H0 바닥 보정 확인 ──
    rows = []
    for _, t in T[T.exp == "H0"].iterrows():
        gt = xy(t.label)
        tr = in_trial(TR, t)
        f = in_trial(F, t)
        f = f[f.det_ran] if len(f) else f
        r = {"trial": t.trial_id, "label": t.label, "frames": len(f),
             "검출률": round(float((f.n_peds >= 1).mean()), 3) if len(f) else None}
        if gt and len(tr):
            e = np.hypot(tr.x_m - gt[0], tr.y_m - gt[1])
            best = e.groupby(tr.frame).min()
            r.update({"위치오차 P50(m)": p50(best), "위치오차 P95(m)": p95(best)})
            if cam_pos:
                r["카메라거리(m)"] = round(math.hypot(gt[0] - cam_pos[0], gt[1] - cam_pos[1]), 2)
        rows.append(r)
    h0 = pd.DataFrame(rows)
    if len(h0):
        h0.to_csv(os.path.join(out, "H0_floor.csv"), index=False)
        rep += ["## H0 · 바닥 보정 (사람 발 위치 오차)", "시뮬 실측 평균 위치오차 0.203 m", "", md(h0)]

    # ── V2 Cam2 ──
    rows = []
    for _, t in T[T.exp == "V2"].iterrows():
        ev = in_trial(C2E, t)
        c2 = in_trial(C2, t)
        hi = ev[ev.event == "emr_high"].ts_cap.min() if len(ev) else np.nan
        cr = ev[ev.event == "cross"].ts_cap.min() if len(ev) else np.nan
        kind = str(t.label).lower()
        r = {"trial": t.trial_id, "label": t.label, "frames": len(c2), "EMR최고": c2.emr.max() if len(c2) else None,
             "emr_high": not np.isnan(hi), "cross": not np.isnan(cr),
             "선행(s)": round((cr - hi) / 1000, 2) if not (np.isnan(hi) or np.isnan(cr)) else None,
             "분석 P50(ms)": p50(c2.t_total) if len(c2) else None, "FPS": p50(c2.fps) if len(c2) else None}
        if "depart" in kind or "출발" in kind:
            r["판정"] = "성공(미리)" if (r["emr_high"] and r["cross"] and hi <= cr) else (
                "늦음" if (r["emr_high"] and r["cross"]) else ("놓침" if r["cross"] else "넘음 없음"))
        else:
            r["판정"] = "오탐" if r["emr_high"] else "정상"
        rows.append(r)
    v2 = pd.DataFrame(rows)
    if len(v2):
        v2.to_csv(os.path.join(out, "V2_cam2.csv"), index=False)
        dep = v2[v2.label.astype(str).str.contains("depart|출발", case=False)]
        wt = v2[~v2.label.astype(str).str.contains("depart|출발", case=False)]
        summ = pd.DataFrame([{
            "출발 미리 감지": "%d/%d" % ((dep["판정"] == "성공(미리)").sum(), len(dep)) if len(dep) else "-",
            "선행 P50(s)": p50(dep["선행(s)"]) if len(dep) else None,
            "대기·서성임 오탐": "%d/%d" % ((wt["판정"] == "오탐").sum(), len(wt)) if len(wt) else "-",
            "분석 P50(ms)": p50(v2["분석 P50(ms)"])}])
        rep += ["## V2 · Cam2 대기구역 (EMR [임시 식])", md(summ), "", md(v2)]

    # ── G0 GPS 정지 오차 ──
    rows = []
    for _, t in T[T.exp == "G0"].iterrows():
        gt = xy(t.label)
        g = in_trial(G, t, "recv_ms")
        g = g[g.raw_x.notna()] if len(g) else g
        r = {"trial": t.trial_id, "label": t.label, "fixes": len(g), "정확도(폰) P50": p50(g.acc) if len(g) else None}
        if gt and len(g):
            r["원본 오차 P50"] = p50(np.hypot(g.raw_x - gt[0], g.raw_y - gt[1]))
            r["원본 오차 P95"] = p95(np.hypot(g.raw_x - gt[0], g.raw_y - gt[1]))
            r["맵매칭 오차 P50"] = p50(np.hypot(g.est_x - gt[0], g.est_y - gt[1]))
        rows.append(r)
    g0 = pd.DataFrame(rows)
    if len(g0):
        g0.to_csv(os.path.join(out, "G0_gps_static.csv"), index=False)
        rep += ["## G0 · GPS 정지 오차 (m)", md(g0)]

    # ── C1 충돌 경고 ──
    rows, lat = [], []
    for _, t in T[T.exp == "C1"].iterrows():
        try:
            exp = json.loads(t.expected) if isinstance(t.expected, str) else {}
        except ValueError:
            exp = {}
        a = A[(A.trial_id == t.trial_id) & A.level.notna()] if len(A) else pd.DataFrame()
        dd = in_trial(D, t, "now_ms")
        selfs = set(dd.self_track.dropna().astype(int)) if len(dd) else set()
        tr = in_trial(TR, t)
        tr = tr[(tr.ts_cap >= t.go_ms) & (~tr.track_id.isin(selfs))] if len(tr) else tr
        entry = tr[tr.y_m < curb].ts_cap.min() if len(tr) else np.nan
        g = in_trial(G, t, "recv_ms")
        for vid, should in (exp or {"A": None}).items():
            av = a[a.vehicle == vid] if len(a) else a
            got = len(av) > 0
            first = av.sent_ms.min() if got else np.nan
            dv = dd[dd.vehicle == vid] if len(dd) else dd
            tts = np.nan   # 첫 경고 순간 차량 위치·속도로 계산한 '정지선까지 남은 시간'
            if got and len(dv):
                at = dv[dv.now_ms <= first + 1].tail(1)
                if len(at) and at.veh_vx.iloc[0] and at.veh_vx.iloc[0] > 0.3:
                    tts = (stop_x - at.veh_x.iloc[0]) / at.veh_vx.iloc[0]
            order = {"caution": 1, "warning": 2, "danger": 3}
            peak = max(av.level, key=lambda x: order.get(x, 0)) if got else None
            if should is None:
                verdict = "경고" if got else "없음"
            else:
                verdict = ("성공" if got else "놓침") if should else ("오경고" if got else "정상")
            rows.append({
                "trial": t.trial_id, "scenario": t.scenario, "source": t.source, "label": t.label, "차량": vid,
                "기대": {True: "경고", False: "없음", None: "-"}[should], "판정": verdict, "최고단계": peak,
                "첫경고 t(s)": round((first - t.go_ms) / 1000, 2) if got else None,
                "보행자진입-경고(s)": round((entry - first) / 1000, 2) if (got and not np.isnan(entry)) else None,
                "정지선까지(s)": round(float(tts), 2) if not np.isnan(tts) else None,
                "GPS±m P50": p50(g.acc) if len(g) else None,
                "GPS-카메라 거리 P50": p50(dv.self_dist) if len(dv) else None,
                "판정불가%": round(100 * float(dv.status.isin(["UNKNOWN", "NO_VEHICLE"]).mean()), 1) if len(dv) else None,
                "결과메모": t.result_note})
            for _, x in av.iterrows():
                if pd.isna(x.ts_cap):
                    continue
                ks = K[K.seq == x.seq] if len(K) else K
                one = ks.sort_values("recv_ms").one_way_ms.iloc[0] if len(ks) else np.nan
                fr = F[(F.trial_id == t.trial_id) & (F.frame == x.frame) & (F.ts_cap == x.ts_cap)] if len(F) else F
                recv = fr.recv_ms.iloc[0] if len(fr) else np.nan
                lat.append({"trial": t.trial_id, "seq": x.seq, "level": x.level,
                            "촬영→서버": recv - x.ts_cap, "서버판정→전송": x.sent_ms - recv,
                            "촬영→전송": x.sent_ms - x.ts_cap, "서버→폰": one,
                            "촬영→폰": (x.sent_ms - x.ts_cap + one) if not pd.isna(one) else np.nan})
    c1 = pd.DataFrame(rows)
    if len(c1):
        c1.to_csv(os.path.join(out, "C1_conflict.csv"), index=False)
        pos = c1[c1["기대"] == "경고"]
        neg = c1[c1["기대"] == "없음"]
        summ = pd.DataFrame([{
            "위험 감지 성공률": "%d/%d (%.0f%%)" % ((pos["판정"] == "성공").sum(), len(pos),
                                             100 * (pos["판정"] == "성공").mean()) if len(pos) else "-",
            "오경고율": "%d/%d (%.0f%%)" % ((neg["판정"] == "오경고").sum(), len(neg),
                                        100 * (neg["판정"] == "오경고").mean()) if len(neg) else "-",
            "경고 때 정지선까지 P50(s)": p50(pos["정지선까지(s)"]) if len(pos) else None,
            "보행자 진입 전 P50(s)": p50(pos["보행자진입-경고(s)"]) if len(pos) else None,
            "GPS±m P50": p50(c1["GPS±m P50"]), "GPS-카메라 P50(m)": p50(c1["GPS-카메라 거리 P50"])}])
        bysc = c1.groupby(["scenario", "기대"])["판정"].value_counts().unstack(fill_value=0).reset_index()
        rep += ["## C1 · 충돌 경고 End-to-End (차량 = 폰 GPS, 보행자 = Cam1)", md(summ), "", "### 시나리오별", md(bysc),
                "", "### 시험별", md(c1)]
        L = pd.DataFrame(lat)
        if len(L):
            L.to_csv(os.path.join(out, "C1_latency.csv"), index=False)
            ls = pd.DataFrame([{k: "%s / %s" % (p50(L[k]), p95(L[k])) for k in
                                ["촬영→서버", "서버판정→전송", "촬영→전송", "서버→폰", "촬영→폰"]}])
            rep += ["", "### 경고 전달 지연 (ms, P50 / P95)",
                    "서버→폰 = 앱이 받은 즉시 보낸 회신 시각(폰 시계를 ping 으로 보정) - 서버 전송 시각. 화면 표시까지는 +수십 ms",
                    "", md(ls)]

    # ── 전체 성능 요약 ──
    if len(F):
        rep += ["## 전체 Cam1 처리 요약 (모든 프레임)",
                md(pd.DataFrame([{"프레임": len(F), "FPS(중앙)": p50(F.fps), "YOLO P50": p50(F.t_infer),
                                  "전체 P50": p50(F.t_total), "전체 P95": p95(F.t_total),
                                  "촬영→서버 P50": p50(F.recv_ms - F.ts_cap), "촬영→서버 P95": p95(F.recv_ms - F.ts_cap),
                                  "최고온도": F.temp_c.max()}]))]
    if len(C2):
        rep += ["## 전체 Cam2 처리 요약",
                md(pd.DataFrame([{"프레임": len(C2), "FPS(중앙)": p50(C2.fps), "MOG2 P50": p50(C2.t_mog),
                                  "LK P50": p50(C2.t_lk), "전체 P50": p50(C2.t_total), "전체 P95": p95(C2.t_total)}]))]

    if args.pi:
        vids = sorted(glob.glob(os.path.join(args.pi, "video", "*_overlay.avi")))
        rep += ["## 녹화 영상 (Pi)", "\n".join("- `%s`" % os.path.relpath(v, args.pi) for v in vids) or "_(없음)_"]

    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("\n저장:", out)


if __name__ == "__main__":
    main()
