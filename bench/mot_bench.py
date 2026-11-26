#!/usr/bin/env python3
# bench/mot_bench.py — 공개 벤치마크(MOTChallenge MOT15, 고정 카메라 구간)에 Cam1 Vision 방식을 그대로 돌려
#                      '정확도(추적 표준 지표)'와 '처리 시간(이 장치에서)'을 방식별로 비교한다.
#
#   Pi 에서:   cd ~/realtest && source ~/bgx-venv/bin/activate && python3 bench/mot_bench.py
#
# 비교하는 방식 (검출 모델·추적기·설정은 모두 같고, 'YOLO 에 화면의 어느 부분을 넣느냐'만 다름)
#   full     : 프레임 전체를 416 으로 줄여 YOLO 1회                      ← 기준선
#   adaptive : 움직임+추적 위치를 모두 감싼 큰 ROI 하나를 잘라 YOLO 1회   ← 포스터 표의 ADAPTIVE
#   crops    : 사람(ROI)마다 따로 잘라 YOLO 를 ROI 수만큼                 ← 포스터 표의 OURS
#   packed   : 사람(ROI)들을 416×416 한 장에 모아 YOLO 1회                ← 포스터 표의 OURS_PACKED (최종 구조)
#
# 원칙
#   - 벤치마크에 맞춘 조정은 하지 않는다. Cam1 설정은 시뮬 데모(simdemo/site_sim.json)에서 쓰던 값 그대로.
#   - 고정 카메라 학습용 구간을 미리 정한 목록 전부 돌리고, 전부 보고한다 (유리한 구간만 고르지 않음).
#   - 정답(GT)은 벤치마크가 준 것. 지표는 CLEAR MOT(IoU 0.5) 와 IDF1 — 이 스크립트가 직접 계산 (공식 평가 도구와
#     소수점 수준 차이가 날 수 있음. 방식끼리의 상대 비교용).
#   - 처리 시간은 '이 장치에서 Cam1Pipeline.process() 한 번'의 시간 (영상 읽기 제외).

import argparse
import configparser
import csv
import json
import math
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bgx.cam1_pipeline import Cam1Pipeline  # noqa: E402

# 벤치마크별 설정. seqs = 학습용(정답 공개) 구간 중 고정 카메라인 것: 이름 → (fps, 설명). fps 는 seqinfo.ini 가 없을 때만 씀
BENCH = {
    "mot15": {
        "title": "MOTChallenge MOT15 (2DMOT2015) train, 고정 카메라 구간", "fmt": "mot15", "sub": "",
        "seqs": {
            "PETS09-S2L1": (7, "높은 곳의 고정 카메라, 보행자 최대 8명 (우리 설치 조건과 가장 비슷)"),
            "TUD-Stadtmitte": (25, "약 2 m 높이 고정 카메라, 거리 보행자"),
            "TUD-Campus": (25, "옆에서 본 보행자, 짧은 구간"),
            "KITTI-17": (10, "고정 카메라, 맑은 날 보행자"),
            "Venice-2": (30, "광장, 1920×1080, 사람 많음 (프레임당 약 12명)"),
            "ADL-Rundle-6": (30, "낮은 각도 보행자 거리, 1920×1080, 프레임당 약 10명"),
        },
        "modes": "full,adaptive,crops,packed", "max_frames": 0,
    },
    "mot17": {
        "title": "MOTChallenge MOT17 train, 고정 카메라 구간", "fmt": "mot16", "sub": "mot17",
        "seqs": {
            "MOT17-02": (30, "광장, 1920×1080, 프레임당 약 30명"),
            "MOT17-04": (30, "밤 보행자 거리, 높은 곳에서, 1920×1080, 프레임당 약 45명"),
            "MOT17-09": (30, "상가 앞 낮은 각도, 1920×1080, 프레임당 약 10명"),
        },
        "modes": "full,adaptive,crops,packed", "max_frames": 600,
    },
    "mot20": {
        "title": "MOTChallenge MOT20 train (전부 고정 카메라, 매우 붐빔)", "fmt": "mot16", "sub": "mot20",
        "seqs": {
            "MOT20-01": (25, "역 실내, 매우 붐빔"),
            "MOT20-02": (25, "역 실내, 매우 붐빔"),
            "MOT20-03": (25, "밤 광장 출구, 매우 붐빔"),
            "MOT20-05": (25, "밤 광장, 매우 붐빔"),
        },
        # crops(사람마다 따로 추론)는 뺌: 프레임당 100명이 넘어 Pi 에서 프레임 하나에 수 초가 걸림
        "modes": "full,adaptive,packed", "max_frames": 600, "extra_distractor": (6,),
    },
}
BENCH["sompt22"] = {
    # 감시용 고정 카메라(6~8 m 기둥) 보행자 추적 데이터셋. 구글 드라이브 배포라 자동으로 받지 않음:
    # 노트북에서 받아 Pi 의 bench/data/sompt22/ 아래에 풀어 두면, 정답(gt)이 있는 구간 폴더를 스스로 찾음
    "title": "SOMPT22 train (감시용 고정 카메라, 6~8 m 기둥)", "fmt": "auto", "sub": "sompt22", "seqs": {},
    "modes": "full,adaptive,crops,packed", "max_frames": 300, "dynamic": True,
}
SEQS = BENCH["mot15"]["seqs"]
PRIMARY = "PETS09-S2L1"
VID_EXT = (".mp4", ".avi", ".mkv", ".mov", ".m4v")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
URLS = {
    "mot15": ["https://motchallenge.net/data/2DMOT2015.zip", "https://motchallenge.net/data/MOT15.zip",
              "https://motchallenge.net/data/2D_MOT_2015.zip"],
    "mot16": ["https://motchallenge.net/data/MOT16.zip"],
    "mot17labels": ["https://motchallenge.net/data/MOT17Labels.zip"],
    "mot17": ["https://motchallenge.net/data/MOT17.zip"],
    "mot20": ["https://motchallenge.net/data/MOT20.zip"],
}
# MOT16/17/20 정답 파일의 분류 번호. 1 = 보행자(평가 대상). 아래는 '헷갈리는 대상': 여기에 맞은 검출은 오검출로 세지 않고 뺌
DISTRACTOR = (2, 7, 8, 12)          # 탈것 위의 사람, 가만히 있는 사람, 사람 모양 물체, 반사된 모습 (공식 평가 도구와 같은 처리)
MODE_KO = {"full": "FULL (전체 화면)", "fixed": "FIXED", "adaptive": "ADAPTIVE", "crops": "OURS (ROI별 추론)",
           "packed": "OURS_PACKED"}
SIZE_BINS = ((0, 60, "small"), (60, 120, "mid"), (120, 1e9, "large"))     # GT 상자 높이(px) — 미리 정한 구간


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ───────────── 데이터 ─────────────
def download(url, dst):
    import urllib.request
    tmp = Path(str(dst) + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **({"Range": "bytes=%d-" % have} if have else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        if have and r.status != 206:
            have = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        if total < 50e6:
            raise RuntimeError("받은 크기가 너무 작음 (%d bytes) — 주소가 바뀐 듯" % total)
        log("[받기] %s  %.0f MB%s" % (url, total / 1e6, " (이어받기 %.0f MB 부터)" % (have / 1e6) if have else ""))
        t0, last = time.time(), 0
        with open(tmp, "ab" if have else "wb") as f:
            n = have
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                n += len(b)
                if n - last > total / 20:
                    last = n
                    sp = (n - have) / max(1e-6, time.time() - t0) / 1e6
                    log("[받기] %3.0f%%  %.0f / %.0f MB  (%.1f MB/s, 남은 시간 약 %.0f 분)" %
                        (100 * n / total, n / 1e6, total / 1e6, sp, (total - n) / 1e6 / max(sp, 0.01) / 60))
    tmp.rename(dst)


def free_gb(path):
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize / 1e9


def fetch(kind, data, need_gb, allow_download):
    """압축 파일 준비 (있으면 그대로). 공간이 모자라면 이 스크립트가 받아 둔 다른 압축 파일(이미 풀어 놓은 것)을 지워 자리를 냄."""
    names = [u.rsplit("/", 1)[1] for u in URLS[kind]]
    for n in names:
        if (data / n).exists() and (data / n).stat().st_size > 5e6:
            return data / n
    if not allow_download:
        return None
    if free_gb(data) < need_gb:
        for z in sorted(data.glob("*.zip")):
            log("[공간] 남은 %.1f GB < 필요 %.1f GB → 이미 풀어 둔 %s (%.1f GB) 삭제" % (
                free_gb(data), need_gb, z.name, z.stat().st_size / 1e9))
            z.unlink()
            if free_gb(data) >= need_gb:
                break
    if free_gb(data) < need_gb:
        log("[공간] 저장 공간 부족: 남은 %.1f GB, 필요 약 %.1f GB → 이 받기는 건너뜀" % (free_gb(data), need_gb))
        return None
    for u, n in zip(URLS[kind], names):
        try:
            download(u, data / n)
            return data / n
        except Exception as e:  # noqa: BLE001
            log("[받기] 실패 %s → %s" % (u, e))
    return None


def extract(zp, folder, out, max_frames=0, want=("img1/", "gt/", "seqinfo.ini")):
    """zp 안의 .../train/<folder>/ 에서 필요한 것만 out/ 으로. max_frames 가 있으면 앞쪽 프레임만. 푼 파일 수를 돌려줌."""
    key = "/train/%s/" % folder
    n_out = 0
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            full = "/" + n
            if key not in full or n.endswith("/"):
                continue
            rel = full.split(key, 1)[1]
            if not any(rel.startswith(w) for w in want):
                continue
            if max_frames and rel.startswith("img1/"):
                stem = Path(rel).stem
                if stem.isdigit() and int(stem) > max_frames:
                    continue
            dst = out / rel
            if dst.exists() and dst.stat().st_size > 0:
                n_out += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as f:
                f.write(src.read())
            n_out += 1
    return n_out


def n_frames(d):
    return len(list((d / "img1").glob("*.jpg"))) if (d / "img1").exists() else 0


def ensure_data(data, bench, seqs, allow_download, max_frames=0):
    """구간 폴더(data/<구간>/img1, gt/gt.txt)를 준비. 돌려주는 값 = 정답 출처 설명 (보고서에 적음)."""
    data.mkdir(parents=True, exist_ok=True)

    def ready(s):
        d = data / s
        return (d / "gt" / "gt.txt").exists() and n_frames(d) >= (max_frames or 1)

    note_f = data / "gt_source.txt"
    missing = [s for s in seqs if not ready(s)]
    if not missing:
        return note_f.read_text().strip() if note_f.exists() else ""
    note = ""
    if bench == "mot15":
        zp = next((p for p in sorted(data.glob("*.zip")) if p.stat().st_size > 50e6), None) or \
            fetch("mot15", data, 3.5, allow_download)
        if zp is None:
            raise SystemExit("벤치마크 데이터를 받지 못함.\n  노트북 브라우저에서 https://motchallenge.net/data/MOT15/ 의 "
                             "'Get all data' 로 2DMOT2015.zip 을 받아 Pi 의 %s/ 에 넣고 다시 실행" % data)
        for s in missing:
            n = extract(zp, s, data / s)
            log("[풀기] %s: 파일 %d개" % (s, n) if n else "[풀기][주의] 압축 파일에 %s 가 없음 → 이 구간은 건너뜀" % s)
    elif bench == "mot17":
        # 영상은 MOT16 과 같음 (MOT17 = MOT16 영상 + 더 정확한 정답). 작은 쪽으로: MOT16.zip(1.9 GB) + MOT17 정답만(약 10 MB)
        zi = fetch("mot16", data, 3.5, allow_download)
        zl = fetch("mot17labels", data, 0.2, allow_download) if zi else None
        if zi is not None:
            for s in missing:
                n = extract(zi, s.replace("MOT17", "MOT16"), data / s, max_frames)
                got = 0
                if zl is not None:
                    (data / s / "gt" / "gt.txt").unlink(missing_ok=True)
                    for suf in ("-FRCNN", "-SDP", "-DPM"):
                        got = extract(zl, s + suf, data / s, want=("gt/",))
                        if got:
                            break
                    if not got:                       # 정답 묶음에 없으면 MOT16 정답으로 되돌림
                        extract(zi, s.replace("MOT17", "MOT16"), data / s, want=("gt/",))
                log("[풀기] %s: 파일 %d개 (정답 = %s)" % (s, n, "MOT17" if got else "MOT16"))
                note = "MOT16 영상 + MOT17 정답" if got else "MOT16 영상 + MOT16 정답 (MOT17 정답 파일을 받지 못함 → 결과는 MOT16 로 불러야 함)"
        else:
            zf = fetch("mot17", data, 7.5, allow_download)
            if zf is None:
                raise SystemExit("MOT17 데이터를 받지 못함 (MOT16.zip · MOT17.zip 모두 실패 또는 공간 부족)")
            for s in missing:
                n = extract(zf, s + "-FRCNN", data / s, max_frames)
                log("[풀기] %s: 파일 %d개" % (s, n))
            note = "MOT17.zip"
    elif bench == "mot20":
        zp = fetch("mot20", data, 6.5, allow_download)
        if zp is None:
            raise SystemExit("MOT20 데이터를 받지 못함 (받기 실패 또는 공간 부족)")
        for s in missing:
            n = extract(zp, s, data / s, max_frames)
            log("[풀기] %s: 파일 %d개" % (s, n))
        note = "MOT20.zip"
    if note:
        note_f.write_text(note)
    return note


def find_gt(d):
    for c in (d / "gt" / "gt.txt", d / "gt.txt", d / (d.name + ".txt")):
        if c.exists():
            return c
    g = sorted((d / "gt").glob("*.txt")) if (d / "gt").is_dir() else []
    return g[0] if g else None


def find_media(d):
    """구간 폴더에서 프레임 사진들 또는 영상 파일을 찾음 → ("img", [파일들]) / ("video", 경로) / None"""
    for sub in ("img1", "img", "images", "frames", "."):
        q = d / sub
        if q.is_dir():
            imgs = sorted(f for f in q.iterdir() if f.suffix.lower() in IMG_EXT)
            if len(imgs) >= 2:
                return "img", imgs
    for q in (d, d / "img1", d / "video", d.parent):
        if q.is_dir():
            vids = sorted(f for f in q.iterdir() if f.suffix.lower() in VID_EXT and
                          (q != d.parent or f.stem == d.name))
            if vids:
                return "video", vids[0]
    return None


def find_seqs(root):
    """root 아래에서 '정답 파일 + 프레임(또는 영상)'이 있는 구간 폴더를 모두 찾음 (폴더가 몇 겹이든). 이름순."""
    out = {}
    for dp, dn, fn in os.walk(root):
        d = Path(dp)
        if d.name in ("gt", "img1", "det"):
            continue
        if find_gt(d) is not None and find_media(d) is not None:
            out[d.name] = d
            dn[:] = []
    return dict(sorted(out.items()))


class Frames:
    """프레임 공급 (사진 묶음 또는 영상 파일). 순서대로만 읽음."""

    def __init__(self, kind, src):
        self.kind, self.src = kind, src
        if kind == "img":
            im = cv2.imread(str(src[0]))
            self.n, self.W, self.H, self.fps = len(src), im.shape[1], im.shape[0], None
        else:
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise RuntimeError("영상을 열 수 없음: %s" % src)
            self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.W, self.H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or None
            cap.release()

    def iter(self, limit=0, step=1):
        n = min(self.n, limit) if limit else self.n
        if self.kind == "img":
            for i in range(0, n, step):
                yield i, cv2.imread(str(self.src[i]))
            return
        cap = cv2.VideoCapture(str(self.src))
        i = 0
        while i < n:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                yield i, fr
            i += 1
        cap.release()


def load_seq(d, name, bench="mot15"):
    B = BENCH[bench]
    d = Path(d)
    info = {"name": name, "fps": B["seqs"].get(name, (None, ""))[0]}
    ini = d / "seqinfo.ini"
    if ini.exists():
        cp = configparser.ConfigParser()
        cp.read(ini)
        if cp.has_section("Sequence") and cp["Sequence"].get("frameRate"):
            info["fps"] = float(cp["Sequence"]["frameRate"])
    med = find_media(d)
    if med is None:
        raise RuntimeError("%s: 프레임 사진이나 영상 파일이 없음 (%s)" % (name, d))
    fr = Frames(*med)
    info.update(frames=fr, media=med[0], W=fr.W, H=fr.H, n=fr.n, fps=info["fps"] or fr.fps or 30.0)
    rows = []
    for ln in open(find_gt(d), encoding="utf-8", errors="ignore"):
        r = [x for x in ln.replace(",", " ").split() if x]      # 쉼표·공백 어느 쪽으로 나눈 파일이든
        if len(r) >= 6:
            try:
                float(r[0]), float(r[5])
            except ValueError:
                continue                                         # 제목 줄
            rows.append(r)
    fmt = B["fmt"]
    ncol = min(len(r) for r in rows) if rows else 0
    flags = {float(r[6]) for r in rows} if ncol > 6 else {1.0}
    classes = {int(float(r[7])) for r in rows} if ncol > 7 else {1}
    if fmt == "auto":
        # 분류 열이 있고 값이 여러 가지면 MOT16 방식(1 = 보행자). 한 가지뿐이면 전부 보행자로 봄
        fmt = "mot16" if len(classes) > 1 else "mot15"
    use_flag = len(flags) > 1                         # 표시 열 값이 전부 같으면(0 뿐이어도) 거르지 않음
    info["gt_format"] = "열 %d개, 표시값 %s, 분류값 %s → %s" % (ncol, sorted(flags)[:6], sorted(classes)[:12],
                                                          "MOT16 방식" if fmt == "mot16" else "전부 보행자")
    gt, alls = {}, {}
    dis = set(DISTRACTOR) | set(B.get("extra_distractor", ()))
    first = min(int(float(r[0])) for r in rows) if rows else 1
    off = 1 - first if first == 0 else 0              # 프레임 번호가 0 부터면 1 부터로 맞춤
    for row in rows:
        fno, tid = int(float(row[0])) + off, int(float(row[1]))
        x, y, w, h = map(float, row[2:6])
        flag = float(row[6]) if ncol > 6 else 1.0
        if fmt == "mot16":
            # 공식 평가 도구와 같은 처리: 평가 대상 = 보행자(분류 1)이면서 표시가 0 이 아닌 줄.
            # 그 밖의 줄도 '모든 정답'에 남겨 두었다가, 헷갈리는 대상(dis)에 맞은 검출을 채점 전에 뺀다.
            cls = int(float(row[7])) if ncol > 7 else 1
            alls.setdefault(fno, []).append((cls in dis, x, y, x + w, y + h))
            if cls != 1 or (use_flag and flag == 0):
                continue
        elif use_flag and flag == 0:                   # 평가에서 빼라고 표시된 줄
            continue
        gt.setdefault(fno, []).append((tid, x, y, x + w, y + h))
    info["gt"], info["gt_all"] = gt, alls
    return info


def drop_distractors(hyps, alls, thr=0.5):
    """검출 중 '헷갈리는 대상' 정답과 짝이 된 것을 뺌 (모든 정답과 1:1 로 짝지은 뒤 판단 — 공식 평가 도구 방식)."""
    if not hyps or not alls:
        return hyps
    M = iou_mat([a[1:] for a in alls], [h[1:] for h in hyps])
    M = np.where(M >= thr, M, 0.0)
    rm = {j for i, j in assign(-M) if M[i, j] > 0 and alls[i][0]}
    return [h for j, h in enumerate(hyps) if j not in rm]


# ───────────── 지표 ─────────────
def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def assign(cost):
    """최소 비용 짝짓기. scipy 가 있으면 헝가리안, 없으면 탐욕."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:  # noqa: BLE001
        out, used_r, used_c = [], set(), set()
        for idx in np.argsort(cost, axis=None):
            r, c = divmod(int(idx), cost.shape[1])
            if r not in used_r and c not in used_c:
                out.append((r, c))
                used_r.add(r)
                used_c.add(c)
        return out


class ClearMot:
    """CLEAR MOT (MOTA·IDSW·Recall·Precision) + IDF1 + MT/ML + 크기별 Recall. MOTChallenge 방식:
    직전 프레임의 짝이 아직 IoU 기준을 넘으면 그 짝을 유지, 나머지는 IoU 최대 짝짓기."""

    def __init__(self, thr=0.5):
        self.thr = thr
        self.gt = self.tp = self.fp = self.fn = self.idsw = self.hyp = 0
        self.last = {}            # gt id → 마지막으로 짝이 된 추적 id
        self.prev = {}            # 직전 프레임의 짝 (gt id → 추적 id)
        self.pair = {}            # (gt id, 추적 id) → 겹친 프레임 수 (IDF1 용)
        self.gt_len, self.gt_hit = {}, {}
        self.sz = {k: [0, 0] for _, _, k in SIZE_BINS}     # 크기 구간 → [정답 수, 맞춘 수]

    def step(self, gts, hyps):
        gid = [g[0] for g in gts]
        hid = [h[0] for h in hyps]
        M = iou_mat([g[1:] for g in gts], [h[1:] for h in hyps])
        for i, j in zip(*np.nonzero(M >= self.thr)):
            k = (gid[i], hid[j])
            self.pair[k] = self.pair.get(k, 0) + 1
        match = {}
        used_h = set()
        for i, g in enumerate(gid):                      # 1) 직전 짝 유지
            h = self.prev.get(g)
            if h is not None and h in hid:
                j = hid.index(h)
                if M[i, j] >= self.thr and j not in used_h:
                    match[i] = j
                    used_h.add(j)
        ri = [i for i in range(len(gid)) if i not in match]
        rj = [j for j in range(len(hid)) if j not in used_h]
        if ri and rj:                                   # 2) 나머지
            sub = M[np.ix_(ri, rj)]
            cost = np.where(sub >= self.thr, 1.0 - sub, 1e6)
            for a, b in assign(cost):
                if sub[a, b] >= self.thr:
                    match[ri[a]] = rj[b]
        self.gt += len(gid)
        self.hyp += len(hid)
        self.tp += len(match)
        self.fn += len(gid) - len(match)
        self.fp += len(hid) - len(match)
        self.prev = {}
        for i, g in enumerate(gid):
            self.gt_len[g] = self.gt_len.get(g, 0) + 1
            h_px = gts[i][4] - gts[i][2]
            key = next(k for lo, hi, k in SIZE_BINS if lo <= h_px < hi)
            self.sz[key][0] += 1
            if i in match:
                h = hid[match[i]]
                self.gt_hit[g] = self.gt_hit.get(g, 0) + 1
                self.sz[key][1] += 1
                if g in self.last and self.last[g] != h:
                    self.idsw += 1
                self.last[g] = h
                self.prev[g] = h

    def result(self):
        g = max(1, self.gt)
        gids = sorted({k[0] for k in self.pair} | set(self.gt_len))
        hids = sorted({k[1] for k in self.pair})
        idtp = 0
        if gids and hids:
            C = np.zeros((len(gids), len(hids)))
            for (a, b), n in self.pair.items():
                C[gids.index(a), hids.index(b)] = n
            idtp = int(sum(C[r, c] for r, c in assign(-C)))
        frac = [self.gt_hit.get(k, 0) / n for k, n in self.gt_len.items()]
        out = {"GT": self.gt, "TP": self.tp, "FP": self.fp, "FN": self.fn, "IDSW": self.idsw,
               "MOTA": round(100 * (1 - (self.fn + self.fp + self.idsw) / g), 1),
               "IDF1": round(100 * 2 * idtp / max(1, self.gt + self.hyp), 1),
               "Recall": round(100 * self.tp / g, 1),
               "Precision": round(100 * self.tp / max(1, self.tp + self.fp), 1),
               "tracks_gt": len(self.gt_len), "MT": sum(f >= 0.8 for f in frac), "ML": sum(f <= 0.2 for f in frac)}
        for k, (n, hit) in self.sz.items():
            out["gt_" + k] = n
            out["recall_" + k] = round(100 * hit / n, 1) if n else None
        return out

    def merge(self, o):
        """여러 구간 합계용 (IDF1·MT/ML 은 구간별 짝을 그대로 더함: 구간마다 id 가 겹치지 않게 접두어를 붙여 넣는다)."""
        raise NotImplementedError


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def cpu_temp():
    try:
        return round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000, 1)
    except Exception:  # noqa: BLE001
        return None


def device_name():
    try:
        m = open("/proc/device-tree/model").read().strip("\x00\n ")
        if m:
            return m
    except Exception:  # noqa: BLE001
        pass
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.lower().startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.machine()


def camera_motion(seq, limit=0):
    """고정 카메라인지 확인: 줄인 영상에서 화면 전체 움직임의 중앙값(px). 고정이면 거의 0."""
    fr = seq["frames"]
    n = min(fr.n, limit) if limit else fr.n
    step = max(1, n // 40)
    prev, vals = None, []
    for _, im in fr.iter(n, step):
        g = cv2.cvtColor(cv2.resize(im, (320, 240)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 8)
            if pts is not None and len(pts) >= 10:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None)
                d = np.linalg.norm((nxt - pts).reshape(-1, 2)[st.ravel() == 1], axis=1)
                if len(d):
                    vals.append(float(np.median(d)))
        prev = g
        if len(vals) >= 40:
            break
    return round(float(np.median(vals)), 2) if vals else None


# ───────────── 실행 ─────────────
def run_mode(seq, mode, det, base_cfg, args, outdir):
    cfg = dict(base_cfg, mode=mode, det_policy="every", fixed_roi=None, demo_speed_min=0.0, demo_speed_max=0.0,
               draw_curb=False)
    pipe = Cam1Pipeline(det, None, (seq["W"], seq["H"]), cfg)
    met = ClearMot(args.iou)
    tot, inf, nrun, area, nroi = [], [], [], [], []
    fall = 0
    nf = min(seq["n"], args.max_frames) if args.max_frames else seq["n"]
    shots = {int(nf * q) for q in (0.3, 0.6)} if args.save_frames else set()
    temp0 = cpu_temp()
    wall0 = time.perf_counter()
    done = 0
    for i, frame in seq["frames"].iter(nf):
        done += 1
        r = pipe.process(frame, i / seq["fps"])
        tm = r["timing"]
        tot.append(tm["total"])
        if r["det_ran"]:
            inf.append(tm["infer"] or 0.0)
            nrun.append(r.get("n_infer", 0 if (tm["infer"] or 0) == 0 else 1))
        if r.get("roi_area") is not None:
            area.append(r["roi_area"])
            nroi.append(r["n_rois"])
        fall += bool(r["fallback"]) or bool(r.get("full_scan") and mode != "full")
        hyps = [(p["track_id"], *p["bbox"]) for p in r["peds"] if not p.get("extrapolated")]
        hyps = drop_distractors(hyps, seq["gt_all"].get(i + 1), args.iou)
        met.step(seq["gt"].get(i + 1, []), hyps)
        if i in shots:
            img = pipe.draw(frame, r, "")
            for g in seq["gt"].get(i + 1, []):
                cv2.rectangle(img, (int(g[1]), int(g[2])), (int(g[3]), int(g[4])), (255, 255, 255), 1)
            cv2.imwrite(str(outdir / ("%s_%s_f%04d.jpg" % (seq["name"], mode, i + 1))), img)
    res = met.result()
    nf = max(1, done)
    res.update(seq=seq["name"], mode=mode, frames=nf, W=seq["W"], H=seq["H"], fps_src=seq["fps"],
               fps=round(1000 / max(1e-6, float(np.mean(tot))), 2), ms_mean=round(float(np.mean(tot)), 1),
               ms_p50=round(pct(tot, 50), 1), ms_p95=round(pct(tot, 95), 1),
               yolo_ms_p50=round(pct(inf, 50), 1) if inf else None,
               yolo_runs_per_frame=round(float(np.sum(nrun)) / nf, 2),
               roi_area_pct=round(100 * float(np.mean(area)), 1) if area else None,
               rois_mean=round(float(np.mean(nroi)), 2) if nroi else None,
               fullframe_pct=round(100 * fall / nf, 1) if mode != "full" else 100.0,
               wall_s=round(time.perf_counter() - wall0, 1), temp_start=temp0, temp_end=cpu_temp())
    return res, met


def combine(rows):
    """구간 합계: 개수는 더하고, 비율 지표는 합계로 다시 계산. 시간은 프레임 수로 가중 평균."""
    out = {}
    for mode in dict.fromkeys(r["mode"] for r in rows):
        rs = [r for r in rows if r["mode"] == mode]
        s = {k: sum(r[k] for r in rs) for k in ("GT", "TP", "FP", "FN", "IDSW", "frames", "tracks_gt", "MT", "ML")}
        g = max(1, s["GT"])
        s.update(seq="전체(%d개 구간)" % len(rs), mode=mode,
                 MOTA=round(100 * (1 - (s["FN"] + s["FP"] + s["IDSW"]) / g), 1),
                 Recall=round(100 * s["TP"] / g, 1), Precision=round(100 * s["TP"] / max(1, s["TP"] + s["FP"]), 1),
                 IDF1=round(sum(r["IDF1"] * (r["GT"] + r["TP"] + r["FP"]) for r in rs) /
                            max(1, sum(r["GT"] + r["TP"] + r["FP"] for r in rs)), 1),
                 ms_mean=round(sum(r["ms_mean"] * r["frames"] for r in rs) / s["frames"], 1))
        s["fps"] = round(1000 / s["ms_mean"], 2)
        for k in ("small", "mid", "large"):
            n = sum(r["gt_" + k] for r in rs)
            hit = sum((r["recall_" + k] or 0) * r["gt_" + k] / 100 for r in rs)
            s["gt_" + k], s["recall_" + k] = n, (round(100 * hit / n, 1) if n else None)
        out[mode] = s
    return list(out.values())


COLS = [("mode", "방식"), ("MOTA", "MOTA"), ("IDF1", "IDF1"), ("Recall", "Recall"), ("Precision", "Prec."),
        ("IDSW", "ID 변경"), ("recall_small", "작은 사람 Recall"), ("fps", "FPS"), ("ms_p50", "처리 ms P50"),
        ("ms_p95", "P95"), ("yolo_runs_per_frame", "YOLO 횟수/프레임")]


def table(rows, md=False):
    hdr = [h for _, h in COLS]
    body = []
    for r in rows:
        body.append([MODE_KO.get(r["mode"], r["mode"]) if k == "mode" else ("-" if r.get(k) is None else str(r[k]))
                     for k, _ in COLS])
    if md:
        return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] +
                         ["| " + " | ".join(b) + " |" for b in body])
    w = [max(len(x) + sum(ord(ch) > 0x2E80 for ch in x) for x in col) for col in zip(hdr, *body)]

    def fmt(cells):
        return "  ".join(c + " " * (w[i] - len(c) - sum(ord(ch) > 0x2E80 for ch in c)) for i, c in enumerate(cells))
    return "\n".join([fmt(hdr), "-" * (sum(w) + 2 * len(w))] + [fmt(b) for b in body])


def make_selftest(data):
    """합성 구간 (색 상자 = 사람). 스크립트·지표가 맞게 도는지 확인용. 벤치마크 결과가 아님."""
    rng = np.random.default_rng(3)
    name = "SELFTEST"
    d = data / name
    (d / "img1").mkdir(parents=True, exist_ok=True)
    (d / "gt").mkdir(exist_ok=True)
    W, H, N = 768, 576, 120
    bg = rng.integers(95, 125, (H, W, 3), dtype=np.uint8)
    cols = [(40, 40, 230), (40, 200, 40), (230, 60, 40), (30, 200, 230), (200, 40, 200)]
    ppl = [dict(x=rng.uniform(60, W - 120), y=rng.uniform(60, H - 160), vx=rng.uniform(-6, 6), vy=rng.uniform(-3, 3),
                w=int(rng.uniform(14, 34)), c=cols[i]) for i in range(5)]
    rows = []
    for f in range(1, N + 1):
        img = bg.copy()
        for i, p in enumerate(ppl):
            if not (i == 4 and f < 30):
                p["x"] = float(np.clip(p["x"] + p["vx"], 5, W - p["w"] - 5))
                p["y"] = float(np.clip(p["y"] + p["vy"], 5, H - p["w"] * 2.6 - 5))
            x, y, w, h = int(p["x"]), int(p["y"]), p["w"], int(p["w"] * 2.6)
            cv2.rectangle(img, (x, y), (x + w, y + h), p["c"], -1)
            rows.append("%d,%d,%d,%d,%d,%d,1,-1,-1,-1" % (f, i + 1, x, y, w, h))
        cv2.imwrite(str(d / "img1" / ("%06d.jpg" % f)), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (d / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
    (d / "seqinfo.ini").write_text("[Sequence]\nname=%s\nimDir=img1\nframeRate=10\nseqLength=%d\nimWidth=%d\n"
                                   "imHeight=%d\nimExt=.jpg\n" % (name, N, W, H))
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="mot15", choices=["mot15", "sompt22"],
                    help="mot15 = MOTChallenge MOT15 / sompt22 = SOMPT22 (감시용 고정 카메라)")
    ap.add_argument("--data", default=str(ROOT / "bench" / "data"))
    ap.add_argument("--seqs", default=None, help="쉼표로 구분. 기본 = 그 벤치마크의 고정 카메라 학습 구간 전부")
    ap.add_argument("--modes", default=None, help="기본 = 벤치마크별 설정 (mot20 은 crops 제외)")
    ap.add_argument("--site", default=str(ROOT / "simdemo" / "site_sim.json"), help="Cam1 설정을 가져올 파일 (시뮬 데모 값 그대로)")
    ap.add_argument("--detector", default="yolo", choices=["yolo", "color"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=None, help="구간마다 앞쪽 N 프레임만 (기본: mot15 전체, mot17·mot20 은 600)")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-save-frames", dest="save_frames", action="store_false")
    ap.add_argument("--selftest", action="store_true", help="합성 영상으로 스크립트만 점검 (YOLO·데이터 필요 없음)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    B = BENCH[args.bench]
    data = Path(args.data) / B["sub"] if B["sub"] else Path(args.data)
    if args.max_frames is None:
        args.max_frames = B["max_frames"]
    gt_note = ""
    site = json.loads(Path(args.site).read_text(encoding="utf-8"))["cam1"]
    base_cfg = dict(site.get("pipeline", {}), canvas=site.get("imgsz", 416))
    seq_dirs = {}
    if args.selftest:
        seq_names = [make_selftest(data)]
        args.detector = "color"
    elif B.get("dynamic"):
        data.mkdir(parents=True, exist_ok=True)
        for z in sorted(data.glob("*.zip")):          # 받아 온 압축 파일이 있으면 먼저 풂 (한 번만)
            mark = data / (z.name + ".풀었음")
            if not mark.exists():
                log("[풀기] %s (%.1f GB) …" % (z.name, z.stat().st_size / 1e9))
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(data)
                mark.write_text("ok")
        seq_dirs = find_seqs(data)
        want = [s for s in (args.seqs or "").split(",") if s]
        seq_names = [s for s in seq_dirs if not want or s in want]
        if not seq_names:
            raise SystemExit("%s 아래에서 구간을 찾지 못함.\n  '정답 파일(gt/gt.txt) + 프레임 사진(img1/) 또는 영상 파일'이 들어 있는 "
                             "구간 폴더가 있어야 함. 지금 들어 있는 것:\n  %s" % (
                                 data, "\n  ".join(sorted(str(p.relative_to(data)) for p in data.rglob("*")
                                                         if p.is_file())[:40]) or "(비어 있음)"))
    else:
        seq_names = [s for s in (args.seqs or ",".join(B["seqs"])).split(",") if s]
        gt_note = ensure_data(data, args.bench, seq_names, not args.no_download, args.max_frames)
        seq_names = [s for s in seq_names if (data / s / "gt" / "gt.txt").exists() and n_frames(data / s)]
    if not seq_names:
        raise SystemExit("돌릴 구간이 없음")
    modes = [m for m in (args.modes or B["modes"]).split(",") if m]
    outdir = Path(args.out or (ROOT / "bench" / "results" / (time.strftime("%m%d_%H%M%S") + "_" + args.bench)))
    outdir.mkdir(parents=True, exist_ok=True)

    if args.detector == "yolo":
        from bgx.detector import UltralyticsDetector
        model = str(ROOT / site["model"])
        log("[모델] %s (입력 %d, conf %.2f)" % (site["model"], site.get("imgsz", 416), site.get("conf", 0.1)))
        det = UltralyticsDetector(model, site.get("imgsz", 416), site.get("conf", 0.1))
        det_name = "%s @%d" % (Path(site["model"]).name, site.get("imgsz", 416))
    else:
        from bgx.detector import ColorBoxDetector
        det = ColorBoxDetector()
        det_name = "ColorBox (합성 점검용)"
    dev = device_name()
    log("[장치] %s / 방식 %s / 구간 %s" % (dev, modes, seq_names))

    rows, meta_seq = [], {}
    t_start, stopped = time.time(), False
    for si, name in enumerate(seq_names):
      try:
          seq = load_seq(seq_dirs.get(name, data / name), name, "mot15" if args.selftest else args.bench)
          cam = camera_motion(seq, args.max_frames)
          meta_seq[name] = {"W": seq["W"], "H": seq["H"], "fps": seq["fps"], "frames": seq["n"],
                            "gt_boxes": sum(len(v) for v in seq["gt"].values()), "camera_motion_px": cam,
                            "note": B["seqs"].get(name, (0, ""))[1]}
          used = min(seq["n"], args.max_frames) if args.max_frames else seq["n"]
          meta_seq[name]["frames"] = used
          meta_seq[name]["gt_boxes"] = sum(len(v) for k, v in seq["gt"].items() if k <= used)
          meta_seq[name]["gt_format"] = seq["gt_format"]
          log("   정답 형식: %s / 입력: %s" % (seq["gt_format"], "영상 파일" if seq["media"] == "video" else "프레임 사진"))
          log("━━ %s  %dx%d  %d프레임 @%g fps  정답 상자 %d개  화면 전체 움직임 중앙값 %s px%s" % (
              name, seq["W"], seq["H"], used, seq["fps"], meta_seq[name]["gt_boxes"], cam,
              "  ← 카메라가 움직이는 듯 (결과 해석 주의)" if cam is not None and cam > 0.5 else ""))
          if meta_seq[name]["gt_boxes"] == 0:
              log("   [주의] 이 구간은 쓸 수 있는 정답 상자가 0개 → 건너뜀 (정답 형식을 확인해야 함)")
              del meta_seq[name]
              continue
          for mode in modes:
              res, _ = run_mode(seq, mode, det, base_cfg, args, outdir)
              rows.append(res)
              log("   %-22s MOTA %5.1f  IDF1 %5.1f  Recall %5.1f  ID변경 %3d  작은사람 %s  | %5.1f FPS  %5.1f ms(P50)  YOLO %.1f회/프레임  전체화면 %s%%  %s°C" % (
                  MODE_KO.get(mode, mode), res["MOTA"], res["IDF1"], res["Recall"], res["IDSW"],
                  "-" if res["recall_small"] is None else "%.1f" % res["recall_small"], res["fps"], res["ms_p50"],
                  res.get("yolo_runs_per_frame", 1), res.get("fullframe_pct", "-"), res["temp_end"]))
          left = len(seq_names) - si - 1
          if left:
              log("   (남은 구간 %d개, 예상 남은 시간 약 %.0f분 — 중간에 Ctrl+C 를 눌러도 끝난 구간까지는 저장됨)" % (
                  left, (time.time() - t_start) / (si + 1) * left / 60))
      except KeyboardInterrupt:
        stopped = True
        log("[중단] Ctrl+C — 끝까지 돌린 구간만 저장")
        break

    done_seqs = {n for n in meta_seq if sum(1 for r in rows if r["seq"] == n) == len(modes)}
    rows = [r for r in rows if r["seq"] in done_seqs]
    meta_seq = {n: m for n, m in meta_seq.items() if n in done_seqs}
    seq_names = [n for n in seq_names if n in meta_seq]
    if not rows:
        raise SystemExit("평가한 구간이 없음")
    allrows = combine(rows) if len(seq_names) > 1 else []
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(outdir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows + allrows:
            w.writerow(r)
    meta = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "device": dev, "detector": det_name, "iou": args.iou,
            "benchmark": B["title"] + (" — " + gt_note if gt_note else ""), "bench": args.bench,
            "tracker": "ByteTrack (bgx.bytetrack)",
            "cam1_cfg": base_cfg, "det_policy": "every (매 프레임 YOLO)", "modes": modes, "seqs": meta_seq,
            "selftest": bool(args.selftest), "max_frames": args.max_frames or None, "stopped_early": stopped}
    (outdir / "results.json").write_text(json.dumps({"meta": meta, "rows": rows, "all": allrows}, ensure_ascii=False,
                                                    indent=1), encoding="utf-8")
    md = ["# 벤치마크 결과 — %s" % meta["when"], "",
          "- 벤치마크: %s" % meta["benchmark"], "- 장치: %s" % dev, "- 검출 모델: %s, 추적: ByteTrack" % det_name,
          "- 방식만 다르고 모델·추적기·설정은 같음. Cam1 설정은 시뮬 데모 값 그대로 (벤치마크에 맞춘 조정 없음).",
          "- 지표: CLEAR MOT (IoU %.1f), IDF1 — 이 스크립트가 직접 계산. '작은 사람' = 정답 상자 높이 60 px 미만." % args.iou,
          "- 처리 시간: 이 장치에서 프레임 하나를 처리하는 시간 (영상 읽기 제외).",
          "- 구간마다 앞쪽 %d프레임만 사용." % args.max_frames if args.max_frames else "- 구간 전체 프레임 사용.", ""]
    for name in seq_names:
        m = meta_seq[name]
        md += ["## %s%s" % (name, "  (주 평가 구간)" if name == PRIMARY else ""),
               "%s · %dx%d · %d프레임 @%g fps · 정답 상자 %d개 · 화면 전체 움직임 %s px" % (
                   m["note"], m["W"], m["H"], m["frames"], m["fps"], m["gt_boxes"], m["camera_motion_px"]), "",
               table([r for r in rows if r["seq"] == name], md=True), ""]
    if allrows:
        md += ["## 전체 합계 (%d개 구간)" % len(seq_names), "", table(allrows, md=True), ""]
    (outdir / "results.md").write_text("\n".join(md), encoding="utf-8")

    print()
    for name in seq_names:
        print("== %s%s" % (name, "  (주 평가 구간)" if name == PRIMARY else ""))
        print(table([r for r in rows if r["seq"] == name]))
        print()
    if allrows:
        print("== 전체 합계 (%d개 구간)" % len(seq_names))
        print(table(allrows))
        print()
    log("[저장] %s  (results.md / results.csv / results.json%s)" % (outdir, " / 장면 그림" if args.save_frames else ""))


if __name__ == "__main__":
    main()
