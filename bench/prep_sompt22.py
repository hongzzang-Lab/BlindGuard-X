#!/usr/bin/env python3
# bench/prep_sompt22.py — (노트북에서 실행) 내려받은 SOMPT22 를 풀고, 평가에 필요한 부분만 Pi 로 보냄.
#
#   cd <저장소 폴더>
#   python3 bench/prep_sompt22.py                       ← ~/Downloads 에서 방금 받은 zip 을 스스로 찾음
#   python3 bench/prep_sompt22.py --src ~/Downloads/SOMPT22-xxx.zip ~/Downloads/SOMPT22-yyy.zip
#   python3 bench/prep_sompt22.py --no-send             ← 풀고 무엇이 들어 있는지만 확인
#
# 하는 일
#   1) zip 을 ~/Downloads/sompt22_data/ 에 풂 (zip 안의 zip 도 풂)
#   2) '정답 파일 + 프레임(또는 영상)'이 있는 구간 폴더를 찾아 표로 보여 줌 (정답이 없는 시험용 구간은 제외)
#   3) 구간마다 정답 파일과 앞쪽 N 프레임(기본 300, 영상 파일이면 통째로)만 Pi 의 ~/realtest/bench/data/sompt22/ 로 보냄
import argparse
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

# 이 파일은 파이썬 기본 기능만 씀 (노트북에 따로 설치할 것 없음). 구간 찾는 규칙은 bench/mot_bench.py 와 같음.
VID_EXT = (".mp4", ".avi", ".mkv", ".mov", ".m4v")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
ARC_EXT = (".zip", ".tar", ".tgz", ".gz")


def find_gt(d):
    for c in (d / "gt" / "gt.txt", d / "gt.txt", d / (d.name + ".txt")):
        if c.exists():
            return c
    g = sorted((d / "gt").glob("*.txt")) if (d / "gt").is_dir() else []
    return g[0] if g else None


def find_media(d):
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
    out = {}
    for dp, dn, fn in os.walk(root):
        d = Path(dp)
        if d.name in ("gt", "img1", "det"):
            continue
        if find_gt(d) is not None and find_media(d) is not None:
            out[d.name] = d
            dn[:] = []
    return dict(sorted(out.items()))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def unzip_all(zp, out, depth=0):
    mark = out / (".풀었음_" + zp.name)
    if mark.exists():
        log("[풀기] %s — 이미 풀었음" % zp.name)
        return
    log("[풀기] %s%s (%.2f GB) … (몇 분 걸릴 수 있음)" % ("  " * depth, zp.name, zp.stat().st_size / 1e9))
    if zipfile.is_zipfile(zp):
        with zipfile.ZipFile(zp) as z:
            z.extractall(out)
            names = z.namelist()
    elif tarfile.is_tarfile(zp):
        with tarfile.open(zp) as t:
            names = t.getnames()
            try:
                t.extractall(out, filter="data")
            except TypeError:              # 옛 파이썬
                t.extractall(out)
    else:
        log("[문제] 풀 수 없는 형식: %s  (zip · tar 만 됨. rar/7z 이면 파일 관리자에서 '여기에 풀기'로 %s 에 풀어 주세요)" % (zp.name, out))
        return
    inner = [out / n for n in names if n.lower().endswith(ARC_EXT)]
    mark.write_text("ok")
    for iz in inner:                       # 압축 안의 압축
        if iz.exists() and depth < 3:
            unzip_all(iz, iz.parent, depth + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", help="받은 zip 파일들 또는 이미 풀어 둔 폴더. 없으면 ~/Downloads 에서 최근 zip 을 찾음")
    ap.add_argument("--work", default=str(Path.home() / "Downloads" / "sompt22_data"))
    ap.add_argument("--frames", type=int, default=300, help="구간마다 Pi 로 보낼 앞쪽 프레임 수")
    ap.add_argument("--pi", default=None, help="계정@Pi주소 (예: pi@192.168.0.20). 보낼 때 필요")
    ap.add_argument("--no-send", action="store_true")
    ap.add_argument("--dest-cmd", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    srcs = [Path(os.path.expanduser(x)) for x in (args.src or [])]
    if not srcs:
        dl = Path.home() / "Downloads"
        now = time.time()
        cand = [p for p in dl.iterdir() if p.is_file() and p.suffix.lower() in ARC_EXT + (".rar", ".7z")
                and p.stat().st_size > 50e6]
        named = [p for p in cand if "sompt" in p.name.lower()]
        srcs = sorted(named or [p for p in cand if now - p.stat().st_mtime < 8 * 3600])
        part = sorted(dl.glob("*.crdownload"))
        if part:
            log("[주의] 아직 받는 중인 파일이 있음: %s → 다 받은 뒤 다시 실행" % ", ".join(p.name for p in part))
            if not srcs:
                return 1
        if not srcs and not find_seqs(work):
            log("~/Downloads 에서 SOMPT22 압축 파일을 찾지 못함. --src 로 파일(또는 풀어 둔 폴더)을 직접 지정하세요.")
            log("지금 ~/Downloads 의 큰 파일: %s" % ([p.name for p in sorted(dl.iterdir()) if p.is_file()
                                                     and p.stat().st_size > 50e6] or "없음"))
            return 1
        if srcs:
            log("[찾음] %s" % ", ".join("%s (%.2f GB)" % (p.name, p.stat().st_size / 1e9) for p in srcs))
    roots = [work]
    for sp in srcs:
        if sp.is_dir():
            roots.append(sp)
        elif sp.is_file():
            unzip_all(sp, work)
        else:
            log("[주의] 무시: %s" % sp)

    seqs = {}
    for r in roots:
        seqs.update(find_seqs(r))
    if not seqs:
        log("[문제] '정답 파일 + 프레임(또는 영상)'이 있는 구간 폴더를 찾지 못함. 풀린 내용 (앞 60개):")
        for p in sorted(work.rglob("*"))[:60]:
            print("   ", p.relative_to(work), "" if p.is_dir() else "(%.1f MB)" % (p.stat().st_size / 1e6))
        return 1

    print()
    print("%-16s %-6s %7s %9s  %s" % ("구간", "입력", "프레임", "정답 줄", "정답 첫 줄"))
    plan, total = [], 0
    for name, d in seqs.items():
        kind, src = find_media(d)
        gtf = find_gt(d)
        lines = [ln for ln in open(gtf, errors="ignore") if ln.strip()]
        print("%-16s %-6s %7s %9d  %s" % (name, "영상" if kind == "video" else "사진",
                                          len(src) if kind == "img" else "-", len(lines),
                                          lines[0].strip()[:60] if lines else ""))
        files = [(gtf, "%s/gt/gt.txt" % name)]
        if (d / "seqinfo.ini").exists():
            files.append((d / "seqinfo.ini", "%s/seqinfo.ini" % name))
        if kind == "img":
            files += [(f, "%s/img1/%s" % (name, f.name)) for f in src[: args.frames]]
        else:
            files.append((src, "%s/%s" % (name, src.name)))
        plan += files
        total += sum(f.stat().st_size for f, _ in files)
    print()
    log("[보낼 것] 구간 %d개, 파일 %d개, %.2f GB (구간마다 앞쪽 %d프레임)" % (len(seqs), len(plan), total / 1e9, args.frames))
    if args.no_send:
        return 0
    if not (args.dest_cmd or args.pi):
        log("[문제] 어디로 보낼지 없음 → --pi 계정@Pi주소 를 붙여 다시 실행 (풀고 확인만 하려면 --no-send)")
        return 1
    cmd = args.dest_cmd or "ssh %s 'mkdir -p ~/realtest/bench/data/sompt22 && cd ~/realtest/bench/data/sompt22 && tar xf -'" % args.pi
    log("[전송] Pi 로 보내는 중 … (비밀번호를 물으면 입력)")
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
    sent, last, t0 = 0, 0, time.time()
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
            for f, arc in plan:
                tar.add(str(f), arcname=arc)
                sent += f.stat().st_size
                if sent - last > total / 10:
                    last = sent
                    log("[전송] %3.0f%%  %.2f / %.2f GB  (%.1f MB/s)" % (100 * sent / total, sent / 1e9, total / 1e9,
                                                                      sent / 1e6 / max(1e-6, time.time() - t0)))
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        log("[문제] 전송 실패 (종료 코드 %d). Pi 주소·비밀번호·저장 공간을 확인" % rc)
        return 1
    print()
    log("[완료] Pi 로 보냄. 다음: Pi 창에서  cd ~/realtest && bash bench/run_on_pi.sh --bench sompt22")
    return 0


if __name__ == "__main__":
    sys.exit(main())
