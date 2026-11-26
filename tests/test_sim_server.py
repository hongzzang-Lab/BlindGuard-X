# tests/test_sim_server.py
#
# 시뮬 데모 서버 흐름 시험 (네트워크·CARLA 없이 ASGI 직접 구동)
#   가짜 CARLA(/api/sim: 차량 A·B GNSS+OBU, 보행자 신호) + 가짜 Pi(/edge: Cam1 장면, Cam2 EMR) + 폰 앱(/ws)
#   → 차량 A 앱에 '경고(Cam2 조기 징후 + 신호 전환)' → '긴급(CPA)' 이 오고, 차량 B 앱에는 아무것도 안 오는지
#   python3 tests/test_sim_server.py   (약 10초)

import asyncio
import csv
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ["BGX_LOGDIR"] = tempfile.mkdtemp(prefix="bgx_sim_")
os.environ["BGX_CALIB_DIR"] = tempfile.mkdtemp(prefix="bgx_sim_calib_")
os.environ["BGX_SITE"] = os.path.join(ROOT, "simdemo", "site_sim.json")

import server.exp_server as SV  # noqa: E402
from test_integration import WSClient, http  # noqa: E402  (같은 ASGI 흉내 도구)


def now_ms():
    return int(time.time() * 1000)


GEOM = {"curb_y": 0.0, "road_far_y": -14.0, "crosswalk_x": [-2.0, 2.0], "lane_y": -5.25, "stop_x": None,
        "lanes": [{"y": -1.75, "dir": -1, "width": 3.5, "name": "차로1 ←"}, {"y": -5.25, "dir": -1, "width": 3.5, "name": "차로2 ←"},
                  {"y": -8.75, "dir": 1, "width": 3.5, "name": "차로3 →"}, {"y": -12.25, "dir": 1, "width": 3.5, "name": "차로4 →"}],
        "near_band_y": [-7.0, 0.0], "waiting_xy": [0.0, 1.2], "waiting_poly": [[-3.5, 0], [3.5, 0], [3.5, 3], [-3.5, 3]]}


async def main():
    SV.SV = SV
    ticker = asyncio.create_task(SV.ticker())
    phA, phB, edge = WSClient("/ws"), WSClient("/ws"), WSClient("/edge")
    for c in (phA, phB, edge):
        await c.run()
    await phA.send({"type": "hello", "v": 1, "device_id": "phone-A"})
    await phB.send({"type": "hello", "v": 1, "device_id": "phone-B"})
    await edge.send({"type": "hello_edge", "edge_id": "carla-j189", "cams": {}, "has_H": True})
    await asyncio.sleep(0.2)
    assert [a.vehicle for a in SV.S["apps"]] == ["A", "B"]

    r = await http("POST", "/api/sim", {"geom": GEOM, "phase": "setup", "vehicles": []})
    assert r["ok"] and SV.S["source"] == "carla"
    cfg = await http("GET", "/api/config")
    assert cfg["floor"]["road_far_y"] == -14.0 and cfg["geom_rev"] == 1 and cfg["fusion"]["near_band_y"] == [-7.0, 0.0]
    # 보정 적용이 Pi 로 감 (carla_demo 가 하는 것)
    r = await http("POST", "/api/calib/cam2", {"polygon": [[10, 10], [600, 10], [600, 400], [10, 400]],
                                              "boundary": [[10, 10], [600, 10]]})
    assert r["ok"] and r["sent"]
    r = await http("POST", "/api/sim", {"cmd": "take_start", "take": 1, "t": 0.0, "vehicles": [],
                                        "signal": {"ped": "GREEN", "veh": "RED"}, "events": ["take_start"],
                                        "obstacles": [{"id": "truck", "x": 7.0, "y": -1.75, "len": 8.0, "wid": 2.5}]})
    await asyncio.sleep(0.1)
    cmds = [m.get("cmd") for m in edge.out if m.get("type") == "cmd"]
    assert "record" in cmds and "reset_bg" in cmds and "set_geom" in cmds, cmds
    assert SV.S["trial"] and SV.S["trial"]["exp"] == "SIM"

    t0 = time.time()
    frame = 0
    while True:
        t = time.time() - t0
        if t > 6.0:
            break
        ped_sig = "FLASH" if t < 1.0 else "RED"
        xa = 40 - 8.33 * t
        xb = -45 + 8.33 * t
        await http("POST", "/api/sim", {"take": 1, "t": round(t, 2), "phase": "main",
                                        "signal": {"ped": ped_sig, "veh": "GREEN" if t > 1.2 else "RED"},
                                        "vehicles": [{"id": "A", "x": xa, "y": -5.25, "vx": -8.33, "vy": 0.0, "speed": 8.33},
                                                     {"id": "B", "x": xb, "y": -12.25, "vx": 8.33, "vy": 0.0, "speed": 8.33}],
                                        "events": ["ped_approach"] if frame == 3 else []})
        # Pi: Cam2 (t 0.4~1.6 출발 예측) · Cam1 (t ≥ 1.5 횡단 중인 보행자)
        frame += 1
        tc = now_ms() - 60
        emr = 0.3 + (0.5 if 0.4 <= t <= 1.6 else 0)
        await edge.send({"type": "cam2", "cam": "cam2", "frame": frame, "ts_cap": tc, "ts_done": tc + 6, "warmup": False,
                         "present": True, "area": 900, "signed_px": -20.0, "toward_px": 3.0, "n_pts": 30, "persist": 5,
                         "prox": 0.8, "tow": 1.0, "per": 1.0, "emr": emr, "emr_state": emr >= 0.6, "crossed": False,
                         "events": ["emr_high"] if frame == 5 else [], "timing": {"mog": 2, "lk": 2, "total": 5},
                         "fps": 10, "dropped": 0, "rec": True})
        peds = []
        if t >= 1.5:
            peds = [{"track_id": 3, "x_m": 0.2, "y_m": -1.5 * (t - 1.5), "vx": 0.0, "vy": -1.5, "v_ok": True,
                     "conf": 0.8, "u": 1, "v": 1, "bbox": [1, 1, 2, 2]}]
        await edge.send({"type": "scene", "cam": "cam1", "frame": frame, "ts_cap": tc, "ts_done": tc + 30,
                         "ts_send": tc + 31, "offset_ms": 0, "mode": "packed", "every": 1, "det_ran": True,
                         "full_scan": False, "fallback": False, "n_rois": 1, "roi_area": 0.1, "pack_scale": 1.2,
                         "canvas_fill": 0.5, "n_det": len(peds), "timing": {"flow": 3, "roi": 1, "pack": 1, "infer": 27,
                                                                             "track": 1, "homo": 0.2, "total": 33},
                         "peds": peds, "fps": 10, "dropped": 0, "temp_c": 60, "rec": True, "has_H": True})
        await asyncio.sleep(0.1)
    st = await http("GET", "/api/monitor")
    await http("POST", "/api/sim", {"cmd": "take_end", "take": 1, "t": 6.0, "vehicles": [], "events": ["take_end"]})
    await asyncio.sleep(0.2)

    lvA = [(m.get("risk") or {}).get("level") for m in phA.out if m.get("type") == "alert" and m.get("risk")]
    lvB = [(m.get("risk") or {}).get("level") for m in phB.out if m.get("type") == "alert" and m.get("risk")]
    firstA = next(m for m in phA.out if m.get("type") == "alert" and m.get("risk"))
    print("차량 A 앱이 받은 단계:", lvA, "첫 근거:", firstA["risk"].get("reasons"))
    print("차량 B 앱이 받은 단계:", lvB)
    assert lvA and lvA[0] == "warning" and set(firstA["risk"]["reasons"]) == {"EMR", "SIGNAL"}, firstA
    assert "danger" in lvA
    dmsg = next(m for m in phA.out if m.get("type") == "alert" and (m.get("risk") or {}).get("level") == "danger")
    assert "CPA" in dmsg["risk"]["reasons"]
    assert not lvB, lvB
    # 관제 화면 데이터
    assert st["source"] == "carla" and st["sim"]["connected"] and st["sim"]["signal"]["ped"] == "RED"
    vA = next(v for v in st["vehicles"] if v["id"] == "A")
    vB = next(v for v in st["vehicles"] if v["id"] == "B")
    assert vB["level"] is None and vB["in_band"] is False
    assert st["sim"]["obstacles"][0]["id"] == "truck"
    assert any("[시뮬]" in e["text"] for e in st["events"])
    assert any("Cam2 조기 징후" in e["text"] for e in st["events"])
    # 기록
    d = SV.LOGDIR
    rows = list(csv.DictReader(open(os.path.join(d, "fusion.csv"), encoding="utf-8")))
    assert any(r["reasons"] == "EMR|SIGNAL" for r in rows) and any("CPA" in r["reasons"] for r in rows)
    assert not any(r["vehicle"] == "B" and r["level"] for r in rows)
    trials = list(csv.DictReader(open(os.path.join(d, "trials.csv"), encoding="utf-8")))
    assert trials and trials[-1]["exp"] == "SIM", trials
    alerts = list(csv.DictReader(open(os.path.join(d, "alerts.csv"), encoding="utf-8")))
    assert all(a["vehicle"] == "A" for a in alerts)
    print("fusion.csv %d줄, alerts.csv %d줄, sim_states.csv OK" % (len(rows), len(alerts)))
    print("기록 폴더:", d)
    ticker.cancel()
    for c in (phA, phB, edge):
        await c.close()
    print("ok test_sim_server")


if __name__ == "__main__":
    asyncio.run(main())
