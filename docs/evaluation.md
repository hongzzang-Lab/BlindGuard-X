# Evaluation — MOT15 & SOMPT22

![Raspberry Pi 5](https://img.shields.io/badge/Device-Raspberry%20Pi%205-238636?style=flat-square) ![Self-computed](https://img.shields.io/badge/Metrics-Self--computed-6e7781?style=flat-square)

Four input strategies for Cam1 were run unchanged on public pedestrian-tracking data, measuring accuracy and per-frame processing time on a Raspberry Pi 5. Measured on November 26, 2026.

> This is **not a leaderboard submission**. Only training sequences with public ground truth were used, and metrics were computed by `bench/mot_bench.py`. Small differences from the official TrackEval tool are possible.

---

## Question

**Does giving the detector more pixels per person improve detection and identity tracking on a CPU-only edge device—and what does packing those regions into one inference cost?**

## Setup

| Item | Value |
|---|---|
| Device | Raspberry Pi 5 Model B (no GPU) |
| Detector | YOLO26n, NCNN, 416×416 input, confidence 0.10 |
| Tracker | ByteTrack (`bgx/bytetrack.py`) |
| Settings | The simulation-demo configuration, unchanged; no per-dataset tuning; detector run on every frame |
| Metrics | CLEAR MOT at IoU 0.5, IDF1 |
| Timing | Time to process one frame, excluding image decoding; FPS = 1000 / mean ms |

The four strategies differ **only** in what the detector sees: `FULL`, `ADAPTIVE`, `OURS` (per-ROI inference) and `OURS_PACKED`. See [vision method](vision-method.md).

## MOT15 — six static-camera sequences, complete (2,315 frames, 18,824 boxes)

| Input strategy | MOTA | IDF1 | Recall | Precision | ID switches | FPS |
|---|---|---|---|---|---|---|
| FULL | 52.4 | 52.9 | 62.6 | **86.9** | 153 | **30.6** |
| ADAPTIVE | 54.2 | 55.4 | 65.3 | 86.3 | 148 | 25.3 |
| OURS (per-ROI inference) | 53.1 | **59.8** | **72.9** | 79.3 | **137** | 11.5 |
| OURS_PACKED | **56.0** | 59.5 | 67.4 | 86.4 | 141 | 25.2 |

Per sequence (MOTA / IDF1 / Recall / Precision / ID switches / FPS)

| Sequence | FULL | ADAPTIVE | OURS | OURS_PACKED |
|---|---|---|---|---|
| PETS09-S2L1 (768×576, 795 frames) | 76.9 / 60.2 / 88.4 / 89.4 / 43 / 30.6 | 81.3 / 66.9 / 90.6 / 91.2 / 26 / 25.4 | 81.2 / 73.8 / 91.9 / 90.1 / 29 / 12.5 | 77.9 / 72.4 / 91.3 / 87.7 / 26 / 25.4 |
| TUD-Stadtmitte (640×480, 179) | 77.3 / 69.4 / 82.3 / 95.2 / 9 / 31.1 | 77.9 / 65.8 / 82.6 / 96.0 / 14 / 26.9 | 77.4 / 71.2 / 82.9 / 94.8 / 10 / 14.5 | 78.5 / 68.4 / 82.6 / 96.2 / 9 / 26.9 |
| TUD-Campus (640×480, 71) | 58.8 / 66.3 / 79.4 / 80.1 / 3 / 31.1 | 58.8 / 66.3 / 79.4 / 80.1 / 3 / 26.9 | 62.1 / 70.3 / 79.4 / 82.8 / 3 / 17.8 | 60.2 / 69.8 / 79.4 / 81.7 / 5 / 26.7 |
| KITTI-17 (1224×370, 145) | 64.4 / 77.8 / 71.9 / 92.5 / 11 / 31.8 | 64.9 / 78.2 / 73.1 / 91.7 / 11 / 28.1 | 58.1 / 79.1 / 79.2 / 79.8 / 7 / 15.6 | 65.7 / 77.4 / 76.1 / 89.5 / 10 / 27.8 |
| Venice-2 (1920×1080, 600) | 34.1 / 41.9 / 42.4 / 84.2 / 24 / 30.3 | 33.5 / 42.0 / 45.8 / 79.5 / 36 / 24.4 | 35.9 / 54.0 / 60.6 / 71.4 / 36 / 9.8 | 40.6 / 49.8 / 50.9 / 83.7 / 29 / 24.2 |
| ADL-Rundle-6 (1920×1080, 525) | 48.6 / 50.2 / 61.5 / 84.1 / 63 / 30.4 | 52.0 / 54.1 / 64.3 / 85.2 / 58 / 24.6 | 45.7 / 48.6 / 69.8 / 75.2 / 52 / 10.4 | 51.8 / 53.0 / 64.1 / 85.2 / 62 / 24.7 |

Processing time — PETS09-S2L1, three runs (accuracy metrics were identical across runs)

| Input strategy | FPS (three runs) | Median ms per frame | Detector calls per frame |
|---|---|---|---|
| FULL | 30.65 / 30.60 / 30.76 | 32.5 | 1.0 |
| ADAPTIVE | 25.47 / 25.36 / 25.47 | 39.2 | 1.0 |
| OURS (per-ROI inference) | 12.54 / 12.50 / 12.54 | 69.2 | 2.36 |
| OURS_PACKED | 25.42 / 25.39 / 25.46 | 39.4 | 1.0 |

## SOMPT22 — nine training sequences, first 300 frames (10 s) of each (2,700 frames, 98,085 boxes)

Static surveillance cameras on 6–8 m poles. About 36 pedestrians per frame on average, far more crowded than the design target (a few people at a crosswalk). Single run.

| Input strategy | MOTA | IDF1 | Recall | Precision | ID switches | FPS |
|---|---|---|---|---|---|---|
| FULL | 18.0 | 27.9 | 19.5 | **93.6** | 182 | **30.2** |
| ADAPTIVE | 19.6 | 30.0 | 21.4 | 92.9 | **162** | 24.8 |
| OURS (per-ROI inference) | **29.3** | **42.6** | **33.0** | 90.6 | 295 | 8.7 |
| OURS_PACKED | 23.5 | 35.1 | 26.1 | 91.9 | 243 | 24.3 |

Per sequence (MOTA / Recall / median ms per frame)

| Sequence | Boxes | FULL | ADAPTIVE | OURS | OURS_PACKED | PACKED frames that fell back to full-frame |
|---|---|---|---|---|---|---|
| 02 (1280×720) | 8,252 | 13.4 / 17.3 / 32.9 | 15.7 / 22.6 / 38.7 | 17.3 / 27.3 / 97.9 | 16.5 / 25.5 / 39.1 | 40.3 % |
| 04 | 7,353 | 12.7 / 15.2 / 32.5 | 27.4 / 28.7 / 40.4 | 44.9 / 49.3 / 73.0 | 37.6 / 41.8 / 41.7 | 34.0 % |
| 05 | 9,025 | 18.7 / 23.6 / 33.4 | 15.5 / 19.3 / 39.7 | 30.0 / 40.4 / 158.2 | 33.4 / 41.5 / 42.2 | 10.0 % |
| 07 | 15,709 | 13.8 / 14.8 / 33.3 | 13.0 / 13.9 / 39.9 | 28.3 / 30.3 / 220.4 | 19.0 / 20.0 / 41.4 | 49.0 % |
| 08 | 17,269 | 10.2 / 11.3 / 32.7 | 11.8 / 13.3 / 40.3 | 20.6 / 22.9 / 102.8 | 15.9 / 17.0 / 41.5 | 10.0 % |
| 10 | 17,596 | 16.7 / 16.9 / 33.4 | 16.5 / 16.8 / 41.6 | 19.5 / 20.2 / 101.8 | 17.2 / 17.6 / 41.8 | 92.0 % |
| 11 | 8,249 | 32.5 / 33.2 / 33.3 | 39.3 / 40.4 / 41.0 | 45.5 / 46.9 / 72.4 | 39.0 / 40.2 / 41.1 | 97.0 % |
| 12 | 11,853 | 28.9 / 29.3 / 33.9 | 28.9 / 29.3 / 41.9 | 47.9 / 52.1 / 133.7 | 28.9 / 29.3 / 42.1 | 100.0 % |
| 13 | 2,779 | 33.4 / 35.4 / 32.2 | 32.9 / 37.6 / 39.7 | 16.7 / 21.1 / 68.3 | 18.2 / 24.9 / 39.6 | 11.3 % |

All sequences except 02 are 1920×1080. The fallback rate includes the periodic full-frame pass (one frame in ten in this configuration).

## Main observation

**Per-person ROI finds more people.** From FULL to per-ROI inference, Recall rose from 62.6 to 72.9 on MOT15 and from 19.5 to 33.0 on SOMPT22; IDF1 rose from 52.9 to 59.8 and from 27.9 to 42.6.

**Packing keeps part of that gain in one inference.** OURS_PACKED ran at 2.2× (MOT15) and 2.8× (SOMPT22) the speed of per-ROI inference, with MOTA 3.6 and 5.5 points above FULL. At the same speed as ADAPTIVE, its MOTA was 1.8 and 3.9 points higher.

**Takeaway:** On this device and detector, region selection trades a small amount of speed for accuracy. It does not make detection faster than plain full-frame input.

## Limitations

1. **FULL is the fastest.** Selecting and packing regions adds about 7–8 ms per frame.
2. **Packing gives back part of the Recall gain** (72.9 → 67.4 on MOT15, 33.0 → 26.1 on SOMPT22). The more people, the more each ROI must shrink.
3. **On SOMPT22-13 both ROI strategies were clearly worse than FULL** (MOTA 33.4 → 16.7 / 18.2). The cause has not been investigated.
4. **On SOMPT22 sequences 10, 11 and 12, OURS_PACKED fell back to full-frame on more than 90 % of frames.** Packing was effectively not exercised there.
5. **On MOT15 PETS09-S2L1, OURS_PACKED scored below ADAPTIVE** (MOTA 77.9 vs 81.3) because of additional false positives.
6. **ID switches increased on SOMPT22** (182 → 243 / 295). More people were detected, so the counts are not comparable; IDF1 is the identity metric used here.
7. **Recall for pedestrians under 60 px tall stayed at 2–4 % on SOMPT22** even with ROI. Small-pedestrian detection is not solved.
8. **Absolute scores are low.** The smallest YOLO model at 416 input was used; results are not comparable with leaderboard methods.
9. **A full-frame detector at a larger input was not measured.** Whether ROI Packing reaches a given accuracy more cheaply than enlarging the input is untested.
10. **SOMPT22 used only the first 10 s of each sequence, in a single run.** No statistical significance is claimed.

## Other measurements

| What | Result | Condition |
|---|---|---|
| CARLA clip (poster table) | OURS_PACKED 15.41 FPS · Recall 0.6429 · 0 ID switches / ADAPTIVE 21.14 · 0.6883 · 0 / FIXED 31.83 · 0.3636 · 1 | One 240-frame sequence |
| Real webcam, Cam1 on Pi 5 | Detection rate 88.5 %, 0 ID switches, about 10 FPS | One recorded clip |
| Real webcam, Cam2 | Early-departure cue did not trigger | Real-camera test |

In the CARLA clip ADAPTIVE had higher Recall than OURS_PACKED, while on the two public datasets the order was reversed in aggregate. The ranking depends on the sequence.

## Reproducing

See [`bench/README.md`](../bench/README.md) (Korean). Datasets are not included: MOT15 is downloaded by the script, and SOMPT22 must be obtained from its distributor (non-commercial research licence). Each dataset took about 15 minutes on the Pi 5.

**[System flow](system-flow.md)** · **[Vision method](vision-method.md)** · **[Main README](../README.md)**

---

_Numbers from different datasets or evaluation protocols should not be compared directly._
