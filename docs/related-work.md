# Related Work

Packing regions of interest into one frame for a single detector pass is **not** an idea introduced by this project. This page lists the closest prior work found in a search on November 26, 2026. Most entries were checked at abstract level only, so the stated differences may be incomplete.

## Packing regions into one inference

| Work | Summary | How this project differs |
|---|---|---|
| [Pack and Detect (PaD)](https://arxiv.org/abs/1809.01701), 2018 | ROIs from the previous frame's detections are packed into a reduced-size frame; falls back to the full frame when packing fails | Goal there is compute reduction; no motion step for finding new objects |
| [FlexPatch](https://juheonyi.github.io/files/FlexPatch.pdf), INFOCOM 2022 | Patches likely to suffer tracking failure, plus candidate cells for new objects, are bin-packed into a fixed-size cluster; smartphone, Tiny YOLO-v4, Lucas–Kanade tracking | Detection runs every few frames; new objects are found from edge intensity |
| [MOSAIC](https://arxiv.org/abs/2305.03222), MMSys 2023 | Background subtraction and a tracker select regions that are bin-packed onto a 640×640 canvas; Jetson TX2, YOLOv5s | Main goal is multiplexing several cameras; GPU device |
| [Mondrian](https://arxiv.org/abs/2403.07598), 2024 | ROIs are rescaled to the smallest size that keeps accuracy and packed into canvases | Goal is throughput; ROIs are shrunk rather than enlarged |
| [UFPMP-Det](https://arxiv.org/abs/2112.10415), AAAI 2022 | A coarse detector's regions are merged into a mosaic that is detected again | Still drone images; a detector pass is spent on finding regions |
| [Canvas-based attention scheduling](https://par.nsf.gov/servlets/purl/10547298), RTAS 2024 | Real-time scheduling framework that packs focus regions into smaller canvas frames | Centred on scheduling theory |

## Choosing regions from motion

| Work | Summary |
|---|---|
| [Self-cueing attention scheduling](https://par.nsf.gov/servlets/purl/10348765), RTAS 2022 | An optical-flow module decides which sub-frame regions to inspect and how critical they are |

## Cropping without packing

These correspond to the `ADAPTIVE` and per-ROI strategies in this repository.

- [Remix](https://www.microsoft.com/en-us/research/publication/flexible-high-resolution-object-detection-on-edge-devices-with-tunable-latency/), MobiCom 2021 — non-uniform image partitioning under a latency budget
- [Elf](https://doi.org/10.1145/3447993.3448628), MobiCom 2021 — regions predicted from past detections, partitioned and offloaded to several servers
- [EdgeDuet](https://zhouzimu.github.io/paper/infocom21-wang.pdf), INFOCOM 2021 — small-object tiles offloaded to an edge server
- [AutoFocus](https://arxiv.org/abs/1812.01600), ICCV 2019 — finer scales only where small objects are likely
- [ClusDet](https://arxiv.org/abs/1904.08008), ICCV 2019 — clustered regions detected separately
- [SAHI](https://arxiv.org/abs/2202.06934), 2022 — generic sliced inference

## Where this project stands

Every individual element here—ROI packing, motion-based region selection, track memory, full-frame fallback—exists in prior work. What this project contributes is narrower:

- an implementation that builds ROIs from **Optical Flow and Track Memory together** and packs them into a 416×416 canvas;
- running it on a **GPU-less Raspberry Pi 5** (CPU, NCNN);
- connecting it to a roadside use case: fixed cameras → pedestrian position and velocity → per-vehicle CPA → an alert on the driver's phone;
- a like-for-like comparison of four input strategies on two public datasets ([evaluation](evaluation.md)).

> No claim of novelty for ROI packing itself is made. A search cannot prove absence; closer work may exist.

[Back to README](../README.md) · [Vision method](vision-method.md) · [Evaluation](evaluation.md)
