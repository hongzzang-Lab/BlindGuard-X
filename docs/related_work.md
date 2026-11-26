# 선행 연구

ROI를 한 장에 모아 검출기를 한 번만 돌리는 발상은 이 프로젝트가 처음이 아니다. 2026-11-26에 찾은 범위에서 가까운 연구를 적는다. 대부분 초록 수준으로 확인했다.

## ROI를 한 장에 모아 추론 (Packing)

| 연구 | 요약 | 이 프로젝트와 다른 점 |
|---|---|---|
| [Pack and Detect (PaD)](https://arxiv.org/abs/1809.01701), 2018 | 이전 프레임 검출로 ROI를 정해 줄인 프레임 한 장에 모음. 실패하면 전체 프레임 | 목적이 연산 절감. 움직임으로 새 물체를 찾는 단계 없음 |
| [FlexPatch](https://juheonyi.github.io/files/FlexPatch.pdf), INFOCOM 2022 | 추적 실패가 예상되는 상자와 새 물체 후보 칸을 고정 크기 묶음에 bin packing. 스마트폰, Tiny YOLO-v4, LK 추적 | 검출은 몇 프레임마다. 새 물체를 에지로 찾음 |
| [MOSAIC](https://arxiv.org/abs/2305.03222), MMSys 2023 | 배경 차분 + 추적기로 ROI를 찾아 640×640 캔버스에 bin packing. Jetson TX2, YOLOv5s | 여러 카메라를 한 캔버스에 모으는 것이 주목적. GPU 장치 |
| [Mondrian](https://arxiv.org/abs/2403.07598), 2024 | ROI를 정확도가 유지되는 최소 크기로 줄여 캔버스에 채움 | 목적이 처리량. ROI를 키우지 않고 줄임 |
| [UFPMP-Det](https://arxiv.org/abs/2112.10415), AAAI 2022 | 거친 검출 결과를 묶어 모자이크 한 장으로 만들어 재검출 | 드론 정지 영상. ROI를 찾는 데 검출기를 한 번 더 씀 |
| [Canvas-based Attention Scheduling](https://par.nsf.gov/servlets/purl/10547298), RTAS 2024 | 관심 영역을 작은 캔버스 프레임에 모아 처리하는 실시간 스케줄링 틀 | 스케줄링 이론 중심 |

## 움직임으로 볼 영역 정하기

| 연구 | 요약 |
|---|---|
| [Self-Cueing Attention Scheduling](https://par.nsf.gov/servlets/purl/10348765), RTAS 2022 | Optical Flow 기반 모듈이 프레임의 어느 부분을 볼지와 중요도를 정함 |

## 잘라서 따로 추론 (이 저장소의 ADAPTIVE · ROI별 추론에 해당)

- [Remix](https://www.microsoft.com/en-us/research/publication/flexible-high-resolution-object-detection-on-edge-devices-with-tunable-latency/), MobiCom 2021 — 지연 예산 안에서 이미지를 불균등 분할
- [Elf](https://doi.org/10.1145/3447993.3448628), MobiCom 2021 — 과거 검출로 영역을 예측해 나눠서 여러 서버로
- [EdgeDuet](https://zhouzimu.github.io/paper/infocom21-wang.pdf), INFOCOM 2021 — 작은 물체 부분만 타일로 잘라 엣지 서버로
- [AutoFocus](https://arxiv.org/abs/1812.01600), ICCV 2019 — 작은 물체가 있을 법한 영역만 높은 배율로
- [ClusDet](https://arxiv.org/abs/1904.08008), ICCV 2019 — 물체가 몰린 영역을 찾아 따로 검출
- [SAHI](https://arxiv.org/abs/2202.06934), 2022 — 조각내어 추론하는 범용 틀

## 이 프로젝트의 위치

개별 요소(ROI Packing, 움직임 기반 ROI, 추적 기억, 전체 화면 복귀)는 모두 선행 연구에 있다. 이 프로젝트가 한 일은 다음이다.

- Optical Flow와 Track Memory를 함께 써서 ROI를 만들고 416×416 한 장에 Packing하는 조합을 구현
- GPU 없는 Raspberry Pi 5(CPU, NCNN)에서 동작
- 도로변 고정 카메라 → 보행자 위치·속도 → 차량별 CPA → 운전자 폰 경고까지 연결
- 공개 데이터셋 2종으로 네 방식을 같은 조건에서 비교 (`docs/benchmark.md`)
