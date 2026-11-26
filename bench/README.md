# bench — 공개 벤치마크로 Cam1 방식 비교

우리 Cam1 Vision 방식(ROI 를 사람마다 만들어 한 장에 모아 YOLO 1회)을 공개 보행자 추적 벤치마크
**MOTChallenge MOT15 (2DMOT2015)** 의 고정 카메라 학습 구간에 그대로 돌려, 방식별 정확도와 처리 시간을 잰다.

## 실행 (Pi 에서)
```bash
cd ~/realtest && bash bench/run_on_pi.sh
```
- 데이터가 없으면 `bench/data/` 로 자동으로 받는다 (약 1.3 GB, 한 번만).
- 결과: `bench/results/<날짜_시각>/results.md` (표), `results.csv`, `results.json`, 장면 그림(jpg).

## 무엇을 비교하나
| 이름 | 내용 | 포스터 표 |
|---|---|---|
| full | 프레임 전체를 416 으로 줄여 YOLO 1회 | (기준선) |
| adaptive | 움직임+추적 위치를 감싼 큰 ROI 하나, YOLO 1회 | ADAPTIVE |
| crops | 사람(ROI)마다 따로 잘라 YOLO 를 ROI 수만큼 | OURS |
| packed | ROI 들을 416×416 한 장에 모아 YOLO 1회 | OURS_PACKED |

검출 모델(YOLO26n-NCNN 416), 추적기(ByteTrack), Cam1 설정(`simdemo/site_sim.json`)은 모두 같다.
벤치마크에 맞춘 조정은 하지 않는다.

## 지표
- 정확도: MOTA, IDF1, Recall, Precision, ID 변경(IDSW) — CLEAR MOT, IoU 0.5. 이 스크립트가 직접 계산한다
  (공식 평가 도구와 소수점 수준 차이가 날 수 있음. 방식끼리의 상대 비교용).
- 작은 사람 Recall: 정답 상자 높이 60 px 미만인 보행자만의 Recall (해상도 가설을 직접 보는 값).
- 처리 시간: 이 장치에서 프레임 하나를 처리하는 시간 (영상 읽기 제외), FPS = 1000 / 평균 ms.

## 읽을 때 주의
- 고정 카메라 구간만 쓴다 (우리 방식은 Optical Flow 로 움직임을 찾으므로 카메라가 움직이면 성립하지 않음).
- 가장 작은 YOLO 를 416 입력으로 쓰므로 절대 점수는 순위표의 방법들보다 낮다. 볼 것은 방식 간 차이.
- 구간당 1회 실행. 통계적 유의성은 주장할 수 없다 → "예비 결과".
- 사람 수가 많은 구간(Venice-2, ADL-Rundle-6)은 설계 범위(횡단보도의 몇 명) 밖이라 한 장에 다 못 모으면
  전체 화면으로 되돌아간다. 표의 '전체 화면 비율'(csv 의 fullframe_pct)로 확인.

## 두 번째 벤치마크 — SOMPT22 (감시용 고정 카메라)
MOT 계열이 아닌 데이터셋. 6~8 m 기둥에 단 고정 카메라로 찍은 보행자 영상이라 우리 설치 조건(길가 고정 카메라)과 가깝다.
구글 드라이브로 배포해서 자동으로 받을 수 없다 → 노트북 브라우저로 받아, 필요한 부분만 Pi 로 보낸다.

```bash
# 노트북: https://sompt22.github.io/ 의 "Download Training Set" 으로 ~/Downloads 에 받은 뒤
cd <저장소 폴더>
bash bench/send_to_pi.sh                 # 스크립트 보내기
python3 bench/prep_sompt22.py            # 풀기 → 구간 확인 → 구간마다 앞쪽 300프레임 + 정답을 Pi 로

# Pi
cd ~/realtest && bash bench/run_on_pi.sh --bench sompt22
```
- 정답이 공개된 학습용 구간 전부를 돌린다 (시험용 구간은 정답이 비공개라 쓸 수 없음).
- 구간마다 앞쪽 300프레임(10초)만 쓴다. 표에 그렇게 적힌다.
- 프레임당 사람이 평균 수십 명이라 우리 설계 범위(횡단보도의 몇 명)보다 훨씬 붐빈다. 한 장에 다 못 모으면
  전체 화면으로 되돌아가며, 그 비율이 실행 화면의 '전체화면 %'로 나온다.
- 중간에 Ctrl+C 를 눌러도 끝난 구간까지는 저장된다.
- 라이선스: 비상업 연구용.
