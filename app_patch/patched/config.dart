// lib/config.dart
//
// 아직 진짜로 연결되지 않은 부분을 "흉내(데모)"로 돌릴지 정하는 스위치 모음.
// 하나씩 실제 기능이 붙을 때마다 false로 바꾸면 된다.
//
//   kDemoServer  : AI 서버 대신 1초 뒤 "연결됨"으로 흉내
//                  → FastAPI WebSocket 서버를 붙이면 false
//   kDemoAlerts  : 서버 경고 대신 정해진 순서(탐지 구역 아님 → 사각지대 → 주의 → 경고 → 긴급)로
//                  경고를 흉내. 화면·음성·기록이 제대로 도는지 확인용
//                  → 서버에서 실제 경고 JSON을 받으면 false
//   (kDemoServer·kDemoAlerts를 false로 하면 아래 kServerUrl의 테스트 서버/실제 서버에 접속한다.
//    그때는 kDemoLinkStates도 false로 해야 진짜 연결 상태가 화면에 보인다.)
//   kDemoGpsOnWeb: 크롬(웹)에서 확인할 때는 GPS를 흉내.
//                  안드로이드(에뮬레이터·실제 폰)에서는 이 값과 상관없이 항상 진짜 GPS를 쓴다.

import 'package:flutter/foundation.dart' show kIsWeb;

const bool kDemoServer = false;
const bool kDemoAlerts = false;
const bool kDemoGpsOnWeb = true;

/// 서버 주소 (결정 사항 "서버 주소")
/// - 에뮬레이터: 10.0.2.2 = 에뮬레이터에서 본 "개발 PC 자신" (Android 공식 문서)
/// - 크롬(웹): 같은 PC이므로 localhost
/// - 실제 폰: PC의 와이파이 IP로 바꾼다 (예: ws://192.168.0.10:8000/ws). 폰과 PC가 같은 와이파이여야 함
/// 8000 = 서버 실행 도구 uvicorn의 기본 포트
const String kServerUrlEmulator = 'ws://10.0.2.2:8000/ws';
const String kServerUrlWeb = 'ws://localhost:8000/ws';

/// 클라우드(Render 무료) 서버 주소.
/// - https 주소이므로 WebSocket도 암호화된 wss:// 를 쓴다 (ws:// 는 암호화 안 된 것)
/// - 끝의 /ws 는 서버 코드의 @app.websocket("/ws") 와 같아야 한다
/// - 15분간 접속이 없으면 서버가 잠들어서, 첫 접속은 깨어나는 동안 "끊김"이 잠깐 보일 수 있다
/// [공개 저장소] 팀 서버 주소는 지웠다. 자기 서버 주소로 바꾸거나,
///   빌드할 때 --dart-define=SERVER_URL=wss://<내 서버>/ws 로 지정한다.
const String kServerUrlCloud = 'wss://YOUR-SERVER.example.com/ws';

/// true: 클라우드 서버에 접속 (PC 서버를 안 켜도 됨, 실제 폰으로 어디서나 테스트)
/// false: 내 PC의 테스트 서버에 접속 (에뮬레이터 10.0.2.2 / 크롬 localhost)
const bool kUseCloudServer = true;

/// 앱이 실제로 접속할 주소. 위 스위치 하나로 고른다.
/// [실물 테스트 11/21] 빌드할 때 서버 주소를 직접 지정 (값을 안 주면 아래 기존 규칙 그대로)
///   flutter build apk --release --dart-define=SERVER_URL=ws://10.42.0.1:8000/ws
///   flutter run --dart-define=SERVER_URL=ws://10.42.0.1:8000/ws
/// 10.42.0.1 = 우분투 노트북 '핫스팟'의 노트북 주소 (실험 서버 server/exp_server.py)
const String kServerUrlOverride = String.fromEnvironment('SERVER_URL');

String get kServerUrl => kServerUrlOverride.isNotEmpty
    ? kServerUrlOverride
    : (kUseCloudServer
        ? kServerUrlCloud
        : (kIsWeb ? kServerUrlWeb : kServerUrlEmulator));

/// [데모] 주행 중 GPS "약함", 서버 "끊김" 상태를 일부러 만들어서
/// 바텀시트 색(노랑/빨강)과 끊김 배너·음성이 제대로 나오는지 확인한다.
/// 순서는 demo_alert_source.dart 맨 위 설명 참고 (38초 GPS 약함, 44초 서버 끊김, 50초 복구).
/// → 안드로이드에서 진짜 GPS 상태를 보고 싶을 때는 false로 바꾼다. (true면 흉내 값이 진짜 값을 덮어씀)
const bool kDemoLinkStates = false;

/// [데모] true면 "탐지 구역 아님 / 구역 안"을 데모 순서(0~6초 밖, 6초부터 안)대로 흉내.
/// false면 진짜 GPS 위치로 판단한다 (lib/zones.dart의 구역 + services/zone_service.dart).
/// 위험 경고(주의·경고·긴급) 자체는 서버가 없으므로 계속 데모 순서를 쓰되,
/// "구역 밖"이면 경고를 띄우지 않는다.
const bool kDemoZone = false;

/// GPS 정확도(오차 반경, m)가 이 값 이하면 "양호", 넘으면 "약함".
/// [임시 기준] 감시 구역 폭보다 확실히 작아야 한다. GPS 오차 실험 후 팀이 정한다.
const double kGoodGpsAccuracyM = 20;

/// 이 속도(m/s) 이상으로 움직일 때만 GPS 진행 방향을 믿는다. 느리면 마지막 방향을 유지한다.
/// 근거: 갤럭시 A25 실측 (10/23 분석)
///   - 버스 탑승 기록에서 폰이 준 방향과 실제 이동 방향(앞뒤 위치로 계산)의 차이를 속도별로 비교
///     · 구역 판단의 방향 허용각 ±45°를 넘게 틀린 비율
///       1.5~2 m/s: 2.7% / 2~3 m/s: 2.0% / 3~4 m/s: 0.6% / 4 m/s 이상: 0% (전 구간)
///   - 4 m/s는 "이 기록에서 방향이 허용각을 넘게 틀린 적이 없는 가장 낮은 속도"
///   - 정지 25시간 중 가짜로 속도가 이 기준을 넘은 횟수: 1.5 m/s → 49번 / 4 m/s → 20번
/// 단점: 4 m/s보다 느리게 움직이는 동안(천천히 회전 등)은 방향이 갱신되지 않고 마지막 방향을 쓴다.
/// 한계: 폰 1대, 버스 탑승 기록. 승용차 거치 조건은 미확인.
const double kHeadingMinSpeedMps = 4.0;

/// GPS 값이 이 시간 동안 안 오면 "끊김"으로 본다.
/// 근거: 갤럭시 A25 정지 25시간 실측 (10/18 01:14 ~ 10/19 02:20, 위치 72,256개)
///   - 평소 간격은 1초, 17시 이후로는 5~6초 간격이 자주 나옴 (폰이 값을 덜 자주 줌)
///   - 기준별 "끊김" 판정 횟수:  5초 → 3,820번 / 6초 → 10번 / 7초 → 5번 / 8초 → 4번 / 10초 → 4번
///   - 8초부터는 더 늘려도 횟수가 안 줄어듦 = 남은 4번은 진짜로 값이 끊긴 경우
///   - 그래서 "가짜 끊김은 없애면서, 진짜 끊김은 가장 빨리 알아채는" 8초로 정함
///   - 이 4번을 뺀 가장 긴 간격은 7.2초였음
/// 한계: 폰 1대(A25), 한 장소, 정지 상태 기록. 다른 폰·주행 중에는 다를 수 있다.
const Duration kGpsLostAfter = Duration(seconds: 8);
