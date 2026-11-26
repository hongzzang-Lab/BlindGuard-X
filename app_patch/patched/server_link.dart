// lib/services/server_link.dart
//
// 서버(Risk Engine)와 WebSocket으로 연결하는 곳. 메시지 형식은 "서버↔앱 메시지 형식 (초안 v1)" 문서 그대로.
//
//   앱 → 서버 : hello(연결 직후) · location(위치가 올 때마다, 약 1초) · drive(주행 시작·종료) · ping(1초마다)
//   서버 → 앱 : alert(구역·위험·물체) · pong(ping에 대한 답)
//
// 설계 의도
// - 화면 코드는 서버를 몰라도 되게 했다. 여기서 JSON을 앱의 AlertState로 바꿔서 내보내므로,
//   주행 화면은 데모 경고를 받던 것과 똑같이 받기만 하면 된다.
// - 연결이 끊기면 2초 뒤 자동으로 다시 접속한다. (주행 중 서버가 잠깐 재시작돼도 앱을 껐다 켤 필요 없게)
// - 서버 메시지가 3초 동안 하나도 안 오면 "끊김"으로 본다. (결정 사항 N = 3초 [임시])
//   pong이 1초마다 오므로, 정상이면 3초 안에 반드시 무언가 온다.
//
// 사용 패키지: web_socket_channel

import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart' show debugPrint;
import 'package:web_socket_channel/web_socket_channel.dart';
import '../models/drive_models.dart';
import '../zones.dart';
import 'device_id.dart';
import 'location_service.dart';

class ServerLink {
  ServerLink(this.url);
  final String url;

  static const _version = 1;
  static const _lostAfter = Duration(seconds: 3); // 결정 사항: 서버 끊김 판단 3초 [임시]
  static const _retryAfter = Duration(seconds: 2); // 재접속 간격 (디자인 선택값)

  /// 익명 기기 ID. 설치 후 처음 한 번 만들어 저장한 값을 계속 쓴다 (device_id.dart).
  /// connect()에서 읽어 오므로, 그 전에는 빈 값이다. (연결 전엔 어차피 메시지를 안 보냄)
  String deviceId = '';

  WebSocketChannel? _ch;
  StreamSubscription? _sub;
  Timer? _pingTimer, _lostTimer, _retryTimer;
  bool _closed = false;
  bool _driving = false; // 주행 중인지 기억 → 재접속하면 서버에 다시 알려줌

  final _alerts = StreamController<AlertState>.broadcast();
  final _quality = StreamController<LinkQuality>.broadcast();
  final _ping = StreamController<int>.broadcast();
  final _positions = StreamController<({double lat, double lng})>.broadcast();

  /// 서버가 보낸 경고 (앱 모양으로 바꾼 것)
  Stream<AlertState> get alerts => _alerts.stream;

  /// 서버 연결 상태 (양호 / 끊김)
  Stream<LinkQuality> get quality => _quality.stream;

  /// 서버 연결 속도 (ping을 보내고 pong을 받기까지 걸린 ms)
  Stream<int> get pingMs => _ping.stream;

  /// 서버 GPS 필터가 다듬은 내 위치 (위치를 보낼 때마다 1번씩 돌아옴)
  /// 화면의 구역 표시를 서버 판단과 같은 위치로 하려고 쓴다.
  Stream<({double lat, double lng})> get positions => _positions.stream;

  // ───────────── 연결 ─────────────

  Future<void> connect() async {
    if (_closed) return;
    try {
      if (deviceId.isEmpty) {
        deviceId = await DeviceId.get();
        debugPrint('[서버] 기기 ID: $deviceId');
      }
      debugPrint('[서버] 접속 시도: $url');
      final ch = WebSocketChannel.connect(Uri.parse(url));
      await ch.ready; // 실제로 연결될 때까지 기다림 (실패하면 아래 catch로)
      _ch = ch;
      _sub = ch.stream.listen(_onMessage,
          onDone: _onDisconnected, onError: (_) => _onDisconnected());
      debugPrint('[서버] 연결됨');
      _send({'type': 'hello', 'device_id': deviceId});
      // 주행 중에 끊겼다 다시 붙은 경우: 서버는 새 연결을 "주행 전"으로 알므로 다시 알려준다
      if (_driving) _sendDriveMsg(true);
      _pingTimer = Timer.periodic(const Duration(seconds: 1), (_) {
        _send({'type': 'ping', 't': DateTime.now().millisecondsSinceEpoch});
      });
      _armLost();
    } catch (e) {
      debugPrint('[서버] 접속 실패: $e');
      _onDisconnected();
    }
  }

  void _onDisconnected() {
    _cleanup();
    _quality.add(LinkQuality.lost);
    if (_closed) return;
    _retryTimer = Timer(_retryAfter, connect); // 잠시 뒤 다시 접속
  }

  void _armLost() {
    _lostTimer?.cancel();
    _lostTimer = Timer(_lostAfter, () {
      debugPrint('[서버] ${_lostAfter.inSeconds}초 동안 응답 없음 → 끊김');
      _quality.add(LinkQuality.lost);
    });
  }

  void _cleanup() {
    _pingTimer?.cancel();
    _lostTimer?.cancel();
    _retryTimer?.cancel();
    _sub?.cancel();
    _ch = null;
  }

  Future<void> close() async {
    _closed = true;
    _cleanup();
    await _alerts.close();
    await _quality.close();
    await _ping.close();
    await _positions.close();
  }

  // ───────────── 앱 → 서버 ─────────────

  void _send(Map<String, dynamic> msg) {
    final ch = _ch;
    if (ch == null) return; // 연결 안 됐으면 버림 (위치는 1초마다 새로 오므로 쌓아둘 필요 없음)
    msg['v'] = _version;
    ch.sink.add(jsonEncode(msg));
  }

  void sendLocation(LocationSnapshot s) {
    if (s.lat == null || s.lng == null) return;
    _send({
      'type': 'location',
      'device_id': deviceId,
      'ts': DateTime.now().millisecondsSinceEpoch,
      'lat': s.lat,
      'lng': s.lng,
      'accuracy_m': s.accuracyM,
      'speed_mps': s.speedMps,
      'heading_deg': s.headingDeg,
      // [실물 테스트 11/21] 실험 서버용 추가 값 (예전 서버는 모르는 값이라 무시)
      'heading_raw_deg': s.rawHeadingDeg,
      'heading_acc': s.headingAccuracyDeg,
      'speed_acc': s.speedAccuracyMps,
      'fix_ts': s.fixTsMs,
    });
  }

  void sendDrive(bool start) {
    _driving = start;
    _sendDriveMsg(start);
  }

  void _sendDriveMsg(bool start) {
    _send({
      'type': 'drive',
      'device_id': deviceId,
      'ts': DateTime.now().millisecondsSinceEpoch,
      'state': start ? 'start' : 'end',
    });
  }

  // ───────────── 서버 → 앱 ─────────────

  void _onMessage(dynamic raw) {
    _armLost(); // 무엇이든 오면 "살아 있음"
    _quality.add(LinkQuality.good);
    final Map<String, dynamic> m;
    try {
      m = jsonDecode(raw as String) as Map<String, dynamic>;
    } catch (_) {
      debugPrint('[서버] 읽을 수 없는 메시지: $raw');
      return;
    }
    switch (m['type']) {
      case 'pong':
        final t = m['t'];
        if (t is int) _ping.add(DateTime.now().millisecondsSinceEpoch - t);
      case 'alert':
        _alerts.add(_toAlertState(m));
        // [실물 테스트 11/21] 받은 즉시 회신 → 서버가 '서버→폰 전달 지연'을 잼 (예전 서버는 모르는 메시지로 무시)
        final risk = m['risk'];
        _send({
          'type': 'alert_ack',
          'seq': m['seq'],
          'level': risk is Map ? risk['level'] : null,
          't': DateTime.now().millisecondsSinceEpoch,
        });
      case 'position':
        final lat = (m['lat'] as num?)?.toDouble();
        final lng = (m['lng'] as num?)?.toDouble();
        if (lat != null && lng != null) _positions.add((lat: lat, lng: lng));
      default:
        debugPrint('[서버] 모르는 메시지: ${m['type']}');
    }
  }

  // ───────────── JSON → 앱 모양 ─────────────

  static const _kindKo = {
    'pedestrian': '보행자',
    'bicycle': '자전거',
    'kickboard': '킥보드',
    'motorcycle': '오토바이',
    'vehicle': '차량',
  };

  static AlertState _toAlertState(Map<String, dynamic> m) {
    final zone = (m['zone'] as Map?) ?? const {};
    final risk = m['risk'] as Map?;

    final state = switch (zone['state']) {
      'alert' => ZoneState.alert,
      'monitoring' => ZoneState.monitoring,
      _ => ZoneState.outside,
    };

    // 레이더 최대 거리 = 그 구역 반경 (결정 사항). 구역을 모르면 첫 구역 반경.
    final zoneId = zone['id'];
    final radarMaxM = kZones
        .firstWhere((z) => z.id == zoneId, orElse: () => kZones.first)
        .radiusM;

    final objects = <DetectedObject>[
      for (final o in (m['objects'] as List? ?? const []))
        DetectedObject(
          kind: _kindKo[o['kind']] ?? '물체',
          angleDeg: (o['angle_deg'] as num?)?.toDouble() ?? 0,
          // 레이더는 0~1 비율로 그리므로 m를 최대 거리로 나눈다 (바깥으로 넘치면 가장자리에)
          distanceRatio:
              (((o['distance_m'] as num?)?.toDouble() ?? radarMaxM) / radarMaxM)
                  .clamp(0.0, 1.0),
          target: o['target'] == true,
        ),
    ];

    return AlertState(
      zone: state,
      level: switch (risk?['level']) {
        'caution' => RiskLevel.caution,
        'warning' => RiskLevel.warning,
        'danger' => RiskLevel.danger,
        _ => null,
      },
      side: switch (risk?['side']) {
        'front' => Side.front,
        'right' => Side.right,
        'back' => Side.back,
        'left' => Side.left,
        _ => null,
      },
      objectKind: _kindKo[risk?['object_kind']],
      objects: objects,
    );
  }
}
