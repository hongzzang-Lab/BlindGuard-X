// lib/services/location_service.dart
//
// GPS 위치를 계속 받아서 "GPS 상태 / 정확도 / 진행 방향"으로 바꿔주는 곳.
//
// 백그라운드 동작의 핵심이 여기 있다.
// - 주행을 시작하면 start(background: true)로 다시 시작한다.
// - 그러면 안드로이드가 상단 알림바에 "BlindGuard 실행 중" 알림을 띄우고(포그라운드 서비스),
//   다른 내비 앱을 켜도 이 앱이 꺼지지 않고 계속 GPS를 받는다.
// - 앱이 살아 있으니 경고 확인과 음성 안내도 계속 돈다.
// (geolocator 패키지의 foregroundNotificationConfig 기능을 사용)
//
// 칼만 필터는 아직 넣지 않았다. 서버(Python)에서 하기로 했으므로 여기서는 원본 GPS를 그대로 쓴다.

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:geolocator/geolocator.dart';
import '../config.dart';
import '../models/drive_models.dart';
import '../zones.dart';

/// 한 순간의 GPS 상태
class LocationSnapshot {
  const LocationSnapshot({
    required this.quality,
    this.accuracyM,
    this.headingDeg,
    this.lat,
    this.lng,
    this.speedMps,
  });
  final LinkQuality quality; // 양호 / 약함 / 끊김
  final double? lat; // 위도. 감시 구역 판단에 사용. 모르면 null
  final double? lng; // 경도
  final double? speedMps; // 속도(m/s). 서버에 보내는 location 메시지용
  final double? accuracyM; // 오차 반경(m). 펼친 바텀시트에 표시
  final double? headingDeg; // 진행 방향(북=0°, 시계 방향). 아직 모르면 null

  static const lost = LocationSnapshot(quality: LinkQuality.lost);
}

/// 각도(0~360)를 8방위 글자로. 진행 방향 표시용 (북, 북동, 동 …)
/// 표시는 2글자까지로 정했다 → 2글자 이내 이름이 있는 방위는 8개(45°씩)뿐이다.
/// (감시 구역의 방향 판단은 글자가 아니라 각도 숫자로 하므로 표시와 상관없다)
String headingText(double deg) {
  const names = ['북', '북동', '동', '남동', '남', '남서', '서', '북서'];
  // 각 방위가 자기 각도 ±22.5° 범위를 맡도록 반 칸(22.5°)만큼 밀어서 나눈다
  final i = (((deg % 360) + 360 + 22.5) % 360 ~/ 45) % 8;
  return names[i];
}

class LocationService {
  final _out = StreamController<LocationSnapshot>.broadcast();

  /// 화면은 이 흐름(stream)을 듣고 있다가 값이 오면 다시 그린다
  Stream<LocationSnapshot> get stream => _out.stream;

  StreamSubscription<Position>? _sub;
  Timer? _lostTimer;
  Timer? _demoTimer;
  double? _lastHeading; // 정지 중에도 유지할 마지막 방향

  /// 크롬 데모일 때만 true (안드로이드에서는 항상 진짜 GPS)
  bool get _useDemo => kIsWeb && kDemoGpsOnWeb;

  /// GPS 받기 시작. background가 true면 안드로이드에서 상시 알림과 함께 백그라운드 유지.
  Future<void> start({required bool background}) async {
    await stop();

    if (_useDemo) {
      // [데모] 1초 뒤 GPS 양호 + 방향 북동 + 위치는 테스트 구역 중심으로 흉내
      // (크롬에서도 "구역 안" 화면을 확인할 수 있게)
      _demoTimer = Timer(const Duration(seconds: 1), () {
        _emit(LocationSnapshot(
            quality: LinkQuality.good,
            accuracyM: 8,
            headingDeg: 42,
            lat: kZones.first.lat,
            lng: kZones.first.lng));
      });
      return;
    }

    try {
      // 1) 휴대폰 위치 기능이 켜져 있는지
      if (!await Geolocator.isLocationServiceEnabled()) {
        _emit(LocationSnapshot.lost);
        return;
      }
      // 2) 위치 권한 (권한 화면에서 이미 받았겠지만, 혹시 꺼져 있으면 한 번 더 요청)
      var perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        perm = await Geolocator.requestPermission();
      }
      if (perm == LocationPermission.denied ||
          perm == LocationPermission.deniedForever) {
        _emit(LocationSnapshot.lost);
        return;
      }

      // 3) 위치 흐름 시작
      _sub = Geolocator.getPositionStream(
        locationSettings: _settings(background),
      ).listen(
        _onPosition,
        onError: (_) => _emit(LocationSnapshot.lost),
      );
      _armLostTimer();
    } catch (e) {
      debugPrint('GPS 시작 실패: $e');
      _emit(LocationSnapshot.lost);
    }
  }

  LocationSettings _settings(bool background) {
    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      return AndroidSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 0, // 조금만 움직여도 매번 값 받기
        intervalDuration: const Duration(seconds: 1),
        // 이 설정이 있으면 "포그라운드 서비스"로 돌아서 백그라운드에서도 안 꺼진다
        foregroundNotificationConfig: background
            ? const ForegroundNotificationConfig(
                notificationTitle: 'BlindGuard 실행 중',
                notificationText: '위험할 때 음성으로 알려드려요',
                notificationChannelName: 'BlindGuard 주행',
                enableWakeLock: true, // 화면이 꺼져도 CPU가 잠들지 않게
                setOngoing: true, // 사용자가 알림을 밀어서 지우지 못하게
              )
            : null,
      );
    }
    return const LocationSettings(
      accuracy: LocationAccuracy.high,
      distanceFilter: 0,
    );
  }

  void _onPosition(Position p) {
    _armLostTimer();
    final quality = p.accuracy <= kGoodGpsAccuracyM
        ? LinkQuality.good
        : LinkQuality.weak;
    // 충분히 빠르게 움직일 때만 방향을 새로 믿고, 아니면 마지막 방향 유지
    if (p.speed >= kHeadingMinSpeedMps && p.heading >= 0) {
      _lastHeading = p.heading;
    }
    _emit(LocationSnapshot(
      quality: quality,
      accuracyM: p.accuracy,
      headingDeg: _lastHeading,
      lat: p.latitude,
      lng: p.longitude,
      speedMps: p.speed,
    ));
  }

  /// 일정 시간 GPS 값이 안 오면 "끊김"으로 바꾸는 타이머. 값이 올 때마다 다시 맞춘다.
  void _armLostTimer() {
    _lostTimer?.cancel();
    _lostTimer = Timer(kGpsLostAfter, () => _emit(LocationSnapshot.lost));
  }

  void _emit(LocationSnapshot s) {
    if (!_out.isClosed) _out.add(s);
  }

  /// GPS 받기 중지. 포그라운드 서비스와 상시 알림도 같이 사라진다.
  Future<void> stop() async {
    _lostTimer?.cancel();
    _demoTimer?.cancel();
    await _sub?.cancel();
    _sub = null;
  }

  Future<void> dispose() async {
    await stop();
    await _out.close();
  }
}
