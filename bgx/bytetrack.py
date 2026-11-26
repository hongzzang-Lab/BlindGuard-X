# bgx/bytetrack.py
#
# ByteTrack (Zhang et al., ECCV 2022) 를 의존성 없이 직접 구현한 것.
# ROI Packing 을 쓰면 탐지기 입력(포장된 캔버스)의 좌표가 매 프레임 바뀌므로,
# 탐지 결과를 원본 프레임 좌표로 되돌린 '다음에' 추적해야 한다.
# 그래서 ultralytics 의 model.track() 대신 이 추적기를 쓴다 (ultralytics 버전에 따라 내부 API가 바뀌는 문제도 피함).
#
# 핵심 아이디어 (BYTE):
#   1) 신뢰도 높은 탐지(≥ high)로 기존 트랙과 먼저 짝짓기 (IoU, 점수 융합)
#   2) 짝 못 찾은 트랙을 신뢰도 '낮은' 탐지(low ~ high)로 한 번 더 짝짓기 → 가려짐·흐림에 강함
#   3) 남은 높은 탐지 중 new_track 이상만 새 트랙, 짝 못 찾은 트랙은 track_buffer 프레임 동안 '잃어버림' 상태로 보관
# 기본값은 ultralytics bytetrack.yaml 과 같다.

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # noqa: BLE001
    _lsa = None


# ───────────── 칼만 필터 (x, y, 종횡비 a, 높이 h) — ByteTrack 원본과 같은 형태 ─────────────
class KalmanXYAH:
    def __init__(self):
        n, dt = 4, 1.0
        self.F = np.eye(2 * n)
        for i in range(n):
            self.F[i, n + i] = dt
        self.Hm = np.eye(n, 2 * n)
        self.w_pos = 1.0 / 20
        self.w_vel = 1.0 / 160

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.w_pos * h, 2 * self.w_pos * h, 1e-2, 2 * self.w_pos * h,
               10 * self.w_vel * h, 10 * self.w_vel * h, 1e-5, 10 * self.w_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.w_pos * h, self.w_pos * h, 1e-2, self.w_pos * h,
               self.w_vel * h, self.w_vel * h, 1e-5, self.w_vel * h]
        Q = np.diag(np.square(std))
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.w_pos * h, self.w_pos * h, 1e-1, self.w_pos * h]))
        S = self.Hm @ cov @ self.Hm.T + R
        K = cov @ self.Hm.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.Hm @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


def _xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def _xyah_to_xyxy(m):
    x, y, a, h = m[:4]
    w = a * h
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a)[:, None, :]
    b = np.asarray(b)[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def linear_assignment(cost, thresh):
    """cost <= thresh 인 짝만 인정. scipy 가 없으면 탐욕 매칭."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    matches = []
    if _lsa is not None:
        c = cost.copy()
        c[c > thresh] = thresh + 1e4
        r, k = _lsa(c)
        for i, j in zip(r, k):
            if cost[i, j] <= thresh:
                matches.append((i, j))
    else:
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        used_r, used_c = set(), set()
        for i, j in order:
            if cost[i, j] > thresh:
                break
            if i not in used_r and j not in used_c:
                matches.append((int(i), int(j)))
                used_r.add(i)
                used_c.add(j)
    mr = {i for i, _ in matches}
    mc = {j for _, j in matches}
    return matches, [i for i in range(cost.shape[0]) if i not in mr], [j for j in range(cost.shape[1]) if j not in mc]


NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


class STrack:
    def __init__(self, xyxy, score):
        self.z0 = _xyxy_to_xyah(np.asarray(xyxy, float))
        self.score = float(score)
        self.mean = self.cov = None
        self.state = NEW
        self.is_activated = False
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    @property
    def xyxy(self):
        if self.mean is None:
            return _xyah_to_xyxy(self.z0)
        return _xyah_to_xyxy(self.mean)


class BYTETracker:
    def __init__(self, track_high=0.25, track_low=0.1, new_track=0.25, track_buffer=30, match_thresh=0.8,
                 fuse_score=True):
        self.high, self.low, self.new = track_high, track_low, new_track
        self.buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.kf = KalmanXYAH()
        self.tracked, self.lost = [], []
        self.frame_id = 0
        self._next_id = 0

    def reset(self):
        self.__init__(self.high, self.low, self.new, self.buffer, self.match_thresh, self.fuse_score)

    def _nid(self):
        self._next_id += 1
        return self._next_id

    def _activate(self, t):
        t.track_id = self._nid()
        t.mean, t.cov = self.kf.initiate(t.z0)
        t.state = TRACKED
        t.is_activated = self.frame_id == 1
        t.frame_id = t.start_frame = self.frame_id
        t.tracklet_len = 0

    def _update(self, t, det, reactivate=False):
        t.mean, t.cov = self.kf.update(t.mean, t.cov, det.z0)
        t.score = det.score
        t.state = TRACKED
        t.is_activated = True
        t.frame_id = self.frame_id
        t.tracklet_len = 0 if reactivate else t.tracklet_len + 1

    def _dist(self, tracks, dets, fuse):
        iou = iou_matrix([t.xyxy for t in tracks], [d.xyxy for d in dets])
        if fuse and len(dets):
            iou = iou * np.array([d.score for d in dets])[None, :]
        return 1.0 - iou

    def update(self, dets, steps=1):
        """dets: (N, 5) [x1, y1, x2, y2, score] 원본 프레임 좌표.
        steps: 지난 업데이트 이후 지난 프레임 수 (조건부 추론으로 YOLO를 건너뛴 만큼 칼만 예측을 더 돌림)
        반환: 활성 트랙 목록 [(x1, y1, x2, y2, track_id, score), ...]"""
        steps = max(1, int(steps))
        self.frame_id += steps
        dets = np.asarray(dets, float).reshape(-1, 5)
        s = dets[:, 4]
        high = [STrack(d[:4], d[4]) for d in dets[s >= self.high]]
        low = [STrack(d[:4], d[4]) for d in dets[(s >= self.low) & (s < self.high)]]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]
        pool = tracked + [t for t in self.lost if t not in tracked]
        for t in pool:
            if t.state != TRACKED:
                t.mean[7] = 0  # 잃어버린 트랙은 높이 속도 0 (원본과 같음)
            for _ in range(steps):
                t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        activated, refound, lost, removed = [], [], [], []
        # 1단계: 높은 신뢰도
        m, u_trk, u_det = linear_assignment(self._dist(pool, high, self.fuse_score), self.match_thresh)
        for i, j in m:
            t = pool[i]
            if t.state == TRACKED:
                self._update(t, high[j])
                activated.append(t)
            else:
                self._update(t, high[j], reactivate=True)
                refound.append(t)
        # 2단계: 낮은 신뢰도 (지금 TRACKED 였던 트랙만)
        r_tracked = [pool[i] for i in u_trk if pool[i].state == TRACKED]
        m, u_trk2, _ = linear_assignment(self._dist(r_tracked, low, False), 0.5)
        for i, j in m:
            self._update(r_tracked[i], low[j])
            activated.append(r_tracked[i])
        for i in u_trk2:
            t = r_tracked[i]
            if t.state != LOST:
                t.state = LOST
                lost.append(t)
        # 확정 안 된 트랙: 남은 높은 탐지와
        rest = [high[j] for j in u_det]
        m, u_unc, u_det2 = linear_assignment(self._dist(unconfirmed, rest, self.fuse_score), 0.7)
        for i, j in m:
            self._update(unconfirmed[i], rest[j])
            activated.append(unconfirmed[i])
        for i in u_unc:
            unconfirmed[i].state = REMOVED
            removed.append(unconfirmed[i])
        # 새 트랙
        for j in u_det2:
            d = rest[j]
            if d.score >= self.new:
                self._activate(d)
                activated.append(d)
        # 오래 잃어버린 트랙 제거
        for t in self.lost:
            if self.frame_id - t.frame_id > self.buffer:
                t.state = REMOVED
                removed.append(t)

        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _join(self.tracked, activated)
        self.tracked = _join(self.tracked, refound)
        self.lost = [t for t in self.lost if t not in self.tracked]
        self.lost = _join(self.lost, lost)
        self.lost = [t for t in self.lost if t.state != REMOVED]
        self.tracked, self.lost = _remove_dup(self.tracked, self.lost)
        out = []
        for t in self.tracked:
            if t.is_activated:
                b = t.xyxy
                out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), t.track_id, t.score))
        return out

    def lost_boxes(self):
        """잃어버린(가려진) 트랙의 예측 위치 → Track Memory 에 씀."""
        return [(t.track_id, t.xyxy) for t in self.lost if t.is_activated]


def _join(a, b):
    seen = {id(t) for t in a}
    return a + [t for t in b if id(t) not in seen]


def _remove_dup(tracked, lost):
    if not tracked or not lost:
        return tracked, lost
    d = 1 - iou_matrix([t.xyxy for t in tracked], [t.xyxy for t in lost])
    dup_t, dup_l = set(), set()
    for i, j in zip(*np.where(d < 0.15)):
        ta = tracked[i].frame_id - tracked[i].start_frame
        tb = lost[j].frame_id - lost[j].start_frame
        (dup_l if ta > tb else dup_t).add(j if ta > tb else i)
    return ([t for i, t in enumerate(tracked) if i not in dup_t],
            [t for j, t in enumerate(lost) if j not in dup_l])
