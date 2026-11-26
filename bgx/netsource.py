# bgx/netsource.py
#
# CARLA 가상 카메라 영상을 네트워크(TCP)로 받는 입력. pi/edge.py 의 FrameSource 와 같은 사용법.
#   --cam1 tcp:8765   → Pi 가 8765 포트에서 기다리고, 노트북 simdemo/carla_demo.py 가 접속해서 보냄
#
# 패킷 (simdemo/carla_demo.py 의 FrameSender 와 약속)
#   [4바이트 길이 N (big-endian)] [N 바이트: 8바이트 double 보낸 시각(노트북, 초) + 4바이트 CARLA 프레임 번호 + JPEG]
#
# - 웹캠과 똑같이 '최신 프레임만' 보관 (처리가 밀리면 옛 프레임은 버림 → dropped 로 셈)
# - 촬영 시각(ts_cap_local)은 'Pi 가 프레임을 다 받은 시각'. 노트북→Pi 전송(수 ms)은 촬영→서버 지연에 들어가지 않는다.
#   (웹캠도 드라이버가 프레임을 넘겨준 시각을 쓰므로 같은 기준)
# - 연결이 끊기면 다시 기다린다 (CARLA 스크립트를 다시 켜도 Pi 를 재시작할 필요 없음)

import socket
import struct
import threading
import time

HEAD = struct.Struct("!I")
META = struct.Struct("!dI")


def parse_tcp_src(src):
    """'tcp:8765' 또는 'tcp:0.0.0.0:8765' → (host, port)."""
    parts = src.split(":")[1:]
    if len(parts) == 1:
        return "0.0.0.0", int(parts[0])
    return parts[0], int(parts[1])


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("닫힘")
        buf.extend(chunk)
    return bytes(buf)


class NetFrameSource(threading.Thread):
    is_file = False
    eof = False

    def __init__(self, src, w, h, fps, **_):
        super().__init__(daemon=True)
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        self.host, self.port = parse_tcp_src(src)
        self.fps = fps
        self.W, self.H = w, h
        self.cond = threading.Condition()
        self.frame, self.t, self.n = None, 0.0, 0
        self.carla_frame = None
        self.sender_t = None
        self.running = True
        self.clients = 0
        self.size_warned = False
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((self.host, self.port))
        self.srv.listen(1)
        self.srv.settimeout(1.0)
        self.status = "대기 중 (포트 %d)" % self.port
        print("[net] %s:%d 에서 CARLA 영상 기다리는 중" % (self.host, self.port), flush=True)

    def run(self):
        cv2, np = self.cv2, self.np
        while self.running:
            try:
                conn, addr = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self.clients += 1
            self.status = "수신 중 (%s)" % addr[0]
            print("[net] 포트 %d ← %s 접속" % (self.port, addr[0]), flush=True)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(5.0)
            try:
                while self.running:
                    (n,) = HEAD.unpack(recv_exact(conn, HEAD.size))
                    data = recv_exact(conn, n)
                    t_sent, cf = META.unpack_from(data)
                    img = cv2.imdecode(np.frombuffer(data, np.uint8, offset=META.size), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if (img.shape[1], img.shape[0]) != (self.W, self.H):
                        if not self.size_warned:
                            print("[net] 포트 %d: 받은 크기 %dx%d ≠ 설정 %dx%d → 맞춰 줄임" %
                                  (self.port, img.shape[1], img.shape[0], self.W, self.H), flush=True)
                            self.size_warned = True
                        img = cv2.resize(img, (self.W, self.H))
                    t = time.time()
                    with self.cond:
                        self.frame, self.t, self.n = img, t, self.n + 1
                        self.carla_frame, self.sender_t = cf, t_sent
                        self.cond.notify_all()
            except (ConnectionError, socket.timeout, OSError) as e:
                print("[net] 포트 %d 연결 끊김 (%s) → 다시 기다림" % (self.port, e), flush=True)
                self.status = "대기 중 (포트 %d)" % self.port
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def get(self, last_n, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.n != last_n, timeout=timeout)
            if self.n == last_n:
                return None, None, last_n
            return self.frame, self.t, self.n

    def stop(self):
        self.running = False
        try:
            self.srv.close()
        except OSError:
            pass
