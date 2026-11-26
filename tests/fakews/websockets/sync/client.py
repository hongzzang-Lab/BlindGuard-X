# 시험용 가짜 websockets.sync.client — 같은 프로세스 안의 ASGI 서버(asyncio 루프 스레드)로 연결
import asyncio, json, queue
BRIDGE = {"loop": None, "app": None}

class Conn:
    def __init__(self, url):
        path = "/" + url.split("/", 3)[3]
        self.outq = queue.Queue()
        loop = BRIDGE["loop"]
        self.loop = loop
        self.inq = None
        fut = asyncio.run_coroutine_threadsafe(self._start(path), loop)
        fut.result(5)

    async def _start(self, path):
        self.inq = asyncio.Queue()
        scope = {"type": "websocket", "path": path, "raw_path": path.encode(), "headers": [], "query_string": b"",
                 "client": ("10.42.0.20", 4000), "server": ("t", 8000), "scheme": "ws", "root_path": "",
                 "subprotocols": [], "asgi": {"version": "3.0"}}
        await self.inq.put({"type": "websocket.connect"})
        async def receive():
            return await self.inq.get()
        async def send(m):
            if m["type"] == "websocket.send":
                self.outq.put(m["text"])
        self.task = asyncio.ensure_future(BRIDGE["app"](scope, receive, send))

    def send(self, text):
        self.loop.call_soon_threadsafe(self.inq.put_nowait, {"type": "websocket.receive", "text": text})

    def recv(self, timeout=None):
        try:
            return self.outq.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError()

def connect(url, open_timeout=None, max_size=None, **kw):
    start = BRIDGE.pop("start", None)
    if start:            # 처음 연결할 때 서버 루프를 켬 (edge.py 가 카메라 프로세스를 fork 한 '뒤')
        start()
    return Conn(url)
