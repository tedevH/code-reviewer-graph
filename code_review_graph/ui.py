"""Live dashboard: a local web page that shows every agent working in real time.

Serves one HTML page and a Server-Sent Events stream of the run's EventBus.
Binds to 127.0.0.1 only; nothing is exposed to the network.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from code_review_graph.events import EventBus


def _handler(bus: EventBus, html: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server naming)
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", html)
            elif self.path.startswith("/events"):
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            seq = int(self.headers.get("Last-Event-ID", "-1")) + 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    events = bus.since(seq, timeout=15)
                    if not events:
                        self.wfile.write(b": ping\n\n")
                    for e in events:
                        self.wfile.write(f"id: {e['seq']}\ndata: {json.dumps(e)}\n\n".encode())
                        seq = e["seq"] + 1
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, *args):  # keep the terminal quiet
            pass

    return Handler


class Dashboard:
    def __init__(self, bus: EventBus, port: int = 8765, open_browser: bool = True):
        self.bus, self.port, self.open_browser = bus, port, open_browser
        self.server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> "Dashboard":
        html = resources.files("code_review_graph").joinpath("dashboard.html").read_bytes()
        handler = _handler(self.bus, html)
        for port in range(self.port, self.port + 20):
            try:
                self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
                break
            except OSError:
                continue
        if self.server is None:
            raise RuntimeError(f"No free port in {self.port}-{self.port + 19}")
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        if self.open_browser:
            webbrowser.open(self.url)
        return self

    def hold(self) -> None:
        """Keep serving after the run so the results stay on screen."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if self.server:
                self.server.shutdown()
