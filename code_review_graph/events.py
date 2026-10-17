"""In-process event bus. Agents publish; the dashboard and terminal subscribe.

Events are plain dicts with a monotonically increasing `seq`, so a client
that reconnects (or opens late) replays everything it missed.

  run         {title, kind, provider, model, repo, meta}
  agent       {id, name, role, color, status, task?}   registration and status changes
  delta       {id, kind: "text" | "thinking", text}    streamed tokens (coalesced)
  agent_done  {id, status: "done" | "error", tokens, seconds, task?, error?}
  download    {stage: "server" | "download" | "done", model, message, completed?, total?}
  log         {message, level}
  result      {markdown}
  run_done    {seconds}
"""

from __future__ import annotations

import sys
import threading
import time


class EventBus:
    def __init__(self):
        self._events: list[dict] = []
        self._cond = threading.Condition()
        self.started = time.time()

    def emit(self, type: str, **data) -> None:
        with self._cond:
            self._events.append({"seq": len(self._events), "type": type,
                                 "t": round(time.time() - self.started, 3), **data})
            self._cond.notify_all()

    def since(self, seq: int, timeout: float = 15.0) -> list[dict]:
        with self._cond:
            if len(self._events) <= seq:
                self._cond.wait(timeout)
            return self._events[seq:]

    def log(self, message: str, level: str = "info") -> None:
        self.emit("log", message=message, level=level)


class TerminalReporter:
    """Prints agent lifecycle to stderr so stdout stays clean for the review itself."""

    def __init__(self, bus: EventBus, stream=sys.stderr):
        self.bus, self.stream = bus, stream
        self.names: dict[str, str] = {}
        self._dl_step: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "TerminalReporter":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            for e in self.bus.since(seq, timeout=0.5):
                seq = e["seq"] + 1
                self._print(e)
        for e in self.bus.since(seq, timeout=0):
            self._print(e)

    def _print(self, e: dict) -> None:
        t = e["type"]
        if t == "agent":
            if "name" in e:
                self.names[e["id"]] = e["name"]
            if e["status"] == "waiting" and e.get("task"):
                self._line(f"{self.names.get(e['id'], e['id']):<14} → {e['task']}")
        elif t == "agent_done":
            name = self.names.get(e["id"], e["id"])
            what = f" {e['task']}" if e.get("task") else ""
            if e["status"] == "error":
                self._line(f"{name:<14} ✗{what}: {e.get('error')}")
            else:
                self._line(f"{name:<14} ✓{what} ({e['tokens']} tok, {e['seconds']:.1f}s)")
        elif t == "log":
            self._line(e["message"])
        elif t == "download":
            self._download(e)

    def _download(self, e: dict) -> None:
        if e["stage"] == "server":
            self._line(e["message"])
        elif e["stage"] == "done":
            self._line(f"Downloaded {e['model']}")
        elif e.get("total", 0) > 50_000_000:   # the weights; skip tiny config layers
            pct = int(100 * e["completed"] / e["total"])
            step = pct // 10
            if step != self._dl_step.get(e["model"]):
                self._dl_step[e["model"]] = step
                gb = lambda b: f"{b / 1e9:.1f}"  # noqa: E731
                self._line(f"Downloading {e['model']}: {pct}% ({gb(e['completed'])}/{gb(e['total'])} GB)")
        elif e["message"] == "starting download":
            self._line(f"{e['model']} isn't on this machine yet; downloading it now")

    def _line(self, text: str) -> None:
        print(f"[{time.time() - self.bus.started:6.1f}s] {text}", file=self.stream, flush=True)
