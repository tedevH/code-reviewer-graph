"""Local provider behaviour against a fake Ollama server (no real model needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from code_review_graph.llm import LLMError, OllamaLLM, recommended_model


class FakeOllama:
    def __init__(self, installed):
        self.installed = list(installed)
        self.pulls = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _ndjson(self, chunks):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for c in chunks:
                    self.wfile.write((json.dumps(c) + "\n").encode())

            def do_GET(self):
                if self.path == "/api/version":
                    self._json({"version": "test"})
                elif self.path == "/api/tags":
                    self._json({"models": [{"name": n} for n in fake.installed]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/pull":
                    fake.pulls.append(body["model"])
                    fake.installed.append(body["model"])
                    self._ndjson([{"status": "pulling manifest"},
                                  {"status": "pulling abc", "total": 100, "completed": 50},
                                  {"status": "pulling abc", "total": 100, "completed": 100},
                                  {"status": "success"}])
                elif self.path == "/api/chat":
                    assert body["options"]["num_ctx"] >= 8192
                    self._ndjson([{"message": {"content": "NO "}}, {"message": {"content": "FINDINGS"}},
                                  {"done": True}])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def ollama():
    servers = []

    def make(installed=()):
        f = FakeOllama(installed)
        servers.append(f)
        return f

    yield make
    for f in servers:
        f.server.shutdown()


def test_downloads_missing_model_with_progress(ollama):
    fake = ollama(installed=["gemma3:4b"])
    llm = OllamaLLM("qwen2.5-coder:7b", host=fake.url)
    events = []
    llm.prepare(events.append)
    assert fake.pulls == ["qwen2.5-coder:7b"]
    assert events[-1]["stage"] == "done"
    assert any(e.get("total") == 100 for e in events)
    tokens = []
    assert llm.complete("sys", "hi", on_token=lambda k, t: tokens.append(t)) == "NO FINDINGS"
    assert tokens == ["NO ", "FINDINGS"]


def test_skips_download_when_installed(ollama):
    fake = ollama(installed=["qwen2.5-coder:7b"])
    OllamaLLM("qwen2.5-coder:7b", host=fake.url).prepare()
    assert fake.pulls == []


def test_no_pull_refuses(ollama):
    fake = ollama(installed=[])
    with pytest.raises(LLMError, match="ollama pull"):
        OllamaLLM("qwen2.5-coder:7b", host=fake.url, auto_pull=False).prepare()


def test_bare_model_name_matches_latest_tag(ollama):
    fake = ollama(installed=["llama3.2:latest"])
    OllamaLLM("llama3.2", host=fake.url).prepare()
    assert fake.pulls == []


@pytest.mark.parametrize("ram,model", [(4, "qwen2.5-coder:1.5b"), (8, "qwen2.5-coder:3b"),
                                       (15.6, "qwen2.5-coder:7b"), (24, "qwen2.5-coder:7b"),
                                       (32, "qwen2.5-coder:14b"), (128, "qwen2.5-coder:32b")])
def test_model_fits_ram(ram, model):
    assert recommended_model(ram)[0] == model
