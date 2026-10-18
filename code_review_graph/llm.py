"""Local model providers. Every LLM call in crg goes through `LLM.complete`.

crg only runs open models on your own machine:

  ollama   (default) Ollama. crg starts the server if it isn't running and
           downloads the model if you don't have it yet.
  openai   any local server that speaks the OpenAI API: LM Studio, llama.cpp
           `llama-server`, vLLM, LocalAI, MLX `mlx_lm.server`. Load the model
           in that server yourself.

Both stream tokens, so the dashboard can show each agent writing live.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import time
from typing import Callable, Protocol

import httpx

DEFAULT_PROVIDER = os.environ.get("CRG_PROVIDER", "ollama")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OPENAI_BASE_URL = os.environ.get("CRG_BASE_URL", "http://localhost:1234/v1")
# Ollama's default context is small and it truncates silently, which would cut the
# diff out of a review prompt. Ask for enough room for context pack + diff.
LOCAL_CONTEXT = int(os.environ.get("CRG_NUM_CTX", "16384"))

# Default model by installed RAM: (minimum GiB, Ollama model, approx download size).
# Code-tuned and openly licensed; the largest one that leaves room for the OS and
# several parallel agent contexts.
MODEL_LADDER = [
    (64, "qwen2.5-coder:32b", "20 GB"),
    (32, "qwen2.5-coder:14b", "9 GB"),
    (16, "qwen2.5-coder:7b", "4.7 GB"),
    (8, "qwen2.5-coder:3b", "1.9 GB"),
    (0, "qwen2.5-coder:1.5b", "1 GB"),
]

OnToken = Callable[[str, str], None]                 # (kind: "text" | "thinking", delta)
OnProgress = Callable[[dict], None]                  # {"stage", "model", "completed", "total", "message"}


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    provider: str
    model: str
    context_tokens: int | None

    def prepare(self, on_progress: OnProgress | None = None) -> None: ...

    def complete(self, system: str, prompt: str, max_tokens: int = 16000,
                 on_token: OnToken | None = None) -> str: ...


def ram_gib() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 8.0


def recommended_model(ram: float | None = None) -> tuple[str, str]:
    ram = ram_gib() if ram is None else ram
    for min_gib, model, size in MODEL_LADDER:
        if ram >= min_gib * 0.95:   # "16 GB" machines report a little under 16 GiB
            return model, size
    return MODEL_LADDER[-1][1], MODEL_LADDER[-1][2]


def _canonical(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def _split_think(text: str) -> str:
    """Drop <think>...</think> blocks some local reasoning models put in their answer."""
    while "<think>" in text and "</think>" in text:
        start, end = text.index("<think>"), text.index("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


class OllamaLLM:
    provider = "ollama"

    def __init__(self, model: str | None = None, host: str = OLLAMA_HOST, num_ctx: int = LOCAL_CONTEXT,
                 parallel: int = 4, auto_pull: bool = True):
        self.host = host.rstrip("/")
        self.context_tokens = num_ctx
        self.parallel = parallel
        self.auto_pull = auto_pull
        self.model = model or os.environ.get("CRG_MODEL") or recommended_model()[0]

    # --- getting ready -------------------------------------------------------

    def _reachable(self) -> bool:
        try:
            return httpx.get(f"{self.host}/api/version", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def _start_server(self, on_progress: OnProgress | None) -> None:
        binary = shutil.which("ollama")
        if binary is None:
            raise LLMError(
                "Ollama isn't installed. Install it, then run crg again:\n"
                "  macOS:  brew install ollama   (or the app from https://ollama.com/download)\n"
                "  Linux:  curl -fsSL https://ollama.com/install.sh | sh"
            )
        env = dict(os.environ)
        env.setdefault("OLLAMA_NUM_PARALLEL", str(self.parallel))   # let agents run concurrently
        proc = subprocess.Popen([binary, "serve"], env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        atexit.register(proc.terminate)   # we started it, so we stop it
        if on_progress:
            on_progress({"stage": "server", "model": self.model,
                         "message": f"Started Ollama (OLLAMA_NUM_PARALLEL={env['OLLAMA_NUM_PARALLEL']})"})
        for _ in range(60):
            if self._reachable():
                return
            if proc.poll() is not None:
                raise LLMError("`ollama serve` exited immediately. Run it yourself to see why.")
            time.sleep(0.25)
        raise LLMError(f"Started Ollama, but it never answered at {self.host}.")

    def installed(self) -> list[str]:
        tags = httpx.get(f"{self.host}/api/tags", timeout=10).json().get("models", [])
        return [m["name"] for m in tags]

    def pull(self, on_progress: OnProgress | None = None) -> None:
        """Download the model through Ollama, reporting byte progress."""
        last_emit = 0.0
        with httpx.stream("POST", f"{self.host}/api/pull", json={"model": self.model, "stream": True},
                          timeout=httpx.Timeout(None, connect=10)) as r:
            if r.status_code != 200:
                raise LLMError(f"Ollama couldn't pull {self.model}: {r.read().decode(errors='replace')[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise LLMError(f"Downloading {self.model} failed: {chunk['error']}")
                now = time.monotonic()
                final = chunk.get("status") == "success"
                layer_done = bool(chunk.get("total")) and chunk.get("completed") == chunk.get("total")
                if on_progress and (final or layer_done or now - last_emit > 0.25):
                    last_emit = now
                    on_progress({"stage": "done" if final else "download", "model": self.model,
                                 "message": chunk.get("status", ""),
                                 "completed": chunk.get("completed", 0), "total": chunk.get("total", 0)})

    def prepare(self, on_progress: OnProgress | None = None) -> None:
        if not self._reachable():
            self._start_server(on_progress)
        if _canonical(self.model) in {_canonical(n) for n in self.installed()}:
            return
        if not self.auto_pull:
            raise LLMError(f"{self.model} isn't downloaded. Run `ollama pull {self.model}` or drop --no-pull.")
        if on_progress:
            on_progress({"stage": "download", "model": self.model, "message": "starting download",
                         "completed": 0, "total": 0})
        self.pull(on_progress)

    def unload(self) -> None:
        """Free this model's memory now (e.g. before loading a larger grader)."""
        try:
            httpx.post(f"{self.host}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
        except httpx.HTTPError:
            pass

    # --- generation ----------------------------------------------------------

    def complete(self, system: str, prompt: str, max_tokens: int = 16000,
                 on_token: OnToken | None = None) -> str:
        body = {
            "model": self.model,
            "stream": True,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "options": {"num_ctx": self.context_tokens, "num_predict": max_tokens, "temperature": 0.2},
        }
        out: list[str] = []
        try:
            with httpx.stream("POST", f"{self.host}/api/chat", json=body, timeout=httpx.Timeout(600, connect=10)) as r:
                if r.status_code != 200:
                    raise LLMError(f"Ollama {r.status_code}: {r.read().decode(errors='replace')[:300]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMError(f"Ollama: {chunk['error']}")
                    msg = chunk.get("message") or {}
                    if msg.get("thinking") and on_token:
                        on_token("thinking", msg["thinking"])
                    if msg.get("content"):
                        out.append(msg["content"])
                        if on_token:
                            on_token("text", msg["content"])
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        return _split_think("".join(out))


class OpenAICompatLLM:
    """LM Studio, llama.cpp `llama-server`, vLLM, LocalAI, MLX (`/v1/chat/completions`)."""

    provider = "openai"

    def __init__(self, model: str | None = None, base_url: str = OPENAI_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.context_tokens = None
        self.headers = {"Authorization": f"Bearer {os.environ.get('CRG_API_KEY', 'local')}"}
        self.model = model or os.environ.get("CRG_MODEL") or ""

    def prepare(self, on_progress: OnProgress | None = None) -> None:
        try:
            data = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=5).json()
        except httpx.HTTPError as e:
            raise LLMError(f"No local OpenAI-compatible server at {self.base_url}. Start LM Studio's "
                           "server, `llama-server`, or `mlx_lm.server` first.") from e
        models = [m["id"] for m in data.get("data", []) if "embed" not in m["id"]]
        if not models:
            raise LLMError(f"{self.base_url} has no model loaded. Load or download one in that app first.")
        if not self.model:
            self.model = models[0]
        elif self.model not in models:
            raise LLMError(f"{self.base_url} doesn't have {self.model}. Available: {', '.join(models)}")

    def complete(self, system: str, prompt: str, max_tokens: int = 16000,
                 on_token: OnToken | None = None) -> str:
        body = {
            "model": self.model, "stream": True, "max_tokens": max_tokens, "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        out: list[str] = []
        try:
            with httpx.stream("POST", f"{self.base_url}/chat/completions", json=body,
                              headers=self.headers, timeout=httpx.Timeout(600, connect=10)) as r:
                if r.status_code != 200:
                    raise LLMError(f"{self.base_url} {r.status_code}: {r.read().decode(errors='replace')[:300]}")
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("reasoning_content") and on_token:
                        on_token("thinking", delta["reasoning_content"])
                    if delta.get("content"):
                        out.append(delta["content"])
                        if on_token:
                            on_token("text", delta["content"])
        except httpx.HTTPError as e:
            raise LLMError(f"Request to {self.base_url} failed: {e}") from e
        return _split_think("".join(out))


def make_llm(provider: str | None = None, model: str | None = None, base_url: str | None = None,
             parallel: int = 4, auto_pull: bool = True) -> LLM:
    provider = provider or DEFAULT_PROVIDER
    if provider == "ollama":
        return OllamaLLM(model, host=base_url or OLLAMA_HOST, parallel=parallel, auto_pull=auto_pull)
    if provider == "openai":
        return OpenAICompatLLM(model, base_url=base_url or OPENAI_BASE_URL)
    raise LLMError(f"Unknown provider {provider!r}. crg runs local models only: ollama or openai.")
