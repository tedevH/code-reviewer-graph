"""Model providers. Every LLM call in crg goes through `LLM.complete`.

  ollama     local models via Ollama (default; free, nothing leaves your machine)
  openai     any OpenAI-compatible local server: LM Studio, llama.cpp, vLLM, LocalAI
  anthropic  Claude via the Anthropic API

All providers stream, so the dashboard can show each agent writing live.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Protocol

import httpx

DEFAULT_PROVIDER = os.environ.get("CRG_PROVIDER", "ollama")
CLAUDE_MODEL = "claude-opus-5-5"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OPENAI_BASE_URL = os.environ.get("CRG_BASE_URL", "http://localhost:1234/v1")
# Ollama's default context is small and it truncates silently, which would cut the
# diff out of a review prompt. Ask for enough room for context pack + diff.
LOCAL_CONTEXT = int(os.environ.get("CRG_NUM_CTX", "16384"))

OnToken = Callable[[str, str], None]   # (kind: "text" | "thinking", delta)


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class LLM(Protocol):
    provider: str
    model: str
    context_tokens: int | None

    def complete(self, system: str, prompt: str, max_tokens: int = 16000,
                 on_token: OnToken | None = None) -> str: ...


def _split_think(text: str) -> str:
    """Drop <think>...</think> blocks some local reasoning models put in their answer."""
    while "<think>" in text and "</think>" in text:
        start, end = text.index("<think>"), text.index("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


class OllamaLLM:
    provider = "ollama"

    def __init__(self, model: str | None = None, host: str = OLLAMA_HOST, num_ctx: int = LOCAL_CONTEXT):
        self.host = host.rstrip("/")
        self.context_tokens = num_ctx
        self.model = model or os.environ.get("CRG_MODEL") or self._pick_model()

    def _pick_model(self) -> str:
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=5).json().get("models", [])
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama isn't reachable at {self.host}. Start it with `ollama serve`.") from e
        chat = [m["name"] for m in tags if "embed" not in m["name"]]
        if not chat:
            raise LLMError("No Ollama models installed. Try `ollama pull qwen2.5-coder:7b`.")
        return chat[0]

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
    """LM Studio, llama.cpp `llama-server`, vLLM, LocalAI, ... (`/v1/chat/completions`)."""

    provider = "openai"

    def __init__(self, model: str | None = None, base_url: str = OPENAI_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.context_tokens = None
        self.headers = {"Authorization": f"Bearer {os.environ.get('CRG_API_KEY', 'local')}"}
        self.model = model or os.environ.get("CRG_MODEL") or self._pick_model()

    def _pick_model(self) -> str:
        try:
            data = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=5).json()
        except httpx.HTTPError as e:
            raise LLMError(f"No OpenAI-compatible server at {self.base_url}.") from e
        models = [m["id"] for m in data.get("data", []) if "embed" not in m["id"]]
        if not models:
            raise LLMError(f"{self.base_url} lists no models. Load one in your server first.")
        return models[0]

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


class AnthropicLLM:
    provider = "anthropic"
    context_tokens = None

    def __init__(self, model: str | None = None, effort: str = "low", client=None):
        import anthropic

        self.model = model or CLAUDE_MODEL
        self.effort = effort
        # Resolves ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = client or anthropic.Anthropic()

    def complete(self, system: str, prompt: str, max_tokens: int = 16000,
                 on_token: OnToken | None = None) -> str:
        # `fallbacks: "default"` retries a safety-classifier decline on Anthropic's
        # recommended fallback model, which matters for a tool that reads security code.
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            for text in stream.text_stream:
                if on_token:
                    on_token("text", text)
            response = stream.get_final_message()
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"model declined ({getattr(details, 'category', None)})")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "max_tokens":
            text += "\n\n_(output truncated at max_tokens)_"
        return text


def make_llm(provider: str | None = None, model: str | None = None, effort: str = "low",
             base_url: str | None = None) -> LLM:
    provider = provider or DEFAULT_PROVIDER
    if provider == "ollama":
        return OllamaLLM(model, host=base_url or OLLAMA_HOST)
    if provider == "openai":
        return OpenAICompatLLM(model, base_url=base_url or OPENAI_BASE_URL)
    if provider == "anthropic":
        return AnthropicLLM(model, effort=effort)
    raise LLMError(f"Unknown provider {provider!r}. Use ollama, openai or anthropic.")
