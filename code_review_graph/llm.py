"""Thin wrapper around the Claude API. Everything LLM-shaped goes through here."""

from __future__ import annotations

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"


class LLMRefusal(RuntimeError):
    pass


class LLM:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low", client=None):
        self.model = model
        self.effort = effort
        # Resolves ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = client or anthropic.Anthropic()

    def complete(self, system: str, prompt: str, max_tokens: int = 16000) -> str:
        # Streaming keeps long reviews clear of HTTP timeouts. `fallbacks: "default"`
        # retries a safety-classifier decline on Anthropic's recommended fallback
        # model, which matters for a tool that reads security-sensitive code.
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"model declined ({getattr(details, 'category', None)})")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "max_tokens":
            text += "\n\n_(output truncated at max_tokens)_"
        return text
