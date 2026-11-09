"""LLM provider interface used by generation (stage 5), verification (stage 6) and the
eval judge. Each provider takes a system prompt, a user prompt and a Pydantic schema, and
returns the model's JSON text plus token usage and latency.

- `AnthropicProvider`: Claude via the Anthropic API, JSON constrained by `output_config`.
- `GeminiProvider`: Gemini via `google-genai`, on Vertex AI by default or the Google AI
  Studio API when `GEMINI_API_KEY` is set; JSON constrained by `response_json_schema`.
- `FakeProvider`: scripted responses for tests, so no test ever makes a paid call.

Model IDs are never hard-coded: callers pass them in from `GENERATE_MODEL`,
`VERIFY_MODEL` or `--model`, and `get_provider` routes by the ID's prefix.

No automatic fallback to another model on refusal: in a model benchmark, a silent model
switch would mix results across models. A refusal is returned as `refused=True`.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from pydantic import BaseModel

from clausewitz.config import get_settings


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up; retrying within minutes won't help."""


@dataclass
class LLMResult:
    text: str  # the model's JSON output (empty on refusal)
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    refused: bool = False
    stop_reason: str | None = None


class Provider(Protocol):
    def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> LLMResult: ...


def json_schema(schema: type[BaseModel]) -> dict:
    """JSON Schema for a Pydantic model, with every object closed (no extra keys).

    Structured-output APIs require `additionalProperties: false` on every object.
    """
    out = schema.model_json_schema()

    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(out)
    return out


class AnthropicProvider:
    def __init__(self, client=None):
        import anthropic

        self._client = client or anthropic.Anthropic(max_retries=4)

    def complete_json(self, model, system, user, schema, max_tokens) -> LLMResult:
        t = time.perf_counter()
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": json_schema(schema)}},
        )
        latency = (time.perf_counter() - t) * 1000
        refused = response.stop_reason == "refusal"
        text = "" if refused else next((b.text for b in response.content if b.type == "text"), "")
        return LLMResult(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency,
            refused=refused,
            stop_reason=response.stop_reason,
        )


class GeminiProvider:
    # Free-tier keys hit per-minute limits often; retry 429/5xx with backoff. Latency is
    # measured on the successful attempt only, so rate-limit waits don't count as latency.
    MAX_ATTEMPTS = 8
    sleep = staticmethod(time.sleep)

    def __init__(self, client=None):
        if client is None:
            from google import genai

            settings = get_settings()
            if settings.gemini_api_key:  # Google AI Studio (has a free tier)
                client = genai.Client(api_key=settings.gemini_api_key)
            else:  # Vertex AI with application-default credentials
                client = genai.Client(
                    vertexai=True,
                    project=settings.google_cloud_project,
                    location=settings.google_cloud_location,
                )
        self._client = client

    def complete_json(self, model, system, user, schema, max_tokens) -> LLMResult:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_json_schema=json_schema(schema),
        )
        for attempt in range(self.MAX_ATTEMPTS):
            t = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=model, contents=user, config=config
                )
                break
            except errors.APIError as e:
                if e.code == 429 and "PerDay" in str(e):
                    raise QuotaExhausted(f"{model}: daily quota exhausted") from e
                if e.code not in (429, 500, 503) or attempt == self.MAX_ATTEMPTS - 1:
                    raise
                self.sleep(min(60.0, 4.0 * 2**attempt))
        latency = (time.perf_counter() - t) * 1000
        usage = response.usage_metadata
        candidate = response.candidates[0] if response.candidates else None
        finish = str(candidate.finish_reason) if candidate and candidate.finish_reason else None
        refused = candidate is None or (finish is not None and "SAFETY" in finish)
        return LLMResult(
            text="" if refused else (response.text or ""),
            model=model,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            # Gemini bills thinking tokens as output; count them like Claude's.
            output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
            if usage
            else 0,
            latency_ms=latency,
            refused=refused,
            stop_reason=finish,
        )


class FakeProvider:
    """Test double. `respond(system, user, schema) -> str` returns the JSON text."""

    def __init__(self, respond: Callable[[str, str, type[BaseModel]], str]):
        self.respond = respond
        self.calls: list[dict] = []

    def complete_json(self, model, system, user, schema, max_tokens) -> LLMResult:
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema})
        text = self.respond(system, user, schema)
        return LLMResult(
            text=text,
            model=model,
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(text.split()),
            latency_ms=0.0,
        )


@lru_cache
def _anthropic() -> AnthropicProvider:
    return AnthropicProvider()


@lru_cache
def _gemini() -> GeminiProvider:
    return GeminiProvider()


def get_provider(model: str) -> Provider:
    """Route a model ID to its provider by prefix."""
    if not model:
        raise ValueError("model ID is empty; set GENERATE_MODEL / VERIFY_MODEL or pass --model")
    if model.startswith("claude"):
        return _anthropic()
    if model.startswith("gemini"):
        return _gemini()
    raise ValueError(f"no provider for model {model!r} (expected claude-* or gemini-*)")
