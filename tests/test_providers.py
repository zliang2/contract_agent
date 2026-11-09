"""Provider response parsing against fake SDK clients (no network)."""

from types import SimpleNamespace as NS

import pytest

from clausewitz.llm.generate import GeneratedAnswer
from clausewitz.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    get_provider,
    json_schema,
)


def test_json_schema_closes_every_object():
    schema = json_schema(GeneratedAnswer)

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)

    found = list(objects(schema))
    assert len(found) == 3  # answer, claim, citation
    assert all(o["additionalProperties"] is False for o in found)


class FakeAnthropicClient:
    def __init__(self, response):
        self.response = response
        self.kwargs = None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def anthropic_response(stop_reason="end_turn", text='{"a": 1}'):
    return NS(
        stop_reason=stop_reason,
        model="claude-x",
        content=[NS(type="thinking"), NS(type="text", text=text)],
        usage=NS(input_tokens=120, output_tokens=30),
    )


def test_anthropic_provider_sends_schema_and_reads_usage():
    client = FakeAnthropicClient(anthropic_response())
    r = AnthropicProvider(client).complete_json("claude-x", "sys", "user", GeneratedAnswer, 99)
    assert (r.text, r.input_tokens, r.output_tokens, r.refused) == ('{"a": 1}', 120, 30, False)
    fmt = client.kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert client.kwargs["max_tokens"] == 99 and client.kwargs["system"] == "sys"


def test_anthropic_refusal():
    client = FakeAnthropicClient(anthropic_response("refusal", text="partial"))
    r = AnthropicProvider(client).complete_json("claude-x", "s", "u", GeneratedAnswer, 9)
    assert r.refused and r.text == ""


class FakeGeminiClient:
    def __init__(self, response):
        self.response = response
        self.kwargs = None
        self.models = self

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def gemini_response(finish="STOP"):
    return NS(
        text='{"a": 2}',
        candidates=[NS(finish_reason=finish)],
        usage_metadata=NS(prompt_token_count=200, candidates_token_count=40,
                          thoughts_token_count=10),
    )  # fmt: skip


def test_gemini_provider_counts_thinking_as_output():
    client = FakeGeminiClient(gemini_response())
    r = GeminiProvider(client).complete_json("gemini-x", "sys", "user", GeneratedAnswer, 50)
    assert (r.text, r.input_tokens, r.output_tokens, r.refused) == ('{"a": 2}', 200, 50, False)
    cfg = client.kwargs["config"]
    assert cfg.response_mime_type == "application/json" and cfg.system_instruction == "sys"


def test_gemini_safety_block_is_a_refusal():
    r = GeminiProvider(FakeGeminiClient(gemini_response("SAFETY"))).complete_json(
        "gemini-x", "s", "u", GeneratedAnswer, 5
    )
    assert r.refused and r.text == ""


def test_get_provider_routes_by_prefix():
    with pytest.raises(ValueError, match="empty"):
        get_provider("")
    with pytest.raises(ValueError, match="no provider"):
        get_provider("gpt-x")


def test_gemini_retries_rate_limits_then_succeeds():
    from google.genai import errors

    class Flaky(FakeGeminiClient):
        def __init__(self):
            super().__init__(gemini_response())
            self.n = 0

        def generate_content(self, **kwargs):
            self.n += 1
            if self.n < 3:
                raise errors.APIError(429, {"error": {"message": "rate limited"}})
            return self.response

    client = Flaky()
    provider = GeminiProvider(client)
    waits = []
    provider.sleep = waits.append
    r = provider.complete_json("gemini-x", "s", "u", GeneratedAnswer, 5)
    assert r.text == '{"a": 2}' and client.n == 3 and waits == [4.0, 8.0]


def test_gemini_does_not_retry_bad_requests():
    from google.genai import errors

    class Bad(FakeGeminiClient):
        def generate_content(self, **kwargs):
            raise errors.APIError(400, {"error": {"message": "bad"}})

    provider = GeminiProvider(Bad(None))
    provider.sleep = lambda s: None
    with pytest.raises(errors.APIError):
        provider.complete_json("gemini-x", "s", "u", GeneratedAnswer, 5)
