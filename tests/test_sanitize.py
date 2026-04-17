"""Tests for agentverify.cassette.sanitize — cassette sanitization."""

from __future__ import annotations

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.sanitize import (
    DEFAULT_PATTERNS,
    SanitizePattern,
    sanitize_interactions,
    _redact_value,
)
from agentverify.models import TokenUsage


class TestRedactValue:
    """Tests for the _redact_value helper."""

    def test_string_redaction(self):
        compiled = [(__import__("re").compile(r"sk-[A-Za-z0-9_-]{20,}"), "sk-***REDACTED***")]
        result = _redact_value("key is sk-abcdefghijklmnopqrstuvwxyz", compiled)
        assert "sk-***REDACTED***" in result
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result

    def test_dict_redaction(self):
        compiled = [(__import__("re").compile(r"secret123"), "***")]
        result = _redact_value({"key": "my secret123 value"}, compiled)
        assert result == {"key": "my *** value"}

    def test_list_redaction(self):
        compiled = [(__import__("re").compile(r"token"), "***")]
        result = _redact_value(["my token here", "no match"], compiled)
        assert result == ["my *** here", "no match"]

    def test_nested_structure(self):
        compiled = [(__import__("re").compile(r"secret"), "***")]
        data = {"a": [{"b": "my secret"}]}
        result = _redact_value(data, compiled)
        assert result == {"a": [{"b": "my ***"}]}

    def test_non_string_passthrough(self):
        compiled = [(__import__("re").compile(r"x"), "y")]
        assert _redact_value(42, compiled) == 42
        assert _redact_value(None, compiled) is None
        assert _redact_value(3.14, compiled) == 3.14


class TestSanitizeInteractions:
    """Tests for sanitize_interactions()."""

    def _make_interaction(
        self,
        messages: list | None = None,
        content: str | None = None,
        tools: list | None = None,
    ) -> tuple[NormalizedRequest, NormalizedResponse]:
        req = NormalizedRequest(
            messages=messages or [],
            model="gpt-4.1",
            tools=tools,
        )
        resp = NormalizedResponse(content=content)
        return (req, resp)

    def test_openai_key_redacted(self):
        interactions = [
            self._make_interaction(
                messages=[{"role": "system", "content": "Key: sk-proj-abcdefghijklmnopqrstuvwxyz1234"}],
                content="Response with sk-proj-abcdefghijklmnopqrstuvwxyz1234",
            )
        ]
        result = sanitize_interactions(interactions)
        req, resp = result[0]
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz1234" not in req.messages[0]["content"]
        assert "sk-***REDACTED***" in req.messages[0]["content"]
        assert "sk-***REDACTED***" in resp.content

    def test_anthropic_key_redacted(self):
        interactions = [
            self._make_interaction(
                messages=[{"content": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"}],
            )
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert "sk-ant-***REDACTED***" in req.messages[0]["content"]

    def test_aws_key_redacted(self):
        interactions = [
            self._make_interaction(
                messages=[{"content": "AKIAIOSFODNN7EXAMPLE"}],
            )
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert "AKIA***REDACTED***" in req.messages[0]["content"]

    def test_bearer_token_redacted(self):
        interactions = [
            self._make_interaction(
                messages=[{"content": "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc"}],
            )
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert "Bearer ***REDACTED***" in req.messages[0]["content"]

    def test_tools_sanitized(self):
        interactions = [
            self._make_interaction(
                tools=[{"description": "Uses sk-proj-abcdefghijklmnopqrstuvwxyz1234"}],
            )
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert "sk-***REDACTED***" in req.tools[0]["description"]

    def test_response_tool_calls_sanitized(self):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(
            tool_calls=[{"name": "auth", "arguments": '{"token": "sk-proj-abcdefghijklmnopqrstuvwxyz1234"}'}],
        )
        result = sanitize_interactions([(req, resp)])
        _, sanitized_resp = result[0]
        assert "sk-***REDACTED***" in sanitized_resp.tool_calls[0]["arguments"]

    def test_token_usage_preserved(self):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(
            content="ok",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        result = sanitize_interactions([(req, resp)])
        _, sanitized_resp = result[0]
        assert sanitized_resp.token_usage.input_tokens == 10
        assert sanitized_resp.token_usage.output_tokens == 5

    def test_none_content_preserved(self):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(content=None, tool_calls=None)
        result = sanitize_interactions([(req, resp)])
        _, sanitized_resp = result[0]
        assert sanitized_resp.content is None
        assert sanitized_resp.tool_calls is None

    def test_custom_patterns(self):
        custom = [SanitizePattern(name="custom", pattern=r"my-secret-\d+", replacement="***")]
        interactions = [
            self._make_interaction(
                messages=[{"content": "value is my-secret-12345"}],
            )
        ]
        result = sanitize_interactions(interactions, patterns=custom)
        req, _ = result[0]
        assert "***" in req.messages[0]["content"]
        assert "my-secret-12345" not in req.messages[0]["content"]

    def test_empty_patterns_returns_unchanged(self):
        interactions = [
            self._make_interaction(messages=[{"content": "sk-proj-abcdefghijklmnopqrstuvwxyz1234"}])
        ]
        result = sanitize_interactions(interactions, patterns=[])
        req, _ = result[0]
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz1234" in req.messages[0]["content"]

    def test_no_sensitive_data_unchanged(self):
        interactions = [
            self._make_interaction(
                messages=[{"content": "Hello, world!"}],
                content="Response text",
            )
        ]
        result = sanitize_interactions(interactions)
        req, resp = result[0]
        assert req.messages[0]["content"] == "Hello, world!"
        assert resp.content == "Response text"

    def test_model_preserved(self):
        interactions = [
            self._make_interaction(messages=[])
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert req.model == "gpt-4.1"

    def test_parameters_sanitized(self):
        req = NormalizedRequest(
            messages=[],
            model="m",
            parameters={"api_key": "sk-proj-abcdefghijklmnopqrstuvwxyz1234"},
        )
        resp = NormalizedResponse(content="ok")
        result = sanitize_interactions([(req, resp)])
        sanitized_req, _ = result[0]
        assert "sk-***REDACTED***" in sanitized_req.parameters["api_key"]

    def test_raw_metadata_sanitized(self):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(
            content="ok",
            raw_metadata={"header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghijklmnop"},
        )
        result = sanitize_interactions([(req, resp)])
        _, sanitized_resp = result[0]
        assert "Bearer ***REDACTED***" in sanitized_resp.raw_metadata["header"]


class TestDefaultPatterns:
    """Verify DEFAULT_PATTERNS structure."""

    def test_has_patterns(self):
        assert len(DEFAULT_PATTERNS) > 0

    def test_all_are_sanitize_pattern(self):
        for p in DEFAULT_PATTERNS:
            assert isinstance(p, SanitizePattern)
            assert p.name
            assert p.pattern
            assert p.replacement
