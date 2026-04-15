"""Tests for the LiteLLM provider adapter.

Since LiteLLM uses an OpenAI-compatible format, the normalize/denormalize
methods are delegated to the OpenAI adapter.  Tests here verify that
delegation works correctly and that the patch() context manager targets
``litellm.completion``.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as unittest_mock_patch

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.adapters.litellm import LiteLLMAdapter
from agentverify.models import TokenUsage


@pytest.fixture
def adapter() -> LiteLLMAdapter:
    return LiteLLMAdapter()


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------


class TestName:
    def test_name(self, adapter: LiteLLMAdapter) -> None:
        assert adapter.name == "litellm"


# ---------------------------------------------------------------------------
# normalize_request (delegation to OpenAI adapter)
# ---------------------------------------------------------------------------


def _make_litellm_response(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a mock that mimics a LiteLLM (OpenAI-compatible) response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    if tool_calls:
        tc_mocks = []
        for tc in tool_calls:
            tc_mock = MagicMock()
            tc_mock.function.name = tc["name"]
            tc_mock.function.arguments = tc["arguments"]
            tc_mocks.append(tc_mock)
        mock.choices[0].message.tool_calls = tc_mocks
    else:
        mock.choices[0].message.tool_calls = None
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    return mock


class TestNormalizeRequest:
    def test_basic_request(self, adapter: LiteLLMAdapter) -> None:
        raw = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4.1",
        }
        result = adapter.normalize_request(raw)
        assert result.messages == [{"role": "user", "content": "Hello"}]
        assert result.model == "gpt-4.1"
        assert result.tools is None
        assert result.parameters == {}

    def test_request_with_tools(self, adapter: LiteLLMAdapter) -> None:
        raw = {
            "messages": [{"role": "user", "content": "Weather?"}],
            "model": "anthropic/claude-sonnet-4-6-20250514",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
        }
        result = adapter.normalize_request(raw)
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# normalize_response (delegation to OpenAI adapter)
# ---------------------------------------------------------------------------


class TestNormalizeResponse:
    def test_text_response(self, adapter: LiteLLMAdapter) -> None:
        raw = _make_litellm_response(content="Hello world")
        result = adapter.normalize_response(raw)
        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 10
        assert result.token_usage.output_tokens == 5

    def test_tool_call_response(self, adapter: LiteLLMAdapter) -> None:
        raw = _make_litellm_response(
            tool_calls=[
                {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
            ]
        )
        result = adapter.normalize_response(raw)
        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# denormalize_response (delegation to OpenAI adapter)
# ---------------------------------------------------------------------------


class TestDenormalizeResponse:
    def test_text_response(self, adapter: LiteLLMAdapter) -> None:
        normalized = NormalizedResponse(
            content="Hello",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        result = adapter.denormalize_response(normalized)
        assert result.choices[0].message.content == "Hello"
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    def test_tool_call_response(self, adapter: LiteLLMAdapter) -> None:
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
            ],
            token_usage=TokenUsage(input_tokens=20, output_tokens=10),
        )
        result = adapter.denormalize_response(normalized)
        assert result.choices[0].finish_reason == "tool_calls"
        tc = result.choices[0].message.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"location": "Tokyo"}'


# ---------------------------------------------------------------------------
# Round-trip: normalize_response → denormalize_response preserves semantics
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_text_round_trip(self, adapter: LiteLLMAdapter) -> None:
        raw = _make_litellm_response(content="Hello world", prompt_tokens=50, completion_tokens=20)
        normalized = adapter.normalize_response(raw)
        restored = adapter.denormalize_response(normalized)
        re_normalized = adapter.normalize_response(restored)

        assert re_normalized.content == normalized.content
        assert re_normalized.tool_calls == normalized.tool_calls
        assert re_normalized.token_usage == normalized.token_usage

    def test_tool_call_round_trip(self, adapter: LiteLLMAdapter) -> None:
        raw = _make_litellm_response(
            tool_calls=[
                {"name": "search", "arguments": '{"q": "test"}'},
                {"name": "fetch", "arguments": '{"url": "https://example.com"}'},
            ],
            prompt_tokens=100,
            completion_tokens=30,
        )
        normalized = adapter.normalize_response(raw)
        restored = adapter.denormalize_response(normalized)
        re_normalized = adapter.normalize_response(restored)

        assert re_normalized.content == normalized.content
        assert re_normalized.tool_calls == normalized.tool_calls
        assert re_normalized.token_usage == normalized.token_usage


# ---------------------------------------------------------------------------
# patch() context manager
# ---------------------------------------------------------------------------


class TestPatch:
    """Test the patch() context manager for record and replay modes."""

    def test_patch_record_mode(self, adapter: LiteLLMAdapter) -> None:
        """In RECORD mode, the real API is called and the response is recorded."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.RECORD

        fake_response = _make_litellm_response(content="recorded answer")

        with unittest_mock_patch(
            "litellm.completion",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import litellm

                result = litellm.completion(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4.1",
                )

        recorder.record.assert_called_once()

    def test_patch_replay_mode(self, adapter: LiteLLMAdapter) -> None:
        """In REPLAY mode, the cassette is looked up and denormalized."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = NormalizedResponse(
            content="replayed answer",
            token_usage=TokenUsage(input_tokens=5, output_tokens=3),
        )

        with unittest_mock_patch(
            "litellm.completion",
            return_value=MagicMock(),
        ):
            with adapter.patch(recorder):
                import litellm

                result = litellm.completion(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4.1",
                )

        assert result.choices[0].message.content == "replayed answer"
        recorder.lookup.assert_called_once()

    def test_patch_replay_missing_raises(self, adapter: LiteLLMAdapter) -> None:
        """In REPLAY mode, if lookup returns None, CassetteMissingRequestError is raised."""
        from agentverify.cassette.recorder import CassetteMode
        from agentverify.errors import CassetteMissingRequestError

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None

        with unittest_mock_patch(
            "litellm.completion",
            return_value=MagicMock(),
        ):
            with adapter.patch(recorder):
                import litellm

                with pytest.raises(CassetteMissingRequestError):
                    litellm.completion(
                        messages=[{"role": "user", "content": "hi"}],
                        model="gpt-4.1",
                    )

    def test_patch_auto_mode(self, adapter: LiteLLMAdapter) -> None:
        """AUTO mode behaves like RECORD — calls real API and records."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.AUTO

        fake_response = _make_litellm_response(content="auto answer")

        with unittest_mock_patch(
            "litellm.completion",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import litellm

                result = litellm.completion(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4.1",
                )

        recorder.record.assert_called_once()

    def test_patch_replay_fallback(self, adapter: LiteLLMAdapter) -> None:
        """In REPLAY mode with FALLBACK, if lookup returns None, the real API is called."""
        from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None
        recorder.on_missing = OnMissingRequest.FALLBACK

        fake_response = _make_litellm_response(content="fallback answer")

        with unittest_mock_patch(
            "litellm.completion",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import litellm

                result = litellm.completion(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4.1",
                )

        recorder.record.assert_called_once()
