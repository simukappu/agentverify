"""Tests for the OpenAI provider adapter.

Covers normalize_request, normalize_response, denormalize_response,
and the patch() context manager.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as unittest_mock_patch

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.adapters.openai import OpenAIAdapter
from agentverify.models import TokenUsage


@pytest.fixture
def adapter() -> OpenAIAdapter:
    return OpenAIAdapter()


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------


class TestName:
    def test_name(self, adapter: OpenAIAdapter) -> None:
        assert adapter.name == "openai"


# ---------------------------------------------------------------------------
# normalize_request
# ---------------------------------------------------------------------------


class TestNormalizeRequest:
    def test_basic_request(self, adapter: OpenAIAdapter) -> None:
        raw = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4.1",
        }
        result = adapter.normalize_request(raw)
        assert result.messages == [{"role": "user", "content": "Hello"}]
        assert result.model == "gpt-4.1"
        assert result.tools is None
        assert result.parameters == {}

    def test_request_with_tools(self, adapter: OpenAIAdapter) -> None:
        raw = {
            "messages": [{"role": "user", "content": "Weather?"}],
            "model": "gpt-4.1",
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
        assert "properties" in result.tools[0]["parameters"]

    def test_extra_parameters(self, adapter: OpenAIAdapter) -> None:
        raw = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "gpt-4.1",
            "temperature": 0.0,
            "max_tokens": 100,
        }
        result = adapter.normalize_request(raw)
        assert result.parameters == {"temperature": 0.0, "max_tokens": 100}

    def test_empty_request(self, adapter: OpenAIAdapter) -> None:
        result = adapter.normalize_request({})
        assert result.messages == []
        assert result.model == ""
        assert result.tools is None
        assert result.parameters == {}


# ---------------------------------------------------------------------------
# normalize_response
# ---------------------------------------------------------------------------


def _make_openai_response(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a mock that mimics an OpenAI ChatCompletion object."""
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


class TestNormalizeResponse:
    def test_text_response(self, adapter: OpenAIAdapter) -> None:
        raw = _make_openai_response(content="Hello world")
        result = adapter.normalize_response(raw)
        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 10
        assert result.token_usage.output_tokens == 5

    def test_tool_call_response(self, adapter: OpenAIAdapter) -> None:
        raw = _make_openai_response(
            tool_calls=[
                {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
            ]
        )
        result = adapter.normalize_response(raw)
        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["arguments"] == '{"location": "Tokyo"}'

    def test_no_usage(self, adapter: OpenAIAdapter) -> None:
        raw = _make_openai_response(content="Hi")
        raw.usage = None
        result = adapter.normalize_response(raw)
        assert result.token_usage is None


# ---------------------------------------------------------------------------
# denormalize_response
# ---------------------------------------------------------------------------


class TestDenormalizeResponse:
    def test_text_response(self, adapter: OpenAIAdapter) -> None:
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
        assert result.usage.total_tokens == 15

    def test_tool_call_response(self, adapter: OpenAIAdapter) -> None:
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
            ],
            token_usage=TokenUsage(input_tokens=20, output_tokens=10),
        )
        result = adapter.denormalize_response(normalized)
        assert result.choices[0].message.content is None
        assert result.choices[0].finish_reason == "tool_calls"
        tc = result.choices[0].message.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"location": "Tokyo"}'
        assert tc.type == "function"
        assert tc.id == "call_0"

    def test_no_usage(self, adapter: OpenAIAdapter) -> None:
        normalized = NormalizedResponse(content="Hi")
        result = adapter.denormalize_response(normalized)
        assert result.usage is None

    def test_metadata_preserved(self, adapter: OpenAIAdapter) -> None:
        normalized = NormalizedResponse(
            content="Hi",
            raw_metadata={"model": "gpt-4.1", "id": "chatcmpl-123"},
        )
        result = adapter.denormalize_response(normalized)
        assert result.model == "gpt-4.1"
        assert result.id == "chatcmpl-123"


# ---------------------------------------------------------------------------
# Round-trip: normalize_response → denormalize_response preserves semantics
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_text_round_trip(self, adapter: OpenAIAdapter) -> None:
        raw = _make_openai_response(content="Hello world", prompt_tokens=50, completion_tokens=20)
        normalized = adapter.normalize_response(raw)
        restored = adapter.denormalize_response(normalized)
        re_normalized = adapter.normalize_response(restored)

        assert re_normalized.content == normalized.content
        assert re_normalized.tool_calls == normalized.tool_calls
        assert re_normalized.token_usage == normalized.token_usage

    def test_tool_call_round_trip(self, adapter: OpenAIAdapter) -> None:
        raw = _make_openai_response(
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

    def test_patch_record_mode(self, adapter: OpenAIAdapter) -> None:
        """In RECORD mode, the real API is called and the response is recorded."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.RECORD

        fake_response = _make_openai_response(content="recorded answer")

        with unittest_mock_patch(
            "openai.resources.chat.completions.Completions.create",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import openai

                result = openai.resources.chat.completions.Completions.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4.1",
                )

        recorder.record.assert_called_once()

    def test_patch_replay_mode(self, adapter: OpenAIAdapter) -> None:
        """In REPLAY mode, the cassette is looked up and denormalized."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = NormalizedResponse(
            content="replayed answer",
            token_usage=TokenUsage(input_tokens=5, output_tokens=3),
        )

        with unittest_mock_patch(
            "openai.resources.chat.completions.Completions.create",
            return_value=MagicMock(),
        ):
            with adapter.patch(recorder):
                import openai

                result = openai.resources.chat.completions.Completions.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4.1",
                )

        assert result.choices[0].message.content == "replayed answer"
        recorder.lookup.assert_called_once()

    def test_patch_replay_missing_raises(self, adapter: OpenAIAdapter) -> None:
        """In REPLAY mode, if lookup returns None, CassetteMissingRequestError is raised."""
        from agentverify.cassette.recorder import CassetteMode
        from agentverify.errors import CassetteMissingRequestError

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None

        with unittest_mock_patch(
            "openai.resources.chat.completions.Completions.create",
            return_value=MagicMock(),
        ):
            with adapter.patch(recorder):
                import openai

                with pytest.raises(CassetteMissingRequestError):
                    openai.resources.chat.completions.Completions.create(
                        messages=[{"role": "user", "content": "hi"}],
                        model="gpt-4.1",
                    )

    def test_patch_auto_mode(self, adapter: OpenAIAdapter) -> None:
        """AUTO mode behaves like RECORD — calls real API and records."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.AUTO

        fake_response = _make_openai_response(content="auto answer")

        with unittest_mock_patch(
            "openai.resources.chat.completions.Completions.create",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import openai

                result = openai.resources.chat.completions.Completions.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4.1",
                )

        recorder.record.assert_called_once()

    def test_patch_replay_fallback(self, adapter: OpenAIAdapter) -> None:
        """In REPLAY mode with FALLBACK, if lookup returns None, the real API is called."""
        from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None
        recorder.on_missing = OnMissingRequest.FALLBACK

        fake_response = _make_openai_response(content="fallback answer")

        with unittest_mock_patch(
            "openai.resources.chat.completions.Completions.create",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import openai

                result = openai.resources.chat.completions.Completions.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4.1",
                )

        recorder.record.assert_called_once()


# ---------------------------------------------------------------------------
# LegacyAPIResponse / with_raw_response handling (langchain-openai v1.x path)
# ---------------------------------------------------------------------------


class TestLegacyAPIResponseHandling:
    """langchain-openai v1.x calls ``client.with_raw_response.create(...)``.

    The wrapper sets an ``X-Stainless-Raw-Response`` extra header and
    expects a ``LegacyAPIResponse``-shaped object back that exposes
    ``.parse()``. These tests cover both the record-side unwrap and the
    replay-side re-wrap.
    """

    def test_normalize_response_unwraps_legacy_api_response(
        self, adapter: OpenAIAdapter
    ) -> None:
        """``normalize_response`` must call ``.parse()`` on
        ``LegacyAPIResponse``-like wrappers so recording works via the
        ``with_raw_response`` code path.
        """
        inner = _make_openai_response(content="wrapped answer")

        class _LegacyLike:
            """Stand-in with ``.parse()`` but no ``.choices`` attribute."""

            def parse(self):
                return inner

        result = adapter.normalize_response(_LegacyLike())
        assert result.content == "wrapped answer"

    def test_replay_via_raw_response_returns_wrapper(
        self, adapter: OpenAIAdapter
    ) -> None:
        """In REPLAY mode, when the caller went through ``with_raw_response``
        (as indicated by the ``X-Stainless-Raw-Response`` header), the
        adapter returns a ``LegacyAPIResponse``-compatible wrapper.
        """
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = NormalizedResponse(
            content="raw-replayed",
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

        with unittest_mock_patch(
            "openai.resources.chat.completions.Completions.create",
            return_value=MagicMock(),
        ):
            with adapter.patch(recorder):
                import openai

                wrapped = openai.resources.chat.completions.Completions.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4.1",
                    extra_headers={"X-Stainless-Raw-Response": "true"},
                )

        # Caller should be able to .parse() the wrapper
        assert hasattr(wrapped, "parse")
        assert isinstance(wrapped.headers, dict)
        parsed = wrapped.parse()
        assert parsed.choices[0].message.content == "raw-replayed"
