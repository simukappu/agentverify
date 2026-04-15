"""Tests for the Anthropic provider adapter.

Covers normalize_request, normalize_response, denormalize_response,
and the patch() context manager.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as unittest_mock_patch

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.adapters.anthropic import AnthropicAdapter
from agentverify.models import TokenUsage


@pytest.fixture
def adapter() -> AnthropicAdapter:
    return AnthropicAdapter()


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------


class TestName:
    def test_name(self, adapter: AnthropicAdapter) -> None:
        assert adapter.name == "anthropic"


# ---------------------------------------------------------------------------
# normalize_request
# ---------------------------------------------------------------------------


class TestNormalizeRequest:
    def test_basic_request(self, adapter: AnthropicAdapter) -> None:
        raw = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "claude-sonnet-4-6-20250514",
        }
        result = adapter.normalize_request(raw)
        assert result.messages == [{"role": "user", "content": "Hello"}]
        assert result.model == "claude-sonnet-4-6-20250514"
        assert result.tools is None
        assert result.parameters == {}

    def test_request_with_tools(self, adapter: AnthropicAdapter) -> None:
        raw = {
            "messages": [{"role": "user", "content": "Weather?"}],
            "model": "claude-sonnet-4-6-20250514",
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        }
        result = adapter.normalize_request(raw)
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "get_weather"
        assert "properties" in result.tools[0]["parameters"]

    def test_extra_parameters(self, adapter: AnthropicAdapter) -> None:
        raw = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "claude-sonnet-4-6-20250514",
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        result = adapter.normalize_request(raw)
        assert result.parameters == {"temperature": 0.0, "max_tokens": 1024}

    def test_empty_request(self, adapter: AnthropicAdapter) -> None:
        result = adapter.normalize_request({})
        assert result.messages == []
        assert result.model == ""
        assert result.tools is None
        assert result.parameters == {}


# ---------------------------------------------------------------------------
# normalize_response
# ---------------------------------------------------------------------------


def _make_anthropic_response(
    content_blocks: list[dict] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Build a mock that mimics an Anthropic Message object.

    content_blocks should be a list of dicts like:
      [{"type": "text", "text": "Hello"}]
      [{"type": "tool_use", "id": "toolu_xxx", "name": "get_weather", "input": {"location": "Tokyo"}}]
    """
    mock = MagicMock()

    blocks = []
    if content_blocks:
        for cb in content_blocks:
            block = MagicMock()
            block.type = cb["type"]
            if cb["type"] == "text":
                block.text = cb["text"]
            elif cb["type"] == "tool_use":
                block.name = cb["name"]
                block.input = cb["input"]
                block.id = cb.get("id", "toolu_xxx")
            blocks.append(block)

    mock.content = blocks
    mock.usage.input_tokens = input_tokens
    mock.usage.output_tokens = output_tokens
    mock.stop_reason = "end_turn"
    mock.model = "claude-sonnet-4-6-20250514"
    mock.id = "msg_xxx"
    return mock


class TestNormalizeResponse:
    def test_text_response(self, adapter: AnthropicAdapter) -> None:
        raw = _make_anthropic_response(
            content_blocks=[{"type": "text", "text": "Hello world"}]
        )
        result = adapter.normalize_response(raw)
        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 10
        assert result.token_usage.output_tokens == 5

    def test_tool_use_response(self, adapter: AnthropicAdapter) -> None:
        raw = _make_anthropic_response(
            content_blocks=[
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "get_weather",
                    "input": {"location": "Tokyo"},
                }
            ]
        )
        result = adapter.normalize_response(raw)
        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["arguments"] == '{"location": "Tokyo"}'

    def test_mixed_content_response(self, adapter: AnthropicAdapter) -> None:
        raw = _make_anthropic_response(
            content_blocks=[
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "get_weather",
                    "input": {"location": "Tokyo"},
                },
            ]
        )
        result = adapter.normalize_response(raw)
        assert result.content == "Let me check the weather."
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

    def test_multiple_text_blocks(self, adapter: AnthropicAdapter) -> None:
        raw = _make_anthropic_response(
            content_blocks=[
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "beautiful "},
                {"type": "text", "text": "world"},
            ]
        )
        result = adapter.normalize_response(raw)
        assert result.content == "Hello beautiful world"

    def test_text_then_tool_use(self, adapter: AnthropicAdapter) -> None:
        """After concatenating text, the loop continues to a tool_use block."""
        raw = _make_anthropic_response(
            content_blocks=[
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": " Part 2"},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "search",
                    "input": {"q": "test"},
                },
            ]
        )
        result = adapter.normalize_response(raw)
        assert result.content == "Part 1 Part 2"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

    def test_unknown_block_type_ignored(self, adapter: AnthropicAdapter) -> None:
        """Unknown block types are silently skipped."""
        raw = _make_anthropic_response(
            content_blocks=[
                {"type": "text", "text": "Hello"},
            ]
        )
        # Add an unknown block type after the text block
        unknown_block = MagicMock()
        unknown_block.type = "image"
        raw.content.append(unknown_block)
        result = adapter.normalize_response(raw)
        assert result.content == "Hello"
        assert result.tool_calls is None

    def test_no_usage(self, adapter: AnthropicAdapter) -> None:
        raw = _make_anthropic_response(
            content_blocks=[{"type": "text", "text": "Hi"}]
        )
        raw.usage = None
        result = adapter.normalize_response(raw)
        assert result.token_usage is None


# ---------------------------------------------------------------------------
# denormalize_response
# ---------------------------------------------------------------------------


class TestDenormalizeResponse:
    def test_text_response(self, adapter: AnthropicAdapter) -> None:
        normalized = NormalizedResponse(
            content="Hello",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        result = adapter.denormalize_response(normalized)
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_tool_use_response(self, adapter: AnthropicAdapter) -> None:
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
            ],
            token_usage=TokenUsage(input_tokens=20, output_tokens=10),
        )
        result = adapter.denormalize_response(normalized)
        assert len(result.content) == 1
        assert result.content[0].type == "tool_use"
        assert result.content[0].name == "get_weather"
        assert result.content[0].input == {"location": "Tokyo"}
        assert result.content[0].id == "toolu_0"
        assert result.stop_reason == "tool_use"

    def test_no_usage(self, adapter: AnthropicAdapter) -> None:
        normalized = NormalizedResponse(content="Hi")
        result = adapter.denormalize_response(normalized)
        assert result.usage is None

    def test_metadata_preserved(self, adapter: AnthropicAdapter) -> None:
        normalized = NormalizedResponse(
            content="Hi",
            raw_metadata={"model": "claude-sonnet-4-6-20250514", "id": "msg_123"},
        )
        result = adapter.denormalize_response(normalized)
        assert result.model == "claude-sonnet-4-6-20250514"
        assert result.id == "msg_123"

    def test_tool_use_invalid_json_arguments(self, adapter: AnthropicAdapter) -> None:
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "get_weather", "arguments": "not-valid-json"},
            ],
        )
        result = adapter.denormalize_response(normalized)
        assert result.content[0].type == "tool_use"
        assert result.content[0].input == {}

    def test_tool_use_dict_arguments(self, adapter: AnthropicAdapter) -> None:
        """When arguments is already a dict (not a JSON string), use it directly."""
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            ],
        )
        result = adapter.denormalize_response(normalized)
        assert result.content[0].type == "tool_use"
        assert result.content[0].input == {"location": "Tokyo"}


# ---------------------------------------------------------------------------
# Round-trip: normalize_response → denormalize_response preserves semantics
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_text_round_trip(self, adapter: AnthropicAdapter) -> None:
        raw = _make_anthropic_response(
            content_blocks=[{"type": "text", "text": "Hello world"}],
            input_tokens=50,
            output_tokens=20,
        )
        normalized = adapter.normalize_response(raw)
        restored = adapter.denormalize_response(normalized)
        re_normalized = adapter.normalize_response(restored)

        assert re_normalized.content == normalized.content
        assert re_normalized.tool_calls == normalized.tool_calls
        assert re_normalized.token_usage == normalized.token_usage

    def test_tool_call_round_trip(self, adapter: AnthropicAdapter) -> None:
        raw = _make_anthropic_response(
            content_blocks=[
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"q": "test"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "fetch",
                    "input": {"url": "https://example.com"},
                },
            ],
            input_tokens=100,
            output_tokens=30,
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

    def test_patch_record_mode(self, adapter: AnthropicAdapter) -> None:
        """In RECORD mode, the real API is called and the response is recorded."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.RECORD

        fake_response = _make_anthropic_response(
            content_blocks=[{"type": "text", "text": "recorded answer"}]
        )

        with unittest_mock_patch(
            "anthropic.resources.messages.Messages.create",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import anthropic

                result = anthropic.resources.messages.Messages.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-sonnet-4-6-20250514",
                )

        recorder.record.assert_called_once()

    def test_patch_replay_mode(self, adapter: AnthropicAdapter) -> None:
        """In REPLAY mode, the cassette is looked up and denormalized."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = NormalizedResponse(
            content="replayed answer",
            token_usage=TokenUsage(input_tokens=5, output_tokens=3),
        )

        with unittest_mock_patch(
            "anthropic.resources.messages.Messages.create",
            return_value=MagicMock(),
        ):
            with adapter.patch(recorder):
                import anthropic

                result = anthropic.resources.messages.Messages.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-sonnet-4-6-20250514",
                )

        assert result.content[0].type == "text"
        assert result.content[0].text == "replayed answer"
        recorder.lookup.assert_called_once()

    def test_patch_replay_missing_raises(self, adapter: AnthropicAdapter) -> None:
        """In REPLAY mode, if lookup returns None, CassetteMissingRequestError is raised."""
        from agentverify.cassette.recorder import CassetteMode
        from agentverify.errors import CassetteMissingRequestError

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None

        with unittest_mock_patch(
            "anthropic.resources.messages.Messages.create",
            return_value=MagicMock(),
        ):
            with adapter.patch(recorder):
                import anthropic

                with pytest.raises(CassetteMissingRequestError):
                    anthropic.resources.messages.Messages.create(
                        messages=[{"role": "user", "content": "hi"}],
                        model="claude-sonnet-4-6-20250514",
                    )

    def test_patch_auto_mode(self, adapter: AnthropicAdapter) -> None:
        """AUTO mode behaves like RECORD — calls real API and records."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.AUTO

        fake_response = _make_anthropic_response(
            content_blocks=[{"type": "text", "text": "auto answer"}]
        )

        with unittest_mock_patch(
            "anthropic.resources.messages.Messages.create",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import anthropic

                result = anthropic.resources.messages.Messages.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-sonnet-4-6-20250514",
                )

        recorder.record.assert_called_once()

    def test_patch_replay_fallback(self, adapter: AnthropicAdapter) -> None:
        """In REPLAY mode with FALLBACK, if lookup returns None, the real API is called."""
        from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None
        recorder.on_missing = OnMissingRequest.FALLBACK

        fake_response = _make_anthropic_response(
            content_blocks=[{"type": "text", "text": "fallback answer"}]
        )

        with unittest_mock_patch(
            "anthropic.resources.messages.Messages.create",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import anthropic

                result = anthropic.resources.messages.Messages.create(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-sonnet-4-6-20250514",
                )

        recorder.record.assert_called_once()
