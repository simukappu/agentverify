"""Tests for the Amazon Bedrock Converse API provider adapter.

Covers normalize_request, normalize_response, denormalize_response,
and the patch() context manager.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as unittest_mock_patch

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.adapters.bedrock import BedrockAdapter
from agentverify.models import TokenUsage


@pytest.fixture
def adapter() -> BedrockAdapter:
    return BedrockAdapter()


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------


class TestName:
    def test_name(self, adapter: BedrockAdapter) -> None:
        assert adapter.name == "bedrock"


# ---------------------------------------------------------------------------
# normalize_request
# ---------------------------------------------------------------------------


class TestNormalizeRequest:
    def test_basic_request(self, adapter: BedrockAdapter) -> None:
        raw = {
            "modelId": "anthropic.claude-sonnet-4-6",
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        }
        result = adapter.normalize_request(raw)
        assert result.messages == [{"role": "user", "content": "Hello"}]
        assert result.model == "anthropic.claude-sonnet-4-6"
        assert result.tools is None
        assert result.parameters == {}

    def test_request_with_tools(self, adapter: BedrockAdapter) -> None:
        raw = {
            "modelId": "anthropic.claude-sonnet-4-6",
            "messages": [{"role": "user", "content": [{"text": "Weather?"}]}],
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "get_weather",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {"location": {"type": "string"}},
                                }
                            },
                        }
                    }
                ]
            },
        }
        result = adapter.normalize_request(raw)
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "get_weather"
        assert "properties" in result.tools[0]["parameters"]

    def test_extra_parameters(self, adapter: BedrockAdapter) -> None:
        raw = {
            "modelId": "anthropic.claude-sonnet-4-6",
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
            "inferenceConfig": {"maxTokens": 1000, "temperature": 0.0},
        }
        result = adapter.normalize_request(raw)
        assert result.parameters == {
            "inferenceConfig": {"maxTokens": 1000, "temperature": 0.0}
        }

    def test_empty_request(self, adapter: BedrockAdapter) -> None:
        result = adapter.normalize_request({})
        assert result.messages == []
        assert result.model == ""
        assert result.tools is None
        assert result.parameters == {}

    def test_multi_text_blocks(self, adapter: BedrockAdapter) -> None:
        raw = {
            "modelId": "model-id",
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Hello"}, {"text": "World"}],
                }
            ],
        }
        result = adapter.normalize_request(raw)
        assert result.messages == [{"role": "user", "content": "Hello\nWorld"}]

    def test_non_text_content_blocks(self, adapter: BedrockAdapter) -> None:
        """Content blocks with non-text entries (e.g. images) are kept as-is."""
        raw = {
            "modelId": "model-id",
            "messages": [
                {
                    "role": "user",
                    "content": [{"image": {"format": "png", "source": {"bytes": b"data"}}}],
                }
            ],
        }
        result = adapter.normalize_request(raw)
        assert result.messages == [
            {"role": "user", "content": [{"image": {"format": "png", "source": {"bytes": b"data"}}}]}
        ]


# ---------------------------------------------------------------------------
# normalize_response
# ---------------------------------------------------------------------------


def _make_bedrock_response(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    input_tokens: int = 50,
    output_tokens: int = 20,
) -> dict:
    """Build a dict that mimics a Bedrock Converse response."""
    content_blocks = []
    if content is not None:
        content_blocks.append({"text": content})
    if tool_calls:
        for tc in tool_calls:
            content_blocks.append(
                {
                    "toolUse": {
                        "toolUseId": tc.get("id", "tooluse_xxx"),
                        "name": tc["name"],
                        "input": tc["input"],
                    }
                }
            )

    stop_reason = "tool_use" if tool_calls else "end_turn"

    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": content_blocks,
            }
        },
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
        "stopReason": stop_reason,
    }


class TestNormalizeResponse:
    def test_text_response(self, adapter: BedrockAdapter) -> None:
        raw = _make_bedrock_response(content="Hello world")
        result = adapter.normalize_response(raw)
        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 50
        assert result.token_usage.output_tokens == 20

    def test_tool_call_response(self, adapter: BedrockAdapter) -> None:
        raw = _make_bedrock_response(
            tool_calls=[
                {"name": "get_weather", "input": {"location": "Tokyo"}},
            ]
        )
        result = adapter.normalize_response(raw)
        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["arguments"] == '{"location": "Tokyo"}'

    def test_no_usage(self, adapter: BedrockAdapter) -> None:
        raw = _make_bedrock_response(content="Hi")
        del raw["usage"]
        result = adapter.normalize_response(raw)
        assert result.token_usage is None

    def test_multiple_text_blocks_in_response(self, adapter: BedrockAdapter) -> None:
        """Multiple text blocks in a response are concatenated."""
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello"}, {"text": "World"}],
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        result = adapter.normalize_response(raw)
        assert result.content == "Hello\nWorld"

    def test_mixed_text_and_tool_use(self, adapter: BedrockAdapter) -> None:
        """Response with both text and toolUse blocks."""
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Let me check the weather."},
                        {"toolUse": {"toolUseId": "tu1", "name": "get_weather", "input": {"location": "Tokyo"}}},
                        {"text": "Done."},
                    ],
                }
            },
            "usage": {"inputTokens": 30, "outputTokens": 15, "totalTokens": 45},
        }
        result = adapter.normalize_response(raw)
        assert result.content == "Let me check the weather.\nDone."
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_multiple_tool_use_blocks(self, adapter: BedrockAdapter) -> None:
        """Response with multiple toolUse blocks."""
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "tu1", "name": "get_weather", "input": {"location": "Tokyo"}}},
                        {"toolUse": {"toolUseId": "tu2", "name": "get_news", "input": {"topic": "tech"}}},
                    ],
                }
            },
            "usage": {"inputTokens": 30, "outputTokens": 15, "totalTokens": 45},
        }
        result = adapter.normalize_response(raw)
        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_news"

    def test_unknown_block_type_ignored(self, adapter: BedrockAdapter) -> None:
        """Unknown content block types are silently ignored."""
        raw = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "tu1", "name": "get_weather", "input": {}}},
                        {"guardContent": {"text": {"text": "blocked"}}},
                    ],
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        result = adapter.normalize_response(raw)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.content is None


# ---------------------------------------------------------------------------
# denormalize_response
# ---------------------------------------------------------------------------


class TestDenormalizeResponse:
    def test_text_response(self, adapter: BedrockAdapter) -> None:
        normalized = NormalizedResponse(
            content="Hello",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        result = adapter.denormalize_response(normalized)
        assert result["output"]["message"]["role"] == "assistant"
        assert result["output"]["message"]["content"] == [{"text": "Hello"}]
        assert result["stopReason"] == "end_turn"
        assert result["usage"]["inputTokens"] == 10
        assert result["usage"]["outputTokens"] == 5
        assert result["usage"]["totalTokens"] == 15

    def test_tool_call_response(self, adapter: BedrockAdapter) -> None:
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
            ],
            token_usage=TokenUsage(input_tokens=20, output_tokens=10),
        )
        result = adapter.denormalize_response(normalized)
        assert result["stopReason"] == "tool_use"
        blocks = result["output"]["message"]["content"]
        assert len(blocks) == 1
        assert blocks[0]["toolUse"]["name"] == "get_weather"
        assert blocks[0]["toolUse"]["input"] == {"location": "Tokyo"}

    def test_no_usage(self, adapter: BedrockAdapter) -> None:
        normalized = NormalizedResponse(content="Hi")
        result = adapter.denormalize_response(normalized)
        assert result["usage"] is None

    def test_invalid_json_arguments(self, adapter: BedrockAdapter) -> None:
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "some_tool", "arguments": "not-valid-json"},
            ],
        )
        result = adapter.denormalize_response(normalized)
        blocks = result["output"]["message"]["content"]
        assert blocks[0]["toolUse"]["input"] == {}

    def test_dict_arguments(self, adapter: BedrockAdapter) -> None:
        """When arguments is already a dict, it's used directly."""
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "some_tool", "arguments": {"key": "value"}},
            ],
        )
        result = adapter.denormalize_response(normalized)
        blocks = result["output"]["message"]["content"]
        assert blocks[0]["toolUse"]["input"] == {"key": "value"}


# ---------------------------------------------------------------------------
# Round-trip: normalize_response → denormalize_response preserves semantics
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_text_round_trip(self, adapter: BedrockAdapter) -> None:
        raw = _make_bedrock_response(
            content="Hello world", input_tokens=50, output_tokens=20
        )
        normalized = adapter.normalize_response(raw)
        restored = adapter.denormalize_response(normalized)
        re_normalized = adapter.normalize_response(restored)

        assert re_normalized.content == normalized.content
        assert re_normalized.tool_calls == normalized.tool_calls
        assert re_normalized.token_usage == normalized.token_usage

    def test_tool_call_round_trip(self, adapter: BedrockAdapter) -> None:
        raw = _make_bedrock_response(
            tool_calls=[
                {"name": "search", "input": {"q": "test"}},
                {"name": "fetch", "input": {"url": "https://example.com"}},
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

    def test_patch_record_mode(self, adapter: BedrockAdapter) -> None:
        """In RECORD mode, the real API is called and the response is recorded."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.RECORD

        fake_response = _make_bedrock_response(content="recorded answer")

        with unittest_mock_patch(
            "botocore.client.BaseClient._make_api_call",
            return_value=fake_response,
        ) as mock_api:
            with adapter.patch(recorder):
                import botocore.client

                client = MagicMock(spec=botocore.client.BaseClient)
                result = botocore.client.BaseClient._make_api_call(
                    client,
                    "Converse",
                    {
                        "modelId": "anthropic.claude-sonnet-4-6",
                        "messages": [
                            {"role": "user", "content": [{"text": "hi"}]}
                        ],
                    },
                )

        recorder.record.assert_called_once()

    def test_patch_replay_mode(self, adapter: BedrockAdapter) -> None:
        """In REPLAY mode, the cassette is looked up and denormalized."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = NormalizedResponse(
            content="replayed answer",
            token_usage=TokenUsage(input_tokens=5, output_tokens=3),
        )

        with unittest_mock_patch(
            "botocore.client.BaseClient._make_api_call",
            return_value={},
        ):
            with adapter.patch(recorder):
                import botocore.client

                client = MagicMock(spec=botocore.client.BaseClient)
                result = botocore.client.BaseClient._make_api_call(
                    client,
                    "Converse",
                    {
                        "modelId": "anthropic.claude-sonnet-4-6",
                        "messages": [
                            {"role": "user", "content": [{"text": "hi"}]}
                        ],
                    },
                )

        assert result["output"]["message"]["content"] == [{"text": "replayed answer"}]
        recorder.lookup.assert_called_once()

    def test_patch_replay_missing_raises(self, adapter: BedrockAdapter) -> None:
        """In REPLAY mode, if lookup returns None, CassetteMissingRequestError is raised."""
        from agentverify.cassette.recorder import CassetteMode
        from agentverify.errors import CassetteMissingRequestError

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None

        with unittest_mock_patch(
            "botocore.client.BaseClient._make_api_call",
            return_value={},
        ):
            with adapter.patch(recorder):
                import botocore.client

                client = MagicMock(spec=botocore.client.BaseClient)
                with pytest.raises(CassetteMissingRequestError):
                    botocore.client.BaseClient._make_api_call(
                        client,
                        "Converse",
                        {
                            "modelId": "anthropic.claude-sonnet-4-6",
                            "messages": [
                                {"role": "user", "content": [{"text": "hi"}]}
                            ],
                        },
                    )

    def test_patch_auto_mode(self, adapter: BedrockAdapter) -> None:
        """AUTO mode behaves like RECORD — calls real API and records."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.AUTO

        fake_response = _make_bedrock_response(content="auto answer")

        with unittest_mock_patch(
            "botocore.client.BaseClient._make_api_call",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import botocore.client

                client = MagicMock(spec=botocore.client.BaseClient)
                result = botocore.client.BaseClient._make_api_call(
                    client,
                    "Converse",
                    {
                        "modelId": "anthropic.claude-sonnet-4-6",
                        "messages": [
                            {"role": "user", "content": [{"text": "hi"}]}
                        ],
                    },
                )

        recorder.record.assert_called_once()

    def test_patch_replay_fallback(self, adapter: BedrockAdapter) -> None:
        """In REPLAY mode with FALLBACK, if lookup returns None, the real API is called."""
        from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None
        recorder.on_missing = OnMissingRequest.FALLBACK

        fake_response = _make_bedrock_response(content="fallback answer")

        with unittest_mock_patch(
            "botocore.client.BaseClient._make_api_call",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import botocore.client

                client = MagicMock(spec=botocore.client.BaseClient)
                result = botocore.client.BaseClient._make_api_call(
                    client,
                    "Converse",
                    {
                        "modelId": "anthropic.claude-sonnet-4-6",
                        "messages": [
                            {"role": "user", "content": [{"text": "hi"}]}
                        ],
                    },
                )

        recorder.record.assert_called_once()

    def test_patch_non_converse_passthrough(self, adapter: BedrockAdapter) -> None:
        """Non-Converse operations pass through to the original implementation."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.RECORD

        original_response = {"TableNames": ["my-table"]}

        with unittest_mock_patch(
            "botocore.client.BaseClient._make_api_call",
            return_value=original_response,
        ):
            with adapter.patch(recorder):
                import botocore.client

                client = MagicMock(spec=botocore.client.BaseClient)
                result = botocore.client.BaseClient._make_api_call(
                    client, "ListTables", {}
                )

        # Non-Converse calls should NOT be recorded
        recorder.record.assert_not_called()
        assert result == original_response
