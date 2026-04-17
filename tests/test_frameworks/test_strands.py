"""Tests for the Strands Agents framework adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentverify.frameworks.strands import from_strands
from agentverify.models import ExecutionResult, TokenUsage, ToolCall


def _make_strands_result(
    messages: list | None = None,
    metrics: dict | None = None,
    message: dict | None = None,
) -> SimpleNamespace:
    """Build a fake Strands AgentResult-like object."""
    state = SimpleNamespace(messages=messages or [])
    result = SimpleNamespace(state=state)
    if metrics is not None:
        result.metrics = metrics
    if message is not None:
        result.message = message
    return result


class TestFromStrands:
    """Tests for from_strands()."""

    def test_basic_tool_calls(self):
        """Extract tool calls from state.messages."""
        result = _make_strands_result(
            messages=[
                {
                    "content": [
                        {
                            "toolUse": {
                                "name": "list_directory",
                                "input": {"path": "/tmp"},
                            }
                        }
                    ]
                },
                {
                    "content": [
                        {
                            "toolUse": {
                                "name": "read_file",
                                "input": {"path": "/tmp/a.txt"},
                            }
                        }
                    ]
                },
            ]
        )

        er = from_strands(result)

        assert isinstance(er, ExecutionResult)
        assert len(er.tool_calls) == 2
        assert er.tool_calls[0] == ToolCall("list_directory", {"path": "/tmp"})
        assert er.tool_calls[1] == ToolCall("read_file", {"path": "/tmp/a.txt"})

    def test_token_usage(self):
        """Extract token usage from metrics."""
        result = _make_strands_result(
            metrics={"inputTokens": 100, "outputTokens": 50},
        )

        er = from_strands(result)

        assert er.token_usage is not None
        assert er.token_usage.input_tokens == 100
        assert er.token_usage.output_tokens == 50
        assert er.token_usage.total_tokens == 150

    def test_token_usage_none_when_no_metrics(self):
        """token_usage is None when metrics attribute is absent."""
        result = _make_strands_result()
        # Remove metrics attribute entirely
        if hasattr(result, "metrics"):
            delattr(result, "metrics")

        er = from_strands(result)

        assert er.token_usage is None

    def test_token_usage_none_when_metrics_empty(self):
        """token_usage is None when metrics is falsy."""
        result = _make_strands_result(metrics={})

        er = from_strands(result)

        assert er.token_usage is None

    def test_final_output(self):
        """Extract final output from message.content text blocks."""
        result = _make_strands_result(
            message={
                "content": [
                    {"text": "Here is the analysis."},
                ]
            },
        )

        er = from_strands(result)

        assert er.final_output == "Here is the analysis."

    def test_final_output_uses_last_text_block(self):
        """When multiple text blocks exist, use the last one."""
        result = _make_strands_result(
            message={
                "content": [
                    {"text": "First block."},
                    {"text": "Second block."},
                ]
            },
        )

        er = from_strands(result)

        assert er.final_output == "Second block."

    def test_final_output_none_when_no_message(self):
        """final_output is None when message attribute is absent."""
        result = _make_strands_result()
        if hasattr(result, "message"):
            delattr(result, "message")

        er = from_strands(result)

        assert er.final_output is None

    def test_final_output_none_when_message_empty(self):
        """final_output is None when message is falsy."""
        result = _make_strands_result(message={})

        er = from_strands(result)

        assert er.final_output is None

    def test_tool_use_without_input(self):
        """toolUse without input key defaults to empty dict."""
        result = _make_strands_result(
            messages=[
                {
                    "content": [
                        {"toolUse": {"name": "get_status"}}
                    ]
                },
            ]
        )

        er = from_strands(result)

        assert len(er.tool_calls) == 1
        assert er.tool_calls[0] == ToolCall("get_status", {})

    def test_mixed_content_blocks(self):
        """Only toolUse blocks are extracted; text blocks are ignored."""
        result = _make_strands_result(
            messages=[
                {
                    "content": [
                        {"text": "Let me check."},
                        {"toolUse": {"name": "search", "input": {"q": "test"}}},
                        {"text": "Done."},
                    ]
                },
            ]
        )

        er = from_strands(result)

        assert len(er.tool_calls) == 1
        assert er.tool_calls[0].name == "search"

    def test_non_dict_message_skipped(self):
        """Non-dict messages in state.messages are skipped."""
        result = _make_strands_result(
            messages=["not a dict", 42, None],
        )

        er = from_strands(result)

        assert er.tool_calls == []

    def test_non_list_content_skipped(self):
        """Messages with non-list content are skipped."""
        result = _make_strands_result(
            messages=[
                {"content": "just a string"},
                {"content": 42},
            ],
        )

        er = from_strands(result)

        assert er.tool_calls == []

    def test_full_round_trip(self):
        """Full extraction with tool calls, metrics, and final output."""
        result = _make_strands_result(
            messages=[
                {
                    "content": [
                        {
                            "toolUse": {
                                "name": "list_directory",
                                "input": {"path": "/"},
                            }
                        }
                    ]
                },
                {
                    "content": [
                        {"toolResult": {"content": [{"text": "file1.txt"}]}}
                    ]
                },
                {
                    "content": [
                        {
                            "toolUse": {
                                "name": "read_file",
                                "input": {"path": "/file1.txt"},
                            }
                        }
                    ]
                },
            ],
            metrics={"inputTokens": 200, "outputTokens": 100},
            message={
                "content": [
                    {"text": "The directory contains one file."},
                ]
            },
        )

        er = from_strands(result)

        assert len(er.tool_calls) == 2
        assert er.tool_calls[0].name == "list_directory"
        assert er.tool_calls[1].name == "read_file"
        assert er.token_usage == TokenUsage(input_tokens=200, output_tokens=100)
        assert er.final_output == "The directory contains one file."

    def test_message_not_dict(self):
        """final_output is None when message is not a dict."""
        result = _make_strands_result()
        result.message = "not a dict"

        er = from_strands(result)

        assert er.final_output is None

    def test_final_output_none_when_content_empty_list(self):
        """final_output is None when message content is an empty list."""
        result = _make_strands_result(
            message={"content": []},
        )

        er = from_strands(result)

        assert er.final_output is None

    def test_final_output_none_when_no_text_blocks(self):
        """final_output is None when content has blocks but none with text."""
        result = _make_strands_result(
            message={
                "content": [
                    {"toolUse": {"name": "some_tool"}},
                    {"image": "data:..."},
                ]
            },
        )

        er = from_strands(result)

        assert er.final_output is None
