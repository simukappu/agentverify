"""Tests for the OpenAI Agents SDK framework adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentverify.frameworks.openai_agents import from_openai_agents
from agentverify.models import ExecutionResult, TokenUsage, ToolCall


def _tool_call_item(name: str, arguments: dict | str = "{}") -> SimpleNamespace:
    """Build a fake ToolCallItem-like object."""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    raw_item = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(type="tool_call_item", raw_item=raw_item)


def _message_item(text: str = "") -> SimpleNamespace:
    return SimpleNamespace(type="message_output_item", raw_item=SimpleNamespace(content=text))


def _usage(input_tokens: int = 0, output_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _run_result(
    new_items: list | None = None,
    final_output: str | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    context_wrapper = SimpleNamespace(usage=usage) if usage is not None else None
    return SimpleNamespace(
        new_items=new_items or [],
        final_output=final_output,
        context_wrapper=context_wrapper,
    )


class TestFromOpenaiAgents:
    """Tests for from_openai_agents()."""

    def test_basic_tool_calls(self):
        result = _run_result(
            new_items=[_tool_call_item("get_weather", {"city": "Tokyo"})],
            final_output="Sunny in Tokyo.",
        )
        er = from_openai_agents(result)
        assert len(er.tool_calls) == 1
        assert er.tool_calls[0] == ToolCall("get_weather", {"city": "Tokyo"})

    def test_multiple_tool_calls(self):
        result = _run_result(
            new_items=[
                _tool_call_item("search", {"q": "AI"}),
                _message_item("thinking..."),
                _tool_call_item("summarize", {"text": "..."}),
            ],
            final_output="Done.",
        )
        er = from_openai_agents(result)
        assert len(er.tool_calls) == 2
        assert er.tool_calls[0].name == "search"
        assert er.tool_calls[1].name == "summarize"

    def test_final_output(self):
        result = _run_result(final_output="The answer is 42.")
        er = from_openai_agents(result)
        assert er.final_output == "The answer is 42."

    def test_final_output_none(self):
        result = _run_result(final_output=None)
        er = from_openai_agents(result)
        assert er.final_output is None

    def test_final_output_non_string_converted(self):
        result = _run_result(final_output={"key": "value"})
        er = from_openai_agents(result)
        assert er.final_output == "{'key': 'value'}"

    def test_token_usage(self):
        result = _run_result(usage=_usage(input_tokens=200, output_tokens=100))
        er = from_openai_agents(result)
        assert er.token_usage == TokenUsage(input_tokens=200, output_tokens=100)

    def test_token_usage_none_when_no_context_wrapper(self):
        result = SimpleNamespace(new_items=[], final_output=None, context_wrapper=None)
        er = from_openai_agents(result)
        assert er.token_usage is None

    def test_token_usage_none_when_zero(self):
        result = _run_result(usage=_usage(input_tokens=0, output_tokens=0))
        er = from_openai_agents(result)
        assert er.token_usage is None

    def test_arguments_as_json_string(self):
        result = _run_result(new_items=[_tool_call_item("search", '{"q": "test"}')])
        er = from_openai_agents(result)
        assert er.tool_calls[0] == ToolCall("search", {"q": "test"})

    def test_arguments_invalid_json_defaults_to_empty(self):
        result = _run_result(new_items=[_tool_call_item("search", "not json")])
        er = from_openai_agents(result)
        assert er.tool_calls[0] == ToolCall("search", {})

    def test_arguments_as_dict(self):
        raw_item = SimpleNamespace(name="tool", arguments={"key": "val"})
        item = SimpleNamespace(type="tool_call_item", raw_item=raw_item)
        result = _run_result(new_items=[item])
        er = from_openai_agents(result)
        assert er.tool_calls[0] == ToolCall("tool", {"key": "val"})

    def test_arguments_non_string_non_dict_defaults_to_empty(self):
        raw_item = SimpleNamespace(name="tool", arguments=12345)
        item = SimpleNamespace(type="tool_call_item", raw_item=raw_item)
        result = _run_result(new_items=[item])
        er = from_openai_agents(result)
        assert er.tool_calls[0] == ToolCall("tool", {})

    def test_tool_call_without_name_skipped(self):
        raw_item = SimpleNamespace(name="", arguments="{}")
        item = SimpleNamespace(type="tool_call_item", raw_item=raw_item)
        result = _run_result(new_items=[item])
        er = from_openai_agents(result)
        assert er.tool_calls == []

    def test_tool_call_with_none_raw_item_skipped(self):
        item = SimpleNamespace(type="tool_call_item", raw_item=None)
        result = _run_result(new_items=[item])
        er = from_openai_agents(result)
        assert er.tool_calls == []

    def test_non_tool_call_items_skipped(self):
        result = _run_result(new_items=[
            _message_item("Hello"),
            SimpleNamespace(type="reasoning_item", raw_item=None),
        ])
        er = from_openai_agents(result)
        assert er.tool_calls == []

    def test_empty_new_items(self):
        result = _run_result(new_items=[])
        er = from_openai_agents(result)
        assert er.tool_calls == []

    def test_none_new_items(self):
        result = _run_result(new_items=None)
        er = from_openai_agents(result)
        assert er.tool_calls == []

    def test_dict_raw_item(self):
        item = SimpleNamespace(
            type="tool_call_item",
            raw_item={"name": "mcp_tool", "arguments": '{"x": 1}'},
        )
        result = _run_result(new_items=[item])
        er = from_openai_agents(result)
        assert er.tool_calls[0] == ToolCall("mcp_tool", {"x": 1})

    def test_full_round_trip(self):
        result = _run_result(
            new_items=[
                _tool_call_item("list_files", {"dir": "/tmp"}),
                _message_item("Found files"),
                _tool_call_item("read_file", {"path": "/tmp/a.txt"}),
            ],
            final_output="The file contains hello world.",
            usage=_usage(input_tokens=300, output_tokens=150),
        )
        er = from_openai_agents(result)
        assert len(er.tool_calls) == 2
        assert er.tool_calls[0].name == "list_files"
        assert er.tool_calls[1].name == "read_file"
        assert er.token_usage == TokenUsage(input_tokens=300, output_tokens=150)
        assert er.final_output == "The file contains hello world."
