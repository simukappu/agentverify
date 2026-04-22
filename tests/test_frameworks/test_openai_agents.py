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



# ---------------------------------------------------------------------------
# Step boundary handling (v0.3.0)
# ---------------------------------------------------------------------------


def _tool_output_item(output) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_call_output_item",
        output=output,
        raw_item=SimpleNamespace(output=output),
    )


class TestOpenAIAgentsSteps:
    def test_tool_call_then_output_then_tool_call_produces_two_steps(self):
        """A tool call, its output, and another tool call produce two steps."""
        result = SimpleNamespace(
            new_items=[
                _tool_call_item("search", {"q": "Tokyo"}),
                _tool_output_item("Tokyo info"),
                _tool_call_item("get_weather", {"city": "Tokyo"}),
                _tool_output_item("sunny"),
            ],
            final_output="done",
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        assert len(er.steps) == 2
        assert [tc.name for tc in er.steps[0].tool_calls] == ["search"]
        assert er.steps[0].tool_results == ["Tokyo info"]
        assert [tc.name for tc in er.steps[1].tool_calls] == ["get_weather"]
        assert er.steps[1].tool_results == ["sunny"]

    def test_message_output_closes_step_with_text(self):
        """A message_output_item with text emits a step containing that text."""
        result = SimpleNamespace(
            new_items=[
                _tool_call_item("search", {"q": "x"}),
                _tool_output_item("r"),
                _message_item("final answer"),
            ],
            final_output="final answer",
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        # One step with the tool call + tool result + text output.
        assert len(er.steps) == 1
        assert er.steps[0].output == "final answer"
        assert er.steps[0].tool_results == ["r"]

    def test_tool_output_without_raw_item_output(self):
        """tool_call_output_item whose raw_item has no output falls back to item.output."""
        item = SimpleNamespace(
            type="tool_call_output_item",
            output="from-item",
            raw_item=SimpleNamespace(),
        )
        result = SimpleNamespace(
            new_items=[_tool_call_item("search", {}), item],
            final_output="x",
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        assert er.steps[0].tool_results == ["from-item"]


class TestOpenAIAgentsExtractText:
    def test_extract_text_from_dict_raw_with_string_content(self):
        """raw_item as dict with content=str returns the string."""
        from agentverify.frameworks.openai_agents import _extract_openai_agents_text

        assert _extract_openai_agents_text({"content": "hello"}) == "hello"

    def test_extract_text_from_none(self):
        from agentverify.frameworks.openai_agents import _extract_openai_agents_text

        assert _extract_openai_agents_text(None) is None

    def test_extract_text_from_list_of_dicts(self):
        from agentverify.frameworks.openai_agents import _extract_openai_agents_text

        raw = SimpleNamespace(content=[{"text": "world"}])
        assert _extract_openai_agents_text(raw) == "world"

    def test_extract_text_from_list_without_text(self):
        from agentverify.frameworks.openai_agents import _extract_openai_agents_text

        raw = SimpleNamespace(content=[{"other": "x"}])
        assert _extract_openai_agents_text(raw) is None

    def test_tool_call_item_missing_raw_skipped(self):
        item = SimpleNamespace(type="tool_call_item", raw_item=None)
        result = SimpleNamespace(
            new_items=[item],
            final_output="x",
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        assert er.steps == []

    def test_tool_call_item_empty_name_skipped(self):
        item = SimpleNamespace(
            type="tool_call_item",
            raw_item=SimpleNamespace(name="", arguments="{}"),
        )
        result = SimpleNamespace(
            new_items=[item],
            final_output="x",
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        assert er.steps == []



class TestExtractTextFallthrough:
    def test_content_not_string_not_list(self):
        """_extract_openai_agents_text returns None for non-str/non-list content."""
        from agentverify.frameworks.openai_agents import _extract_openai_agents_text

        # content is None
        raw = SimpleNamespace(content=None)
        assert _extract_openai_agents_text(raw) is None
        # content is an int (unexpected type)
        raw2 = SimpleNamespace(content=42)
        assert _extract_openai_agents_text(raw2) is None



class TestToolCallItemEmptyName:
    def test_dict_raw_item_empty_name_skipped(self):
        """dict raw_item with empty name is skipped."""
        item = SimpleNamespace(
            type="tool_call_item",
            raw_item={"name": "", "arguments": "{}"},
        )
        result = SimpleNamespace(
            new_items=[item],
            final_output="x",
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        assert er.steps == []


    def test_output_none_with_raw_item_output_dict(self):
        """item.output=None, raw_item is dict with 'output' key."""
        item = SimpleNamespace(
            type="tool_call_output_item",
            output=None,
            raw_item={"output": "from-raw"},
        )
        result = SimpleNamespace(
            new_items=[_tool_call_item("search", {}), item],
            final_output="x",
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        assert er.steps[0].tool_results == ["from-raw"]

    def test_output_none_with_raw_item_output_ns(self):
        """item.output=None, raw_item is namespace with 'output' attribute."""
        item = SimpleNamespace(
            type="tool_call_output_item",
            output=None,
            raw_item=SimpleNamespace(output="from-ns"),
        )
        result = SimpleNamespace(
            new_items=[_tool_call_item("search", {}), item],
            final_output="x",
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        assert er.steps[0].tool_results == ["from-ns"]



class TestMessageOutputMultiple:
    def test_message_output_followed_by_more_items(self):
        """A message_output_item followed by more items iterates back."""
        result = SimpleNamespace(
            new_items=[
                _tool_call_item("search", {}),
                _message_item("mid-answer"),
                _tool_call_item("other", {}),
            ],
            final_output="final",
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        # After message_output_item, second tool_call_item starts a new step.
        assert len(er.steps) >= 1



class TestEmptyMessageOutput:
    def test_empty_message_output_item_skipped(self):
        """A message_output_item with no text + no prior tool activity skips."""
        item = SimpleNamespace(
            type="message_output_item",
            raw_item=SimpleNamespace(content=None),
        )
        result = SimpleNamespace(
            new_items=[item],
            final_output=None,
            context_wrapper=None,
        )
        er = from_openai_agents(result)
        assert er.steps == []
