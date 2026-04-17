"""Tests for the LangGraph framework adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentverify.frameworks.langgraph import from_langgraph
from agentverify.models import ExecutionResult, TokenUsage, ToolCall


def _ai_msg(
    content: str = "",
    tool_calls: list[dict] | None = None,
    usage_metadata: dict | None = None,
) -> SimpleNamespace:
    """Build a fake LangGraph AIMessage-like object."""
    return SimpleNamespace(
        type="ai",
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata=usage_metadata,
    )


def _human_msg(content: str = "") -> SimpleNamespace:
    return SimpleNamespace(type="human", content=content)


def _tool_msg(content: str = "", name: str = "") -> SimpleNamespace:
    return SimpleNamespace(type="tool", content=content, name=name)


class TestFromLanggraph:
    """Tests for from_langgraph()."""

    def test_basic_tool_calls(self):
        result = {
            "messages": [
                _human_msg("What's the weather?"),
                _ai_msg(tool_calls=[
                    {"name": "get_weather", "args": {"city": "Tokyo"}, "id": "call_1"},
                ]),
                _tool_msg("Sunny, 22°C", name="get_weather"),
                _ai_msg(content="The weather in Tokyo is sunny, 22°C."),
            ]
        }
        er = from_langgraph(result)
        assert len(er.tool_calls) == 1
        assert er.tool_calls[0] == ToolCall("get_weather", {"city": "Tokyo"})

    def test_multiple_tool_calls(self):
        result = {
            "messages": [
                _human_msg("Search and summarize"),
                _ai_msg(tool_calls=[
                    {"name": "search", "args": {"q": "AI"}, "id": "c1"},
                    {"name": "summarize", "args": {"text": "..."}, "id": "c2"},
                ]),
                _tool_msg("results"),
                _ai_msg(content="Here is the summary."),
            ]
        }
        er = from_langgraph(result)
        assert len(er.tool_calls) == 2
        assert er.tool_calls[0].name == "search"
        assert er.tool_calls[1].name == "summarize"

    def test_token_usage(self):
        result = {
            "messages": [
                _ai_msg(
                    content="Hello",
                    usage_metadata={"input_tokens": 100, "output_tokens": 50},
                ),
                _ai_msg(
                    content="Done",
                    usage_metadata={"input_tokens": 80, "output_tokens": 40},
                ),
            ]
        }
        er = from_langgraph(result)
        assert er.token_usage == TokenUsage(input_tokens=180, output_tokens=90)

    def test_token_usage_none_when_no_metadata(self):
        result = {"messages": [_ai_msg(content="Hello")]}
        er = from_langgraph(result)
        assert er.token_usage is None

    def test_final_output(self):
        result = {
            "messages": [
                _ai_msg(tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
                _tool_msg("ok"),
                _ai_msg(content="Final answer."),
            ]
        }
        er = from_langgraph(result)
        assert er.final_output == "Final answer."

    def test_final_output_none_when_all_ai_have_tool_calls(self):
        result = {
            "messages": [
                _ai_msg(tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            ]
        }
        er = from_langgraph(result)
        assert er.final_output is None

    def test_final_output_none_when_empty_content(self):
        result = {"messages": [_ai_msg(content="")]}
        er = from_langgraph(result)
        assert er.final_output is None

    def test_empty_messages(self):
        result = {"messages": []}
        er = from_langgraph(result)
        assert er.tool_calls == []
        assert er.token_usage is None
        assert er.final_output is None

    def test_no_messages_key(self):
        result = {}
        er = from_langgraph(result)
        assert er.tool_calls == []

    def test_non_ai_messages_skipped(self):
        result = {
            "messages": [_human_msg("Hi"), _tool_msg("result")]
        }
        er = from_langgraph(result)
        assert er.tool_calls == []
        assert er.final_output is None

    def test_non_dict_args_defaults_to_empty(self):
        result = {
            "messages": [
                _ai_msg(tool_calls=[{"name": "t", "args": "not a dict", "id": "1"}]),
            ]
        }
        er = from_langgraph(result)
        assert er.tool_calls[0] == ToolCall("t", {})

    def test_full_round_trip(self):
        result = {
            "messages": [
                _human_msg("Triage issues"),
                _ai_msg(
                    tool_calls=[{"name": "list_issues", "args": {"repo": "o/r"}, "id": "c1"}],
                    usage_metadata={"input_tokens": 100, "output_tokens": 30},
                ),
                _tool_msg("[issue1, issue2]"),
                _ai_msg(
                    tool_calls=[{"name": "get_issue", "args": {"repo": "o/r", "number": 1}, "id": "c2"}],
                    usage_metadata={"input_tokens": 150, "output_tokens": 40},
                ),
                _tool_msg("issue detail"),
                _ai_msg(
                    content="Issue #1 needs the bug label.",
                    usage_metadata={"input_tokens": 120, "output_tokens": 50},
                ),
            ]
        }
        er = from_langgraph(result)
        assert len(er.tool_calls) == 2
        assert er.tool_calls[0].name == "list_issues"
        assert er.tool_calls[1].name == "get_issue"
        assert er.token_usage == TokenUsage(input_tokens=370, output_tokens=120)
        assert er.final_output == "Issue #1 needs the bug label."
