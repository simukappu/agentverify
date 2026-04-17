"""Tests for the LangChain framework adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentverify.frameworks.langchain import from_langchain
from agentverify.models import ExecutionResult, TokenUsage, ToolCall


def _make_action(tool: str, tool_input: dict | str = None) -> SimpleNamespace:
    """Build a fake LangChain AgentAction-like object."""
    return SimpleNamespace(
        tool=tool,
        tool_input=tool_input if tool_input is not None else {},
    )


def _make_ai_message(
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> SimpleNamespace:
    """Build a fake LangChain AIMessage-like object with usage_metadata."""
    return SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    )


class TestFromLangchain:
    """Tests for from_langchain()."""

    def test_basic_tool_calls(self):
        """Extract tool calls from intermediate_steps."""
        result = {
            "output": "Done.",
            "intermediate_steps": [
                (_make_action("list_issues", {"repo": "owner/repo"}), "obs1"),
                (_make_action("get_issue", {"repo": "owner/repo", "issue_number": 1}), "obs2"),
            ],
        }

        er = from_langchain(result)

        assert isinstance(er, ExecutionResult)
        assert len(er.tool_calls) == 2
        assert er.tool_calls[0] == ToolCall("list_issues", {"repo": "owner/repo"})
        assert er.tool_calls[1] == ToolCall(
            "get_issue", {"repo": "owner/repo", "issue_number": 1}
        )

    def test_final_output(self):
        """Extract final output from result["output"]."""
        result = {"output": "The issues have been triaged.", "intermediate_steps": []}

        er = from_langchain(result)

        assert er.final_output == "The issues have been triaged."

    def test_final_output_none_when_missing(self):
        """final_output is None when output key is absent."""
        result = {"intermediate_steps": []}

        er = from_langchain(result)

        assert er.final_output is None

    def test_token_usage_from_messages(self):
        """Aggregate token usage from AIMessage.usage_metadata."""
        messages = [
            _make_ai_message(input_tokens=100, output_tokens=50),
            SimpleNamespace(),  # HumanMessage — no usage_metadata
            _make_ai_message(input_tokens=80, output_tokens=40),
        ]
        result = {"output": "Done.", "intermediate_steps": []}

        er = from_langchain(result, messages=messages)

        assert er.token_usage is not None
        assert er.token_usage.input_tokens == 180
        assert er.token_usage.output_tokens == 90
        assert er.token_usage.total_tokens == 270

    def test_token_usage_none_when_no_messages(self):
        """token_usage is None when messages is not provided."""
        result = {"output": "Done.", "intermediate_steps": []}

        er = from_langchain(result)

        assert er.token_usage is None

    def test_token_usage_none_when_all_zero(self):
        """token_usage is None when all token counts are zero."""
        messages = [
            _make_ai_message(input_tokens=0, output_tokens=0),
        ]
        result = {"output": "Done.", "intermediate_steps": []}

        er = from_langchain(result, messages=messages)

        assert er.token_usage is None

    def test_non_dict_tool_input_defaults_to_empty(self):
        """Non-dict tool_input is replaced with empty dict."""
        result = {
            "output": "Done.",
            "intermediate_steps": [
                (_make_action("search", "raw query string"), "obs"),
            ],
        }

        er = from_langchain(result)

        assert len(er.tool_calls) == 1
        assert er.tool_calls[0] == ToolCall("search", {})

    def test_empty_intermediate_steps(self):
        """No tool calls when intermediate_steps is empty."""
        result = {"output": "No tools needed.", "intermediate_steps": []}

        er = from_langchain(result)

        assert er.tool_calls == []

    def test_no_intermediate_steps_key(self):
        """No tool calls when intermediate_steps key is absent."""
        result = {"output": "Direct answer."}

        er = from_langchain(result)

        assert er.tool_calls == []

    def test_messages_without_usage_metadata(self):
        """Messages without usage_metadata are skipped."""
        messages = [
            SimpleNamespace(),  # No usage_metadata
            SimpleNamespace(usage_metadata=None),  # Falsy usage_metadata
        ]
        result = {"output": "Done.", "intermediate_steps": []}

        er = from_langchain(result, messages=messages)

        assert er.token_usage is None

    def test_full_round_trip(self):
        """Full extraction with tool calls, token usage, and final output."""
        result = {
            "output": "Issue #1 needs the 'bug' label.",
            "intermediate_steps": [
                (_make_action("list_issues", {"repo": "o/r"}), "issues"),
                (_make_action("get_issue", {"repo": "o/r", "issue_number": 1}), "detail"),
                (_make_action("add_label", {"repo": "o/r", "issue_number": 1, "label": "bug"}), "ok"),
            ],
        }
        messages = [
            _make_ai_message(input_tokens=150, output_tokens=60),
            SimpleNamespace(),
            _make_ai_message(input_tokens=120, output_tokens=45),
        ]

        er = from_langchain(result, messages=messages)

        assert len(er.tool_calls) == 3
        assert er.tool_calls[0].name == "list_issues"
        assert er.tool_calls[1].name == "get_issue"
        assert er.tool_calls[2].name == "add_label"
        assert er.token_usage == TokenUsage(input_tokens=270, output_tokens=105)
        assert er.final_output == "Issue #1 needs the 'bug' label."
