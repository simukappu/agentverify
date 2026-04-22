"""Tests for agentverify._step_builder helpers."""

from __future__ import annotations

from agentverify._step_builder import (
    aggregate_token_usage,
    final_output_from_steps,
    parse_tool_call_arguments,
    tool_calls_from_response,
)
from agentverify.models import Step, TokenUsage


class TestParseToolCallArguments:
    def test_dict_returned_as_is(self):
        assert parse_tool_call_arguments({"x": 1}) == {"x": 1}

    def test_json_string_parsed(self):
        assert parse_tool_call_arguments('{"y": 2}') == {"y": 2}

    def test_invalid_json_returns_empty(self):
        assert parse_tool_call_arguments("{not json") == {}

    def test_json_non_dict_returns_empty(self):
        """JSON that parses to a non-dict (e.g. list or scalar) returns empty."""
        assert parse_tool_call_arguments("[1, 2, 3]") == {}
        assert parse_tool_call_arguments("42") == {}

    def test_non_string_non_dict_returns_empty(self):
        assert parse_tool_call_arguments(None) == {}
        assert parse_tool_call_arguments(42) == {}


class TestToolCallsFromResponse:
    def test_none_returns_empty(self):
        assert tool_calls_from_response(None) == []

    def test_empty_list(self):
        assert tool_calls_from_response([]) == []

    def test_skips_entries_without_name(self):
        """Entries missing a 'name' key are silently skipped."""
        result = tool_calls_from_response([
            {"name": "a", "arguments": {}},
            {"arguments": {"x": 1}},  # no name — skipped
            {"name": "b"},
        ])
        assert [tc.name for tc in result] == ["a", "b"]

    def test_skips_non_dict_entries(self):
        result = tool_calls_from_response([
            {"name": "a"},
            "not-a-dict",
        ])
        assert [tc.name for tc in result] == ["a"]


class TestAggregateTokenUsage:
    def test_returns_none_when_no_step_has_usage(self):
        steps = [Step(index=0, source="llm"), Step(index=1, source="llm")]
        assert aggregate_token_usage(steps) is None

    def test_sums_across_steps(self):
        steps = [
            Step(index=0, source="llm", token_usage=TokenUsage(input_tokens=5, output_tokens=3)),
            Step(index=1, source="llm", token_usage=TokenUsage(input_tokens=7, output_tokens=2)),
        ]
        total = aggregate_token_usage(steps)
        assert total.input_tokens == 12
        assert total.output_tokens == 5


class TestFinalOutputFromSteps:
    def test_last_non_none_output(self):
        steps = [
            Step(index=0, source="llm", output="first"),
            Step(index=1, source="llm", output=None),
            Step(index=2, source="llm", output="last"),
        ]
        assert final_output_from_steps(steps) == "last"

    def test_all_none(self):
        steps = [Step(index=0, source="llm"), Step(index=1, source="llm")]
        assert final_output_from_steps(steps) is None
