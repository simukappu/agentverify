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


# ---------------------------------------------------------------------------
# classify_tool_result_error + build_tool_results_meta
# ---------------------------------------------------------------------------

from agentverify._step_builder import (
    build_tool_results_meta,
    classify_tool_result_error,
)


class TestClassifyToolResultError:
    def test_is_error_true(self):
        assert classify_tool_result_error({"is_error": True}) is True

    def test_is_error_false(self):
        assert classify_tool_result_error({"is_error": False}) is False

    def test_status_error(self):
        assert classify_tool_result_error({"status": "error"}) is True

    def test_status_success(self):
        assert classify_tool_result_error({"status": "success"}) is False

    def test_status_ok(self):
        assert classify_tool_result_error({"status": "ok"}) is False

    def test_nonempty_error_key(self):
        assert classify_tool_result_error({"error": "boom"}) is True

    def test_empty_error_key_is_non_error(self):
        assert classify_tool_result_error({"error": None}) is False
        assert classify_tool_result_error({"error": ""}) is False

    def test_unknown_dict(self):
        assert classify_tool_result_error({"data": 1}) is None

    def test_plain_string_unknown(self):
        assert classify_tool_result_error("sunny, 22C") is None

    def test_json_object_string_classified(self):
        assert classify_tool_result_error('{"is_error": true}') is True
        assert classify_tool_result_error('{"status": "error"}') is True

    def test_malformed_json_object_string_is_unknown(self):
        # starts with { but is not valid JSON
        assert classify_tool_result_error('{not valid json') is None

    def test_json_non_object_string_is_unknown(self):
        # valid JSON but not starting with { → not parsed
        assert classify_tool_result_error("[1, 2, 3]") is None

    def test_empty_string_unknown(self):
        assert classify_tool_result_error("") is None

    def test_non_str_non_dict_unknown(self):
        assert classify_tool_result_error(42) is None
        assert classify_tool_result_error(None) is None
        assert classify_tool_result_error([1, 2]) is None


class TestBuildToolResultsMeta:
    def test_empty_returns_none(self):
        assert build_tool_results_meta([]) is None

    def test_classifies_each_result(self):
        meta = build_tool_results_meta([{"is_error": True}, "plain", {"status": "success"}])
        assert meta == [{"is_error": True}, {}, {"is_error": False}]

    def test_explicit_overrides_classifier(self):
        meta = build_tool_results_meta(["plain", "plain"], explicit=[True, None])
        # first uses explicit True; second falls back to classifier → unknown
        assert meta == [{"is_error": True}, {}]

    def test_explicit_shorter_than_results(self):
        meta = build_tool_results_meta(["a", "b"], explicit=[True])
        assert meta == [{"is_error": True}, {}]
