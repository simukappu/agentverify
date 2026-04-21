"""Tests for agentverify.assertions — assert_tool_calls, assert_cost, assert_no_tool_call, assert_final_output, assert_all."""

import pytest

from agentverify.assertions import (
    assert_all,
    assert_cost,
    assert_final_output,
    assert_latency,
    assert_no_tool_call,
    assert_tool_calls,
)
from agentverify.errors import (
    CostBudgetError,
    FinalOutputError,
    LatencyBudgetError,
    MultipleAssertionError,
    SafetyRuleViolationError,
    ToolCallSequenceError,
)
from agentverify.matchers import ANY, OrderMode
from agentverify.models import ExecutionResult, TokenUsage, ToolCall


# ---------------------------------------------------------------------------
# assert_tool_calls — EXACT mode
# ---------------------------------------------------------------------------


class TestAssertToolCallsExact:
    def test_exact_match(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="a", arguments={"x": 1})]
        )
        assert_tool_calls(result, [ToolCall(name="a", arguments={"x": 1})])

    def test_exact_mismatch_name(self):
        result = ExecutionResult(tool_calls=[ToolCall(name="a")])
        with pytest.raises(ToolCallSequenceError) as exc_info:
            assert_tool_calls(result, [ToolCall(name="b")])
        assert exc_info.value.first_mismatch_index == 0

    def test_exact_mismatch_length_more_expected(self):
        result = ExecutionResult(tool_calls=[ToolCall(name="a")])
        with pytest.raises(ToolCallSequenceError) as exc_info:
            assert_tool_calls(
                result, [ToolCall(name="a"), ToolCall(name="b")]
            )
        assert exc_info.value.first_mismatch_index == 1

    def test_exact_mismatch_length_more_actual(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="a"), ToolCall(name="b")]
        )
        with pytest.raises(ToolCallSequenceError):
            assert_tool_calls(result, [ToolCall(name="a")])

    def test_exact_empty(self):
        result = ExecutionResult(tool_calls=[])
        assert_tool_calls(result, [])

    def test_exact_with_any_matcher(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="search", arguments={"q": "hello"})]
        )
        assert_tool_calls(
            result, [ToolCall(name="search", arguments={"q": ANY})]
        )

    def test_exact_with_partial_args(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="a", arguments={"x": 1, "y": 2})]
        )
        assert_tool_calls(
            result,
            [ToolCall(name="a", arguments={"x": 1})],
            partial_args=True,
        )

    def test_exact_partial_args_missing_key(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="a", arguments={"y": 2})]
        )
        with pytest.raises(ToolCallSequenceError):
            assert_tool_calls(
                result,
                [ToolCall(name="a", arguments={"x": 1})],
                partial_args=True,
            )

    def test_exact_partial_args_wrong_value(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="a", arguments={"x": 99})]
        )
        with pytest.raises(ToolCallSequenceError):
            assert_tool_calls(
                result,
                [ToolCall(name="a", arguments={"x": 1})],
                partial_args=True,
            )

    def test_exact_length_mismatch_first_element_differs(self):
        """Length differs AND first element also differs — mismatch at 0."""
        result = ExecutionResult(tool_calls=[ToolCall(name="x")])
        with pytest.raises(ToolCallSequenceError) as exc_info:
            assert_tool_calls(
                result, [ToolCall(name="a"), ToolCall(name="b")]
            )
        assert exc_info.value.first_mismatch_index == 0


# ---------------------------------------------------------------------------
# assert_tool_calls — IN_ORDER mode
# ---------------------------------------------------------------------------


class TestAssertToolCallsInOrder:
    def test_subsequence_match(self):
        result = ExecutionResult(
            tool_calls=[
                ToolCall(name="a"),
                ToolCall(name="b"),
                ToolCall(name="c"),
            ]
        )
        assert_tool_calls(
            result, [ToolCall(name="a"), ToolCall(name="c")], order=OrderMode.IN_ORDER
        )

    def test_subsequence_no_match(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="a"), ToolCall(name="b")]
        )
        with pytest.raises(ToolCallSequenceError):
            assert_tool_calls(
                result, [ToolCall(name="c")], order=OrderMode.IN_ORDER
            )

    def test_empty_expected(self):
        result = ExecutionResult(tool_calls=[ToolCall(name="a")])
        assert_tool_calls(result, [], order=OrderMode.IN_ORDER)

    def test_wrong_order(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="b"), ToolCall(name="a")]
        )
        with pytest.raises(ToolCallSequenceError):
            assert_tool_calls(
                result,
                [ToolCall(name="a"), ToolCall(name="b")],
                order=OrderMode.IN_ORDER,
            )


# ---------------------------------------------------------------------------
# assert_tool_calls — ANY_ORDER mode
# ---------------------------------------------------------------------------


class TestAssertToolCallsAnyOrder:
    def test_any_order_match(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="b"), ToolCall(name="a")]
        )
        assert_tool_calls(
            result,
            [ToolCall(name="a"), ToolCall(name="b")],
            order=OrderMode.ANY_ORDER,
        )

    def test_any_order_missing(self):
        result = ExecutionResult(tool_calls=[ToolCall(name="a")])
        with pytest.raises(ToolCallSequenceError):
            assert_tool_calls(
                result, [ToolCall(name="z")], order=OrderMode.ANY_ORDER
            )


# ---------------------------------------------------------------------------
# assert_cost
# ---------------------------------------------------------------------------


class TestAssertCost:
    def test_within_token_budget(self):
        result = ExecutionResult(
            token_usage=TokenUsage(input_tokens=50, output_tokens=30)
        )
        assert_cost(result, max_tokens=100)

    def test_exceeds_token_budget(self):
        result = ExecutionResult(
            token_usage=TokenUsage(input_tokens=80, output_tokens=30)
        )
        with pytest.raises(CostBudgetError) as exc_info:
            assert_cost(result, max_tokens=100)
        assert exc_info.value.exceeded_by == 10

    def test_within_cost_budget(self):
        result = ExecutionResult(total_cost_usd=0.05)
        assert_cost(result, max_cost_usd=0.10)

    def test_exceeds_cost_budget(self):
        result = ExecutionResult(total_cost_usd=0.15)
        with pytest.raises(CostBudgetError):
            assert_cost(result, max_cost_usd=0.10)

    def test_no_token_usage_no_error(self):
        result = ExecutionResult()
        assert_cost(result, max_tokens=100)

    def test_no_cost_no_error(self):
        result = ExecutionResult()
        assert_cost(result, max_cost_usd=1.0)

    def test_no_limits_no_error(self):
        result = ExecutionResult(
            token_usage=TokenUsage(input_tokens=999, output_tokens=999),
            total_cost_usd=999.0,
        )
        assert_cost(result)

    def test_strict_mode_missing_token_usage(self):
        result = ExecutionResult()
        with pytest.raises(CostBudgetError, match="strict mode"):
            assert_cost(result, max_tokens=100, strict=True)

    def test_strict_mode_missing_cost(self):
        result = ExecutionResult()
        with pytest.raises(CostBudgetError, match="strict mode"):
            assert_cost(result, max_cost_usd=1.0, strict=True)

    def test_strict_mode_with_data_present(self):
        result = ExecutionResult(
            token_usage=TokenUsage(input_tokens=50, output_tokens=30),
            total_cost_usd=0.05,
        )
        assert_cost(result, max_tokens=100, max_cost_usd=0.10, strict=True)


# ---------------------------------------------------------------------------
# assert_latency
# ---------------------------------------------------------------------------


class TestAssertLatency:
    def test_within_budget(self):
        result = ExecutionResult(duration_ms=1500.0)
        assert_latency(result, max_ms=3000)

    def test_exceeds_budget(self):
        result = ExecutionResult(duration_ms=3500.0)
        with pytest.raises(LatencyBudgetError) as exc_info:
            assert_latency(result, max_ms=3000)
        assert exc_info.value.exceeded_by_ms == 500.0
        assert exc_info.value.actual_ms == 3500.0
        assert exc_info.value.limit_ms == 3000

    def test_equal_to_budget_passes(self):
        result = ExecutionResult(duration_ms=3000.0)
        assert_latency(result, max_ms=3000)

    def test_no_duration_silent_pass(self):
        result = ExecutionResult()
        assert_latency(result, max_ms=3000)

    def test_strict_mode_missing_duration(self):
        result = ExecutionResult()
        with pytest.raises(LatencyBudgetError, match="strict mode"):
            assert_latency(result, max_ms=3000, strict=True)

    def test_strict_mode_with_data_present(self):
        result = ExecutionResult(duration_ms=1500.0)
        assert_latency(result, max_ms=3000, strict=True)


# ---------------------------------------------------------------------------
# assert_no_tool_call
# ---------------------------------------------------------------------------


class TestAssertNoToolCall:
    def test_no_forbidden_calls(self):
        result = ExecutionResult(tool_calls=[ToolCall(name="safe")])
        assert_no_tool_call(result, forbidden_tools=["dangerous"])

    def test_forbidden_call_detected(self):
        result = ExecutionResult(tool_calls=[ToolCall(name="rm_rf")])
        with pytest.raises(SafetyRuleViolationError) as exc_info:
            assert_no_tool_call(result, forbidden_tools=["rm_rf"])
        assert len(exc_info.value.violations) == 1
        assert exc_info.value.violations[0]["tool_name"] == "rm_rf"
        assert exc_info.value.violations[0]["position"] == 0

    def test_multiple_forbidden_calls(self):
        result = ExecutionResult(
            tool_calls=[
                ToolCall(name="rm_rf"),
                ToolCall(name="safe"),
                ToolCall(name="drop_db"),
            ]
        )
        with pytest.raises(SafetyRuleViolationError) as exc_info:
            assert_no_tool_call(result, forbidden_tools=["rm_rf", "drop_db"])
        assert len(exc_info.value.violations) == 2

    def test_empty_forbidden_list(self):
        result = ExecutionResult(tool_calls=[ToolCall(name="anything")])
        assert_no_tool_call(result, forbidden_tools=[])


# ---------------------------------------------------------------------------
# assert_final_output
# ---------------------------------------------------------------------------


class TestAssertFinalOutput:
    def test_contains_match(self):
        result = ExecutionResult(final_output="The weather in Tokyo is sunny.")
        assert_final_output(result, contains="Tokyo")

    def test_contains_no_match(self):
        result = ExecutionResult(final_output="The weather in Tokyo is sunny.")
        with pytest.raises(FinalOutputError, match="does not contain"):
            assert_final_output(result, contains="Berlin")

    def test_equals_match(self):
        result = ExecutionResult(final_output="hello")
        assert_final_output(result, equals="hello")

    def test_equals_no_match(self):
        result = ExecutionResult(final_output="hello")
        with pytest.raises(FinalOutputError, match="does not equal"):
            assert_final_output(result, equals="world")

    def test_matches_regex(self):
        result = ExecutionResult(final_output="Temperature: 22°C")
        assert_final_output(result, matches=r"\d+°C")

    def test_matches_regex_no_match(self):
        result = ExecutionResult(final_output="No temperature info")
        with pytest.raises(FinalOutputError, match="does not match pattern"):
            assert_final_output(result, matches=r"\d+°C")

    def test_final_output_is_none(self):
        result = ExecutionResult(final_output=None)
        with pytest.raises(FinalOutputError, match="final_output is None"):
            assert_final_output(result, contains="anything")

    def test_no_criteria_raises_value_error(self):
        result = ExecutionResult(final_output="hello")
        with pytest.raises(ValueError, match="at least one of"):
            assert_final_output(result)

    def test_multiple_criteria_all_pass(self):
        result = ExecutionResult(final_output="Tokyo 22°C")
        assert_final_output(result, contains="Tokyo", matches=r"\d+°C")

    def test_multiple_criteria_one_fails(self):
        result = ExecutionResult(final_output="Tokyo sunny")
        with pytest.raises(FinalOutputError):
            assert_final_output(result, contains="Tokyo", matches=r"\d+°C")


# ---------------------------------------------------------------------------
# assert_all
# ---------------------------------------------------------------------------


class TestAssertAll:
    def test_all_pass(self):
        result = ExecutionResult(tool_calls=[ToolCall(name="a")])
        assert_all(
            result,
            lambda r: assert_tool_calls(r, [ToolCall(name="a")]),
            lambda r: assert_no_tool_call(r, forbidden_tools=["bad"]),
        )

    def test_collects_multiple_failures(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="bad")],
            token_usage=TokenUsage(input_tokens=999, output_tokens=999),
        )
        with pytest.raises(MultipleAssertionError) as exc_info:
            assert_all(
                result,
                lambda r: assert_no_tool_call(r, forbidden_tools=["bad"]),
                lambda r: assert_cost(r, max_tokens=10),
            )
        assert len(exc_info.value.errors) == 2

    def test_no_assertions(self):
        result = ExecutionResult()
        assert_all(result)  # should not raise
