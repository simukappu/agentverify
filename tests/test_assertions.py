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
from agentverify.matchers import ANY, MATCHES, OrderMode
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

    def test_exact_with_matches_regex(self):
        """MATCHES matcher verifies regex substring match on a string arg."""
        result = ExecutionResult(
            tool_calls=[
                ToolCall(name="http_request", arguments={"url": "https://api.weather.gov/points/47,122"}),
                ToolCall(name="http_request", arguments={"url": "https://api.weather.gov/forecast/SEW"}),
            ]
        )
        assert_tool_calls(
            result,
            [
                ToolCall(name="http_request", arguments={"url": MATCHES(r"/points/")}),
                ToolCall(name="http_request", arguments={"url": MATCHES(r"/forecast")}),
            ],
        )

    def test_exact_with_matches_regex_fails(self):
        result = ExecutionResult(
            tool_calls=[ToolCall(name="http_request", arguments={"url": "https://example.com/"})]
        )
        with pytest.raises(ToolCallSequenceError):
            assert_tool_calls(
                result,
                [ToolCall(name="http_request", arguments={"url": MATCHES(r"/points/")})],
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



# ---------------------------------------------------------------------------
# assert_step
# ---------------------------------------------------------------------------


from agentverify import (
    Step,
    assert_step,
    assert_step_output,
    assert_step_uses_result_from,
)
from agentverify.errors import (
    StepDependencyError,
    StepIndexError,
    StepNameAmbiguousError,
    StepNameNotFoundError,
    StepOutputError,
)


class TestAssertStep:
    def _make(self, steps):
        return ExecutionResult(steps=steps)

    def test_by_index_single_tool_match(self):
        result = self._make([
            Step(index=0, source="llm", tool_calls=[ToolCall("search", {"q": "a"})]),
        ])
        assert_step(result, step=0, expected_tool=ToolCall("search", {"q": "a"}))

    def test_by_name_match(self):
        result = self._make([
            Step(index=0, name="lookup", source="probe", tool_calls=[ToolCall("search", {"q": "a"})]),
        ])
        assert_step(result, name="lookup", expected_tool=ToolCall("search", {"q": "a"}))

    def test_expected_tools_list(self):
        result = self._make([
            Step(index=0, source="llm", tool_calls=[
                ToolCall("a"), ToolCall("b"),
            ]),
        ])
        assert_step(result, step=0, expected_tools=[ToolCall("a"), ToolCall("b")])

    def test_empty_expected_tools_for_pure_compute_step(self):
        result = self._make([
            Step(index=0, name="cache_hit", source="probe", tool_calls=[]),
        ])
        assert_step(result, name="cache_hit", expected_tools=[])

    def test_step_index_out_of_range(self):
        result = self._make([Step(index=0, source="llm")])
        with pytest.raises(StepIndexError):
            assert_step(result, step=5, expected_tools=[])

    def test_step_name_not_found(self):
        result = self._make([Step(index=0, name="foo", source="probe")])
        with pytest.raises(StepNameNotFoundError):
            assert_step(result, name="bar", expected_tools=[])

    def test_step_name_ambiguous(self):
        result = self._make([
            Step(index=0, name="foo", source="probe"),
            Step(index=1, name="foo", source="probe"),
        ])
        with pytest.raises(StepNameAmbiguousError):
            assert_step(result, name="foo", expected_tools=[])

    def test_mismatch_raises_tool_call_sequence_error_with_step_context(self):
        result = self._make([
            Step(index=0, name="call", source="probe", tool_calls=[ToolCall("a")]),
        ])
        with pytest.raises(ToolCallSequenceError) as exc_info:
            assert_step(result, name="call", expected_tool=ToolCall("b"))
        assert exc_info.value.step_index == 0
        assert exc_info.value.step_name == "call"

    def test_requires_exactly_one_specifier(self):
        result = self._make([Step(index=0, source="llm")])
        with pytest.raises(ValueError):
            assert_step(result, step=0, name="x", expected_tools=[])
        with pytest.raises(ValueError):
            assert_step(result, expected_tools=[])

    def test_requires_exactly_one_expected(self):
        result = self._make([Step(index=0, source="llm")])
        with pytest.raises(ValueError):
            assert_step(result, step=0)
        with pytest.raises(ValueError):
            assert_step(
                result, step=0,
                expected_tool=ToolCall("a"),
                expected_tools=[ToolCall("a")],
            )

    def test_in_order_subsequence(self):
        result = self._make([
            Step(index=0, source="llm", tool_calls=[
                ToolCall("a"), ToolCall("b"), ToolCall("c"),
            ]),
        ])
        assert_step(
            result, step=0,
            expected_tools=[ToolCall("a"), ToolCall("c")],
            order=OrderMode.IN_ORDER,
        )

    def test_any_order(self):
        result = self._make([
            Step(index=0, source="llm", tool_calls=[
                ToolCall("b"), ToolCall("a"),
            ]),
        ])
        assert_step(
            result, step=0,
            expected_tools=[ToolCall("a"), ToolCall("b")],
            order=OrderMode.ANY_ORDER,
        )

    def test_partial_args(self):
        result = self._make([
            Step(index=0, source="llm", tool_calls=[
                ToolCall("search", {"q": "x", "limit": 10}),
            ]),
        ])
        assert_step(
            result, step=0,
            expected_tool=ToolCall("search", {"q": "x"}),
            partial_args=True,
        )


class TestAssertStepOutput:
    def _make(self, steps):
        return ExecutionResult(steps=steps)

    def test_contains(self):
        result = self._make([
            Step(index=0, source="llm", output="The weather is sunny"),
        ])
        assert_step_output(result, step=0, contains="sunny")

    def test_equals(self):
        result = self._make([
            Step(index=0, source="llm", output="exactly this"),
        ])
        assert_step_output(result, step=0, equals="exactly this")

    def test_matches_regex(self):
        result = self._make([
            Step(index=0, source="llm", output="Temperature: 22°C"),
        ])
        assert_step_output(result, step=0, matches=r"\d+°C")

    def test_by_name(self):
        result = self._make([
            Step(index=0, name="postproc", source="probe", output="done"),
        ])
        assert_step_output(result, name="postproc", contains="done")

    def test_no_output_fails(self):
        result = self._make([Step(index=0, source="llm", output=None)])
        with pytest.raises(StepOutputError):
            assert_step_output(result, step=0, contains="x")

    def test_contains_mismatch(self):
        result = self._make([Step(index=0, source="llm", output="hello")])
        with pytest.raises(StepOutputError):
            assert_step_output(result, step=0, contains="world")

    def test_requires_at_least_one_matcher(self):
        result = self._make([Step(index=0, source="llm", output="x")])
        with pytest.raises(ValueError):
            assert_step_output(result, step=0)


class TestAssertStepUsesResultFrom:
    def test_string_flows_through_input_context(self):
        # step 0 produces "abc123"; step 1's input_context mentions it.
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", output="abc123"),
            Step(
                index=1, source="llm",
                input_context={"messages": [{"role": "user", "content": "got abc123"}]},
            ),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)

    def test_tool_result_flows_into_next_step_args(self):
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_results=[{"city_id": "TYO-42"}]),
            Step(
                index=1, source="llm",
                tool_calls=[ToolCall("get_weather", {"city": "TYO-42"})],
            ),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0, via="tool_result")

    def test_no_dependency_raises(self):
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", output="abc"),
            Step(
                index=1, source="llm",
                tool_calls=[ToolCall("x", {"q": "unrelated"})],
            ),
        ])
        with pytest.raises(StepDependencyError):
            assert_step_uses_result_from(result, step=1, depends_on=0)

    def test_via_restriction(self):
        # Produced value exists only in tool_result, not in output.
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_results=["KEY-99"], output="unrelated"),
            Step(
                index=1, source="llm",
                tool_calls=[ToolCall("fetch", {"id": "KEY-99"})],
            ),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0, via="tool_result")
        with pytest.raises(StepDependencyError):
            assert_step_uses_result_from(result, step=1, depends_on=0, via="output")

    def test_by_step_name(self):
        result = ExecutionResult(steps=[
            Step(index=0, name="plan", source="probe", output="target=X"),
            Step(
                index=1, name="exec", source="llm",
                input_context={"messages": [{"content": "target=X"}]},
            ),
        ])
        assert_step_uses_result_from(result, step="exec", depends_on="plan")

    def test_invalid_via_raises(self):
        result = ExecutionResult(steps=[Step(index=0, source="llm")])
        with pytest.raises(ValueError):
            assert_step_uses_result_from(result, step=0, depends_on=0, via="bogus")

    def test_structural_equality_for_non_strings(self):
        produced_dict = {"id": 42, "name": "x"}
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_results=[produced_dict]),
            Step(
                index=1, source="llm",
                tool_calls=[ToolCall("use", {"payload": produced_dict})],
            ),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)



# ---------------------------------------------------------------------------
# Step assertion edge cases for coverage
# ---------------------------------------------------------------------------


class TestAssertStepEdgeCases:
    def test_in_order_mismatch_in_step(self):
        """IN_ORDER mode raises when expected is not a subsequence."""
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_calls=[ToolCall("b"), ToolCall("a")]),
        ])
        with pytest.raises(ToolCallSequenceError):
            assert_step(
                result, step=0,
                expected_tools=[ToolCall("a"), ToolCall("b")],
                order=OrderMode.IN_ORDER,
            )

    def test_any_order_missing_in_step(self):
        """ANY_ORDER mode raises when an expected tool is absent."""
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_calls=[ToolCall("a")]),
        ])
        with pytest.raises(ToolCallSequenceError):
            assert_step(
                result, step=0,
                expected_tools=[ToolCall("a"), ToolCall("b")],
                order=OrderMode.ANY_ORDER,
            )

    def test_in_order_empty_expected_passes(self):
        """IN_ORDER mode with empty expected always passes."""
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_calls=[ToolCall("a")]),
        ])
        assert_step(result, step=0, expected_tools=[], order=OrderMode.IN_ORDER)

    def test_step_name_not_found_lists_available(self):
        """Error messages include available step names."""
        from agentverify.errors import StepNameNotFoundError

        result = ExecutionResult(steps=[
            Step(index=0, name="a", source="probe"),
            Step(index=1, name="b", source="probe"),
        ])
        with pytest.raises(StepNameNotFoundError) as exc_info:
            assert_step(result, name="missing", expected_tools=[])
        assert "a" in str(exc_info.value)
        assert "b" in str(exc_info.value)

    def test_step_name_not_found_empty(self):
        """Error messages when no steps have names."""
        from agentverify.errors import StepNameNotFoundError

        result = ExecutionResult(steps=[Step(index=0, source="llm")])
        with pytest.raises(StepNameNotFoundError):
            assert_step(result, name="missing", expected_tools=[])

    def test_assert_step_uses_result_from_invalid_spec_type(self):
        """Non-int, non-str step spec raises TypeError."""
        result = ExecutionResult(steps=[Step(index=0, source="llm")])
        with pytest.raises(TypeError):
            assert_step_uses_result_from(result, step=1.5, depends_on=0)

    def test_assert_step_output_equals_mismatch(self):
        """equals mismatch raises StepOutputError."""
        from agentverify.errors import StepOutputError

        result = ExecutionResult(steps=[
            Step(index=0, source="llm", output="hello"),
        ])
        with pytest.raises(StepOutputError):
            assert_step_output(result, step=0, equals="world")

    def test_assert_step_output_matches_mismatch(self):
        """matches mismatch raises StepOutputError."""
        from agentverify.errors import StepOutputError

        result = ExecutionResult(steps=[
            Step(index=0, source="llm", output="hello"),
        ])
        with pytest.raises(StepOutputError):
            assert_step_output(result, step=0, matches=r"\d+")

    def test_value_appears_in_direct_equality_non_string(self):
        """Dependency detection direct equality works for non-strings."""
        produced = {"x": 1}
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_results=[produced]),
            Step(index=1, source="llm", input_context=produced),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)

    def test_value_appears_in_produced_empty_string(self):
        """Empty strings do not count as flow."""
        from agentverify.errors import StepDependencyError

        result = ExecutionResult(steps=[
            Step(index=0, source="llm", output=""),
            Step(index=1, source="llm", input_context={"messages": ["hello"]}),
        ])
        with pytest.raises(StepDependencyError):
            assert_step_uses_result_from(result, step=1, depends_on=0)

    def test_value_appears_in_tuple_consumed(self):
        """Tuples inside consumed values are searched."""
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_results=["KEY"]),
            Step(
                index=1, source="llm",
                tool_calls=[ToolCall("x", {"payload": ("KEY", "other")})],
            ),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)

    def test_produced_output_via_restricts_tool_result(self):
        """via='output' ignores tool_results."""
        from agentverify.errors import StepDependencyError

        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_results=["only-in-tool-result"]),
            Step(
                index=1, source="llm",
                tool_calls=[ToolCall("x", {"v": "only-in-tool-result"})],
            ),
        ])
        with pytest.raises(StepDependencyError):
            assert_step_uses_result_from(result, step=1, depends_on=0, via="output")

    def test_produced_tool_call_result_flows(self):
        """tool_calls[*].result is considered a produced value."""
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_calls=[
                ToolCall("fetch", {}, result="RESULT-123"),
            ]),
            Step(
                index=1, source="llm",
                tool_calls=[ToolCall("use", {"v": "RESULT-123"})],
            ),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)

    def test_stringify_handles_non_json_serializable(self):
        """_stringify falls back to repr for non-JSON-serializable values."""
        # Using a set as consumed (not JSON serializable) with a string produced
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", output="NEEDLE"),
            # Dict contains a set which would make direct JSON fail
            Step(index=1, source="llm", input_context={"data": {"NEEDLE", "x"}}),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)



class TestStepExactLengthMismatch:
    def test_mismatch_in_middle_reports_correct_index(self):
        """Length mismatch with first few elements matching reports first diff index."""
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_calls=[
                ToolCall("a"), ToolCall("b"), ToolCall("WRONG"),
            ]),
        ])
        with pytest.raises(ToolCallSequenceError) as exc_info:
            assert_step(result, step=0, expected_tools=[
                ToolCall("a"), ToolCall("b"),
            ])
        # actual has 3, expected has 2 — mismatch_index should be 2 (end of expected)
        assert exc_info.value.first_mismatch_index == 2

    def test_mismatch_first_element_differs(self):
        """Length mismatch where the FIRST element already differs."""
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_calls=[ToolCall("x")]),
        ])
        with pytest.raises(ToolCallSequenceError) as exc_info:
            assert_step(result, step=0, expected_tools=[
                ToolCall("a"), ToolCall("b"),
            ])
        assert exc_info.value.first_mismatch_index == 0



class TestValueAppearsInBranches:
    def test_string_in_string_direct_match(self):
        """Produced string directly matches consumed string."""
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", output="Tokyo"),
            Step(index=1, source="llm", tool_calls=[
                ToolCall("x", {"q": "Tokyo weather"}),
            ]),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)

    def test_string_in_non_string_container_path(self):
        """Produced string inside consumed container (list)."""
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", output="NEEDLE"),
            Step(index=1, source="llm", tool_calls=[
                ToolCall("x", {"payload": ["NEEDLE", "other"]}),
            ]),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)

    def test_container_produced_searched_via_leaves(self):
        """Non-primitive produced values are split into leaves for matching."""
        result = ExecutionResult(steps=[
            # produced tool_result is a dict; its leaves (values) are strings/ints
            Step(index=0, source="llm", tool_results=[{"id": 42, "name": "Tokyo"}]),
            Step(index=1, source="llm", tool_calls=[
                # consumed arg mentions "Tokyo" (a leaf of the produced dict)
                ToolCall("x", {"q": "Find Tokyo hotels"}),
            ]),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)

    def test_stringify_typeerror_fallback(self):
        """_stringify falls back to repr when JSON raises.

        Triggered when the consumed dict contains a circular reference
        (ValueError) or a non-serializable non-string type (TypeError).
        """
        # Construct a dict with a circular reference — json.dumps raises
        rec: dict = {}
        rec["self"] = rec  # circular reference → raises ValueError

        # Produced string must NOT appear in any dict VALUES directly
        # (otherwise the dict values iteration finds it before _stringify
        # runs).  Put it in a KEY so the fallback path fires.
        result = ExecutionResult(steps=[
            Step(index=0, source="llm", output="KEY_HINT"),
            Step(index=1, source="llm", tool_calls=[
                ToolCall("x", {"payload": rec}),
                # Second argument is a dict where the produced string
                # appears only as a KEY — not a value, so the dict-values
                # walk misses it and we fall back to _stringify.
                ToolCall("y", {"KEY_HINT": "unrelated"}),
            ]),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)



class TestValueMatchesStringNoMatch:
    def test_string_in_string_no_match(self):
        """Produced string NOT in consumed string — direct path returns False."""
        from agentverify.assertions import _value_matches

        assert _value_matches("foo", "bar baz") is False

    def test_non_string_non_equal_returns_false(self):
        """Non-string produced with no direct equality returns False."""
        from agentverify.assertions import _value_matches

        assert _value_matches(42, "not a number") is False
        assert _value_matches(42, 43) is False



class TestStringifyFallback:
    def test_json_failure_falls_back_to_repr(self):
        """_stringify falls back to repr when JSON serialization fails."""
        from agentverify.assertions import _stringify

        # Circular reference triggers ValueError inside json.dumps
        rec: dict = {}
        rec["self"] = rec
        # default=str doesn't rescue circular references, so we fall
        # back to repr().
        result = _stringify(rec)
        assert "..." in result or "self" in result  # repr shape



class TestProducedValuesBranches:
    def test_via_output_with_none_output(self):
        """_produced_values via='output' with output=None returns empty."""
        from agentverify.assertions import _produced_values

        step_obj = Step(index=0, source="llm", output=None)
        assert _produced_values(step_obj, via="output") == []

    def test_via_tool_result_skips_none_tool_call_results(self):
        """_produced_values skips tool_call.result when it's None."""
        from agentverify.assertions import _produced_values

        step_obj = Step(
            index=0,
            source="llm",
            tool_calls=[
                ToolCall("a", {}, result=None),  # skipped
                ToolCall("b", {}, result="x"),
            ],
            tool_results=[],
        )
        assert _produced_values(step_obj, "tool_result") == ["x"]



class TestLeavesBranches:
    def test_leaves_list_tuple(self):
        """_leaves recursively yields from list/tuple."""
        from agentverify.assertions import _leaves

        assert list(_leaves([1, [2, 3], (4, 5)])) == [1, 2, 3, 4, 5]

    def test_produced_list_is_split_into_leaves(self):
        """A list produced value is split into leaves via _leaves."""
        result = ExecutionResult(steps=[
            # Produced value is a list containing primitives
            Step(index=0, source="llm", tool_results=[["Tokyo", "Osaka"]]),
            Step(index=1, source="llm", tool_calls=[
                ToolCall("search", {"city": "Tokyo"}),
            ]),
        ])
        assert_step_uses_result_from(result, step=1, depends_on=0)
