"""Tests for agentverify.errors — all custom exception classes."""

import pytest

from agentverify.errors import (
    AgentVerifyError,
    CassetteMissingRequestError,
    CostBudgetError,
    FinalOutputError,
    LatencyBudgetError,
    MultipleAssertionError,
    SafetyRuleViolationError,
    ToolCallSequenceError,
)
from agentverify.models import ToolCall


class TestAgentVerifyError:
    def test_inherits_assertion_error(self):
        assert issubclass(AgentVerifyError, AssertionError)

    def test_can_raise_and_catch(self):
        with pytest.raises(AgentVerifyError):
            raise AgentVerifyError("test")


class TestToolCallSequenceError:
    def test_message_with_args(self):
        expected = [ToolCall(name="search", arguments={"q": "hello"})]
        actual = [ToolCall(name="fetch", arguments={"url": "http://x"})]
        err = ToolCallSequenceError(expected=expected, actual=actual, first_mismatch_index=0)
        assert "Tool call sequence mismatch at index 0" in str(err)
        assert "search" in str(err)
        assert "fetch" in str(err)

    def test_message_no_args(self):
        expected = [ToolCall(name="a")]
        actual = [ToolCall(name="b")]
        err = ToolCallSequenceError(expected=expected, actual=actual, first_mismatch_index=0)
        assert "a()" in str(err)
        assert "b()" in str(err)

    def test_message_length_mismatch(self):
        expected = [ToolCall(name="a"), ToolCall(name="b")]
        actual = [ToolCall(name="a")]
        err = ToolCallSequenceError(expected=expected, actual=actual, first_mismatch_index=1)
        assert "index 1" in str(err)

    def test_attributes(self):
        expected = [ToolCall(name="a")]
        actual = [ToolCall(name="b")]
        err = ToolCallSequenceError(expected=expected, actual=actual, first_mismatch_index=0)
        assert err.expected == expected
        assert err.actual == actual
        assert err.first_mismatch_index == 0

    def test_format_tool_call_with_non_string_arg(self):
        expected = [ToolCall(name="calc", arguments={"n": 42})]
        actual = [ToolCall(name="other")]
        err = ToolCallSequenceError(expected=expected, actual=actual, first_mismatch_index=0)
        assert "n=42" in str(err)

    def test_first_mismatch_marker_on_actual(self):
        expected = [ToolCall(name="a")]
        actual = [ToolCall(name="b")]
        err = ToolCallSequenceError(expected=expected, actual=actual, first_mismatch_index=0)
        msg = str(err)
        assert "← first mismatch" in msg
        assert "← actual" in msg


class TestCostBudgetError:
    def test_token_budget_message(self):
        err = CostBudgetError(actual=150, limit=100, exceeded_by=50)
        assert "Token budget exceeded" in str(err)
        assert "150" in str(err)
        assert "100" in str(err)
        assert "50" in str(err)

    def test_cost_budget_message(self):
        err = CostBudgetError(actual=1.50, limit=1.00, exceeded_by=0.50)
        assert "Cost budget exceeded" in str(err)
        assert "$1.50" in str(err)
        assert "$1.00" in str(err)

    def test_attributes(self):
        err = CostBudgetError(actual=200, limit=100, exceeded_by=100)
        assert err.actual == 200
        assert err.limit == 100
        assert err.exceeded_by == 100

    def test_zero_limit_no_division_error(self):
        err = CostBudgetError(actual=10, limit=0, exceeded_by=10)
        assert "0" in str(err)

    def test_float_zero_limit(self):
        err = CostBudgetError(actual=0.5, limit=0.0, exceeded_by=0.5)
        assert "Cost budget exceeded" in str(err)

    def test_custom_message(self):
        err = CostBudgetError(actual=0, limit=100, exceeded_by=0, message="custom error")
        assert str(err) == "custom error"


class TestSafetyRuleViolationError:
    def test_single_violation(self):
        violations = [{"tool_name": "rm_rf", "arguments": {}, "position": 0}]
        err = SafetyRuleViolationError(violations=violations)
        assert "1 forbidden tool call detected" in str(err)
        assert "rm_rf()" in str(err)

    def test_multiple_violations(self):
        violations = [
            {"tool_name": "rm_rf", "arguments": {"path": "/"}, "position": 0},
            {"tool_name": "drop_db", "arguments": {}, "position": 1},
        ]
        err = SafetyRuleViolationError(violations=violations)
        assert "2 forbidden tool calls detected" in str(err)
        assert "rm_rf" in str(err)
        assert "drop_db" in str(err)

    def test_violation_with_string_arg(self):
        violations = [{"tool_name": "rm", "arguments": {"path": "/tmp"}, "position": 0}]
        err = SafetyRuleViolationError(violations=violations)
        assert 'path="/tmp"' in str(err)

    def test_violation_with_int_arg(self):
        violations = [{"tool_name": "kill", "arguments": {"pid": 1234}, "position": 2}]
        err = SafetyRuleViolationError(violations=violations)
        assert "pid=1234" in str(err)

    def test_attributes(self):
        violations = [{"tool_name": "x", "arguments": {}, "position": 0}]
        err = SafetyRuleViolationError(violations=violations)
        assert err.violations == violations


class TestMultipleAssertionError:
    def test_single_error(self):
        inner = AgentVerifyError("inner failure")
        err = MultipleAssertionError(errors=[inner])
        assert "1 assertion failed" in str(err)
        assert "inner failure" in str(err)

    def test_multiple_errors(self):
        e1 = AgentVerifyError("first")
        e2 = AgentVerifyError("second")
        err = MultipleAssertionError(errors=[e1, e2])
        assert "2 assertions failed" in str(err)
        assert "first" in str(err)
        assert "second" in str(err)

    def test_attributes(self):
        e1 = AgentVerifyError("a")
        err = MultipleAssertionError(errors=[e1])
        assert err.errors == [e1]


class TestCassetteMissingRequestError:
    def test_inherits_agent_verify_error(self):
        assert issubclass(CassetteMissingRequestError, AgentVerifyError)

    def test_message(self):
        err = CassetteMissingRequestError("no match found")
        assert "no match found" in str(err)


class TestLatencyBudgetError:
    def test_message(self):
        err = LatencyBudgetError(actual_ms=3500.0, limit_ms=3000.0, exceeded_by_ms=500.0)
        msg = str(err)
        assert "Latency budget exceeded" in msg
        assert "3,500.0 ms" in msg
        assert "3,000.0 ms" in msg
        assert "500.0 ms" in msg

    def test_attributes(self):
        err = LatencyBudgetError(actual_ms=100.0, limit_ms=50.0, exceeded_by_ms=50.0)
        assert err.actual_ms == 100.0
        assert err.limit_ms == 50.0
        assert err.exceeded_by_ms == 50.0

    def test_zero_limit_no_division_error(self):
        err = LatencyBudgetError(actual_ms=10.0, limit_ms=0.0, exceeded_by_ms=10.0)
        assert "Latency budget exceeded" in str(err)

    def test_custom_message(self):
        err = LatencyBudgetError(
            actual_ms=0.0, limit_ms=100.0, exceeded_by_ms=0.0, message="custom latency error"
        )
        assert str(err) == "custom latency error"


class TestFinalOutputError:
    def test_inherits_agent_verify_error(self):
        assert issubclass(FinalOutputError, AgentVerifyError)

    def test_message(self):
        err = FinalOutputError("final_output is None")
        assert "final_output is None" in str(err)


class TestCassetteRequestMismatchError:
    def test_inherits_agent_verify_error(self):
        from agentverify.errors import CassetteRequestMismatchError

        assert issubclass(CassetteRequestMismatchError, AgentVerifyError)

    def test_message_model_mismatch(self):
        from agentverify.errors import CassetteRequestMismatchError

        err = CassetteRequestMismatchError(
            index=0, field="model", recorded="gpt-4.1", actual="gpt-3.5-turbo"
        )
        msg = str(err)
        assert "interaction 0" in msg
        assert "model" in msg
        assert "gpt-4.1" in msg
        assert "gpt-3.5-turbo" in msg
        assert "re-record" in msg.lower()

    def test_message_tools_mismatch(self):
        from agentverify.errors import CassetteRequestMismatchError

        err = CassetteRequestMismatchError(
            index=2,
            field="tools",
            recorded=["calc", "search"],
            actual=["calc", "delete_user"],
        )
        msg = str(err)
        assert "interaction 2" in msg
        assert "tools" in msg

    def test_attributes(self):
        from agentverify.errors import CassetteRequestMismatchError

        err = CassetteRequestMismatchError(
            index=1, field="model", recorded="a", actual="b"
        )
        assert err.index == 1
        assert err.field == "model"
        assert err.recorded == "a"
        assert err.actual == "b"



class TestStepDependencyErrorTruncation:
    def test_truncated_when_repr_exceeds_limit(self):
        """Long repr is truncated with ellipsis."""
        from agentverify.errors import StepDependencyError

        # Long value — should trigger truncation
        long_list = ["x" * 300]
        err = StepDependencyError(
            step=1, depends_on=0, via="any",
            produced=long_list, consumed=[],
        )
        assert "…" in str(err)


class TestToolCallSequenceErrorStepContext:
    def test_step_name_included_in_message(self):
        """Error message includes step name when provided."""
        from agentverify.errors import ToolCallSequenceError

        err = ToolCallSequenceError(
            expected=[],
            actual=[],
            first_mismatch_index=0,
            step_name="plan",
        )
        assert "plan" in str(err)

    def test_step_index_only(self):
        """Error message includes step index alone when name is None."""
        from agentverify.errors import ToolCallSequenceError

        err = ToolCallSequenceError(
            expected=[],
            actual=[],
            first_mismatch_index=0,
            step_index=3,
        )
        assert "step 3" in str(err)
