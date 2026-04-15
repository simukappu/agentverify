"""Assertion engine for agentverify.

Provides assertion functions for verifying tool call sequences,
cost/token budgets, safety guardrails, and batch assertion execution.
"""

from __future__ import annotations

from typing import Callable, Optional

from agentverify.errors import (
    AgentVerifyError,
    CostBudgetError,
    FinalOutputError,
    MultipleAssertionError,
    SafetyRuleViolationError,
    ToolCallSequenceError,
)
from agentverify.matchers import OrderMode
from agentverify.models import ExecutionResult, ToolCall


def _tool_call_matches(
    expected: ToolCall,
    actual: ToolCall,
    partial_args: bool,
) -> bool:
    """Check whether two ToolCalls match.

    Names must match exactly.  Argument matching depends on *partial_args*:
    - False: arguments must be equal (ANY matcher still works via __eq__)
    - True:  only keys present in expected.arguments are checked
    """
    if expected.name != actual.name:
        return False

    if partial_args:
        for key, value in expected.arguments.items():
            if key not in actual.arguments:
                return False
            if value != actual.arguments[key]:
                return False
        return True

    return expected.arguments == actual.arguments


def assert_tool_calls(
    result: ExecutionResult,
    expected: list[ToolCall],
    order: OrderMode = OrderMode.EXACT,
    partial_args: bool = False,
) -> None:
    """Verify the tool call sequence in an ExecutionResult.

    Args:
        result: The ExecutionResult to verify.
        expected: Expected ToolCall list.
        order: Comparison mode (EXACT, IN_ORDER, ANY_ORDER).
        partial_args: When True, only keys present in expected arguments
            are checked against actual arguments.

    Raises:
        ToolCallSequenceError: When a mismatch is detected.
    """
    actual = result.tool_calls

    if order == OrderMode.EXACT:
        _assert_exact(expected, actual, partial_args)
    elif order == OrderMode.IN_ORDER:
        _assert_in_order(expected, actual, partial_args)
    elif order == OrderMode.ANY_ORDER:
        _assert_any_order(expected, actual, partial_args)
    else:  # pragma: no cover — OrderMode is a closed enum
        pass


def _assert_exact(
    expected: list[ToolCall],
    actual: list[ToolCall],
    partial_args: bool,
) -> None:
    """EXACT mode: same length and same tool calls at each position."""
    if len(expected) != len(actual):
        # first_mismatch_index is the first position that differs,
        # or min(len(expected), len(actual)) if one list is shorter.
        mismatch = min(len(expected), len(actual))
        for i in range(mismatch):
            if not _tool_call_matches(expected[i], actual[i], partial_args):
                mismatch = i
                break
        raise ToolCallSequenceError(
            expected=expected,
            actual=actual,
            first_mismatch_index=mismatch,
        )

    for i in range(len(expected)):
        if not _tool_call_matches(expected[i], actual[i], partial_args):
            raise ToolCallSequenceError(
                expected=expected,
                actual=actual,
                first_mismatch_index=i,
            )


def _assert_in_order(
    expected: list[ToolCall],
    actual: list[ToolCall],
    partial_args: bool,
) -> None:
    """IN_ORDER mode: expected must be a subsequence of actual.

    Uses a two-pointer approach: walk through actual looking for each
    expected element in order.
    """
    if not expected:
        return

    exp_idx = 0
    for act_idx in range(len(actual)):
        if exp_idx < len(expected) and _tool_call_matches(
            expected[exp_idx], actual[act_idx], partial_args
        ):
            exp_idx += 1

    if exp_idx < len(expected):
        # The first unmatched expected element is the mismatch point
        raise ToolCallSequenceError(
            expected=expected,
            actual=actual,
            first_mismatch_index=exp_idx,
        )


def _assert_any_order(
    expected: list[ToolCall],
    actual: list[ToolCall],
    partial_args: bool,
) -> None:
    """ANY_ORDER mode: all expected tool calls must exist in actual."""
    for i, exp_tc in enumerate(expected):
        found = any(
            _tool_call_matches(exp_tc, act_tc, partial_args)
            for act_tc in actual
        )
        if not found:
            raise ToolCallSequenceError(
                expected=expected,
                actual=actual,
                first_mismatch_index=i,
            )


def assert_cost(
    result: ExecutionResult,
    max_tokens: Optional[int] = None,
    max_cost_usd: Optional[float] = None,
    strict: bool = False,
) -> None:
    """Verify that cost/token consumption is within budget.

    Args:
        result: The ExecutionResult to verify.
        max_tokens: Maximum allowed total tokens.
        max_cost_usd: Maximum allowed cost in USD.
        strict: When True, raise CostBudgetError if token_usage or
            total_cost_usd is None when the corresponding limit is set.
            Default False silently passes when data is unavailable.

    Raises:
        CostBudgetError: When budget is exceeded or when strict=True
            and the required data is missing.
    """
    if max_tokens is not None:
        if result.token_usage is not None:
            actual_tokens = result.token_usage.total_tokens
            if actual_tokens > max_tokens:
                raise CostBudgetError(
                    actual=actual_tokens,
                    limit=max_tokens,
                    exceeded_by=actual_tokens - max_tokens,
                )
        elif strict:
            raise CostBudgetError(
                actual=0,
                limit=max_tokens,
                exceeded_by=0,
                message="token_usage is None but strict mode requires it",
            )

    if max_cost_usd is not None:
        if result.total_cost_usd is not None:
            if result.total_cost_usd > max_cost_usd:
                raise CostBudgetError(
                    actual=result.total_cost_usd,
                    limit=max_cost_usd,
                    exceeded_by=result.total_cost_usd - max_cost_usd,
                )
        elif strict:
            raise CostBudgetError(
                actual=0.0,
                limit=max_cost_usd,
                exceeded_by=0.0,
                message="total_cost_usd is None but strict mode requires it",
            )


def assert_no_tool_call(
    result: ExecutionResult,
    forbidden_tools: list[str],
) -> None:
    """Verify that no forbidden tools were called.

    All violations are collected and reported at once.

    Args:
        result: The ExecutionResult to verify.
        forbidden_tools: List of forbidden tool names.

    Raises:
        SafetyRuleViolationError: When forbidden tool calls are detected.
    """
    forbidden_set = set(forbidden_tools)
    violations: list[dict] = []

    for position, tc in enumerate(result.tool_calls):
        if tc.name in forbidden_set:
            violations.append(
                {
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "position": position,
                }
            )

    if violations:
        raise SafetyRuleViolationError(violations=violations)


def assert_final_output(
    result: ExecutionResult,
    contains: Optional[str] = None,
    equals: Optional[str] = None,
    matches: Optional[str] = None,
) -> None:
    """Verify the agent's final text output.

    At least one of *contains*, *equals*, or *matches* must be provided.

    Args:
        result: The ExecutionResult to verify.
        contains: Substring that must appear in final_output.
        equals: Exact string that final_output must equal.
        matches: Regex pattern that final_output must match.

    Raises:
        FinalOutputError: When the final output does not meet expectations.
    """
    if contains is None and equals is None and matches is None:
        raise ValueError(
            "assert_final_output requires at least one of: contains, equals, matches"
        )

    if result.final_output is None:
        raise FinalOutputError("final_output is None")

    if equals is not None:
        if result.final_output != equals:
            raise FinalOutputError(
                f"final_output does not equal expected\n"
                f"\n"
                f"  Expected: {equals!r}\n"
                f"  Actual:   {result.final_output!r}"
            )

    if contains is not None:
        if contains not in result.final_output:
            raise FinalOutputError(
                f"final_output does not contain expected substring\n"
                f"\n"
                f"  Substring: {contains!r}\n"
                f"  Actual:    {result.final_output!r}"
            )

    if matches is not None:
        import re

        if not re.search(matches, result.final_output):
            raise FinalOutputError(
                f"final_output does not match pattern\n"
                f"\n"
                f"  Pattern: {matches!r}\n"
                f"  Actual:  {result.final_output!r}"
            )


def assert_all(
    result: ExecutionResult,
    *assertions: Callable[[ExecutionResult], None],
) -> None:
    """Execute multiple assertions, collecting all failures.

    Each assertion is a callable that takes an ExecutionResult and
    returns None or raises an AgentVerifyError.  All assertions are
    executed even if some fail.

    Args:
        result: The ExecutionResult to verify.
        *assertions: Callable assertions to execute.

    Raises:
        MultipleAssertionError: When one or more assertions fail.
    """
    errors: list[AgentVerifyError] = []

    for assertion in assertions:
        try:
            assertion(result)
        except AgentVerifyError as exc:
            errors.append(exc)

    if errors:
        raise MultipleAssertionError(errors=errors)
