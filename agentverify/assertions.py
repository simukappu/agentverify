"""Assertion engine for agentverify.

Provides assertion functions for verifying tool call sequences,
cost/token budgets, safety guardrails, and batch assertion execution.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from agentverify.errors import (
    AgentVerifyError,
    CostBudgetError,
    FinalOutputError,
    LatencyBudgetError,
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


def assert_latency(
    result: ExecutionResult,
    max_ms: float,
    strict: bool = False,
) -> None:
    """Verify that execution latency is within the allowed threshold.

    Args:
        result: The ExecutionResult to verify.
        max_ms: Maximum allowed latency in milliseconds.
        strict: When True, raise LatencyBudgetError if duration_ms is None.
            Default False silently passes when data is unavailable.

    Raises:
        LatencyBudgetError: When latency exceeds the threshold or when
            strict=True and duration_ms is None.
    """
    if result.duration_ms is not None:
        if result.duration_ms > max_ms:
            raise LatencyBudgetError(
                actual_ms=result.duration_ms,
                limit_ms=max_ms,
                exceeded_by_ms=result.duration_ms - max_ms,
            )
    elif strict:
        raise LatencyBudgetError(
            actual_ms=0.0,
            limit_ms=max_ms,
            exceeded_by_ms=0.0,
            message="duration_ms is None but strict mode requires it",
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



# ---------------------------------------------------------------------------
# Step-level assertions
# ---------------------------------------------------------------------------


def _resolve_step(
    result: ExecutionResult,
    step: int | None,
    name: str | None,
):
    """Resolve a step specifier into a Step object.

    Exactly one of ``step`` or ``name`` must be provided.
    """
    from agentverify.errors import (
        StepIndexError,
        StepNameAmbiguousError,
        StepNameNotFoundError,
    )

    if (step is None) == (name is None):
        raise ValueError(
            "Exactly one of `step` or `name` must be provided"
        )

    if step is not None:
        if step < 0 or step >= len(result.steps):
            raise StepIndexError(step=step, total_steps=len(result.steps))
        return result.steps[step]

    # name-based lookup
    matches = [s for s in result.steps if s.name == name]
    if not matches:
        raise StepNameNotFoundError(
            name=name,
            available_names=[s.name for s in result.steps],
        )
    if len(matches) > 1:
        raise StepNameAmbiguousError(
            name=name,
            indices=[s.index for s in matches],
        )
    return matches[0]


def assert_step(
    result: ExecutionResult,
    step: int | None = None,
    name: str | None = None,
    expected_tool: Optional["ToolCall"] = None,
    expected_tools: Optional[list["ToolCall"]] = None,
    order: OrderMode = OrderMode.EXACT,
    partial_args: bool = False,
) -> None:
    """Verify the tool call(s) at a specific step.

    Exactly one of ``step`` (0-indexed) or ``name`` (from
    :func:`step_probe`) must be provided.  Exactly one of
    ``expected_tool`` or ``expected_tools`` must be provided.  Use
    ``expected_tools=[]`` to assert the step made no tool calls.

    Raises:
        StepIndexError: step index out of range.
        StepNameNotFoundError: no step with the given name.
        StepNameAmbiguousError: multiple steps share the name.
        ToolCallSequenceError: the step's tool calls don't match.
    """
    if (expected_tool is None) == (expected_tools is None):
        raise ValueError(
            "Exactly one of `expected_tool` or `expected_tools` must be provided"
        )

    resolved_step = _resolve_step(result, step, name)
    expected_list: list[ToolCall]
    if expected_tool is not None:
        expected_list = [expected_tool]
    else:
        expected_list = list(expected_tools or [])

    actual = resolved_step.tool_calls

    if order == OrderMode.EXACT:
        _assert_step_exact(resolved_step, expected_list, actual, partial_args)
    elif order == OrderMode.IN_ORDER:
        _assert_step_in_order(resolved_step, expected_list, actual, partial_args)
    else:  # OrderMode.ANY_ORDER — OrderMode is a closed enum
        _assert_step_any_order(resolved_step, expected_list, actual, partial_args)


def _assert_step_exact(step_obj, expected, actual, partial_args) -> None:
    if len(expected) != len(actual):
        mismatch = min(len(expected), len(actual))
        for i in range(mismatch):
            if not _tool_call_matches(expected[i], actual[i], partial_args):
                mismatch = i
                break
        raise ToolCallSequenceError(
            expected=expected,
            actual=actual,
            first_mismatch_index=mismatch,
            step_index=step_obj.index,
            step_name=step_obj.name,
        )
    for i in range(len(expected)):
        if not _tool_call_matches(expected[i], actual[i], partial_args):
            raise ToolCallSequenceError(
                expected=expected,
                actual=actual,
                first_mismatch_index=i,
                step_index=step_obj.index,
                step_name=step_obj.name,
            )


def _assert_step_in_order(step_obj, expected, actual, partial_args) -> None:
    if not expected:
        return
    exp_idx = 0
    for act in actual:
        if exp_idx < len(expected) and _tool_call_matches(
            expected[exp_idx], act, partial_args
        ):
            exp_idx += 1
    if exp_idx < len(expected):
        raise ToolCallSequenceError(
            expected=expected,
            actual=actual,
            first_mismatch_index=exp_idx,
            step_index=step_obj.index,
            step_name=step_obj.name,
        )


def _assert_step_any_order(step_obj, expected, actual, partial_args) -> None:
    for i, exp_tc in enumerate(expected):
        if not any(_tool_call_matches(exp_tc, a, partial_args) for a in actual):
            raise ToolCallSequenceError(
                expected=expected,
                actual=actual,
                first_mismatch_index=i,
                step_index=step_obj.index,
                step_name=step_obj.name,
            )


def assert_step_output(
    result: ExecutionResult,
    step: int | None = None,
    name: str | None = None,
    contains: str | None = None,
    equals: str | None = None,
    matches: str | None = None,
) -> None:
    """Verify the intermediate text output of a specific step.

    Exactly one of ``step`` or ``name`` must be provided.  At least one
    of ``contains``, ``equals``, or ``matches`` must be provided.
    """
    from agentverify.errors import StepOutputError

    if contains is None and equals is None and matches is None:
        raise ValueError(
            "assert_step_output requires at least one of: contains, equals, matches"
        )

    resolved_step = _resolve_step(result, step, name)
    output = resolved_step.output

    step_label = f"step {resolved_step.index}"
    if resolved_step.name:
        step_label += f" ({resolved_step.name!r})"

    if output is None:
        raise StepOutputError(f"{step_label} has no output")

    if equals is not None and output != equals:
        raise StepOutputError(
            f"{step_label} output does not equal expected\n"
            f"\n"
            f"  Expected: {equals!r}\n"
            f"  Actual:   {output!r}"
        )
    if contains is not None and contains not in output:
        raise StepOutputError(
            f"{step_label} output does not contain expected substring\n"
            f"\n"
            f"  Substring: {contains!r}\n"
            f"  Actual:    {output!r}"
        )
    if matches is not None:
        import re

        if not re.search(matches, output):
            raise StepOutputError(
                f"{step_label} output does not match pattern\n"
                f"\n"
                f"  Pattern: {matches!r}\n"
                f"  Actual:  {output!r}"
            )


def assert_step_uses_result_from(
    result: ExecutionResult,
    step: int | str,
    depends_on: int | str,
    via: str = "any",
) -> None:
    """Verify that step N's input references data produced by step M.

    Checks that any produced value from step M appears as a consumed
    value in step N.

    * Produced values: step M's ``tool_results``,
      ``tool_calls[*].result``, ``output``.
    * Consumed values: step N's ``input_context``,
      ``tool_calls[*].arguments``.

    Matching is substring for strings, structural equality for
    containers.

    Args:
        step: Step that should consume data.  Accepts int (index) or
            str (name).
        depends_on: Step that should produce data.  Same specifier rules.
        via: One of ``"tool_result"``, ``"output"``, or ``"any"``.
            Restricts which channel of produced values is checked.
    """
    from agentverify.errors import StepDependencyError

    if via not in ("tool_result", "output", "any"):
        raise ValueError(
            f"via must be one of 'tool_result', 'output', 'any' — got {via!r}"
        )

    consumer = _resolve_step_by_spec(result, step)
    producer = _resolve_step_by_spec(result, depends_on)

    produced = _produced_values(producer, via)
    consumed = _consumed_values(consumer)

    for p in produced:
        # A produced value flows when it equals a consumed value, or
        # any of its leaves appears inside a consumed value.
        # Strings are also passed through _leaves so that JSON-encoded
        # tool results can be descended and their primitives extracted.
        candidates: list = []
        if isinstance(p, (int, float, bool)) or p is None:
            candidates.append(p)
        else:
            candidates.extend(_leaves(p))
            if p not in candidates:
                candidates.append(p)  # direct equality fallback
        for cand in candidates:
            for c in consumed:
                if _value_matches(cand, c):
                    return

    raise StepDependencyError(
        step=consumer.index,
        depends_on=producer.index,
        via=via,
        produced=produced,
        consumed=consumed,
    )


def _resolve_step_by_spec(result: ExecutionResult, spec: int | str):
    """Resolve a step spec (int index or str name) into a Step."""
    if isinstance(spec, int):
        return _resolve_step(result, step=spec, name=None)
    if isinstance(spec, str):
        return _resolve_step(result, step=None, name=spec)
    raise TypeError(
        f"step spec must be int or str, got {type(spec).__name__}"
    )


def _produced_values(step_obj, via: str) -> list:
    """Collect values produced by a step that downstream steps might use."""
    produced: list = []
    if via in ("tool_result", "any"):
        produced.extend(step_obj.tool_results)
        for tc in step_obj.tool_calls:
            if tc.result is not None:
                produced.append(tc.result)
    if via in ("output", "any"):
        if step_obj.output is not None:
            produced.append(step_obj.output)
    return produced


def _consumed_values(step_obj) -> list:
    """Collect values that a step consumed as input."""
    consumed: list = []
    if step_obj.input_context is not None:
        consumed.append(step_obj.input_context)
    for tc in step_obj.tool_calls:
        consumed.append(tc.arguments)
    return consumed


def _value_matches(produced, consumed) -> bool:
    """Return True when ``produced`` appears inside ``consumed``.

    - Direct equality always matches.
    - String produced in string consumed → substring match.
    - Primitive produced (str / int / float / bool) in any other
      consumed → substring match on a JSON/repr serialization of
      ``consumed``.
    - Non-primitive produced values are only matched via direct equality
      (callers are expected to pass leaves via :func:`_leaves`).
    """
    if produced == consumed:
        return True

    if isinstance(produced, str):
        if not produced:
            return False
        if isinstance(consumed, str):
            return produced in consumed
        serialized = _stringify(consumed)
        return produced in serialized

    # Non-string primitives: JSON-serialize and substring-search the
    # consumed container to catch values embedded in nested structures.
    if isinstance(produced, (int, float, bool)) and not isinstance(consumed, str):
        needle = _stringify(produced)
        serialized = _stringify(consumed)
        return needle in serialized

    return False


def _stringify(value) -> str:
    """Serialize a nested value into a string for substring search.

    Falls back to ``repr`` if JSON serialization fails (e.g. circular
    references or non-serializable types with no default handler).
    """
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _leaves(value):
    """Yield leaf (non-container) values from a nested structure.

    Strings that successfully parse as JSON are recursively descended —
    this lets us extract primitive values embedded in JSON-serialized
    tool results (common in cassette-based tests).
    """
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _leaves(v)
    elif isinstance(value, str):
        # Try to see if the string is JSON; if so, descend into it.
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                yield value  # keep the raw string itself as a candidate too
                yield from _leaves(parsed)
                return
        yield value
    else:
        yield value
