"""Custom exception classes for agentverify.

All exceptions inherit from AssertionError so that pytest recognizes them
as assertion failures and integrates them into standard test failure reports.
"""

from __future__ import annotations

from typing import Any


class AgentVerifyError(AssertionError):
    """Base exception for all agentverify assertion failures."""

    pass


class ToolCallSequenceError(AgentVerifyError):
    """Raised when the tool call sequence does not match the expected sequence."""

    def __init__(
        self,
        expected: list[Any],
        actual: list[Any],
        first_mismatch_index: int,
        step_index: int | None = None,
        step_name: str | None = None,
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.first_mismatch_index = first_mismatch_index
        self.step_index = step_index
        self.step_name = step_name
        super().__init__(self._build_message())

    def _format_tool_call(self, tc: Any) -> str:
        """Format a single tool call for display."""
        name = getattr(tc, "name", str(tc))
        args = getattr(tc, "arguments", {})
        if args:
            args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in args.items())
            return f"{name}({args_str})"
        return f"{name}()"

    def _build_message(self) -> str:
        idx = self.first_mismatch_index
        header = f"Tool call sequence mismatch at index {idx}"
        if self.step_index is not None or self.step_name is not None:
            parts: list[str] = []
            if self.step_index is not None:
                parts.append(f"step {self.step_index}")
            if self.step_name is not None:
                parts.append(f"name={self.step_name!r}")
            header = f"Tool call sequence mismatch at index {idx} ({', '.join(parts)})"
        lines = [header, ""]

        lines.append("Expected:")
        for i, tc in enumerate(self.expected):
            marker = "     ← first mismatch" if i == idx else ""
            lines.append(f"  [{i}] {self._format_tool_call(tc)}{marker}")

        lines.append("")
        lines.append("Actual:")
        for i, tc in enumerate(self.actual):
            marker = "  ← actual" if i == idx else ""
            lines.append(f"  [{i}] {self._format_tool_call(tc)}{marker}")

        return "\n".join(lines)


class CostBudgetError(AgentVerifyError):
    """Raised when token or cost budget is exceeded."""

    def __init__(
        self,
        actual: int | float,
        limit: int | float,
        exceeded_by: int | float,
        message: str | None = None,
    ) -> None:
        self.actual = actual
        self.limit = limit
        self.exceeded_by = exceeded_by
        self._custom_message = message
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if self._custom_message:
            return self._custom_message
        if isinstance(self.actual, float) or isinstance(self.limit, float):
            # Cost in USD
            pct = (self.exceeded_by / self.limit * 100) if self.limit else 0
            return (
                "Cost budget exceeded\n"
                "\n"
                f"  Actual:  ${self.actual:,.2f}\n"
                f"  Limit:   ${self.limit:,.2f}\n"
                f"  Exceeded by: ${self.exceeded_by:,.2f} ({pct:.1f}%)"
            )
        # Token budget
        pct = (self.exceeded_by / self.limit * 100) if self.limit else 0
        return (
            "Token budget exceeded\n"
            "\n"
            f"  Actual:  {self.actual:,} tokens\n"
            f"  Limit:   {self.limit:,} tokens\n"
            f"  Exceeded by: {self.exceeded_by:,} tokens ({pct:.1f}%)"
        )


class SafetyRuleViolationError(AgentVerifyError):
    """Raised when forbidden tool calls are detected."""

    def __init__(self, violations: list[dict[str, Any]]) -> None:
        self.violations = violations
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        n = len(self.violations)
        lines = [f"{n} forbidden tool call{'s' if n != 1 else ''} detected", ""]
        for i, v in enumerate(self.violations, start=1):
            tool_name = v.get("tool_name", "unknown")
            arguments = v.get("arguments", {})
            position = v.get("position", "?")
            if arguments:
                args_str = ", ".join(
                    f'{k}="{val}"' if isinstance(val, str) else f"{k}={val}"
                    for k, val in arguments.items()
                )
                lines.append(f"  [{i}] {tool_name}({args_str}) at position {position}")
            else:
                lines.append(f"  [{i}] {tool_name}() at position {position}")
        return "\n".join(lines)


class MultipleAssertionError(AgentVerifyError):
    """Raised when multiple assertions fail, collecting all failures."""

    def __init__(self, errors: list[AgentVerifyError]) -> None:
        self.errors = errors
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        n = len(self.errors)
        lines = [f"{n} assertion{'s' if n != 1 else ''} failed:", ""]
        for i, err in enumerate(self.errors, start=1):
            lines.append(f"  [{i}] {type(err).__name__}: {err}")
            lines.append("")
        return "\n".join(lines)


class FinalOutputError(AgentVerifyError):
    """Raised when the agent's final output does not meet expectations."""

    pass


class CassetteMissingRequestError(AgentVerifyError):
    """Raised when a cassette does not have a matching request for replay."""

    pass


class LatencyBudgetError(AgentVerifyError):
    """Raised when execution latency exceeds the allowed threshold."""

    def __init__(
        self,
        actual_ms: float,
        limit_ms: float,
        exceeded_by_ms: float,
        message: str | None = None,
    ) -> None:
        self.actual_ms = actual_ms
        self.limit_ms = limit_ms
        self.exceeded_by_ms = exceeded_by_ms
        self._custom_message = message
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if self._custom_message:
            return self._custom_message
        pct = (self.exceeded_by_ms / self.limit_ms * 100) if self.limit_ms else 0
        return (
            "Latency budget exceeded\n"
            "\n"
            f"  Actual:  {self.actual_ms:,.1f} ms\n"
            f"  Limit:   {self.limit_ms:,.1f} ms\n"
            f"  Exceeded by: {self.exceeded_by_ms:,.1f} ms ({pct:.1f}%)"
        )


class CassetteRequestMismatchError(AgentVerifyError):
    """Raised when a replay request does not match the recorded request.

    This indicates the cassette is stale and should be re-recorded.
    """

    def __init__(
        self,
        index: int,
        field: str,
        recorded: Any,
        actual: Any,
    ) -> None:
        self.index = index
        self.field = field
        self.recorded = recorded
        self.actual = actual
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return (
            f"Cassette request mismatch at interaction {self.index}\n"
            f"\n"
            f"  Field:    {self.field}\n"
            f"  Recorded: {self.recorded!r}\n"
            f"  Actual:   {self.actual!r}\n"
            f"\n"
            f"  The cassette may be stale. Re-record with --cassette-mode=record."
        )


class StepIndexError(AgentVerifyError):
    """Raised when a step index is out of range."""

    def __init__(self, step: int, total_steps: int) -> None:
        self.step = step
        self.total_steps = total_steps
        super().__init__(
            f"Step index {step} is out of range (result has {total_steps} step{'s' if total_steps != 1 else ''})"
        )


class StepNameNotFoundError(AgentVerifyError):
    """Raised when no step with the given name exists."""

    def __init__(self, name: str, available_names: list[str]) -> None:
        self.name = name
        self.available_names = available_names
        named = [n for n in available_names if n is not None]
        hint = (
            f"  Available step names: {named}"
            if named
            else "  No steps in this result have names."
        )
        super().__init__(
            f"No step named {name!r} in execution result\n\n{hint}"
        )


class StepNameAmbiguousError(AgentVerifyError):
    """Raised when multiple steps share the same name."""

    def __init__(self, name: str, indices: list[int]) -> None:
        self.name = name
        self.indices = indices
        super().__init__(
            f"Step name {name!r} is ambiguous — matches {len(indices)} steps at indices {indices}.\n"
            f"\n"
            f"  Use `step=<index>` instead of `name=...` to disambiguate."
        )


class StepOutputError(AgentVerifyError):
    """Raised when a step's intermediate output does not match expectations."""

    pass


class StepDependencyError(AgentVerifyError):
    """Raised when no data flow is detected between two steps."""

    def __init__(
        self,
        step: int,
        depends_on: int,
        via: str,
        produced: list[Any],
        consumed: list[Any],
    ) -> None:
        self.step = step
        self.depends_on = depends_on
        self.via = via
        self.produced = produced
        self.consumed = consumed
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        lines = [
            f"No data flow detected from step {self.depends_on} to step {self.step}",
            "",
            f"  via: {self.via}",
            f"  step {self.depends_on} produced: {self._truncate_list(self.produced)}",
            f"  step {self.step} consumed:  {self._truncate_list(self.consumed)}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _truncate_list(values: list[Any], limit: int = 200) -> str:
        repr_str = repr(values)
        if len(repr_str) > limit:
            return repr_str[:limit] + "…"
        return repr_str
