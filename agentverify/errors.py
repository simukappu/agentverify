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
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.first_mismatch_index = first_mismatch_index
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
        lines = [f"Tool call sequence mismatch at index {idx}", ""]

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
