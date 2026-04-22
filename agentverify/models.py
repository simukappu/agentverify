"""Core data models for agentverify.

Provides structured representations of agent execution results,
steps, tool calls, and token usage for deterministic testing.

An :class:`ExecutionResult` is made of an ordered list of
:class:`Step` objects.  Each step represents a single observable unit
of agent execution (an LLM call, a user-defined probe boundary, or a
pure tool execution).  The flat ``tool_calls`` view is available via a
derived property for convenience and backward compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class ToolCall:
    """A single tool call executed by an agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None


@dataclass
class TokenUsage:
    """Token consumption breakdown."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


StepSource = Literal["llm", "probe", "tool"]


@dataclass(frozen=True)
class Step:
    """A single observable unit of agent execution.

    A step can represent an LLM call (``source="llm"``), a user-defined
    probe boundary (``source="probe"``), or a pure tool execution in a
    workflow-style agent (``source="tool"``).

    Attributes:
        index: 0-based position in ``ExecutionResult.steps``.
        name: Optional human-readable label (from ``step_probe`` or a
            framework node name).
        source: What produced this step boundary.
        tool_calls: Tool calls emitted during this step (may be empty).
        tool_results: Tool results observed during this step.  Used by
            :func:`assert_step_uses_result_from` for data flow checks.
        output: Text output produced by this step (LLM reply, or user
            data via ``ProbeHandle.set_output``).
        duration_ms: Wall-clock duration of this step, if known.
        token_usage: Tokens consumed during this step, if known.
        input_context: Snapshot of input messages / context that arrived
            at this step.  Used by :func:`assert_step_uses_result_from`.
    """

    index: int
    name: Optional[str] = None
    source: StepSource = "llm"
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    output: Optional[str] = None
    duration_ms: Optional[float] = None
    token_usage: Optional[TokenUsage] = None
    input_context: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this step to a plain dict."""
        return {
            "index": self.index,
            "name": self.name,
            "source": self.source,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "result": tc.result}
                for tc in self.tool_calls
            ],
            "tool_results": list(self.tool_results),
            "output": self.output,
            "duration_ms": self.duration_ms,
            "token_usage": (
                {
                    "input_tokens": self.token_usage.input_tokens,
                    "output_tokens": self.token_usage.output_tokens,
                }
                if self.token_usage is not None
                else None
            ),
            "input_context": self.input_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        """Build a Step from a plain dict."""
        tool_calls: list[ToolCall] = []
        for i, tc in enumerate(data.get("tool_calls", [])):
            if not isinstance(tc, dict) or "name" not in tc:
                raise ValueError(
                    f"step.tool_calls[{i}]: each tool call dict must have a 'name' key, got {tc!r}"
                )
            tool_calls.append(
                ToolCall(
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                    result=tc.get("result"),
                )
            )

        token_usage = None
        if data.get("token_usage") is not None:
            tu = data["token_usage"]
            token_usage = TokenUsage(
                input_tokens=tu.get("input_tokens", 0),
                output_tokens=tu.get("output_tokens", 0),
            )

        source = data.get("source", "llm")
        if source not in ("llm", "probe", "tool"):
            raise ValueError(
                f"step.source must be one of 'llm', 'probe', 'tool' — got {source!r}"
            )

        return cls(
            index=data.get("index", 0),
            name=data.get("name"),
            source=source,
            tool_calls=tool_calls,
            tool_results=list(data.get("tool_results", [])),
            output=data.get("output"),
            duration_ms=data.get("duration_ms"),
            token_usage=token_usage,
            input_context=data.get("input_context"),
        )


@dataclass
class ExecutionResult:
    """The result of a single agent execution.

    An ExecutionResult is an ordered list of :class:`Step` objects plus
    aggregate metadata (token usage, cost, duration, final output).

    The flat ``tool_calls`` view is exposed as a read-only property
    derived from ``steps`` — there is no underlying flat storage.

    For backward compatibility with v0.2.0, the constructor also
    accepts a ``tool_calls=[...]`` keyword argument which is wrapped
    into a single synthetic ``Step`` (``source="llm"``) when ``steps``
    is not provided.  New code should use ``steps=[...]`` or
    :meth:`from_flat_tool_calls`.
    """

    steps: list[Step] = field(default_factory=list)
    token_usage: Optional[TokenUsage] = None
    total_cost_usd: Optional[float] = None
    duration_ms: Optional[float] = None
    final_output: Optional[str] = None

    def __init__(
        self,
        steps: list[Step] | None = None,
        token_usage: Optional[TokenUsage] = None,
        total_cost_usd: Optional[float] = None,
        duration_ms: Optional[float] = None,
        final_output: Optional[str] = None,
        *,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        if tool_calls is not None:
            if steps is not None:
                raise TypeError(
                    "ExecutionResult(): pass either 'steps' or 'tool_calls', not both"
                )
            self.steps = [
                Step(index=0, source="llm", tool_calls=list(tool_calls))
            ] if tool_calls else []
        else:
            self.steps = list(steps) if steps else []
        self.token_usage = token_usage
        self.total_cost_usd = total_cost_usd
        self.duration_ms = duration_ms
        self.final_output = final_output

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Flattened tool calls across all steps (read-only derived view)."""
        return [tc for s in self.steps for tc in s.tool_calls]

    @classmethod
    def from_flat_tool_calls(
        cls,
        tool_calls: list[ToolCall],
        *,
        token_usage: Optional[TokenUsage] = None,
        total_cost_usd: Optional[float] = None,
        duration_ms: Optional[float] = None,
        final_output: Optional[str] = None,
    ) -> ExecutionResult:
        """Build an ExecutionResult from a flat tool call list (single step).

        Wraps the flat tool call list into a single synthetic ``Step``.
        Useful for migration from v0.2.0 and for tests that don't care
        about step structure.
        """
        steps = [Step(index=0, source="llm", tool_calls=list(tool_calls))]
        return cls(
            steps=steps,
            token_usage=token_usage,
            total_cost_usd=total_cost_usd,
            duration_ms=duration_ms,
            final_output=final_output,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionResult:
        """Build an ExecutionResult from a dictionary.

        Prefers the ``steps`` key.  If only a legacy ``tool_calls`` key
        is present, it is wrapped into a single synthetic step for
        backward compatibility with v0.2.0 serialized results.  When
        both keys are present, ``steps`` takes precedence.
        """
        token_usage = None
        if data.get("token_usage") is not None:
            tu = data["token_usage"]
            token_usage = TokenUsage(
                input_tokens=tu.get("input_tokens", 0),
                output_tokens=tu.get("output_tokens", 0),
            )

        if "steps" in data and data["steps"] is not None:
            steps = [Step.from_dict(s) for s in data["steps"]]
        elif "tool_calls" in data:
            # Legacy single-step compatibility path (v0.2.0 dicts).
            legacy_calls: list[ToolCall] = []
            for i, tc in enumerate(data.get("tool_calls") or []):
                if not isinstance(tc, dict) or "name" not in tc:
                    raise ValueError(
                        f"tool_calls[{i}]: each tool call dict must have a 'name' key, got {tc!r}"
                    )
                legacy_calls.append(
                    ToolCall(
                        name=tc["name"],
                        arguments=tc.get("arguments", {}),
                        result=tc.get("result"),
                    )
                )
            steps = [Step(index=0, source="llm", tool_calls=legacy_calls)]
        else:
            steps = []

        return cls(
            steps=steps,
            token_usage=token_usage,
            total_cost_usd=data.get("total_cost_usd"),
            final_output=data.get("final_output"),
            duration_ms=data.get("duration_ms"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ExecutionResult:
        """Build an ExecutionResult from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> dict[str, Any]:
        """Convert this ExecutionResult to a dictionary.

        The output uses the ``steps`` schema — flat ``tool_calls`` is
        not emitted.  Use :meth:`from_flat_tool_calls` to load legacy
        v0.2.0 data.
        """
        result: dict[str, Any] = {
            "steps": [s.to_dict() for s in self.steps],
        }

        if self.token_usage is not None:
            result["token_usage"] = {
                "input_tokens": self.token_usage.input_tokens,
                "output_tokens": self.token_usage.output_tokens,
            }
        else:
            result["token_usage"] = None

        result["total_cost_usd"] = self.total_cost_usd
        result["duration_ms"] = self.duration_ms
        result["final_output"] = self.final_output

        return result

    def to_json(self) -> str:
        """Convert this ExecutionResult to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
