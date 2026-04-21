"""Core data models for agentverify.

Provides structured representations of agent execution results,
tool calls, and token usage for deterministic testing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


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


@dataclass
class ExecutionResult:
    """The result of a single agent execution."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    token_usage: Optional[TokenUsage] = None
    total_cost_usd: Optional[float] = None
    duration_ms: Optional[float] = None
    final_output: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionResult:
        """Build an ExecutionResult from a dictionary."""
        tool_calls = []
        for i, tc in enumerate(data.get("tool_calls", [])):
            if not isinstance(tc, dict) or "name" not in tc:
                raise ValueError(
                    f"tool_calls[{i}]: each tool call dict must have a 'name' key, got {tc!r}"
                )
            tool_calls.append(
                ToolCall(
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                    result=tc.get("result"),
                )
            )

        token_usage = None
        if "token_usage" in data and data["token_usage"] is not None:
            tu = data["token_usage"]
            token_usage = TokenUsage(
                input_tokens=tu.get("input_tokens", 0),
                output_tokens=tu.get("output_tokens", 0),
            )

        return cls(
            tool_calls=tool_calls,
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
        """Convert this ExecutionResult to a dictionary."""
        result: dict[str, Any] = {
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                }
                for tc in self.tool_calls
            ],
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
