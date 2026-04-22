"""Internal helpers for building :class:`Step` / :class:`ExecutionResult`.

Used by recorder, MockLLM, and framework adapters.  Not part of the
public API.
"""

from __future__ import annotations

import json
from typing import Any

from agentverify.models import Step, TokenUsage, ToolCall


def parse_tool_call_arguments(raw: Any) -> dict[str, Any]:
    """Parse tool call arguments into a dict.

    ``raw`` may be a dict already, a JSON string, or something else.
    Invalid JSON and non-dict/non-string values return an empty dict.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def tool_calls_from_response(
    response_tool_calls: list[dict[str, Any]] | None,
) -> list[ToolCall]:
    """Convert NormalizedResponse.tool_calls dicts into ToolCall objects."""
    if not response_tool_calls:
        return []
    result: list[ToolCall] = []
    for tc in response_tool_calls:
        if not isinstance(tc, dict) or "name" not in tc:
            continue
        result.append(
            ToolCall(
                name=tc["name"],
                arguments=parse_tool_call_arguments(tc.get("arguments", {})),
                result=tc.get("result"),
            )
        )
    return result


def aggregate_token_usage(steps: list[Step]) -> TokenUsage | None:
    """Sum token usage across all steps, or return None if none have it."""
    total_input = 0
    total_output = 0
    has = False
    for s in steps:
        if s.token_usage is not None:
            has = True
            total_input += s.token_usage.input_tokens
            total_output += s.token_usage.output_tokens
    if not has:
        return None
    return TokenUsage(input_tokens=total_input, output_tokens=total_output)


def final_output_from_steps(steps: list[Step]) -> str | None:
    """Return the last non-None step output, or None if there are none."""
    last: str | None = None
    for s in steps:
        if s.output is not None:
            last = s.output
    return last
