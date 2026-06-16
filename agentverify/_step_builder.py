"""Internal helpers for building :class:`Step` / :class:`ExecutionResult`.

Used by recorder, MockLLM, and framework adapters.  Not part of the public API.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agentverify.models import Step, TokenUsage, ToolCall


def classify_tool_result_error(payload: Any) -> Optional[bool]:
    """Classify whether a tool result payload represents an error.

    This is the single shared, total classifier used by the framework adapters and the cassette/MockLLM backfill so that tool-invocation-outcome detection is consistent across construction paths.

    Returns:
        ``True`` if the payload is a recognizable tool error, ``False`` if it is a recognizable non-error, and ``None`` when the error status cannot be determined (e.g. a plain string tool output).

    It never raises: arbitrary untrusted payloads are treated strictly as data.
    """
    if isinstance(payload, str):
        # A tool result may arrive as a JSON-encoded string (common on
        # cassette replay, where the tool message content is serialized).
        # Try to parse an object out of it; a plain non-JSON string
        # (e.g. "sunny, 22C") stays unknown.
        stripped = payload.strip()
        if stripped[:1] == "{":
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return None
            return classify_tool_result_error(parsed)
        return None
    if isinstance(payload, dict):
        if "is_error" in payload:
            return bool(payload["is_error"])
        status = payload.get("status")
        if isinstance(status, str):
            lowered = status.lower()
            if lowered == "error":
                return True
            if lowered in ("success", "ok"):
                return False
        error_val = payload.get("error")
        if error_val:
            return True
        if "error" in payload and error_val in (None, "", [], {}):
            # Explicit empty error field reads as non-error.
            return False
    return None


def build_tool_results_meta(
    tool_results: list[Any],
    explicit: list[Optional[bool]] | None = None,
) -> Optional[list[dict[str, Any]]]:
    """Build a ``tool_results_meta`` list aligned with ``tool_results``.

    For each result, the error status is taken from ``explicit[i]`` when it is not ``None``, otherwise from :func:`classify_tool_result_error`.  Positions that resolve to ``None`` (unknown) are still emitted as an entry without an ``"is_error"`` key, keeping the list positionally aligned.

    Returns ``None`` when ``tool_results`` is empty, so steps with no tool results carry no metadata.
    """
    if not tool_results:
        return None
    meta: list[dict[str, Any]] = []
    for i, result in enumerate(tool_results):
        status: Optional[bool] = None
        if explicit is not None and i < len(explicit) and explicit[i] is not None:
            status = explicit[i]
        else:
            status = classify_tool_result_error(result)
        meta.append({"is_error": status} if status is not None else {})
    return meta


def parse_tool_call_arguments(raw: Any) -> dict[str, Any]:
    """Parse tool call arguments into a dict.

    ``raw`` may be a dict already, a JSON string, or something else. Invalid JSON and non-dict/non-string values return an empty dict.
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
