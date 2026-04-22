"""OpenAI Agents SDK adapter for agentverify.

Converts an OpenAI Agents SDK ``RunResult`` into an agentverify
``ExecutionResult``.

Step boundary: **each ``MessageOutputItem`` / ``ToolCallItem`` cluster
that belongs to the same model response becomes one step**
(``source="llm"``).  Because the SDK emits tool call items grouped by
the assistant turn that produced them, we start a new step whenever we
see a ``message_output_item`` or when we see a ``tool_call_item``
immediately after ``tool_call_output_item``(s).  ``tool_call_output_item``
values are attached to the preceding step's ``tool_results``.
"""

from __future__ import annotations

import json
from typing import Any

from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def from_openai_agents(result: Any) -> ExecutionResult:
    """Build an ``ExecutionResult`` from an OpenAI Agents SDK ``RunResult``.

    Args:
        result: A ``RunResult`` or ``RunResultStreaming`` from the
            OpenAI Agents SDK ``Runner.run()`` or ``Runner.run_sync()``.
    """
    new_items = getattr(result, "new_items", []) or []

    steps: list[Step] = []
    current_tool_calls: list[ToolCall] = []
    current_tool_results: list[Any] = []
    just_saw_tool_output = False

    def flush_current_step() -> None:
        nonlocal current_tool_calls, current_tool_results
        if current_tool_calls or current_tool_results:
            steps.append(
                Step(
                    index=len(steps),
                    source="llm",
                    tool_calls=current_tool_calls,
                    tool_results=current_tool_results,
                )
            )
        current_tool_calls = []
        current_tool_results = []

    for item in new_items:
        item_type = getattr(item, "type", None)

        if item_type == "tool_call_item":
            # If the previous step had a tool output, this tool_call
            # starts a new step (it's the next assistant turn).
            if just_saw_tool_output:
                flush_current_step()
                just_saw_tool_output = False

            raw = getattr(item, "raw_item", None)
            if raw is None:
                continue
            name = (
                raw.get("name", "")
                if isinstance(raw, dict)
                else getattr(raw, "name", "")
            )
            if not name:
                continue
            args_raw = (
                raw.get("arguments", "{}")
                if isinstance(raw, dict)
                else getattr(raw, "arguments", "{}")
            )
            current_tool_calls.append(
                ToolCall(name=name, arguments=_parse_arguments(args_raw))
            )
        elif item_type == "tool_call_output_item":
            output = getattr(item, "output", None)
            if output is None and hasattr(item, "raw_item"):
                raw = item.raw_item
                output = (
                    raw.get("output")
                    if isinstance(raw, dict)
                    else getattr(raw, "output", None)
                )
            current_tool_results.append(output)
            just_saw_tool_output = True
        elif item_type == "message_output_item":
            # A final-or-intermediate assistant text message: close the
            # current step (with its text output).
            raw = getattr(item, "raw_item", None)
            text = _extract_openai_agents_text(raw)
            if current_tool_calls or current_tool_results or text is not None:
                steps.append(
                    Step(
                        index=len(steps),
                        source="llm",
                        tool_calls=current_tool_calls,
                        tool_results=current_tool_results,
                        output=text,
                    )
                )
                current_tool_calls = []
                current_tool_results = []
                just_saw_tool_output = False

    # Flush any trailing tool_calls/results.
    flush_current_step()

    # Token usage.
    token_usage: TokenUsage | None = None
    context_wrapper = getattr(result, "context_wrapper", None)
    usage = getattr(context_wrapper, "usage", None) if context_wrapper else None
    if usage is not None:
        it = getattr(usage, "input_tokens", 0) or 0
        ot = getattr(usage, "output_tokens", 0) or 0
        if it or ot:
            token_usage = TokenUsage(input_tokens=it, output_tokens=ot)

    # Final output.
    final_output = getattr(result, "final_output", None)
    if final_output is not None and not isinstance(final_output, str):
        final_output = str(final_output)

    return ExecutionResult(
        steps=steps,
        token_usage=token_usage,
        final_output=final_output,
    )


def _extract_openai_agents_text(raw: Any) -> str | None:
    """Extract text from a message_output_item's raw_item.

    OpenAI Agents ``MessageOutputItem.raw_item`` is a
    ``ResponseOutputMessage`` whose ``content`` is a list of
    ``ResponseOutputText`` / other content parts.
    """
    if raw is None:
        return None
    content = (
        raw.get("content")
        if isinstance(raw, dict)
        else getattr(raw, "content", None)
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            text = (
                part.get("text")
                if isinstance(part, dict)
                else getattr(part, "text", None)
            )
            if isinstance(text, str):
                return text
    return None
