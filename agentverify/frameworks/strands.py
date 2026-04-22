"""Strands Agents adapter for agentverify.

Converts a Strands Agents ``AgentResult`` into an agentverify
``ExecutionResult`` without requiring a custom converter function.

Usage::

    from agentverify.frameworks.strands import from_strands

    result = agent("Analyze the files")
    execution_result = from_strands(result)

The Strands AgentResult has the following structure:

- ``result.state.messages``: Conversation history (list of message dicts).
  Each ``assistant`` message's ``content`` is a list of blocks, which
  may contain ``{"toolUse": {...}}`` and/or ``{"text": "..."}`` entries.
  ``user`` messages between two assistant messages often contain
  ``{"toolResult": {...}}`` blocks that carry tool execution results.
- ``result.metrics``: Token consumption dict with ``"inputTokens"`` and
  ``"outputTokens"`` keys (may be absent during mock execution).
- ``result.message``: Final response message dict.

Step boundary: **each ``assistant`` message becomes one step**
(``source="llm"``).  Tool results observed in the next ``user`` message
are attached to the previous step's ``tool_results``.
"""

from __future__ import annotations

from typing import Any

from agentverify._step_builder import aggregate_token_usage
from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall


def _extract_tool_use_calls(content: list[Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for block in content:
        if isinstance(block, dict) and "toolUse" in block:
            tu = block["toolUse"]
            calls.append(ToolCall(name=tu["name"], arguments=tu.get("input", {})))
    return calls


def _extract_text(content: list[Any]) -> str | None:
    last_text: str | None = None
    for block in content:
        if isinstance(block, dict) and "text" in block:
            last_text = block["text"]
    return last_text


def _extract_tool_results(content: list[Any]) -> list[Any]:
    results: list[Any] = []
    for block in content:
        if isinstance(block, dict) and "toolResult" in block:
            results.append(block["toolResult"])
    return results


def from_strands(result: Any) -> ExecutionResult:
    """Build an ``ExecutionResult`` from a Strands Agents ``AgentResult``.

    Args:
        result: A Strands Agents SDK ``AgentResult`` object.
    """
    # Walk messages: each assistant message is one step; tool results
    # that appear after it (in the next user message) are attached to
    # that step's tool_results.
    messages = result.state.messages if hasattr(result, "state") else []

    steps: list[Step] = []
    pending_input_context: dict[str, Any] | None = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        # Detect message kind: if role is missing, infer from content —
        # any toolUse or text block is treated as an assistant message
        # (preserves v0.2.0 adapter behaviour for messages without a role).
        has_tool_use = any(
            isinstance(b, dict) and "toolUse" in b for b in content
        )
        has_text = any(isinstance(b, dict) and "text" in b for b in content)
        has_tool_result = any(
            isinstance(b, dict) and "toolResult" in b for b in content
        )

        is_assistant = role == "assistant" or (
            role is None and (has_tool_use or has_text)
        )
        is_user = role == "user" or (role is None and has_tool_result)

        if is_assistant:
            step = Step(
                index=len(steps),
                source="llm",
                tool_calls=_extract_tool_use_calls(content),
                output=_extract_text(content),
                input_context=pending_input_context,
            )
            steps.append(step)
            pending_input_context = None
        elif is_user:
            tool_results = _extract_tool_results(content)
            if tool_results and steps:
                prev = steps[-1]
                steps[-1] = Step(
                    index=prev.index,
                    name=prev.name,
                    source=prev.source,
                    tool_calls=prev.tool_calls,
                    tool_results=prev.tool_results + tool_results,
                    output=prev.output,
                    duration_ms=prev.duration_ms,
                    token_usage=prev.token_usage,
                    input_context=prev.input_context,
                )
            pending_input_context = {"messages": [msg]}

    # Token usage — Strands provides aggregate metrics at the top level,
    # not per-step.  Attach it to the overall ExecutionResult.
    token_usage: TokenUsage | None = aggregate_token_usage(steps)
    if token_usage is None and hasattr(result, "metrics") and result.metrics:
        metrics = result.metrics
        it = metrics.get("inputTokens", 0)
        ot = metrics.get("outputTokens", 0)
        if it or ot:
            token_usage = TokenUsage(input_tokens=it, output_tokens=ot)

    # Final output — prefer the last text block in result.message.
    final_output: str | None = None
    if hasattr(result, "message") and result.message:
        mc = (
            result.message.get("content")
            if isinstance(result.message, dict)
            else None
        )
        if isinstance(mc, list):
            final_output = _extract_text(mc)

    return ExecutionResult(
        steps=steps,
        token_usage=token_usage,
        final_output=final_output,
    )
