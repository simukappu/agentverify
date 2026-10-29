"""Strands Agents adapter for agentverify.

Converts a Strands Agents ``Agent`` after a run into an agentverify ``ExecutionResult`` without requiring a custom converter function.

Usage::

    from agentverify.frameworks.strands import from_strands

    agent("Analyze the files")
    execution_result = from_strands(agent)

The conversation lives on the agent, not on the ``AgentResult`` a call returns:

- ``agent.messages``: Conversation history (list of message dicts), including earlier calls on the same agent; clear it between tests (``agent.messages.clear()``) for one run per result.
  Each ``assistant`` message's ``content`` is a list of blocks, which may contain ``{"toolUse": {...}}`` and/or ``{"text": "..."}`` entries. ``user`` messages between two assistant messages often contain ``{"toolResult": {...}}`` blocks that carry tool execution results.
- ``agent.event_loop_metrics.accumulated_usage``: Token consumption, with ``"inputTokens"`` and ``"outputTokens"`` keys.

Step boundary: **each ``assistant`` message becomes one step** (``source="llm"``).  Tool results observed in the next ``user`` message are attached to the previous step's ``tool_results``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agentverify._step_builder import aggregate_token_usage
from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall, call_id_or_none


def _extract_tool_use_calls(content: list[Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for block in content:
        if isinstance(block, dict) and "toolUse" in block:
            tu = block["toolUse"]
            calls.append(ToolCall(name=tu["name"], arguments=tu.get("input", {}), call_id=call_id_or_none(tu.get("toolUseId"))))
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


def _messages(source: Any) -> list[Any]:
    if isinstance(getattr(source, "messages", None), list):  # an Agent
        return source.messages
    state = getattr(source, "state", None)
    if state is None:
        return []
    if isinstance(getattr(state, "messages", None), list):
        return state.messages
    raise TypeError(
        "from_strands() reads the conversation from the Agent, and a Strands AgentResult does not carry it: "
        "pass the agent after the call, as in agent(prompt); from_strands(agent)"
    )


def _usage(metrics: Any) -> dict[str, Any]:
    if isinstance(metrics, dict):
        return metrics
    usage = getattr(metrics, "accumulated_usage", None)  # EventLoopMetrics
    return usage if isinstance(usage, dict) else {}


def from_strands(source: Any) -> ExecutionResult:
    """Build an ``ExecutionResult`` from a Strands Agents ``Agent`` after a run.

    Args:
        source: The ``Agent`` the run used. Its ``messages`` hold the conversation; the ``AgentResult`` the call returned does not, and raises ``TypeError``.
    """
    # Walk messages: each assistant message is one step; tool results
    # that appear after it (in the next user message) are attached to
    # that step's tool_results.
    messages = _messages(source)

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
                new_results = prev.tool_results + tool_results
                prev_meta = list(prev.tool_results_meta or [])
                for tr in tool_results:
                    status = tr.get("status") if isinstance(tr, dict) else None
                    if status == "error":
                        prev_meta.append({"is_error": True})
                    elif status == "success":
                        prev_meta.append({"is_error": False})
                    else:
                        prev_meta.append({})
                steps[-1] = replace(prev, tool_results=new_results, tool_results_meta=prev_meta)
            pending_input_context = {"messages": [msg]}

    # Token usage — Strands provides aggregate metrics at the top level,
    # not per-step.  Attach it to the overall ExecutionResult.
    token_usage: TokenUsage | None = aggregate_token_usage(steps)
    metrics = getattr(source, "event_loop_metrics", None) or getattr(source, "metrics", None)
    if token_usage is None and metrics:
        usage = _usage(metrics)
        it = usage.get("inputTokens", 0)
        ot = usage.get("outputTokens", 0)
        if it or ot:
            token_usage = TokenUsage(input_tokens=it, output_tokens=ot)

    # Final output — the last text block of result.message, or of the agent's last assistant message.
    final_output: str | None = None
    message = getattr(source, "message", None)
    if not isinstance(message, dict):
        message = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "assistant"), None)
    if message:
        mc = message.get("content")
        if isinstance(mc, list):
            final_output = _extract_text(mc)

    return ExecutionResult(
        steps=steps,
        token_usage=token_usage,
        final_output=final_output,
    )
