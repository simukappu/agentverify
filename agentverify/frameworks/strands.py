"""Strands Agents adapter for agentverify.

Converts a Strands Agents ``AgentResult`` into an agentverify
``ExecutionResult`` without requiring a custom converter function.

Usage::

    from agentverify.adapters.strands import from_strands

    result = agent("Analyze the files")
    execution_result = from_strands(result)

The Strands AgentResult has the following structure:

- ``result.state.messages``: Conversation history (list of message dicts).
  Each message's ``content`` is a list of blocks; blocks with a
  ``"toolUse"`` key contain ``{"name": ..., "input": ...}``.
- ``result.metrics``: Token consumption dict with ``"inputTokens"`` and
  ``"outputTokens"`` keys (may be absent during mock execution).
- ``result.message``: Final response message dict whose ``"content"``
  list may contain ``{"text": "..."}`` blocks.

Conversion mapping::

    state.messages[*].content[*].toolUse.name   -> tool_calls[*].name
    state.messages[*].content[*].toolUse.input  -> tool_calls[*].arguments
    metrics.inputTokens                         -> token_usage.input_tokens
    metrics.outputTokens                        -> token_usage.output_tokens
    message.content[*].text                     -> final_output
"""

from __future__ import annotations

from typing import Any

from agentverify.models import ExecutionResult, TokenUsage, ToolCall


def from_strands(result: Any) -> ExecutionResult:
    """Build an ``ExecutionResult`` from a Strands Agents ``AgentResult``.

    Args:
        result: A Strands Agents SDK ``AgentResult`` object.

    Returns:
        An ``ExecutionResult`` containing tool calls, token usage,
        and final output text extracted from the Strands result.
    """
    # --- Extract tool calls ---
    tool_calls: list[ToolCall] = []

    for message in result.state.messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue

        for block in content:
            if isinstance(block, dict) and "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        name=tool_use["name"],
                        arguments=tool_use.get("input", {}),
                    )
                )

    # --- Extract token usage ---
    token_usage = None
    if hasattr(result, "metrics") and result.metrics:
        metrics = result.metrics
        token_usage = TokenUsage(
            input_tokens=metrics.get("inputTokens", 0),
            output_tokens=metrics.get("outputTokens", 0),
        )

    # --- Extract final output text ---
    final_output = None
    if hasattr(result, "message") and result.message:
        message_content = (
            result.message.get("content")
            if isinstance(result.message, dict)
            else None
        )
        if isinstance(message_content, list):
            for block in message_content:
                if isinstance(block, dict) and "text" in block:
                    final_output = block["text"]

    return ExecutionResult(
        tool_calls=tool_calls,
        token_usage=token_usage,
        final_output=final_output,
    )
