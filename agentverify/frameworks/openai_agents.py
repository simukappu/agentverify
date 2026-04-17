"""OpenAI Agents SDK adapter for agentverify.

Converts an OpenAI Agents SDK ``RunResult`` into an agentverify
``ExecutionResult`` without requiring a custom converter function.

Usage::

    from agentverify.frameworks.openai_agents import from_openai_agents

    result = await Runner.run(agent, "What's the weather?")
    execution_result = from_openai_agents(result)

The OpenAI Agents SDK ``RunResult`` has the following structure:

- ``result.new_items``: List of ``RunItem`` objects generated during the run.
  ``ToolCallItem`` instances have a ``raw_item`` with ``name`` and
  ``arguments`` attributes (for ``ResponseFunctionToolCall``).
- ``result.final_output``: The agent's final text response (``str`` or
  structured output).
- ``result.context_wrapper.usage``: Token usage with ``input_tokens``,
  ``output_tokens``, and ``total_tokens`` attributes.

Conversion mapping::

    new_items[ToolCallItem].raw_item.name       -> tool_calls[*].name
    new_items[ToolCallItem].raw_item.arguments   -> tool_calls[*].arguments
    context_wrapper.usage.input_tokens           -> token_usage.input_tokens
    context_wrapper.usage.output_tokens          -> token_usage.output_tokens
    final_output                                 -> final_output
"""

from __future__ import annotations

import json
from typing import Any

from agentverify.models import ExecutionResult, TokenUsage, ToolCall


def from_openai_agents(result: Any) -> ExecutionResult:
    """Build an ``ExecutionResult`` from an OpenAI Agents SDK ``RunResult``.

    Args:
        result: A ``RunResult`` or ``RunResultStreaming`` from the
            OpenAI Agents SDK ``Runner.run()`` or ``Runner.run_sync()``.

    Returns:
        An ``ExecutionResult`` containing tool calls, token usage,
        and final output text extracted from the run result.
    """
    # --- Extract tool calls ---
    tool_calls: list[ToolCall] = []

    new_items = getattr(result, "new_items", []) or []
    for item in new_items:
        # Identify ToolCallItem by type attribute
        item_type = getattr(item, "type", None)
        if item_type != "tool_call_item":
            continue

        raw = getattr(item, "raw_item", None)
        if raw is None:
            continue

        # Extract name and arguments from raw_item
        # ResponseFunctionToolCall has .name and .arguments (JSON string)
        name = (
            raw.get("name", "") if isinstance(raw, dict) else getattr(raw, "name", "")
        )
        if not name:
            continue

        arguments_raw = (
            raw.get("arguments", "{}") if isinstance(raw, dict) else getattr(raw, "arguments", "{}")
        )

        # arguments may be a JSON string or already a dict
        if isinstance(arguments_raw, str):
            try:
                arguments = json.loads(arguments_raw)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        elif isinstance(arguments_raw, dict):
            arguments = arguments_raw
        else:
            arguments = {}

        tool_calls.append(ToolCall(name=name, arguments=arguments))

    # --- Extract token usage ---
    token_usage = None
    context_wrapper = getattr(result, "context_wrapper", None)
    usage = getattr(context_wrapper, "usage", None) if context_wrapper else None

    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        if input_tokens > 0 or output_tokens > 0:
            token_usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    # --- Extract final output ---
    final_output = getattr(result, "final_output", None)
    if final_output is not None and not isinstance(final_output, str):
        final_output = str(final_output)

    return ExecutionResult(
        tool_calls=tool_calls,
        token_usage=token_usage,
        final_output=final_output,
    )
