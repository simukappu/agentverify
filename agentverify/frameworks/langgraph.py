"""LangGraph adapter for agentverify.

Converts a LangGraph ``create_react_agent`` result into an agentverify
``ExecutionResult`` without requiring a custom converter function.

Usage::

    from agentverify.frameworks.langgraph import from_langgraph

    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
    execution_result = from_langgraph(result)

LangGraph's ``create_react_agent`` returns a dict with a ``"messages"`` key
containing a list of LangChain message objects:

- ``AIMessage`` with ``tool_calls`` attribute: list of
  ``{"name": str, "args": dict, "id": str}`` dicts.
- ``AIMessage`` with ``usage_metadata`` attribute:
  ``{"input_tokens": int, "output_tokens": int, ...}``.
- ``ToolMessage``: tool execution results (not used for assertions).
- ``HumanMessage``: user input (not used for assertions).

The last ``AIMessage`` without ``tool_calls`` provides the ``final_output``.

Conversion mapping::

    messages[AIMessage].tool_calls[*].name  -> tool_calls[*].name
    messages[AIMessage].tool_calls[*].args  -> tool_calls[*].arguments
    messages[AIMessage].usage_metadata      -> token_usage (summed)
    messages[AIMessage][-1].content         -> final_output
"""

from __future__ import annotations

from typing import Any

from agentverify.models import ExecutionResult, TokenUsage, ToolCall


def from_langgraph(result: dict[str, Any]) -> ExecutionResult:
    """Build an ``ExecutionResult`` from a LangGraph agent result.

    Args:
        result: Return value of ``agent.invoke()`` — a dict with a
            ``"messages"`` key containing LangChain message objects.

    Returns:
        An ``ExecutionResult`` containing tool calls, token usage,
        and final output text extracted from the LangGraph result.
    """
    messages = result.get("messages", [])

    # --- Extract tool calls and token usage ---
    tool_calls: list[ToolCall] = []
    total_input = 0
    total_output = 0
    has_token_usage = False
    final_output: str | None = None

    for msg in messages:
        # Use duck-typing: check for type attribute to identify AIMessage
        # without importing langchain_core.
        msg_type = getattr(msg, "type", None)
        if msg_type != "ai":
            continue

        # Extract tool calls from AIMessage.tool_calls
        msg_tool_calls = getattr(msg, "tool_calls", None)
        if msg_tool_calls:
            for tc in msg_tool_calls:
                name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append(ToolCall(name=name, arguments=args))

        # Accumulate token usage from AIMessage.usage_metadata
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            has_token_usage = True
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)

        # Track last AIMessage without tool_calls as final output
        if not msg_tool_calls:
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content:
                final_output = content

    token_usage = (
        TokenUsage(input_tokens=total_input, output_tokens=total_output)
        if has_token_usage
        else None
    )

    return ExecutionResult(
        tool_calls=tool_calls,
        token_usage=token_usage,
        final_output=final_output,
    )
