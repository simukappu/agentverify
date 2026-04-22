"""LangGraph adapter for agentverify.

Converts a LangGraph ``create_react_agent`` result into an agentverify
``ExecutionResult``.

Step boundary: **each ``AIMessage`` becomes one step**
(``source="llm"``).  ``ToolMessage`` objects that follow an ``AIMessage``
are attached to that step's ``tool_results`` until the next ``AIMessage``.
If the last ``AIMessage`` has no ``tool_calls``, its ``content`` becomes
the ``final_output`` and the step's ``output``.
"""

from __future__ import annotations

from typing import Any

from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall


def from_langgraph(result: dict[str, Any]) -> ExecutionResult:
    """Build an ``ExecutionResult`` from a LangGraph agent result.

    Args:
        result: Return value of ``agent.invoke()`` — a dict with a
            ``"messages"`` key containing LangChain message objects.
    """
    messages = result.get("messages", []) or []

    steps: list[Step] = []
    total_input = 0
    total_output = 0
    has_token_usage = False
    final_output: str | None = None

    for msg in messages:
        msg_type = getattr(msg, "type", None)

        if msg_type == "ai":
            # Extract tool calls from AIMessage.
            raw_tool_calls = getattr(msg, "tool_calls", None) or []
            tool_calls: list[ToolCall] = []
            for tc in raw_tool_calls:
                name = (
                    tc.get("name", "")
                    if isinstance(tc, dict)
                    else getattr(tc, "name", "")
                )
                args = (
                    tc.get("args", {})
                    if isinstance(tc, dict)
                    else getattr(tc, "args", {})
                )
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append(ToolCall(name=name, arguments=args))

            # Extract token usage.
            step_token_usage = None
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                has_token_usage = True
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
                step_token_usage = TokenUsage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                )

            # Text output — use AIMessage.content when it's a plain string.
            content = getattr(msg, "content", None)
            output = content if isinstance(content, str) and content else None

            # Track last AIMessage without tool_calls as final_output.
            if not raw_tool_calls and isinstance(content, str) and content:
                final_output = content

            steps.append(
                Step(
                    index=len(steps),
                    source="llm",
                    tool_calls=tool_calls,
                    output=output,
                    token_usage=step_token_usage,
                )
            )
        elif msg_type == "tool":
            # ToolMessage — attach to the previous AIMessage step.
            if steps:
                prev = steps[-1]
                tool_result = getattr(msg, "content", None)
                steps[-1] = Step(
                    index=prev.index,
                    name=prev.name,
                    source=prev.source,
                    tool_calls=prev.tool_calls,
                    tool_results=prev.tool_results + [tool_result],
                    output=prev.output,
                    duration_ms=prev.duration_ms,
                    token_usage=prev.token_usage,
                    input_context=prev.input_context,
                )

    token_usage = (
        TokenUsage(input_tokens=total_input, output_tokens=total_output)
        if has_token_usage
        else None
    )

    return ExecutionResult(
        steps=steps,
        token_usage=token_usage,
        final_output=final_output,
    )
