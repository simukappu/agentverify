"""LangChain adapter for agentverify.

Converts a LangChain ``AgentExecutor`` output dict into an agentverify
``ExecutionResult``.

Step boundary: **each ``(AgentAction, observation)`` tuple in
``intermediate_steps`` becomes one step** (``source="llm"``).  The
observation is attached as the step's sole ``tool_results`` entry.
The final ``output`` of the executor is set on the
``ExecutionResult.final_output`` field (not on the last step).
"""

from __future__ import annotations

from typing import Any

from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall


def from_langchain(
    result: dict[str, Any],
    messages: list[Any] | None = None,
) -> ExecutionResult:
    """Build an ``ExecutionResult`` from a LangChain AgentExecutor output.

    Args:
        result: Return value of ``AgentExecutor.invoke()`` (dict).
        messages: LangChain conversation history message list (optional).
            Used to aggregate token consumption from
            ``AIMessage.usage_metadata``.
    """
    steps: list[Step] = []
    for i, (action, observation) in enumerate(
        result.get("intermediate_steps", [])
    ):
        arguments = (
            action.tool_input if isinstance(action.tool_input, dict) else {}
        )
        steps.append(
            Step(
                index=i,
                source="llm",
                tool_calls=[ToolCall(name=action.tool, arguments=arguments)],
                tool_results=[observation],
            )
        )

    # Token usage — aggregated from AIMessage.usage_metadata across
    # conversation history (not per-step).
    token_usage: TokenUsage | None = None
    if messages:
        total_input = 0
        total_output = 0
        for msg in messages:
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
        if total_input > 0 or total_output > 0:
            token_usage = TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
            )

    return ExecutionResult(
        steps=steps,
        token_usage=token_usage,
        final_output=result.get("output"),
    )
