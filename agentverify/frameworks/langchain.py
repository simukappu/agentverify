"""LangChain adapter for agentverify.

Converts a LangChain ``AgentExecutor`` output dict into an agentverify
``ExecutionResult`` without requiring a custom converter function.

Usage::

    from agentverify.adapters.langchain import from_langchain

    result = agent_executor.invoke({"input": "Triage the issues"})
    execution_result = from_langchain(result)

The LangChain AgentExecutor returns the following structure:

- ``result["output"]``: Final response text.
- ``result["intermediate_steps"]``: List of ``(AgentAction, observation)``
  tuples recording the tool call history.

Token consumption is not included in the AgentExecutor output dict.
To capture token usage, pass the conversation history ``messages`` list
separately; ``AIMessage.usage_metadata`` will be aggregated.

Conversion mapping::

    intermediate_steps[*][0].tool        -> tool_calls[*].name
    intermediate_steps[*][0].tool_input  -> tool_calls[*].arguments
    AIMessage.usage_metadata.input_tokens  -> token_usage.input_tokens
    AIMessage.usage_metadata.output_tokens -> token_usage.output_tokens
    output                               -> final_output
"""

from __future__ import annotations

from typing import Any

from agentverify.models import ExecutionResult, TokenUsage, ToolCall


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

    Returns:
        An ``ExecutionResult`` containing tool calls, token usage,
        and final output text extracted from the LangChain result.
    """
    # --- Extract tool calls ---
    tool_calls: list[ToolCall] = []

    for action, _observation in result.get("intermediate_steps", []):
        arguments = (
            action.tool_input if isinstance(action.tool_input, dict) else {}
        )
        tool_calls.append(
            ToolCall(
                name=action.tool,
                arguments=arguments,
            )
        )

    # --- Extract token usage ---
    token_usage = None
    if messages:
        total_input = 0
        total_output = 0
        for msg in messages:
            # Only AIMessage instances carry usage_metadata.
            # Use duck-typing to avoid hard dependency on langchain_core.
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
        if total_input > 0 or total_output > 0:
            token_usage = TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
            )

    # --- Extract final output text ---
    return ExecutionResult(
        tool_calls=tool_calls,
        token_usage=token_usage,
        final_output=result.get("output"),
    )
