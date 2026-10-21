"""Pydantic AI adapter for agentverify.

Converts a Pydantic AI run result (``AgentRunResult`` from ``agent.run_sync()`` / ``await agent.run()``) into an agentverify ``ExecutionResult``.

Step boundary: **each ``ModelResponse`` of the run (``new_messages()``) becomes one step** (``source="llm"``); messages from an earlier run passed as ``message_history`` are left out. The ``ToolReturnPart`` and ``RetryPromptPart`` parts of the ``ModelRequest`` that follows become the step's ``tool_results``; a retry prompt and a tool return whose ``outcome`` is not ``"success"`` are errors. Calls of the output tool that returns structured output (``final_result``) are not tool calls.
"""

from __future__ import annotations

import json
from typing import Any

from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall


def _arguments(part: Any) -> dict[str, Any]:
    as_dict = getattr(part, "args_as_dict", None)
    if callable(as_dict):
        try:
            return dict(as_dict())
        except Exception:
            pass
    args = getattr(part, "args", None)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {"input": args}
    return dict(args) if isinstance(args, dict) else {}


def _usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    inputs, outputs = getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
    if not inputs and not outputs:
        return None
    return TokenUsage(input_tokens=inputs or 0, output_tokens=outputs or 0)


def _output_tool(name: Any) -> bool:
    return isinstance(name, str) and name.startswith("final_result")


def from_pydantic_ai(result: Any, agent_name: str | None = None) -> ExecutionResult:
    """Build an ``ExecutionResult`` from a Pydantic AI run result.

    Args:
        result: The run result, or a list of its messages (``all_messages()`` to include earlier runs).
        agent_name: The agent's name for ``Step.agent``; the messages do not carry it.
    """
    messages = result.new_messages() if callable(getattr(result, "new_messages", None)) else list(result or [])
    steps: list[Step] = []
    for message in messages:
        kind = getattr(message, "kind", None)
        parts = getattr(message, "parts", None) or []
        if kind == "response":
            tool_calls = [
                ToolCall(p.tool_name, _arguments(p))
                for p in parts
                if getattr(p, "part_kind", None) == "tool-call" and not _output_tool(p.tool_name)
            ]
            texts = [p.content for p in parts if getattr(p, "part_kind", None) == "text" and isinstance(p.content, str)]
            steps.append(
                Step(
                    index=len(steps),
                    source="llm",
                    tool_calls=tool_calls,
                    output="".join(texts) if texts else None,
                    token_usage=_usage(message),
                    agent=agent_name,
                )
            )
        elif kind == "request" and steps:
            results, meta = [], []
            for part in parts:
                part_kind = getattr(part, "part_kind", None)
                if _output_tool(getattr(part, "tool_name", None)):
                    continue
                if part_kind == "tool-return":
                    results.append(part.content)
                    outcome = getattr(part, "outcome", "success")
                    meta.append({"is_error": outcome != "success"})
                elif part_kind == "retry-prompt" and getattr(part, "tool_name", None):
                    results.append(part.content)
                    meta.append({"is_error": True})
            if results:
                previous = steps[-1]
                steps[-1] = Step(
                    index=previous.index,
                    source="llm",
                    tool_calls=previous.tool_calls,
                    tool_results=previous.tool_results + results,
                    tool_results_meta=(previous.tool_results_meta or []) + meta,
                    output=previous.output,
                    token_usage=previous.token_usage,
                    agent=previous.agent,
                )
    usages = [s.token_usage for s in steps if s.token_usage is not None]
    output = getattr(result, "output", None)
    if output is None:
        final_output = next((s.output for s in reversed(steps) if s.output), None)
    else:
        final_output = output if isinstance(output, str) else str(output)  # structured output, as from_openai_agents does
    return ExecutionResult(
        steps=steps,
        token_usage=TokenUsage(sum(u.input_tokens for u in usages), sum(u.output_tokens for u in usages)) if usages else None,
        final_output=final_output,
    )
