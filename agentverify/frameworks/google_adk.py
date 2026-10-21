"""Google ADK adapter for agentverify.

Converts the events of a Google ADK run (``Runner.run()`` / ``run_async()``) into an agentverify ``ExecutionResult``.

Step boundary: **each event from the model (one with a ``function_call`` or text, authored by an agent) becomes one step** (``source="llm"``), with the event's ``author`` as the agent. Streamed ``partial`` events are skipped; the final event of each model turn carries the whole turn. A ``function_response`` joins the step whose call has its id (else the author's latest step). A change of author is a handoff, except between sub-agents of a ``ParallelAgent`` (different branches).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from agentverify._step_builder import classify_tool_result_error
from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall, handoffs_from_agents


def _parts(event: Any) -> list[Any]:
    content = getattr(event, "content", None)
    return list(getattr(content, "parts", None) or [])


def _parallel(a: str | None, b: str | None) -> bool:
    """True for branches of a ParallelAgent's sub-agents: neither is within the other."""
    if not a or not b or a == b:
        return False
    return not (a.startswith(b + ".") or b.startswith(a + "."))


def from_google_adk(events: Iterable[Any]) -> ExecutionResult:
    """Build an ``ExecutionResult`` from the events of a Google ADK run.

    Args:
        events: The events the runner yielded, in order (collect them from ``runner.run(...)`` or ``async for`` over ``runner.run_async(...)``).
    """
    steps: list[Step] = []
    branches: list[str | None] = []
    owner_by_call_id: dict[str, int] = {}
    for event in events:
        if getattr(event, "partial", False):
            continue
        parts = _parts(event)
        calls = [p.function_call for p in parts if getattr(p, "function_call", None) is not None]
        responses = [p.function_response for p in parts if getattr(p, "function_response", None) is not None]
        texts = [p.text for p in parts if isinstance(getattr(p, "text", None), str) and not getattr(p, "thought", False)]
        author = getattr(event, "author", None)
        if responses:
            for response in responses:
                index = owner_by_call_id.get(getattr(response, "id", None) or "")
                if index is None:
                    index = next((i for i in range(len(steps) - 1, -1, -1) if steps[i].agent == author), len(steps) - 1)
                if index < 0:
                    continue  # a response before any model call
                result = getattr(response, "response", None)
                error = classify_tool_result_error(result)
                step = steps[index]
                steps[index] = replace(
                    step,
                    tool_results=step.tool_results + [result],
                    tool_results_meta=(step.tool_results_meta or []) + [{} if error is None else {"is_error": error}],
                )
            continue
        if not calls and not texts:
            continue
        usage = getattr(event, "usage_metadata", None)
        token_usage = None
        if usage is not None and (getattr(usage, "prompt_token_count", None) or getattr(usage, "candidates_token_count", None)):
            token_usage = TokenUsage(usage.prompt_token_count or 0, usage.candidates_token_count or 0)
        for call in calls:
            if getattr(call, "id", None):
                owner_by_call_id[call.id] = len(steps)
        steps.append(
            Step(
                index=len(steps),
                source="llm",
                tool_calls=[ToolCall(c.name, dict(c.args or {})) for c in calls],
                output="".join(texts) if texts else None,
                token_usage=token_usage,
                agent=author if isinstance(author, str) and author != "user" else None,
            )
        )
        branches.append(getattr(event, "branch", None))
    usages = [s.token_usage for s in steps if s.token_usage is not None]
    handoffs = [
        h
        for h in handoffs_from_agents(steps)
        if not _parallel(branches[h.step], next(branches[k] for k in range(h.step + 1, len(steps)) if steps[k].agent == h.to_agent))
    ]
    return ExecutionResult(
        steps=steps,
        handoffs=handoffs,
        token_usage=TokenUsage(sum(u.input_tokens for u in usages), sum(u.output_tokens for u in usages)) if usages else None,
        final_output=next((s.output for s in reversed(steps) if s.output), None),
    )
