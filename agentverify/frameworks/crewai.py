"""CrewAI adapter for agentverify.

Converts a CrewAI ``CrewOutput`` (from ``crew.kickoff()``) into an agentverify ``ExecutionResult``.

Step boundary: **each assistant message in a task's ``messages`` becomes one step** (``source="llm"``), with the task's agent role as the agent. With native function calling, the ``role: "tool"`` messages that follow become the step's ``tool_results``, matched to their call by ``tool_call_id``; on the ReAct path (a model without function calling), the step's ``Action`` / ``Action Input`` / ``Observation`` lines are the tool call and its result. A call recorded in the task's ``tool_failures`` is an error. When a task's messages end without the answer (the ReAct path does not keep it), the task's output is a last step. Token usage is the crew's total; CrewAI does not report it per call.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from agentverify._step_builder import classify_tool_result_error
from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall, handoffs_from_agents


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {"input": raw}
    return dict(raw) if isinstance(raw, dict) else {}


_REACT = re.compile(r"Action:\s*(?P<name>.+?)\s*\nAction Input:\s*(?P<input>.*?)\s*(?:\nObservation:\s*(?P<observation>.*))?$", re.S)


def _get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def from_crewai(crew_output: Any) -> ExecutionResult:
    """Build an ``ExecutionResult`` from a CrewAI ``CrewOutput``.

    Args:
        crew_output: Return value of ``crew.kickoff()``.
    """
    steps: list[Step] = []
    for task in getattr(crew_output, "tasks_output", None) or []:
        agent = getattr(task, "agent", None)
        call_ids: dict[str, ToolCall] = {}
        task_start = len(steps)
        failures = [(_get(f, "tool_name"), _arguments(_get(f, "tool_args"))) for f in getattr(task, "tool_failures", None) or []]
        for message in getattr(task, "messages", None) or []:
            role = _get(message, "role")
            if role == "assistant":
                calls = []
                for raw_call in _get(message, "tool_calls") or []:
                    call = ToolCall(_get(_get(raw_call, "function"), "name") or "", _arguments(_get(_get(raw_call, "function"), "arguments")))
                    calls.append(call)
                    if _get(raw_call, "id"):
                        call_ids[_get(raw_call, "id")] = call
                content = _get(message, "content")
                results: list[Any] = []
                meta: list[dict[str, Any]] = []
                react = _REACT.search(content) if isinstance(content, str) and not calls else None
                if react is not None:
                    call = ToolCall(react["name"].strip(), _arguments(react["input"]))
                    calls = [call]
                    if react["observation"] is not None:
                        observation = react["observation"].strip()
                        error = True if (call.name, call.arguments) in failures else classify_tool_result_error(observation)
                        results, meta = [observation], [{} if error is None else {"is_error": error}]
                steps.append(
                    Step(
                        index=len(steps),
                        source="llm",
                        tool_calls=calls,
                        tool_results=results,
                        tool_results_meta=meta or None,
                        output=content if isinstance(content, str) and content and react is None else None,
                        agent=agent if isinstance(agent, str) else None,
                    )
                )
            elif role == "tool" and steps:
                previous = steps[-1]
                content = _get(message, "content")
                name = _get(message, "name")
                call = call_ids.get(_get(message, "tool_call_id") or "") or next((c for c in previous.tool_calls if c.name == name), None)
                failed = call is not None and (name, call.arguments) in failures
                error = True if failed else classify_tool_result_error(content)
                steps[-1] = replace(
                    previous,
                    tool_results=previous.tool_results + [content],
                    tool_results_meta=(previous.tool_results_meta or []) + [{} if error is None else {"is_error": error}],
                )
        raw_task = getattr(task, "raw", None)
        if isinstance(raw_task, str) and raw_task and not any(s.output for s in steps[task_start:]):
            steps.append(Step(index=len(steps), source="llm", output=raw_task, agent=agent if isinstance(agent, str) else None))
    usage = getattr(crew_output, "token_usage", None)
    token_usage = None
    if usage is not None and (getattr(usage, "prompt_tokens", 0) or getattr(usage, "completion_tokens", 0)):
        token_usage = TokenUsage(usage.prompt_tokens or 0, usage.completion_tokens or 0)
    raw = getattr(crew_output, "raw", None)
    return ExecutionResult(
        steps=steps,
        handoffs=handoffs_from_agents(steps),
        token_usage=token_usage,
        final_output=raw if isinstance(raw, str) else None,
    )
