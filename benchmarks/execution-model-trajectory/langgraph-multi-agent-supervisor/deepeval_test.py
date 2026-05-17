"""Execution model B: DeepEval deterministic mode (`@observe` + ToolCorrectnessMetric) — LangGraph subject.

Drives the LangGraph multi-agent supervisor against OpenAI end-to-end, captures the trace via DeepEval's ``@observe`` decorator, and asserts via ``ToolCorrectnessMetric`` plus custom checks for argument matching and cross-step data flow.

Both the dev and CI scenarios use the same code path: DeepEval has no documented mechanism for skipping the LLM in CI runs (no trajectory cassette, no trace fixture). The Scenario 2 cost in the results table is the same as Scenario 1 by construction.

CI-secrets observation: ``ToolCorrectnessMetric.__init__`` requires ``OPENAI_API_KEY`` in the environment even in fully deterministic mode. The supervisor itself also needs a real OpenAI key, so for this subject the same secret slot covers both.
"""

from __future__ import annotations

import re
import sys

import pytest

from deepeval.metrics import ToolCorrectnessMetric  # noqa: E402
from deepeval.test_case import (  # noqa: E402
    LLMTestCase,
    ToolCall as DeepEvalToolCall,
)
from deepeval.tracing import observe  # noqa: E402

from _langgraph_agent_factory import USER_PROMPT, build_supervisor  # noqa: E402


def _expected_tools_minimal() -> list[DeepEvalToolCall]:
    """The benchmark assertion's tool-ordering expectation, expressed in DeepEval's flat-list shape.

    The supervisor handoff + research + math chain has many tool calls. The benchmark asserts the load-bearing sequence: transfer_to_research_expert -> at least one web_search -> transfer_to_math_expert -> at least one add. We use name-only expectations because LangGraph's exact tool call args are nondeterministic (handoffs carry no args, web searches use varied queries).
    """
    return [
        DeepEvalToolCall(name="transfer_to_research_expert"),
        DeepEvalToolCall(name="web_search"),
        DeepEvalToolCall(name="transfer_to_math_expert"),
        DeepEvalToolCall(name="add"),
    ]


@observe(type="agent", name="langgraph-multi-agent-supervisor")
def _run_supervisor(prompt: str):
    """Drive the LangGraph supervisor under DeepEval's ``@observe``.

    Returns the raw result dict so the caller can walk ``result["messages"]`` for tool-call history. DeepEval does not auto-instrument LangGraph's tool calls, so this translation is the realistic shape a DeepEval user writes for LangGraph.
    """
    app = build_supervisor()
    return app.invoke({"messages": [{"role": "user", "content": prompt}]})


def _langgraph_messages_to_deepeval_tools(messages: list) -> list[DeepEvalToolCall]:
    """Walk LangGraph's full message history and translate tool calls into DeepEval's ``ToolCall`` shape.

    LangGraph emits an ``AIMessage`` whose ``tool_calls`` attribute is a list of ``{name, args, id}`` dicts, followed by a ``ToolMessage`` whose ``content`` is the tool's output and whose ``tool_call_id`` matches the prior call. We pair them on id so each ``DeepEvalToolCall`` carries both ``input_parameters`` and ``output``.
    """
    pending: dict[str, DeepEvalToolCall] = {}
    out: list[DeepEvalToolCall] = []
    for msg in messages or []:
        # AIMessage carries tool_calls as a list of dicts.
        ai_tool_calls = getattr(msg, "tool_calls", None)
        if ai_tool_calls:
            for tc in ai_tool_calls:
                tc_id = tc.get("id") or ""
                deepeval_tc = DeepEvalToolCall(
                    name=tc.get("name", ""),
                    input_parameters=dict(tc.get("args", {}) or {}),
                )
                pending[tc_id] = deepeval_tc
                out.append(deepeval_tc)
        # ToolMessage carries the result.
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            tc = pending.get(tool_call_id)
            if tc is not None:
                tc.output = getattr(msg, "content", None)
    return out


def _final_text(result) -> str:
    """Best-effort extract of the supervisor's final answer."""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)
    if isinstance(content, str):
        return content
    return str(last)


def _assert_trajectory(tools_called: list[DeepEvalToolCall], final_output: str) -> None:
    """The benchmark assertion (LOC counted: this body)."""
    # --- benchmark assertion (LOC counted: this block) -------------------
    expected_tools = _expected_tools_minimal()
    metric = ToolCorrectnessMetric(
        should_consider_ordering=True,
        include_reason=False,
    )
    test_case = LLMTestCase(
        input=USER_PROMPT,
        actual_output=final_output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    metric.measure(test_case)
    assert metric.is_successful(), (
        f"ToolCorrectnessMetric failed: score={metric.score} "
        f"expected>={metric.threshold}"
    )

    # Argument matching: at least one web_search query mentions FAANG / headcount / employees.
    web_search_queries = [
        (tc.input_parameters or {}).get("query", "")
        for tc in tools_called
        if tc.name == "web_search"
    ]
    assert any(
        re.search(r"(?i)FAANG|headcount|employees", q) for q in web_search_queries
    ), f"No web_search query mentioned FAANG/headcount/employees: {web_search_queries}"

    # Cross-step data flow: the second add step's `a` argument equals the first add step's result. LangGraph returns the add result as a string in tool message content; we reconstruct the running total from the recorded a + b values.
    add_calls = [tc for tc in tools_called if tc.name == "add"]
    assert len(add_calls) >= 2, (
        f"Expected at least 2 add calls; found {len(add_calls)}"
    )
    first_add = add_calls[0].input_parameters or {}
    second_add = add_calls[1].input_parameters or {}
    expected_running_total = first_add["a"] + first_add["b"]
    assert second_add["a"] == expected_running_total, (
        f"Second add's `a` ({second_add['a']}) is not the running total "
        f"of the first add ({first_add['a']} + {first_add['b']} = "
        f"{expected_running_total})"
    )
    # --- end benchmark assertion ----------------------------------------


def _run_and_assert():
    result = _run_supervisor(USER_PROMPT)
    messages = result.get("messages", []) if isinstance(result, dict) else []
    tools_called = _langgraph_messages_to_deepeval_tools(messages)
    _assert_trajectory(tools_called, _final_text(result))


def test_b_langgraph_supervisor_dev():
    """Scenario 1 - dev / first run."""
    _run_and_assert()


def test_b_langgraph_supervisor_ci():
    """Scenario 2 - CI / repeat run.

    DeepEval has no documented mechanism for skipping the LLM in CI; the assertion still requires a fresh trace. Same code path as ``_dev``.
    """
    _run_and_assert()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
