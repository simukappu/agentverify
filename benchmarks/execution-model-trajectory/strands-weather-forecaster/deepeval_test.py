"""Execution model B: DeepEval deterministic mode (`@observe` + ToolCorrectnessMetric) — Strands subject.

Drives the Strands weather agent against Bedrock end-to-end, captures the trace via DeepEval's ``@observe`` decorator, and asserts via ``ToolCorrectnessMetric`` plus a custom data-flow check.

Both the dev and CI scenarios use the same code path: DeepEval has no documented mechanism for skipping the LLM in CI runs (no trajectory cassette, no trace fixture). The Scenario 2 cost in the results table is the same as Scenario 1 by construction.

CI-secrets observation: ``ToolCorrectnessMetric.__init__`` unconditionally calls ``initialize_model()`` and refuses to construct without ``OPENAI_API_KEY`` in the environment, even in fully deterministic mode (no real OpenAI call is made; the deterministic scoring path does not invoke the model). The fake key is sufficient when the test environment cannot supply a real one, but the secret slot is still required.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-fake-deterministic-mode-only")

from deepeval.metrics import ToolCorrectnessMetric  # noqa: E402
from deepeval.test_case import (  # noqa: E402
    LLMTestCase,
    ToolCall as DeepEvalToolCall,
    ToolCallParams,
)
from deepeval.tracing import observe  # noqa: E402

from _strands_agent_factory import USER_PROMPT, build_weather_agent  # noqa: E402

EXPECTED_TOOLS = [
    DeepEvalToolCall(
        name="http_request",
        input_parameters={
            "method": "GET",
            "url": "https://api.weather.gov/points/47.6062,-122.3321",
        },
    ),
    DeepEvalToolCall(
        name="http_request",
        input_parameters={
            "method": "GET",
            "url": "https://api.weather.gov/gridpoints/SEW/125,68/forecast",
        },
    ),
]
# The forecast URL above embeds the NWS-side grid coordinates (``SEW/125,68``) for the Seattle prompt at recording time. ``ToolCorrectnessMetric`` does not expose a regex / pattern matcher for URL arguments, so the deterministic comparison has to use full-equality strings. If NWS reroutes the Seattle coordinates to a different grid in the future, this expected value must be re-recorded; agentverify's equivalent test (``agentverify_test.py``) uses ``MATCHES(r"/forecast")`` and is unaffected by the same drift.


@observe(type="agent", name="strands-weather-forecaster")
def _run_agent(prompt: str):
    """Drive the Strands agent under DeepEval's ``@observe`` so the run is traced.

    A real DeepEval user wraps their top-level agent entry point with ``@observe`` so DeepEval's tracer can attach. Inside the wrapper we still need to translate the framework-specific result shape into DeepEval's ``LLMTestCase`` (DeepEval does not auto-instrument Strands' tool calls). Returns the agent and its result so the caller can read ``agent.messages`` for tool-call history.
    """
    agent = build_weather_agent()
    result = agent(prompt)
    return agent, result


def _strands_messages_to_deepeval_tools(messages: list[Any]) -> list[DeepEvalToolCall]:
    """Walk Strands ``Agent.messages`` and translate to DeepEval's ``ToolCall`` shape.

    Strands records each tool invocation as a ``toolUse`` block on the assistant message and the corresponding result as a ``toolResult`` block on the next user message. We pair them on the tool-use id so each ``DeepEvalToolCall`` carries both ``input_parameters`` (from ``toolUse.input``) and ``output`` (from ``toolResult.content``).
    """
    pending: dict[str, DeepEvalToolCall] = {}
    out: list[DeepEvalToolCall] = []
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if "toolUse" in block:
                tu = block["toolUse"] or {}
                tc = DeepEvalToolCall(
                    name=tu.get("name", ""),
                    input_parameters=dict(tu.get("input", {}) or {}),
                )
                pending[tu.get("toolUseId", "")] = tc
                out.append(tc)
            elif "toolResult" in block:
                tr = block["toolResult"] or {}
                tc = pending.get(tr.get("toolUseId", ""))
                if tc is not None:
                    tc.output = tr.get("content")
    return out


def _final_text(result: Any) -> str:
    """Best-effort extract of the agent's final text response."""
    msg = getattr(result, "message", None) or {}
    if isinstance(msg, dict):
        content = msg.get("content") or []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                return block["text"]
    return str(result)


def _assert_trajectory(tools_called: list[DeepEvalToolCall], final_output: str) -> None:
    """The benchmark assertion (LOC counted: this body)."""
    # --- benchmark assertion (LOC counted: this block) -------------------
    metric = ToolCorrectnessMetric(
        should_consider_ordering=True,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        include_reason=False,
    )
    test_case = LLMTestCase(
        input=USER_PROMPT,
        actual_output=final_output,
        tools_called=tools_called,
        expected_tools=EXPECTED_TOOLS,
    )
    metric.measure(test_case)
    assert metric.is_successful(), (
        f"ToolCorrectnessMetric failed: score={metric.score} "
        f"expected>={metric.threshold}"
    )

    # Cross-step data flow: step 1's URL must contain a substring of step 0's response body. ``ToolCorrectnessMetric`` cannot express this; it is a custom deterministic check.
    assert len(tools_called) >= 2, f"Expected at least 2 tool calls; got {len(tools_called)}"
    step0_result_text = str(tools_called[0].output)
    step1_url = tools_called[1].input_parameters.get("url", "")
    forecast_path = "/gridpoints/SEW/125,68/forecast"
    assert forecast_path in step0_result_text, (
        f"Step 0 result did not contain forecast path {forecast_path!r}"
    )
    assert forecast_path in step1_url, (
        f"Step 1 URL did not consume step 0's forecast path: {step1_url!r}"
    )
    # --- end benchmark assertion ----------------------------------------


def _run_and_assert():
    agent, result = _run_agent(USER_PROMPT)
    tools_called = _strands_messages_to_deepeval_tools(getattr(agent, "messages", []) or [])
    _assert_trajectory(tools_called, _final_text(result))


def test_b_strands_weather_dev():
    """Scenario 1 - dev / first run.

    Drives the agent against the LLM, captures the trace, asserts.
    """
    _run_and_assert()


def test_b_strands_weather_ci():
    """Scenario 2 - CI / repeat run.

    DeepEval has no documented mechanism for skipping the LLM in CI; the assertion still requires a fresh trace. Same code path as ``_dev``.
    """
    _run_and_assert()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
