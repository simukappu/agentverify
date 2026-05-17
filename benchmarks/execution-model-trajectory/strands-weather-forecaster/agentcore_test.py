"""Execution model C: AgentCore Evaluations Custom code-based evaluator (Lambda) — Strands subject.

Drives the Strands weather agent against Bedrock end-to-end, observes the resulting tool calls via the framework's message history, packs them into an OpenTelemetry-shaped span, and calls the AgentCore Evaluations data-plane ``Evaluate`` API against the deployed evaluator.

Both the dev and CI scenarios use the same code path: AgentCore Evaluations is positioned around production-recorded sessions, and there is no documented mechanism for skipping the agent runtime in CI use. The Scenario 2 cost in the results table is the same as Scenario 1 by construction.

Requires the CDK stack under ``agentcore_evaluator_cdk/`` to be deployed first. Reads the evaluator id from ``agentcore_evaluator_cdk/outputs.json`` (written by ``deploy.sh``) or from the ``AGENTCORE_EVALUATOR_ID`` env var.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import boto3
import pytest

from _strands_agent_factory import USER_PROMPT, build_weather_agent

CDK_OUTPUTS_PATH = Path(__file__).parent / "agentcore_evaluator_cdk" / "outputs.json"
STACK_NAME = "AgentverifyExecutionModelBenchWeather"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def _load_evaluator_id() -> str:
    explicit = os.environ.get("AGENTCORE_EVALUATOR_ID")
    if explicit:
        return explicit
    if not CDK_OUTPUTS_PATH.exists():
        pytest.skip(
            f"CDK outputs file not found at {CDK_OUTPUTS_PATH}. "
            "Run agentcore_evaluator_cdk/deploy.sh first, "
            "or set AGENTCORE_EVALUATOR_ID."
        )
    with open(CDK_OUTPUTS_PATH) as f:
        outputs = json.load(f)
    stack_outputs = outputs.get(STACK_NAME) or next(iter(outputs.values()), {})
    eid = stack_outputs.get("EvaluatorId")
    if not eid:
        pytest.skip(f"EvaluatorId not present in {CDK_OUTPUTS_PATH}")
    return eid


def _strands_messages_to_tool_calls(messages: list) -> list[dict]:
    """Walk Strands ``Agent.messages`` and produce the JSON-serialisable tool-call list the Lambda expects.

    Each ``DeepEvalToolCall`` equivalent here is a plain dict with ``name`` / ``arguments`` / ``result``. The same shape is used in the synthesised payload from Scenario 2's previous design and in the Lambda's walker (``_collect_tool_calls``).
    """
    pending: dict[str, dict] = {}
    out: list[dict] = []
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if "toolUse" in block:
                tu = block["toolUse"] or {}
                tc = {
                    "name": tu.get("name", ""),
                    "arguments": dict(tu.get("input", {}) or {}),
                    "result": None,
                }
                pending[tu.get("toolUseId", "")] = tc
                out.append(tc)
            elif "toolResult" in block:
                tr = block["toolResult"] or {}
                tc = pending.get(tr.get("toolUseId", ""))
                if tc is not None:
                    tc["result"] = _flatten_strands_tool_result(tr.get("content"))
    return out


def _flatten_strands_tool_result(content) -> dict:
    """Flatten Strands' tool-result content blocks into a single dict the Lambda assertion can read."""
    text_parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
    return {"body_excerpt": "\n".join(text_parts)}


def _build_session_spans(tool_calls: list[dict]) -> list[dict]:
    """Build an OpenTelemetry-shaped session-spans payload for the AgentCore Evaluations ``Evaluate`` API.

    Tool calls are carried under ``span.attributes["agentverify.tool_calls"]`` as a JSON-encoded string. The AgentCore Evaluations ``Evaluate`` API validates the payload against the OTLP schema and silently drops content placed on ``span_events.body`` during deserialisation, while ``span.attributes`` survive intact, so attributes are the stable carrier for the benchmark's structured payload.
    """
    trace_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    now_ns = time.time_ns()
    span = {
        "name": "agentverify.benchmark.live_trajectory",
        "kind": "INTERNAL",
        "trace_id": trace_id,
        "span_id": span_id,
        "session_id": session_id,
        "scope": {"name": "agentverify.benchmark", "version": "0.1.0"},
        "start_time": now_ns,
        "end_time": now_ns + 1_000_000,
        "attributes": {
            "agentverify.subject": "strands-weather-forecaster",
            "agentverify.benchmark": "execution-model-trajectory",
            "session.id": session_id,
            "agentverify.tool_calls": json.dumps(tool_calls),
        },
        "span_events": [],
        "traceId": trace_id,
    }
    return [span]


def _run_and_assert():
    """Drive the agent, capture tool calls, send to AgentCore Evaluations via the ``Evaluate`` API, assert."""
    evaluator_id = _load_evaluator_id()

    agent = build_weather_agent()
    agent(USER_PROMPT)
    tool_calls = _strands_messages_to_tool_calls(getattr(agent, "messages", []) or [])
    session_spans = _build_session_spans(tool_calls)

    # --- benchmark assertion (LOC counted: this block + lambda_function.py) ---
    client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
    response = client.evaluate(
        evaluatorId=evaluator_id,
        evaluationInput={"sessionSpans": session_spans},
    )
    results = response.get("evaluationResults") or []
    assert results, f"Evaluate returned no results: {response}"
    result = results[0]
    assert result.get("label") == "Pass", (
        f"AgentCore Evaluations evaluator failed: label={result.get('label')!r} "
        f"value={result.get('value')!r} "
        f"explanation={result.get('explanation')!r}"
    )
    # --- end benchmark assertion ----------------------------------------


def test_c_strands_weather_dev():
    """Scenario 1 - dev / first run."""
    _run_and_assert()


def test_c_strands_weather_ci():
    """Scenario 2 - CI / repeat run.

    AgentCore Evaluations has no documented mechanism for skipping the agent runtime in CI; the assertion still requires a fresh session. Same code path as ``_dev``.
    """
    _run_and_assert()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
