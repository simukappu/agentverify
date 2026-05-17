"""Execution model C: AgentCore Evaluations Custom code-based evaluator (Lambda) — LangGraph subject.

Drives the LangGraph multi-agent supervisor against OpenAI end-to-end, observes the resulting tool calls via the message history, packs them into an OpenTelemetry-shaped span, and calls the AgentCore Evaluations data-plane ``Evaluate`` API against the deployed evaluator.

Both the dev and CI scenarios use the same code path: AgentCore Evaluations is positioned around production-recorded sessions, and there is no documented mechanism for skipping the agent runtime in CI use. The Scenario 2 cost in the results table is the same as Scenario 1 by construction.

Requires the CDK stack under ``agentcore_evaluator_cdk/`` to be deployed first.
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

from _langgraph_agent_factory import USER_PROMPT, build_supervisor

CDK_OUTPUTS_PATH = Path(__file__).parent / "agentcore_evaluator_cdk" / "outputs.json"
STACK_NAME = "AgentverifyExecutionModelBenchLangGraph"
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


def _langgraph_messages_to_tool_calls(messages: list) -> list[dict]:
    """Walk LangGraph's full message history and produce JSON-serialisable tool-call dicts.

    Each entry has ``name`` / ``arguments`` / ``result``, matching the shape the Lambda's walker (``_collect_tool_calls``) expects.
    """
    pending: dict[str, dict] = {}
    out: list[dict] = []
    for msg in messages or []:
        ai_tool_calls = getattr(msg, "tool_calls", None)
        if ai_tool_calls:
            for tc in ai_tool_calls:
                tc_id = tc.get("id") or ""
                entry = {
                    "name": tc.get("name", ""),
                    "arguments": dict(tc.get("args", {}) or {}),
                    "result": None,
                }
                pending[tc_id] = entry
                out.append(entry)
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            entry = pending.get(tool_call_id)
            if entry is not None:
                content = getattr(msg, "content", None)
                entry["result"] = {"body_excerpt": str(content) if content is not None else ""}
    return out


def _build_session_spans(tool_calls: list[dict]) -> list[dict]:
    """Build an OpenTelemetry-shaped session-spans payload for the AgentCore Evaluations ``Evaluate`` API."""
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
            "agentverify.subject": "langgraph-multi-agent-supervisor",
            "agentverify.benchmark": "execution-model-trajectory",
            "session.id": session_id,
            "agentverify.tool_calls": json.dumps(tool_calls),
        },
        "span_events": [],
        "traceId": trace_id,
    }
    return [span]


def _run_and_assert():
    """Drive the supervisor, capture tool calls, send to AgentCore Evaluations via the ``Evaluate`` API, assert."""
    evaluator_id = _load_evaluator_id()

    app = build_supervisor()
    result = app.invoke({"messages": [{"role": "user", "content": USER_PROMPT}]})
    messages = result.get("messages", []) if isinstance(result, dict) else []
    tool_calls = _langgraph_messages_to_tool_calls(messages)
    session_spans = _build_session_spans(tool_calls)

    # --- benchmark assertion (LOC counted: this block + lambda_function.py) ---
    client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
    response = client.evaluate(
        evaluatorId=evaluator_id,
        evaluationInput={"sessionSpans": session_spans},
    )
    results = response.get("evaluationResults") or []
    assert results, f"Evaluate returned no results: {response}"
    assertion_result = results[0]
    assert assertion_result.get("label") == "Pass", (
        f"AgentCore Evaluations evaluator failed: label={assertion_result.get('label')!r} "
        f"value={assertion_result.get('value')!r} "
        f"explanation={assertion_result.get('explanation')!r}"
    )
    # --- end benchmark assertion ----------------------------------------


def test_c_langgraph_supervisor_dev():
    """Scenario 1 - dev / first run."""
    _run_and_assert()


def test_c_langgraph_supervisor_ci():
    """Scenario 2 - CI / repeat run.

    AgentCore Evaluations has no documented mechanism for skipping the agent runtime in CI; the assertion still requires a fresh session. Same code path as ``_dev``.
    """
    _run_and_assert()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
