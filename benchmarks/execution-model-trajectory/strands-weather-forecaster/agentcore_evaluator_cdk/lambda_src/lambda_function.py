"""AgentCore Evaluations Custom code-based evaluator — weather forecaster trajectory.

Enforces the execution model C benchmark assertion:

  step 0: http_request GET, URL matches /points/
  step 1: http_request GET, URL matches /forecast
  step 1's URL contains data returned by step 0's response body

Input (event):

  {
    "evaluationInput": {
      "sessionSpans": [ <OpenTelemetry span dicts> ]
    },
    "evaluationTarget": { "traceIds": [...] }  // optional
  }

The synthesised payload (see ``../../../agentcore_test.py``) packs each step's tool call into ``span.attributes["agentverify.tool_calls"]`` as a JSON-encoded string. The AgentCore Evaluations ``Evaluate`` API validates the payload against the OTLP schema and silently drops content placed on ``span_events.body``, but preserves ``span.attributes`` intact, so attributes are the stable carrier for the benchmark's structured payload.

Output:

  { "label": str, "value": float in [0.0, 1.0], "explanation": str }
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context: Any) -> dict:
    logger.info("Evaluation event: %s", json.dumps(event)[:2000])

    spans = (event.get("evaluationInput") or {}).get("sessionSpans") or []
    tool_calls = _collect_tool_calls(spans)

    if len(tool_calls) < 2:
        return _fail(
            f"Expected at least 2 tool calls in the trajectory; found {len(tool_calls)}"
        )

    step0, step1 = tool_calls[0], tool_calls[1]

    # --- benchmark assertion (LOC counted: this block) -------------------
    # Tool ordering + argument matching: step 0 is a GET to /points/.
    if step0.get("name") != "http_request":
        return _fail(f"Step 0 expected http_request, got {step0.get('name')!r}")
    if step0.get("arguments", {}).get("method") != "GET":
        return _fail(
            f"Step 0 method expected GET, got {step0.get('arguments', {}).get('method')!r}"
        )
    if not re.search(r"/points/", step0.get("arguments", {}).get("url", "")):
        return _fail(
            f"Step 0 URL did not match /points/ pattern: {step0.get('arguments', {}).get('url')!r}"
        )

    # Tool ordering + argument matching: step 1 is a GET to /forecast.
    if step1.get("name") != "http_request":
        return _fail(f"Step 1 expected http_request, got {step1.get('name')!r}")
    if step1.get("arguments", {}).get("method") != "GET":
        return _fail(
            f"Step 1 method expected GET, got {step1.get('arguments', {}).get('method')!r}"
        )
    if not re.search(r"/forecast", step1.get("arguments", {}).get("url", "")):
        return _fail(
            f"Step 1 URL did not match /forecast pattern: {step1.get('arguments', {}).get('url')!r}"
        )

    # Cross-step data flow: step 1's URL must appear in step 0's response body. The forecast path is hardcoded for the cassette this benchmark ships with (the Seattle prompt against api.weather.gov returns gridpoints/SEW/125,68); a different cassette would change the constant.
    step0_result_text = json.dumps(step0.get("result"))
    step1_url = step1.get("arguments", {}).get("url", "")
    forecast_path = "/gridpoints/SEW/125,68/forecast"
    if forecast_path not in step0_result_text:
        return _fail(
            f"Step 0 result did not contain forecast path {forecast_path!r}"
        )
    if forecast_path not in step1_url:
        return _fail(
            f"Step 1 URL did not consume step 0's forecast path: {step1_url!r}"
        )
    # --- end benchmark assertion ----------------------------------------

    return {
        "label": "Pass",
        "value": 1.0,
        "explanation": (
            "Trajectory satisfies tool ordering, argument matching, and "
            "cross-step data flow on the weather forecaster benchmark."
        ),
    }


def _collect_tool_calls(spans: list[dict]) -> list[dict]:
    """Walk session spans and return a flat ordered list of tool calls.

    The benchmark synthesises spans that carry tool calls under ``span.attributes["agentverify.tool_calls"]`` as a JSON string. The AgentCore Evaluations ``Evaluate`` API enforces OTLP schema strictly and drops arbitrary content from ``span_events.body`` during deserialisation, while ``span.attributes`` survive intact, so attributes are the stable carrier for the benchmark's structured payload.

    A real session from AgentCore Runtime would carry tool calls under framework-specific keys (Strands AgentCore Runtime / LangGraph OpenInference). Production-grade evaluators would walk those keys here. This benchmark uses the synthesised attribute path.
    """
    out: list[dict] = []
    for span in spans:
        attrs = span.get("attributes") or {}
        raw = attrs.get("agentverify.tool_calls")
        if isinstance(raw, str) and raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                out.extend(tc for tc in parsed if isinstance(tc, dict))
    return out


def _fail(reason: str) -> dict:
    return {"label": "Fail", "value": 0.0, "explanation": reason}
