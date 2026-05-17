"""AgentCore Evaluations Custom code-based evaluator — LangGraph multi-agent supervisor.

Enforces the execution model C benchmark assertion for the LangGraph subject:

  - Tool ordering: transfer_to_research_expert -> at least one web_search -> transfer_to_math_expert -> at least one add.
  - Argument matching: at least one web_search query mentions FAANG / headcount / employees.
  - Cross-step running-total flow: the second add step's ``a`` argument equals the first add step's result (= a + b of the first add).

Input (event):

  {
    "evaluationInput": {
      "sessionSpans": [ <OpenTelemetry span dicts> ]
    },
    "evaluationTarget": { "traceIds": [...] }  // optional
  }

The synthesised payload (see ``../../../agentcore_test.py``) packs each step's tool calls into ``span.attributes["agentverify.tool_calls"]`` as a JSON-encoded string. The AgentCore Evaluations ``Evaluate`` API enforces OTLP schema strictly and silently drops content placed on ``span_events.body``, but ``span.attributes`` survive intact.

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

    if not tool_calls:
        return _fail("No tool calls found in trajectory")

    # --- benchmark assertion (LOC counted: this block) -------------------
    # Tool ordering: the load-bearing sequence appears in order.
    expected_sequence = [
        "transfer_to_research_expert",
        "web_search",
        "transfer_to_math_expert",
        "add",
    ]
    actual_names = [tc.get("name", "") for tc in tool_calls]
    cursor = 0
    for expected in expected_sequence:
        try:
            cursor = actual_names.index(expected, cursor) + 1
        except ValueError:
            return _fail(
                f"Tool sequence missing {expected!r} after position {cursor}; "
                f"actual tool names: {actual_names}"
            )

    # Argument matching: at least one web_search query mentions FAANG / headcount / employees.
    web_search_queries = [
        (tc.get("arguments") or {}).get("query", "")
        for tc in tool_calls
        if tc.get("name") == "web_search"
    ]
    if not any(
        re.search(r"(?i)FAANG|headcount|employees", q) for q in web_search_queries
    ):
        return _fail(
            f"No web_search query mentioned FAANG/headcount/employees: "
            f"{web_search_queries}"
        )

    # Cross-step running-total flow: the second add step's `a` argument equals the first add step's result (a + b of the first add).
    add_calls = [tc for tc in tool_calls if tc.get("name") == "add"]
    if len(add_calls) < 2:
        return _fail(
            f"Expected at least 2 add calls; found {len(add_calls)}"
        )
    first = add_calls[0].get("arguments") or {}
    second = add_calls[1].get("arguments") or {}
    try:
        expected_running_total = first["a"] + first["b"]
    except (KeyError, TypeError) as exc:
        return _fail(f"First add call missing a/b numeric arguments: {first} ({exc})")
    if second.get("a") != expected_running_total:
        return _fail(
            f"Second add's `a` ({second.get('a')!r}) is not the running total "
            f"of the first add ({first.get('a')!r} + {first.get('b')!r} = "
            f"{expected_running_total})"
        )
    # --- end benchmark assertion ----------------------------------------

    return {
        "label": "Pass",
        "value": 1.0,
        "explanation": (
            "Trajectory satisfies tool ordering, argument matching, and "
            "running-total flow on the LangGraph supervisor benchmark."
        ),
    }


def _collect_tool_calls(spans: list[dict]) -> list[dict]:
    """Walk session spans and return a flat ordered list of tool calls.

    See lambda_function in the strands-weather-forecaster subject for the rationale: tool calls are carried under ``span.attributes["agentverify.tool_calls"]`` because the AgentCore Evaluations ``Evaluate`` API drops ``span_events.body`` content during deserialisation.
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
