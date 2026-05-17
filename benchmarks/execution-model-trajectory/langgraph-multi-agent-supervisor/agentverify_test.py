"""Execution model A: agentverify (inline / pytest / SDK patching) — LangGraph subject.

Asserts the canonical multi-agent supervisor benchmark trajectory:

    supervisor → research_expert (handoff)
    research step issues a web_search whose query mentions FAANG / headcount / employees
    supervisor → math_expert (handoff)
    second add step's `a` argument equals the first add step's result (running-total flow)

Scenario 1 (`_dev`) drives the supervisor against OpenAI under ``cassette(mode="record", cassette_dir=tmp)``: the LLM is called, the cassette is captured into a per-run tmp directory, and the assertion runs against the recorded ``ExecutionResult``.

Scenario 2 (`_ci`) replays the canonical example cassette under ``examples/langgraph-multi-agent-supervisor/tests/cassettes/`` deterministically, with no LLM call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentverify import (
    MATCHES,
    OrderMode,
    ToolCall,
    assert_step,
    assert_step_uses_result_from,
)
from agentverify.frameworks.langgraph import from_langgraph

from _langgraph_agent_factory import USER_PROMPT, build_supervisor

EXAMPLE_CASSETTE_DIR = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "langgraph-multi-agent-supervisor"
    / "tests"
    / "cassettes"
)
EXAMPLE_CASSETTE_NAME = "supervisor_faang.yaml"


def _assert_trajectory(result):
    """The benchmark assertion (LOC counted: this body)."""
    # --- benchmark assertion (LOC counted: this block) -------------------
    # Step 0: supervisor routes to research_expert.
    assert_step(
        result,
        step=0,
        expected_tool=ToolCall("transfer_to_research_expert", {}),
        partial_args=True,
    )

    # Research step: at least one web_search whose query mentions FAANG / headcount / employees.
    research_step_idx = next(
        (s.index for s in result.steps if any(tc.name == "web_search" for tc in s.tool_calls)),
        None,
    )
    assert research_step_idx is not None, "Expected a web_search step"
    assert_step(
        result,
        step=research_step_idx,
        expected_tools=[
            ToolCall(
                "web_search",
                {"query": MATCHES(r"(?i)FAANG|headcount|employees")},
            ),
        ],
        order=OrderMode.ANY_ORDER,
        partial_args=True,
    )

    # Math chain: at least 2 add steps; the second's `a` must consume the first's result (running-total flow).
    add_steps = [
        s.index for s in result.steps
        if any(tc.name == "add" for tc in s.tool_calls)
    ]
    assert len(add_steps) >= 2, (
        f"Expected at least 2 add steps for running-total flow; found {len(add_steps)}"
    )
    assert_step_uses_result_from(result, step=add_steps[1], depends_on=add_steps[0])
    # --- end benchmark assertion ----------------------------------------


@pytest.mark.agentverify
def test_a_langgraph_supervisor_dev(cassette, tmp_path):
    """Scenario 1 - dev / first run.

    Drives the LangGraph supervisor against OpenAI under cassette record mode. The cassette is written to ``tmp_path`` so each run is independent and no committed file is mutated.
    """
    with cassette(
        "supervisor_faang_bench.yaml",
        mode="record",
        provider="openai",
        cassette_dir=tmp_path,
    ) as rec:
        app = build_supervisor()
        app.invoke({"messages": [{"role": "user", "content": USER_PROMPT}]})

    _assert_trajectory(rec.to_execution_result())


@pytest.mark.agentverify
def test_a_langgraph_supervisor_ci(cassette):
    """Scenario 2 - CI / repeat run.

    Replays the canonical example cassette deterministically. No LLM call, no OpenAI credentials needed.
    """
    with cassette(
        EXAMPLE_CASSETTE_NAME,
        mode="replay",
        provider="openai",
        cassette_dir=EXAMPLE_CASSETTE_DIR,
    ) as rec:
        pass

    _assert_trajectory(rec.to_execution_result())
