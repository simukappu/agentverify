"""Execution model A: agentverify (inline / pytest / SDK patching) — Strands subject.

Asserts the canonical benchmark trajectory:

    step 0: http_request GET /points/...
    step 1: http_request GET /forecast (URL derived from step 0 result)

Scenario 1 (`_dev`) drives the agent against a real Bedrock model under ``cassette(mode="record", cassette_dir=tmp)``: the LLM is called, the cassette is captured into a per-run tmp directory (and discarded with the test), and the assertion runs against the recorded ``ExecutionResult``.

Scenario 2 (`_ci`) replays the canonical example cassette under ``examples/strands-weather-forecaster/tests/cassettes/`` deterministically, with no LLM call. This mirrors what a CI pipeline does on every PR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentverify import (
    MATCHES,
    ToolCall,
    assert_step,
    assert_step_uses_result_from,
)

from _strands_agent_factory import USER_PROMPT, build_weather_agent

EXAMPLE_CASSETTE_DIR = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "strands-weather-forecaster"
    / "tests"
    / "cassettes"
)
EXAMPLE_CASSETTE_NAME = "weather_seattle.yaml"


def _assert_trajectory(result):
    """The benchmark assertion (LOC counted: this body)."""
    # --- benchmark assertion (LOC counted: this block) -------------------
    assert_step(
        result,
        step=0,
        expected_tool=ToolCall(
            "http_request", {"method": "GET", "url": MATCHES(r"/points/")}
        ),
        partial_args=True,
    )
    assert_step(
        result,
        step=1,
        expected_tool=ToolCall(
            "http_request", {"method": "GET", "url": MATCHES(r"/forecast")}
        ),
        partial_args=True,
    )
    assert_step_uses_result_from(result, step=1, depends_on=0)
    # --- end benchmark assertion ----------------------------------------


@pytest.mark.agentverify
def test_a_strands_weather_dev(cassette, tmp_path):
    """Scenario 1 - dev / first run.

    Drives the Strands weather agent against Bedrock under cassette record mode. The cassette is written to ``tmp_path`` so each run is independent and no committed file is mutated.
    """
    agent = build_weather_agent()
    with cassette(
        "weather_seattle_bench.yaml",
        mode="record",
        provider="bedrock",
        cassette_dir=tmp_path,
    ) as rec:
        agent(USER_PROMPT)

    _assert_trajectory(rec.to_execution_result())


@pytest.mark.agentverify
def test_a_strands_weather_ci(cassette):
    """Scenario 2 - CI / repeat run.

    Replays the canonical example cassette deterministically. No LLM call, no AWS credentials needed.
    """
    with cassette(
        EXAMPLE_CASSETTE_NAME,
        mode="replay",
        provider="bedrock",
        cassette_dir=EXAMPLE_CASSETTE_DIR,
    ) as rec:
        pass

    _assert_trajectory(rec.to_execution_result())
