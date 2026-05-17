"""Bench-local agent factory for the Strands Weather Forecaster subject.

The example under ``examples/strands-weather-forecaster/weather_agent.py`` constructs ``BedrockModel(streaming=False)`` without pinning a model id, which means SDK-default drift could change which Bedrock model is invoked between bench runs. The benchmark needs the model id to be stable across runs (it is one of the variables that affects wall time), so the bench wraps agent construction here and pins the id explicitly.

The system prompt and tool list are imported from the example so the agent under test stays in sync with the example's recorded cassette.

The pinned model id matches the one recorded in ``examples/strands-weather-forecaster/tests/cassettes/weather_seattle.yaml`` so Scenario 2 cassette replay continues to round-trip cleanly.
"""

from __future__ import annotations

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands_tools import http_request

from weather_agent import WEATHER_SYSTEM_PROMPT  # noqa: E402

USER_PROMPT = "What's the weather in Seattle?"
MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


def build_weather_agent() -> Agent:
    """Construct the weather forecaster agent with a pinned Bedrock model id."""
    model = BedrockModel(model_id=MODEL_ID, streaming=False)
    return Agent(
        model=model,
        system_prompt=WEATHER_SYSTEM_PROMPT,
        tools=[http_request],
    )
