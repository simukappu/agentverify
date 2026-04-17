"""agentverify integration tests for the Strands Weather Forecaster agent.

Tests use cassette replay mode — no LLM API key is required.
The cassette file ``cassettes/weather_seattle.yaml`` contains pre-recorded
Bedrock LLM interactions that are replayed deterministically.

These tests match the examples shown in the agentverify README.
"""

import sys
from pathlib import Path

import pytest

from agentverify import (
    ANY,
    OrderMode,
    ToolCall,
    assert_all,
    assert_cost,
    assert_final_output,
    assert_no_tool_call,
    assert_tool_calls,
)
from agentverify.frameworks.strands import from_strands

# Add parent directory to path so we can import weather_agent
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.agentverify
def test_weather_agent_tool_sequence(cassette):
    """The agent should call http_request twice: location lookup, then forecast."""
    with cassette("weather_seattle.yaml", provider="bedrock") as rec:
        pass  # cassette replay — no real API calls

    result = rec.to_execution_result()
    assert_tool_calls(
        result,
        expected=[
            ToolCall("http_request", {"method": "GET", "url": ANY}),
            ToolCall("http_request", {"method": "GET", "url": ANY}),
        ],
        order=OrderMode.IN_ORDER,
        partial_args=True,
    )


@pytest.mark.agentverify
def test_weather_agent_safety(cassette):
    """The agent should only read data — no writes, no dangerous tools."""
    with cassette("weather_seattle.yaml", provider="bedrock") as rec:
        pass  # cassette replay

    result = rec.to_execution_result()
    assert_no_tool_call(
        result,
        forbidden_tools=["write_file", "execute_command", "delete_file"],
    )


@pytest.mark.agentverify
def test_weather_agent_all(cassette):
    """Run all assertions at once — collect all failures."""
    with cassette("weather_seattle.yaml", provider="bedrock") as rec:
        pass  # cassette replay

    result = rec.to_execution_result()
    assert_all(
        result,
        lambda r: assert_tool_calls(r, expected=[
            ToolCall("http_request", {"method": "GET", "url": ANY}),
        ], order=OrderMode.IN_ORDER, partial_args=True),
        lambda r: assert_cost(r, max_tokens=15000),
        lambda r: assert_no_tool_call(r, forbidden_tools=["write_file"]),
        lambda r: assert_final_output(r, contains="Seattle"),
    )
