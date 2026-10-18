"""agentverify tests for the async OpenAI Agents SDK + LiteLLM example.

``Runner.run`` is async and ``LitellmModel`` calls ``litellm.acompletion``. The cassette ``cassettes/weather_tokyo.yaml`` holds one recorded run against Claude Haiku 5.5 on Amazon Bedrock, so these tests replay it with no AWS credentials and no network.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from agentverify import (
    ToolCall,
    assert_cost,
    assert_final_output,
    assert_no_tool_errors,
    assert_step,
    assert_step_uses_result_from,
    assert_tool_calls,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run_agent  # noqa: E402

CASSETTE = "weather_tokyo.yaml"


def _replay(cassette_fixture):
    """Run the async agent inside the cassette; the cassette is a plain context manager, so it wraps ``await`` directly."""

    async def main():
        with cassette_fixture(CASSETTE, provider="litellm") as rec:
            run_result = await run_agent()
        return run_result, rec.to_execution_result()

    return asyncio.run(main())


@pytest.mark.agentverify
def test_agent_calls_weather_tool_for_tokyo(cassette):
    _, result = _replay(cassette)
    assert_tool_calls(result, expected=[ToolCall("get_weather", {"city": "Tokyo"})])
    assert_step(result, step=0, expected_tool=ToolCall("get_weather", {"city": "Tokyo"}))
    assert_no_tool_errors(result)


@pytest.mark.agentverify
def test_answer_step_reads_the_tool_result(cassette):
    _, result = _replay(cassette)
    # Step 1's request carries step 0's tool result back to the model.
    assert_step_uses_result_from(result, step=1, depends_on=0, via="tool_result")


@pytest.mark.agentverify
def test_final_output_is_a_three_line_haiku(cassette):
    run_result, result = _replay(cassette)
    assert run_result.final_output == result.final_output
    assert_final_output(result, matches=r"^[^\n]+\n[^\n]+\n[^\n]+$")


@pytest.mark.agentverify
def test_token_budget(cassette):
    _, result = _replay(cassette)
    assert_cost(result, max_tokens=2000)
