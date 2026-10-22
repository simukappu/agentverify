"""agentverify tests for the Google ADK agent team example.

The cassette ``cassettes/agent_team.yaml`` holds one recorded run of the tutorial's three-turn conversation against Claude Haiku 5.5 on Amazon Bedrock: the LLM calls and the tool calls. These tests replay it with no AWS credentials and no network, and the tools do not run.

``from_google_adk`` builds the result from the runner's events, which name the agent of every step. The cassette alone infers the same route from ADK's ``transfer_to_agent`` calls (last test).
"""

import asyncio
import sys
from pathlib import Path

import pytest

from agentverify import (
    ANY,
    Handoff,
    OrderMode,
    ToolCall,
    assert_cost,
    assert_handoffs,
    assert_no_tool_errors,
    assert_step,
    assert_step_uses_result_from,
    assert_tool_calls,
    assert_tool_result_matches,
)
from agentverify.frameworks.google_adk import from_google_adk

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run_conversation  # noqa: E402

CASSETTE = "agent_team.yaml"

# Turn 1 goes to the greeting agent. Turn 2 starts with it (the last agent to answer), and it hands
# the weather question back to the root agent. Turn 3 goes to the farewell agent.
ROUTE = [
    Handoff("weather_agent_v2", "greeting_agent"),
    Handoff("greeting_agent", "weather_agent_v2"),
    Handoff("weather_agent_v2", "farewell_agent"),
]


def _replay(cassette_fixture):
    """Run the conversation inside the cassette; return the answers, the adapter's result, and the cassette's result."""

    async def main():
        with cassette_fixture(CASSETTE, provider="litellm") as rec:
            events, answers = await run_conversation()
        return answers, from_google_adk(events), rec.to_execution_result()

    return asyncio.run(main())


def _step_calling(result, tool_name):
    return next(s.index for s in result.steps if any(c.name == tool_name for c in s.tool_calls))


@pytest.mark.agentverify
def test_each_turn_uses_its_tool(cassette):
    _, result, _ = _replay(cassette)
    assert_tool_calls(
        result,
        expected=[ToolCall("say_hello"), ToolCall("get_weather", {"city": "New York"}), ToolCall("say_goodbye")],
        order=OrderMode.IN_ORDER,
        partial_args=True,
    )


@pytest.mark.agentverify
def test_delegation_route(cassette):
    _, result, _ = _replay(cassette)
    assert_handoffs(result, ROUTE)


@pytest.mark.agentverify
def test_each_tool_is_called_by_its_own_agent(cassette):
    _, result, _ = _replay(cassette)
    assert_step(result, step=_step_calling(result, "say_hello"), agent="greeting_agent", expected_tool=ToolCall("say_hello", ANY))
    assert_step(result, step=_step_calling(result, "get_weather"), agent="weather_agent_v2", expected_tool=ToolCall("get_weather", {"city": "New York"}))
    assert_step(result, step=_step_calling(result, "say_goodbye"), agent="farewell_agent", expected_tool=ToolCall("say_goodbye", ANY))


@pytest.mark.agentverify
def test_weather_answer_reads_the_tool_result(cassette):
    # The cassette's result, whose steps carry the requests sent to the model.
    answers, _, result = _replay(cassette)
    weather = _step_calling(result, "get_weather")
    assert_no_tool_errors(result)
    assert_tool_result_matches(result, step=weather, contains=["success", "25°C"])
    # The step after the tool call answers from its result, not from a guess.
    assert_step_uses_result_from(result, step=weather + 1, depends_on=weather, via="tool_result")
    assert "25°C" in answers[1]


@pytest.mark.agentverify
def test_token_budget(cassette):
    _, _, result = _replay(cassette)
    assert_cost(result, max_tokens=30_000)


@pytest.mark.agentverify
def test_the_cassette_alone_infers_the_same_route(cassette):
    _, adapter_result, cassette_result = _replay(cassette)
    # The cassette has no agent names, so the first agent is unnamed.
    assert_handoffs(cassette_result, [Handoff(ANY, "greeting_agent"), *ROUTE[1:]])
    assert [c.name for c in cassette_result.tool_calls] == [c.name for c in adapter_result.tool_calls]
