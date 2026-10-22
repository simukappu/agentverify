# Google ADK Agent Team Example

A multi-agent team built with [Google ADK](https://google.github.io/adk-docs/), tested with [agentverify](https://github.com/simukappu/agentverify). A weather agent answers weather questions with its `get_weather` tool, and hands greetings and farewells to two sub-agents, each with its own tool.

Adapted from Step 3 of the official ADK tutorial [Build Your First Intelligent Agent Team](https://google.github.io/adk-docs/tutorials/agent-team/). The tools, the agents' instructions, and the three-turn conversation ("Hello there!", "What is the weather in New York?", "Thanks, bye!") are the tutorial's.

## Why this example?

Which agent answers is decided by the model at run time, through ADK's `transfer_to_agent`. The recorded run shows why that is worth testing: the second turn starts with the greeting agent, the last one to answer, and it hands the weather question back to the weather agent. The tests check that route, that each tool is called by its own agent, and that the weather answer reads the tool's result.

The cassette records the tool calls along with the LLM calls, so replay runs no tools. ADK's `transfer_to_agent` still runs, since it moves the run to the next agent.

`adk eval` scores an agent against eval sets. These tests are complementary: they replay one recorded run in pytest, offline, on every PR.

## Prerequisites

- Python 3.10+
- AWS credentials with access to Claude Haiku 5.5 on Amazon Bedrock (only for re-recording the cassette; not needed for replay)

## Setup

See the [main README](../../README.md#examples) for setup instructions (`git clone`, venv, `pip install`).

## Running the Agent

```bash
cd examples/google-adk-agent-team

export AWS_REGION_NAME=us-east-1   # plus credentials, e.g. AWS_PROFILE
python agent.py
```

Set `LITELLM_MODEL` to use any other LiteLLM model string.

## Deviations from the tutorial

- The models are `LiteLlm(model="bedrock/global.anthropic.claude-haiku-5-5")` instead of Gemini. The tutorial's Step 2 uses the same `LiteLlm` wrapper for other providers.
- The tools do not print, and the team is built by `build_team()` and run by `run_conversation()`, which returns every event so tests can pass them to `from_google_adk`.
- The project depends on `mcp`: recent LiteLLM releases import it for any call with tools without declaring it.

## Running the Tests

```bash
pytest tests -v
```

The tests replay `tests/cassettes/agent_team.yaml`.

| Test | agentverify Assertion | Description |
|---|---|---|
| `test_each_turn_uses_its_tool` | `assert_tool_calls(order=IN_ORDER)` | `say_hello`, then `get_weather(city="New York")`, then `say_goodbye` |
| `test_delegation_route` | `assert_handoffs()` | weather agent → greeting agent → weather agent → farewell agent |
| `test_each_tool_is_called_by_its_own_agent` | `assert_step(agent=...)` | The greeting agent greets, the weather agent looks up the weather, the farewell agent says goodbye |
| `test_weather_answer_reads_the_tool_result` | `assert_no_tool_errors()` + `assert_tool_result_matches()` + `assert_step_uses_result_from()` | The lookup succeeded, and the next step's request carries its result back to the model |
| `test_token_budget` | `assert_cost()` | The conversation stays under 30k tokens |
| `test_the_cassette_alone_infers_the_same_route` | `assert_handoffs()` | Without the events, the cassette reads the same route from the `transfer_to_agent` calls; only the first agent is unnamed |

## Re-recording the Cassette

Every test shares the cassette, so record from a single test:

```bash
AWS_REGION_NAME=us-east-1 pytest tests --cassette-mode=record -k test_delegation_route
```

The model decides the route, so a new recording can take another one (for example, the weather agent answering the second turn directly). The tests then show what changed.
