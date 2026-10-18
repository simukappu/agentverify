# OpenAI Agents SDK on LiteLLM Example (Async)

An async agent built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) that reaches its model through [LiteLLM](https://docs.litellm.ai/), tested with [agentverify](https://github.com/simukappu/agentverify).

Adapted from the official [`model_providers/litellm_provider.py`](https://github.com/openai/openai-agents-python/blob/main/examples/model_providers/litellm_provider.py) sample: one `get_weather` tool, and an agent that answers only in haikus.

## Why this example?

The agent is async from top to bottom. `Runner.run` is a coroutine, and `LitellmModel` calls `litellm.acompletion`. The cassette is a plain context manager, so the test opens it inside a coroutine around `await run_agent()`. No async test plugin is needed.

## Prerequisites

- Python 3.10+
- AWS credentials with access to Claude Haiku 5.5 on Amazon Bedrock (only for re-recording the cassette; not needed for replay)

## Setup

See the [main README](../../README.md#examples) for setup instructions (`git clone`, venv, `pip install`).

## Running the Agent

```bash
cd examples/openai-agents-litellm-provider

export AWS_REGION_NAME=us-east-1   # plus credentials, e.g. AWS_PROFILE
python agent.py
```

Set `LITELLM_MODEL` to use any other LiteLLM model string.

## Deviations from the official sample

- The default model is `bedrock/global.anthropic.claude-haiku-5-5` instead of OpenRouter, so LiteLLM uses the AWS credential chain and no API key is passed.
- The agent is built by `build_agent()` and run by `run_agent()`, so tests can call them.

## Running the Tests

```bash
pytest tests -v
```

The tests replay `tests/cassettes/weather_tokyo.yaml` and check the tool call and its argument, that the answer step read the tool result, the haiku shape of the answer, and a token budget.

## Re-recording the Cassette

```bash
rm tests/cassettes/weather_tokyo.yaml
AWS_REGION_NAME=us-east-1 pytest tests --cassette-mode=record -k tool_for_tokyo
```
