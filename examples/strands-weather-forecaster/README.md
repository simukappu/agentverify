# Strands Weather Forecaster Example

A two-step ReAct agent built with [Strands Agents](https://strandsagents.com/) that calls the US National Weather Service API via the built-in `http_request` tool: first to discover the forecast office for a location, then to fetch the forecast itself. Tested with [agentverify](https://github.com/simukappu/agentverify).

Adapted from the official [Strands Weather Forecaster sample](https://strandsagents.com/docs/examples/python/weather_forecaster/).

## Why this example?

Two LLM calls, one piece of data flowing between them: the simplest shape of a ReAct loop and the cleanest demonstration of:

- `assert_tool_calls` with `MATCHES(pattern)` for regex-based URL matching
- `assert_step` for per-step tool verification
- `assert_step_uses_result_from` to verify that step 1 actually consumes step 0's result

The main README's [Real-World Examples](../../README.md#real-world-examples) section walks through the tests in detail. This README covers setup and execution only.

## Prerequisites

- Python 3.10+
- AWS credentials configured for Amazon Bedrock (only for re-recording the cassette; not needed for replay)

## Setup

See the [main README](../../README.md) for `git clone` and venv setup. Then, from the repo root:

```bash
pip install -e ".[dev]"
pip install -e "examples/strands-weather-forecaster[dev]"
```

## Running the Agent

```bash
cd examples/strands-weather-forecaster

# AWS credentials (profile or env vars, whatever boto3 picks up)
python weather_agent.py
```

The agent prints the forecast for the configured city.

## Running Tests

### Cassette Replay (No AWS Credentials Required)

Tests ship with a pre-recorded Bedrock cassette under `tests/cassettes/`. Just run:

```bash
pytest examples/strands-weather-forecaster/tests
```

The cassette replays deterministically. No Bedrock API calls, zero cost.

### What the Tests Verify

See [Real-World Examples → Strands Weather Forecaster](../../README.md#strands-weather-forecaster--two-step-react) in the main README.

### Recording Mode (Real Bedrock API)

To re-record the cassette with real LLM calls:

1. Make sure AWS credentials are configured (`aws configure` or `AWS_PROFILE`).
2. Re-record from a single test so the cassette file ends up in a deterministic state:
   ```bash
   pytest examples/strands-weather-forecaster/tests -k test_weather_agent_tool_sequence --cassette-mode=record
   ```
3. Commit the updated cassette.

## Framework Adapter

This example uses agentverify's built-in Strands Agents adapter (`from_strands`) under the hood, invoked transparently by the `cassette` fixture. See the [main README's Framework Integration section](../../README.md#framework-integration) for details.
