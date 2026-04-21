# agentverify

[![PyPI](https://img.shields.io/pypi/v/agentverify)](https://pypi.org/project/agentverify/)
[![Downloads](https://img.shields.io/pepy/dt/agentverify)](https://pepy.tech/projects/agentverify)
[![CI](https://github.com/simukappu/agentverify/actions/workflows/ci.yml/badge.svg)](https://github.com/simukappu/agentverify/actions/workflows/ci.yml)
[![Coverage](https://coveralls.io/repos/github/simukappu/agentverify/badge.svg?branch=main)](https://coveralls.io/github/simukappu/agentverify?branch=main)
[![License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

**pytest for AI agents.** Assert tool calls, not vibes.

agentverify is a pytest plugin for deterministic testing of AI agent behavior. Record real LLM calls once, replay them in CI with zero cost, and assert exactly which tools were called, in what order, with what arguments — plus cost budgets and safety guardrails. Framework-agnostic, provider-agnostic, zero LLM cost in CI.

## Why agentverify?

Most AI testing tools evaluate what an LLM *says*. agentverify tests what an agent *does*.

When agents move from prototype to production, the questions change: did the agent call the right tools in the right order? Did it stay within budget? Did it avoid dangerous operations? These are deterministic properties you can assert in CI, the same way you test any other code.

Unlike HTTP-level recorders that capture raw network traffic, agentverify records at the LLM SDK level — capturing tool calls, token usage, and model responses as first-class objects you can assert against. And unlike eval frameworks that score output quality with LLM-as-judge, agentverify asserts deterministic properties: routing correctness, cost control, and safety boundaries.

agentverify brings that discipline to agent development. It works with any framework — [Strands Agents](https://github.com/strands-agents/sdk-python), [LangChain](https://github.com/langchain-ai/langchain), [CrewAI](https://github.com/crewAIInc/crewAI), or plain Python — and any LLM provider. Just build an `ExecutionResult` from your agent's output and write pytest assertions.

## Install

```bash
pip install agentverify
```

## Quick Start — No LLM Required

Copy this into `test_agent.py` and run `pytest`. No API keys, no cassettes — just pure assertions.

```python
from agentverify import (
    ExecutionResult, ToolCall, ANY,
    assert_tool_calls, assert_cost, assert_no_tool_call, assert_final_output,
)

# Build an ExecutionResult from your agent's output (or a dict)
result = ExecutionResult.from_dict({
    "tool_calls": [
        {"name": "get_location", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"lat": 35.6, "lon": 139.7}},
    ],
    "token_usage": {"input_tokens": 50, "output_tokens": 30},
    "total_cost_usd": 0.002,
    "final_output": "The weather in Tokyo is sunny, 22°C.",
})

def test_tool_sequence():
    assert_tool_calls(result, expected=[
        ToolCall("get_location", {"city": "Tokyo"}),
        ToolCall("get_weather", {"lat": ANY, "lon": ANY}),
    ])

def test_budget():
    assert_cost(result, max_tokens=500, max_cost_usd=0.01)

def test_safety():
    assert_no_tool_call(result, forbidden_tools=["delete_user", "drop_table"])

def test_output():
    assert_final_output(result, contains="Tokyo")
```

> **Using a real agent framework?** You don't need to build dicts by hand — use a built-in adapter instead. See [Framework Integration](#framework-integration) below.

```
$ pytest test_agent.py -v
test_agent.py::test_tool_sequence PASSED
test_agent.py::test_budget PASSED
test_agent.py::test_safety PASSED
test_agent.py::test_output PASSED
```

## Real-World Example — Testing the Strands Weather Forecaster

Here's how you'd test the [Strands Weather Forecaster](https://strandsagents.com/docs/examples/python/weather_forecaster/) — an official Strands Agents sample that uses the `http_request` tool to fetch weather data from the National Weather Service API.

```python
import pytest
from agentverify import (
    assert_tool_calls, assert_cost, assert_latency,
    assert_no_tool_call, assert_final_output, assert_all,
    ToolCall, MATCHES,
)

# Import the weather agent from the Strands sample
# (see https://strandsagents.com/docs/examples/python/weather_forecaster/)
from weather_agent import weather_agent


@pytest.mark.agentverify
def test_weather_agent(cassette):
    """Verify tool sequence, budget, latency, safety, and final output in one go."""
    with cassette("weather_seattle.yaml", provider="bedrock") as rec:
        weather_agent("What's the weather in Seattle?")

    result = rec.to_execution_result()
    assert_all(
        result,
        # Two http_request calls with distinct URL shapes — location lookup then forecast
        lambda r: assert_tool_calls(r, expected=[
            ToolCall("http_request", {"method": "GET", "url": MATCHES(r"/points/")}),
            ToolCall("http_request", {"method": "GET", "url": MATCHES(r"/forecast")}),
        ]),
        lambda r: assert_cost(r, max_tokens=15000),
        lambda r: assert_latency(r, max_ms=10_000),
        lambda r: assert_no_tool_call(r, forbidden_tools=["write_file", "execute_command"]),
        lambda r: assert_final_output(r, contains="Seattle"),
    )
```

Prefer a live LLM call over a cassette? Use the built-in Strands adapter — see [Framework Integration](#framework-integration).

## Build an ExecutionResult

Every agentverify assertion takes an `ExecutionResult`. You can build one three ways:

1. **From a dict** — convenient for quick tests and fixtures, as in the [Quick Start](#quick-start--no-llm-required).
2. **From a built-in adapter** — one-liner for Strands Agents, LangChain, LangGraph, OpenAI Agents SDK. See [Framework Integration](#framework-integration).
3. **From a custom converter** — for other frameworks, map your output to the schema below. See [`examples/strands-file-organizer/converter.py`](examples/strands-file-organizer/converter.py) and [`examples/langchain-issue-triage/converter.py`](examples/langchain-issue-triage/converter.py) for ~50-line reference implementations.

`ExecutionResult.from_dict()` accepts these keys:

| Key | Type | Description |
|---|---|---|
| `tool_calls` | `list[dict]` | Each dict has `name` (str, required), `arguments` (dict, optional), `result` (any, optional — stored for reference; not used in assertions) |
| `token_usage` | `dict` or `None` | `{"input_tokens": int, "output_tokens": int}` |
| `total_cost_usd` | `float` or `None` | Total cost in USD (must be set manually — not auto-calculated from tokens or populated from cassettes) |
| `final_output` | `str` or `None` | The agent's final text response |
| `duration_ms` | `float` or `None` | Wall-clock duration in milliseconds (auto-populated by the `cassette` fixture and `MockLLM`) |

`ExecutionResult.from_json(json_string)` and `to_dict()` / `to_json()` are also available for serialization.

## Record & Replay with Cassettes

Record real LLM API calls once. Replay them in CI forever — zero cost, deterministic.

The `cassette` fixture is provided by the agentverify pytest plugin. It creates an `LLMCassetteRecorder` that intercepts LLM SDK calls (not HTTP — it patches the SDK's chat completion method directly). Use `@pytest.mark.agentverify` to mark your test, and call your agent code inside the `with cassette(...)` block. After the block exits, call `rec.to_execution_result()` to build the result for assertions.

```python
import pytest
from agentverify import assert_tool_calls, ToolCall, ANY

@pytest.mark.agentverify
def test_weather_agent(cassette):
    with cassette("weather_agent.yaml", provider="openai") as rec:
        # Replace this with your actual agent invocation, e.g.:
        # agent.run("What's the weather in Tokyo?")
        run_my_agent("What's the weather in Tokyo?")

    # rec.to_execution_result() is called AFTER the with block exits
    result = rec.to_execution_result()
    assert_tool_calls(result, expected=[
        ToolCall("get_location", {"city": "Tokyo"}),
        ToolCall("get_weather", {"lat": ANY, "lon": ANY}),
    ])
```

Record the cassette once, then replay it forever:

```bash
# First run: record real LLM calls to cassette file
pytest --cassette-mode=record

# All subsequent runs: replay from cassette (zero cost, deterministic)
pytest
```

Cassettes are human-readable YAML (or JSON). Commit them to git, review in PRs.

**Cassette modes:**

| Mode | Behavior |
|---|---|
| `AUTO` (default) | If cassette file exists → REPLAY. Otherwise → call real LLM API but don't save (no cassette file is created). |
| `RECORD` | Always call real LLM API and save to cassette file. |
| `REPLAY` | Always replay from cassette file. Raises error if file is missing. |

To create a cassette, use `mode=CassetteMode.RECORD` explicitly or pass `--cassette-mode=record` on the command line. To re-record, simply run with `RECORD` again — the existing file is overwritten.

```bash
# Record cassettes for all tests
pytest --cassette-mode=record

# Replay cassettes (default behavior when cassette files exist)
pytest
```

### Request Matching (Stale Cassette Detection)

Cassette replay verifies request content by default. Each replay request is compared against the recorded request:
- **Model name**: must match exactly (empty model in either side skips the check)
- **Tool names**: the sorted list of tool names must match (empty tools in either side skips the check)

A `CassetteRequestMismatchError` is raised on mismatch with a clear message indicating which field differs and suggesting re-recording.

To disable matching (e.g., during migration):

```bash
# CLI: disable for all tests
pytest --no-cassette-match-requests
```

```python
# Per-test: disable in the cassette factory call
with cassette("my_test.yaml", provider="openai", match_requests=False) as rec:
    run_my_agent("What's the weather?")
```

### Cassette Sanitization

Cassette files are sanitized by default when recording. API keys, tokens, and other sensitive data are automatically redacted before saving to disk.

Built-in patterns cover:
- OpenAI API keys (`sk-...`)
- Anthropic API keys (`sk-ant-...`)
- AWS access keys (`AKIA...`)
- Bearer tokens (`Bearer ...`)

Add custom patterns:

```python
from agentverify import SanitizePattern

custom_patterns = [
    SanitizePattern(name="internal_token", pattern=r"tok-[a-f0-9]{32}", replacement="tok-***REDACTED***"),
]

with cassette("test.yaml", provider="openai", sanitize=custom_patterns) as rec:
    run_my_agent("Do something")
```

Disable sanitization (not recommended):

```python
with cassette("test.yaml", provider="openai", sanitize=False) as rec:
    run_my_agent("Do something")
```

**Other limitations:**

- `total_cost_usd` is not populated from cassettes. Use `assert_cost(max_tokens=...)` for cassette-based budget checks, or set `total_cost_usd` manually in your `ExecutionResult`.
- Cassette sanitization covers common API key patterns by default. Add custom `SanitizePattern` objects for application-specific secrets (internal tokens, database credentials, etc.).

## Assertion Modes

```python
from agentverify import assert_tool_calls, OrderMode, ToolCall, ANY, MATCHES

# Exact match — same tools, same order, same count (default)
assert_tool_calls(result, expected=[...])

# Subsequence — these tools appeared in this order (other calls in between are OK)
assert_tool_calls(result, expected=[...], order=OrderMode.IN_ORDER)

# Set membership — these tools were called (order doesn't matter)
assert_tool_calls(result, expected=[...], order=OrderMode.ANY_ORDER)

# Partial args — only check the keys you care about
assert_tool_calls(result, expected=[
    ToolCall("search", {"query": "Tokyo"}),
], partial_args=True)

# ANY wildcard — ignore a specific argument value
assert_tool_calls(result, expected=[
    ToolCall("get_weather", {"lat": ANY, "lon": ANY}),
])

# MATCHES regex — verify a string argument follows a pattern
# (re.search semantics — use ^/$ anchors for full match)
assert_tool_calls(result, expected=[
    ToolCall("http_request", {"method": "GET", "url": MATCHES(r"/points/")}),
    ToolCall("http_request", {"method": "GET", "url": MATCHES(r"/forecast")}),
])

# Collect all failures at once (doesn't stop at first)
from agentverify import assert_all
assert_all(
    result,
    lambda r: assert_tool_calls(r, expected=[...]),
    lambda r: assert_cost(r, max_tokens=1000),
    lambda r: assert_no_tool_call(r, forbidden_tools=["delete_user"]),
    lambda r: assert_final_output(r, contains="Tokyo"),
)
```

### Strict Cost Assertions

By default, `assert_cost()` silently passes when `token_usage` or `total_cost_usd` is `None` (e.g., during cassette replay where cost data may be unavailable). Use `strict=True` to require that the data is present:

```python
# Fails if token_usage is None, even if the budget would pass
assert_cost(result, max_tokens=500, strict=True)

# Fails if total_cost_usd is None
assert_cost(result, max_cost_usd=0.01, strict=True)
```

### Latency Assertions

`assert_latency()` enforces a response time SLA on your agent. When you use the `cassette` fixture, wall-clock duration is captured automatically on context exit and exposed as `ExecutionResult.duration_ms`:

```python
from agentverify import assert_latency

@pytest.mark.agentverify
def test_weather_agent_latency(cassette):
    with cassette("weather_seattle.yaml", provider="bedrock") as rec:
        weather_agent("What's the weather in Seattle?")

    result = rec.to_execution_result()
    # Fail if the whole agent run took more than 3 seconds
    assert_latency(result, max_ms=3000)
```

During cassette **replay**, the measured duration reflects replay time (typically milliseconds), not the original call time. Capture latency data during **record** mode and persist it alongside the cassette if you want to assert against real LLM response times, or set `duration_ms` manually on your `ExecutionResult`:

```python
result = ExecutionResult.from_dict({
    "tool_calls": [...],
    "duration_ms": 2450.0,
})
assert_latency(result, max_ms=3000)
```

Like `assert_cost()`, `assert_latency()` silently passes when `duration_ms` is `None`. Use `strict=True` to require the data:

```python
assert_latency(result, max_ms=3000, strict=True)
```

### Final Output Assertions

`assert_final_output()` verifies the agent's final text response. Use `contains` for substring checks, `equals` for exact match, or `matches` for regex:

```python
from agentverify import assert_final_output

# Substring check
assert_final_output(result, contains="Tokyo")

# Exact match
assert_final_output(result, equals="The weather in Tokyo is sunny, 22°C.")

# Regex match
assert_final_output(result, matches=r"\d+°C")
```

## Framework Integration

### Built-in Adapters

agentverify ships with built-in adapters for popular agent frameworks. No converter function needed — just import and call:

```python
# Strands Agents
from agentverify.frameworks.strands import from_strands

result = agent("Analyze the files")
execution_result = from_strands(result)

# LangChain
from agentverify.frameworks.langchain import from_langchain

result = agent_executor.invoke({"input": "Triage the issues"})
execution_result = from_langchain(result)

# LangGraph
from agentverify.frameworks.langgraph import from_langgraph

result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
execution_result = from_langgraph(result)

# OpenAI Agents SDK
from agentverify.frameworks.openai_agents import from_openai_agents

result = await Runner.run(agent, "What's the weather?")
execution_result = from_openai_agents(result)
```

| Framework | Adapter | Input |
|---|---|---|
| [Strands Agents](https://github.com/strands-agents/sdk-python) | `from agentverify.frameworks.strands import from_strands` | `AgentResult` |
| [LangChain](https://github.com/langchain-ai/langchain) | `from agentverify.frameworks.langchain import from_langchain` | `AgentExecutor.invoke()` dict |
| [LangGraph](https://github.com/langchain-ai/langgraph) | `from agentverify.frameworks.langgraph import from_langgraph` | `create_react_agent` result dict |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | `from agentverify.frameworks.openai_agents import from_openai_agents` | `RunResult` |

For LangChain, pass the conversation history `messages` list as a second argument to capture token usage:

```python
execution_result = from_langchain(result, messages=memory.chat_memory.messages)
```

### Custom Converters

For other frameworks, agentverify is framework-agnostic. Build an `ExecutionResult` from any agent framework's output using a converter function. The [`examples/`](examples/) directory includes reference converters:

| Framework | Converter | Description |
|---|---|---|
| [Strands Agents](https://github.com/strands-agents/sdk-python) | [`strands-file-organizer/converter.py`](examples/strands-file-organizer/converter.py) | Converts `AgentResult` → `ExecutionResult` |
| [LangChain](https://github.com/langchain-ai/langchain) | [`langchain-issue-triage/converter.py`](examples/langchain-issue-triage/converter.py) | Converts `AgentExecutor` output → `ExecutionResult` |

These converters are small (~50 lines) and easy to adapt for your own framework.

## Supported LLM Providers

| Provider | Extra |
|---|---|
| OpenAI | `pip install agentverify[openai]` |
| Amazon Bedrock | `pip install agentverify[bedrock]` |
| Google Gemini | `pip install agentverify[gemini]` |
| Anthropic | `pip install agentverify[anthropic]` |
| LiteLLM | `pip install agentverify[litellm]` |
| All providers | `pip install agentverify[all]` |

## Tool Mocking — Test Routing Without an LLM

`MockLLM` replays a list of predefined LLM responses that you define in code. No cassette file, no real API call. Useful when you want to test your agent's routing logic — does it call the right tools, in the right order, given a specific LLM response — without recording a cassette first.

```python
import pytest
from agentverify import (
    MockLLM, mock_response, ToolCall,
    assert_tool_calls, assert_no_tool_call,
)

def test_agent_routes_to_weather_tool():
    with MockLLM([
        mock_response(tool_calls=[("get_weather", {"city": "Tokyo"})]),
        mock_response(content="Tokyo is sunny, 22°C."),
    ], provider="openai") as rec:
        my_agent("What's the weather in Tokyo?")

    result = rec.to_execution_result()
    assert_tool_calls(result, expected=[ToolCall("get_weather", {"city": "Tokyo"})])
    assert_no_tool_call(result, forbidden_tools=["delete_user"])
```

`mock_response()` accepts:

- `content` — the final text message the LLM should return.
- `tool_calls` — a list of `(name, arguments)` tuples or `{"name": ..., "arguments": ...}` dicts.
- `input_tokens` / `output_tokens` — optional token usage to report.

Provide one `mock_response(...)` per LLM call your agent is expected to make. If your agent makes more calls than you've queued, `MockLLM` raises `CassetteMissingRequestError` so under-specified tests fail loudly.

### When to use MockLLM vs cassettes

| Use MockLLM when | Use cassettes when |
|---|---|
| You haven't recorded a cassette yet | You want fidelity with real LLM output |
| You want to test hypothetical scenarios (errors, edge cases) | You want to catch prompt-regression bugs |
| You're doing behavior-driven tests of routing | You're regression-testing full agent runs |
| You want zero dependency on any API key | You've already got recordings to replay |

## CI Integration

agentverify is designed for CI pipelines. Commit your cassette files to git and replay them in CI with zero LLM cost.

### GitHub Actions

```yaml
name: Agent Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: pytest --tb=short -v
```

Cassette files in `tests/cassettes/` are replayed automatically — no API keys or secrets needed in CI.

## Error Messages

Clear, structured output when assertions fail:

```
ToolCallSequenceError: Tool call sequence mismatch at index 1

Expected:
  [0] get_location(city="Tokyo")
  [1] get_news(topic="weather")     ← first mismatch

Actual:
  [0] get_location(city="Tokyo")
  [1] search_web(query="Tokyo weather")  ← actual
```

```
CostBudgetError: Token budget exceeded

  Actual:  1,250 tokens
  Limit:   1,000 tokens
  Exceeded by: 250 tokens (25.0%)
```

```
SafetyRuleViolationError: 2 forbidden tool calls detected

  [1] delete_database(table="users") at position 3
  [2] drop_table(name="orders") at position 5
```

```
FinalOutputError: final_output does not contain expected substring

  Substring: 'Berlin'
  Actual:    'The weather in Tokyo is sunny, 22°C.'
```

## Requirements

- Python 3.10+
- pytest 7+

## Examples

The [`examples/`](examples/) directory contains end-to-end examples with real agent frameworks and MCP servers. Each example ships with pre-recorded cassettes — run the tests without any API keys.

| Example | Framework | Description |
|---|---|---|
| [`strands-weather-forecaster`](examples/strands-weather-forecaster/) | Strands Agents + Bedrock | Fetches weather via HTTP, verifies tool sequence and safety. Matches the [Real-World Example](#real-world-example--testing-the-strands-weather-forecaster) in this README |
| [`strands-file-organizer`](examples/strands-file-organizer/) | Strands Agents + Bedrock | Scans a directory via Filesystem MCP, suggests organization. Read-only safety verified |
| [`langchain-issue-triage`](examples/langchain-issue-triage/) | LangChain + OpenAI | Triages GitHub issues via GitHub MCP. Label and priority suggestions |
| [`mcp-server`](examples/mcp-server/) | — | Mock GitHub MCP server for token-free testing |

Try it:

```bash
git clone https://github.com/simukappu/agentverify.git
cd agentverify
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Strands File Organizer
pip install -e "examples/strands-file-organizer/.[dev]"
pytest examples/strands-file-organizer/tests -v
```

```
tests/test_file_organizer.py::test_tool_call_sequence PASSED
tests/test_file_organizer.py::test_token_budget PASSED
tests/test_file_organizer.py::test_safety_read_only PASSED
```

```bash
# LangChain Issue Triage
pip install -e "examples/langchain-issue-triage/.[dev]"
pytest examples/langchain-issue-triage/tests -v
```

```
tests/test_issue_triage.py::TestIssueTriage_MockMCP::test_tool_call_sequence PASSED
tests/test_issue_triage.py::TestIssueTriage_MockMCP::test_safety_read_and_label_only PASSED
```

See each example's README for agent execution instructions and recording mode details.

## Roadmap

- ~~Agent framework adapters — extract `ExecutionResult` directly from Strands Agents, LangChain, and others without writing a converter~~ ✅ Shipped
- ~~Cassette request matching — verify request content during replay to detect stale cassettes~~ ✅ Shipped
- ~~Cassette sanitization — automatic masking of API keys and sensitive data in recorded cassettes~~ ✅ Shipped
- ~~Latency assertion — `assert_latency()` for response time SLAs in production agents~~ ✅ Shipped
- ~~Tool mocking/stubbing — test agent routing logic without calling real tools~~ ✅ Shipped
- Async support — first-class `asyncio` testing for async agents and tools
- Responses API cassette adapter — record/replay for OpenAI Agents SDK (Responses API) with end-to-end example
- Step-level assertions — structured multi-step execution testing with `assert_step()` and intermediate output verification
- Framework adapters for Google ADK and CrewAI — pending async support and stable tool-call APIs from these frameworks
- Cost estimation from tokens — auto-calculate `total_cost_usd` from token usage and model pricing

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

[MIT](LICENSE)

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

Development setup:

```bash
git clone https://github.com/simukappu/agentverify.git
cd agentverify
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
