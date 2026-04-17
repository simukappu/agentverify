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

## 3 Steps to Test Your Agent

### Step 1: Build an ExecutionResult

Build an `ExecutionResult` from your agent's output. You can construct one from a dict, or use a framework-specific converter (see [examples/](examples/) for Strands Agents and LangChain converters).

```python
from agentverify import ExecutionResult

result = ExecutionResult.from_dict({
    "tool_calls": [
        {"name": "get_location", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"lat": 35.6, "lon": 139.7}},
    ],
    "token_usage": {"input_tokens": 50, "output_tokens": 30},
    "total_cost_usd": 0.002,
    "final_output": "The weather in Tokyo is sunny, 22°C.",
})
```

`ExecutionResult.from_dict()` accepts the following keys:

| Key | Type | Description |
|---|---|---|
| `tool_calls` | `list[dict]` | Each dict has `name` (str, required), `arguments` (dict, optional), `result` (any, optional — tool execution result stored for reference; not used in assertions) |
| `token_usage` | `dict` or `None` | `{"input_tokens": int, "output_tokens": int}` |
| `total_cost_usd` | `float` or `None` | Total cost in USD (must be set manually — not auto-calculated from tokens) |
| `final_output` | `str` or `None` | The agent's final text response |

You can also use `ExecutionResult.from_json(json_string)` to parse from a JSON string, and `to_dict()` / `to_json()` for serialization.

#### Building from Your Framework

For frameworks with built-in adapter support (Strands Agents, LangChain), use the adapter directly:

```python
from agentverify.frameworks.strands import from_strands

result = agent("Analyze the files")
execution_result = from_strands(result)
```

For other frameworks, write a small converter function (~20–50 lines) that maps your framework's output to the dict schema above. Here's the general pattern:

```python
from agentverify import ExecutionResult, ToolCall, TokenUsage

def my_framework_to_execution_result(agent_output) -> ExecutionResult:
    # 1. Extract tool calls: map your framework's tool call objects
    #    to ToolCall(name=..., arguments=...)
    tool_calls = [
        ToolCall(name=tc.tool_name, arguments=tc.params)
        for tc in agent_output.tool_history
    ]

    # 2. Extract token usage (if available)
    token_usage = TokenUsage(
        input_tokens=agent_output.metrics.prompt_tokens,
        output_tokens=agent_output.metrics.completion_tokens,
    )

    # 3. Extract final output text
    return ExecutionResult(
        tool_calls=tool_calls,
        token_usage=token_usage,
        final_output=agent_output.response_text,
    )
```

See [`examples/strands-file-organizer/converter.py`](examples/strands-file-organizer/converter.py) and [`examples/langchain-issue-triage/converter.py`](examples/langchain-issue-triage/converter.py) for complete, production-ready converters.

### Step 2: Assert

```python
from agentverify import assert_tool_calls, assert_cost, assert_no_tool_call, assert_final_output, ToolCall, ANY

# Did the agent call the right tools in the right order?
assert_tool_calls(result, expected=[
    ToolCall("get_location", {"city": "Tokyo"}),
    ToolCall("get_weather", {"lat": ANY, "lon": ANY}),
])

# Did it stay within budget?
assert_cost(result, max_tokens=500, max_cost_usd=0.01)

# Did it avoid dangerous tools?
assert_no_tool_call(result, forbidden_tools=["delete_user", "drop_table"])

# Did the final output contain the expected content?
# See "Final Output Assertions" below for equals and regex options.
assert_final_output(result, contains="Tokyo")
```

### Step 3: Record & Replay with Cassettes

Record real LLM API calls once. Replay them in CI forever — zero cost, deterministic.

Cassette replay verifies request content by default — if your model or tools change, replay raises `CassetteRequestMismatchError` instead of silently returning stale responses. Re-record cassettes with `--cassette-mode=record` after significant agent changes. See [Request Matching](#request-matching-stale-cassette-detection) for details.

The `cassette` fixture is a pytest fixture provided by the agentverify plugin. It creates an `LLMCassetteRecorder` that intercepts LLM SDK calls (not HTTP — it patches the SDK's chat completion method directly). Use `@pytest.mark.agentverify` to mark your test, and call your agent code inside the `with cassette(...)` block. After the block exits, call `rec.to_execution_result()` to build the result for assertions.

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
- Be mindful not to include sensitive data (API keys, PII, confidential prompts) in cassette files checked into version control. Cassette sanitization is enabled by default — see [Cassette Sanitization](#cassette-sanitization) below.

## Assertion Modes

```python
from agentverify import assert_tool_calls, OrderMode, ToolCall, ANY

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
- Framework adapters for Google ADK and CrewAI — pending async support and stable tool-call APIs from these frameworks
- Responses API cassette adapter — record/replay for OpenAI Agents SDK (Responses API) with end-to-end example
- Tool mocking/stubbing — test agent routing logic without calling real tools
- Async support — first-class `asyncio` testing for async agents and tools
- ~~Cassette request matching — verify request content during replay to detect stale cassettes~~ ✅ Shipped
- ~~Cassette sanitization — automatic masking of API keys and sensitive data in recorded cassettes~~ ✅ Shipped
- Cost estimation from tokens — auto-calculate `total_cost_usd` from token usage and model pricing
- YAML/JSON test case definitions — declarative test cases for non-Python CI pipelines
- CLI test runner — run agent tests without pytest

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
