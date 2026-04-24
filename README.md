# agentverify

[![PyPI](https://img.shields.io/pypi/v/agentverify)](https://pypi.org/project/agentverify/)
[![Downloads](https://img.shields.io/pepy/dt/agentverify)](https://pepy.tech/projects/agentverify)
[![CI](https://github.com/simukappu/agentverify/actions/workflows/ci.yml/badge.svg)](https://github.com/simukappu/agentverify/actions/workflows/ci.yml)
[![Coverage](https://coveralls.io/repos/github/simukappu/agentverify/badge.svg?branch=main)](https://coveralls.io/github/simukappu/agentverify?branch=main)
[![License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

**pytest for AI agents.** Assert agent actions, not vibes.

agentverify is a pytest plugin for deterministic testing of AI agent actions. Record real LLM calls once, replay them in CI with zero cost, and assert exactly what your agent did — which tools it called, what data flowed between steps, how much it cost, and whether it stayed within your safety rules.

## Why agentverify?

Prompt engineering and eval frameworks tell you whether an LLM said the right thing. Production agents fail for a different set of reasons: they call the wrong tool, skip a step, hallucinate a parameter that was supposed to come from the previous tool's output, or quietly burn through your token budget. Those are deterministic bugs, and they deserve deterministic tests.

agentverify records real LLM SDK calls once and replays them in CI with zero cost. On the recording you can assert:

- **Tool call sequences** — exact order, subsequence, or set membership, with regex / wildcard argument matching
- **Step-level execution** — each LLM call is a step; assert what tool was called at step N, what the step's output was, and that step N's input references data produced by step M
- **Budgets** — token counts, cost in USD, end-to-end latency
- **Safety** — a list of tools that must never be called, no matter what the LLM decides

It ships with built-in adapters for LangChain, LangGraph, Strands Agents, and OpenAI Agents SDK, and supports OpenAI, Amazon Bedrock, Google Gemini, Anthropic, and LiteLLM as LLM providers. Cassettes are human-readable YAML — commit them to git, review in PRs, run in CI without an API key.

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

> **Using a real agent framework?** The [Real-World Examples](#real-world-examples) below show how to test actual agents with built-in adapters for LangChain, LangGraph, Strands Agents, and OpenAI Agents SDK.

```
$ pytest test_agent.py -v
test_agent.py::test_tool_sequence PASSED
test_agent.py::test_budget PASSED
test_agent.py::test_safety PASSED
test_agent.py::test_output PASSED
```

## Real-World Examples

Three end-to-end examples that showcase different agent patterns and testing angles. All run with pre-recorded cassettes — no API keys needed.

### Strands Weather Forecaster — Two-Step ReAct

Test the [Strands Weather Forecaster](https://strandsagents.com/docs/examples/python/weather_forecaster/) — an official Strands Agents sample. The agent uses the `http_request` tool to call the National Weather Service API, which returns the forecast URL for the requested location; the agent then calls that URL to get the actual forecast. Two steps, one piece of data flowing between them — the simplest shape of a ReAct loop.

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

Want to go deeper? Step-level assertions verify the ReAct structure and the data flow between the two steps:

```python
from agentverify import assert_step, assert_step_uses_result_from

@pytest.mark.agentverify
def test_weather_agent_steps(cassette):
    with cassette("weather_seattle.yaml", provider="bedrock") as rec:
        weather_agent("What's the weather in Seattle?")
    result = rec.to_execution_result()

    # First step must call /points/ to discover the forecast office
    assert_step(result, step=0, expected_tool=ToolCall(
        "http_request", {"method": "GET", "url": MATCHES(r"/points/")},
    ), partial_args=True)

    # Second step must call /forecast/ AND use data from the first step's result
    assert_step(result, step=1, expected_tool=ToolCall(
        "http_request", {"method": "GET", "url": MATCHES(r"/forecast")},
    ), partial_args=True)
    assert_step_uses_result_from(result, step=1, depends_on=0)
```

`MATCHES(pattern)` is a regex matcher — see [Assertion Modes](#assertion-modes) for details on `ANY`, `MATCHES`, `OrderMode`, and `partial_args`. Step assertions are covered in [Step-Level Assertions](#step-level-assertions).

### OpenAI Agents SDK LLM as a Judge — Freezing a Probabilistic Refinement Loop

Test the [OpenAI Agents SDK `agent_patterns/llm_as_a_judge.py`](https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/llm_as_a_judge.py) sample: a generator writes a story outline, an evaluator grades it with a structured `{feedback, score}` verdict, and the loop re-runs the generator with that feedback until the evaluator accepts the draft. Everything about this agent is probabilistic — how many rounds it takes, what the feedback says, whether the rewrite incorporates it. Recording one run freezes all of it into a deterministic test.

The headline assertion is the feedback chain: round 2's generator must actually consume round 1's evaluator feedback, not silently regenerate the same outline.

```python
import pytest
from agentverify import assert_step_uses_result_from, assert_step_output

# Import the judge loop from the example
# (see examples/openai-agents-llm-as-a-judge/)
from agent import run_judge_loop_sync


@pytest.mark.agentverify
def test_generator_consumes_evaluator_feedback(cassette):
    """Round 2's regeneration must reference the evaluator's round 1 feedback."""
    with cassette("detective_in_space.yaml", provider="openai") as rec:
        run_judge_loop_sync("A detective story in space.", max_rounds=3)
    result = rec.to_execution_result()

    # Step 1 is the evaluator's first verdict; it must reject on the first try.
    assert_step_output(
        result, step=1, matches=r'"score"\s*:\s*"(needs_improvement|fail)"'
    )
    # Step 2 (generator, round 2) must actually consume step 1's feedback —
    # catches "agent regenerated but ignored the judge" bugs.
    assert_step_uses_result_from(result, step=2, depends_on=1)
```

`assert_step_uses_result_from` walks the multi-line evaluator feedback into the next step's user message even though JSON serialisation would escape the newlines; the match succeeds across that boundary. Because each `Runner.run` call is recorded individually, the example builds the `ExecutionResult` straight off the cassette recorder — `from_openai_agents` is the right adapter for single-run agents, but the cassette layer is the natural single source of truth when the workflow spans several runs.

See [`examples/openai-agents-llm-as-a-judge/`](examples/openai-agents-llm-as-a-judge/) for the full loop, the cassette, and the remaining flat + step-level assertions (budget, safety, pass-verdict reached).

### LangGraph + LangChain — Multi-Agent Handoff and MCP Tool Loops

Full-depth examples for the LangChain ecosystem live under [`examples/`](examples/):

- **[LangGraph multi-agent supervisor](examples/langgraph-multi-agent-supervisor/)** — a supervisor routes to a research expert and a math expert; each `add` in the math chain must consume the previous `add`'s result. `assert_step_uses_result_from` matches the numeric running total (`"231317.0"` produced as a string, consumed as `231317` int/float) across type boundaries, with digit-boundary checks that keep `"231"` from falsely appearing inside `1231`.
- **[LangChain GitHub issue triage](examples/langchain-issue-triage/)** — a `create_react_agent` drives the GitHub MCP server to list issues, read one, propose labels. The step-level tests verify that the `issue_number` passed to `get_issue` actually came from the preceding `list_issues` response, and that destructive tools like `close_issue` are never called.

Each example ships with its own pre-recorded cassette, a detailed `README.md`, and a full test file — pick whichever framework fits your stack.

## Build an ExecutionResult

Every agentverify assertion takes an `ExecutionResult`. You can build one three ways:

1. **From a dict** — convenient for quick tests and fixtures, as in the [Quick Start](#quick-start--no-llm-required).
2. **From a built-in adapter** — one-liner for LangChain, LangGraph, Strands Agents, OpenAI Agents SDK. See [Framework Integration](#framework-integration).
3. **From a custom converter** — for other frameworks, map your output to the schema below. See [`examples/langchain-issue-triage/converter.py`](examples/langchain-issue-triage/converter.py) for a ~50-line reference implementation.

`ExecutionResult.from_dict()` accepts these keys:

| Key | Type | Description |
|---|---|---|
| `steps` | `list[dict]` | Each step dict has `index` (int), `source` (`"llm"` / `"probe"` / `"tool"`), `tool_calls` (list), `tool_results` (list), `output` (str or None), `input_context` (dict or None), `name` (str or None). See [Step-Level Assertions](#step-level-assertions) |
| `tool_calls` | `list[dict]` | **Legacy v0.2.0 form** — when `steps` is absent, a flat `tool_calls` list is wrapped into a single synthetic step. Each dict has `name` (str, required), `arguments` (dict, optional), `result` (any, optional) |
| `token_usage` | `dict` or `None` | `{"input_tokens": int, "output_tokens": int}` |
| `total_cost_usd` | `float` or `None` | Total cost in USD (must be set manually — not auto-calculated from tokens or populated from cassettes) |
| `duration_ms` | `float` or `None` | Wall-clock duration in milliseconds (auto-populated by the `cassette` fixture and `MockLLM`) |
| `final_output` | `str` or `None` | The agent's final text response |

`ExecutionResult.from_json(json_string)` and `to_dict()` / `to_json()` are also available for serialization.

## Record & Replay with Cassettes

Record real LLM API calls once. Replay them in CI forever — zero cost, deterministic.

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

The `cassette` fixture is provided by the agentverify pytest plugin. It creates an `LLMCassetteRecorder` that intercepts LLM SDK calls (not HTTP — it patches the SDK's chat completion method directly). Use `@pytest.mark.agentverify` to mark your test, and call your agent code inside the `with cassette(...)` block. After the block exits, call `rec.to_execution_result()` to build the result for assertions.

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

Pass `sanitize=False` to disable sanitization entirely (not recommended).

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
```

Collect all failures at once instead of stopping at the first:

```python
from agentverify import assert_all
assert_all(
    result,
    lambda r: assert_tool_calls(r, expected=[...]),
    lambda r: assert_cost(r, max_tokens=1000),
    lambda r: assert_latency(r, max_ms=3000),
    lambda r: assert_no_tool_call(r, forbidden_tools=["delete_user"]),
    lambda r: assert_final_output(r, contains="Tokyo"),
)
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

During cassette **replay**, the measured duration reflects replay time (typically milliseconds), not the original call time. Capture latency data during **record** mode, or set `duration_ms` manually via [`ExecutionResult.from_dict()`](#build-an-executionresult).

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

### Strict Mode

`assert_cost()` and `assert_latency()` silently pass when the underlying data (`token_usage`, `total_cost_usd`, `duration_ms`) is `None` — useful during cassette replay where some data may be unavailable. Pass `strict=True` to require the data to be present:

```python
# Fails if token_usage or total_cost_usd is None
assert_cost(result, max_tokens=500, max_cost_usd=0.01, strict=True)

# Fails if duration_ms is None
assert_latency(result, max_ms=3000, strict=True)
```

## Step-Level Assertions

For agents that make multiple LLM calls per execution (ReAct, Plan-and-Execute, multi-hop retrieval, workflow-style), agentverify exposes the full step structure — not just a flat list of tool calls.

`ExecutionResult.steps` is the single source of truth; `result.tool_calls` remains as a derived flat view for simpler cases.

> The [Real-World Examples](#real-world-examples) above show `assert_step` and `assert_step_uses_result_from` in action on Strands and LangGraph. This section covers the full API.

```python
from agentverify import assert_step, assert_step_output, ToolCall, MATCHES

@pytest.mark.agentverify
def test_plan_and_execute(cassette):
    with cassette("trip_planner.yaml", provider="openai") as rec:
        plan_trip("2 days in Tokyo")
    result = rec.to_execution_result()

    # Step 0: agent plans what to search for
    assert_step_output(result, step=0, contains="flights")

    # Step 1: agent calls the flight search tool
    assert_step(result, step=1, expected_tool=ToolCall("search_flights", {"city": "Tokyo"}))

    # Step 2: agent calls the hotel search with a location string it discovered
    assert_step(result, step=2, expected_tool=ToolCall("search_hotels", {"area": MATCHES(r"Shibuya|Shinjuku")}))
```

### Assert step-to-step data flow

`assert_step_uses_result_from()` verifies that one step's input references data produced by another — catches "agent ignored the tool result" bugs.

```python
from agentverify import assert_step_uses_result_from

# Verify step 2 (hotel search) actually uses the result of step 1 (flight search)
assert_step_uses_result_from(result, step=2, depends_on=1)

# Restrict to a specific channel
assert_step_uses_result_from(result, step=2, depends_on=1, via="tool_result")
```

The check searches for any produced value (step M's `tool_results`, `tool_calls[*].result`, or `output`) inside step N's `input_context` or `tool_calls[*].arguments`. Strings are substring-matched; primitives (numbers, booleans) are matched structurally inside containers; JSON-encoded tool results are descended automatically.

Concretely, if step 0's `list_issues` tool returned `[{"number": 1}, {"number": 2}]` and step 1 called `get_issue(issue_number=2)`, the check finds that the number `2` produced by step 0 shows up in step 1's arguments — data flows correctly.

**Works on cassette replay.** Cassette recorders don't record tool execution results directly, but agentverify backfills them from the next step's `input_context` so you can assert data flow without any special adapter setup.

### Workflow-style Agents with `step_probe`

Many production agents aren't pure ReAct loops — they mix LLM calls with caching, state management, validation, and conditional branches. `step_probe()` lets you mark logical step boundaries in your agent code so tests can assert on those non-LLM steps.

```python
# agent.py
from agentverify import step_probe

def run_agent(query: str) -> str:
    with step_probe("fetch_cache") as p:
        cached = cache.get(query)
        p.set_tool_result({"hit": cached is not None})
        if cached:
            return cached
    with step_probe("call_llm"):
        response = llm.invoke(query)
    with step_probe("postprocess", output="formatted"):
        return format_output(response)
```

```python
# test_agent.py
def test_cache_miss_path():
    with MockLLM([mock_response(content="42")], provider="openai") as rec:
        run_agent("what's the answer?")
    result = rec.to_execution_result()
    assert_step(result, name="fetch_cache", expected_tools=[])  # probe with no tool calls
    assert_step(result, name="call_llm", ...)
    assert_step_output(result, name="postprocess", equals="formatted")
```

**`step_probe` is a zero-cost no-op outside of test recording contexts** — when no `LLMCassetteRecorder` or `MockLLM` is active, it's a pass-through context manager. Safe to leave in production code.

## Framework Integration

### Built-in Adapters

agentverify ships with built-in adapters for popular agent frameworks. No converter function needed — just import and call:

| Framework | Adapter | Input |
|---|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | `from agentverify.frameworks.langchain import from_langchain` | `AgentExecutor.invoke()` dict |
| [LangGraph](https://github.com/langchain-ai/langgraph) | `from agentverify.frameworks.langgraph import from_langgraph` | `create_react_agent` result dict |
| [Strands Agents](https://github.com/strands-agents/sdk-python) | `from agentverify.frameworks.strands import from_strands` | `AgentResult` |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | `from agentverify.frameworks.openai_agents import from_openai_agents` | `RunResult` |

For LangChain, pass the conversation history `messages` list as a second argument to capture token usage:

```python
from agentverify.frameworks.langchain import from_langchain

execution_result = from_langchain(result, messages=memory.chat_memory.messages)
```

LangGraph and Strands adapters take the raw agent output directly:

```python
from agentverify.frameworks.langgraph import from_langgraph
from agentverify.frameworks.strands import from_strands

# LangGraph: create_react_agent / create_supervisor result
execution_result = from_langgraph(app.invoke({"messages": [...]}))

# Strands: agent call result
execution_result = from_strands(weather_agent("What's the weather in Seattle?"))
```

### Custom Converters

For other frameworks, build an `ExecutionResult` from your agent's output using a small converter function. See [`examples/langchain-issue-triage/converter.py`](examples/langchain-issue-triage/converter.py) for a ~50-line reference implementation.

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
      - run: pip install "agentverify[all]"
      - run: pip install -r requirements.txt  # your project's deps
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
LatencyBudgetError: Latency budget exceeded

  Actual:  3,450.0 ms
  Limit:   3,000.0 ms
  Exceeded by: 450.0 ms (15.0%)
```

```
CostBudgetError: Token budget exceeded

  Actual:  1,100 tokens
  Limit:   1,000 tokens
  Exceeded by: 100 tokens (10.0%)
```

Other error types follow the same pattern: `SafetyRuleViolationError`, `FinalOutputError`.

## Requirements

- Python 3.10+
- pytest 7+

## Examples

The [`examples/`](examples/) directory contains end-to-end examples with real agent frameworks and MCP servers. Each example ships with pre-recorded cassettes — run the tests without any API keys.

| Example | Framework | Description |
|---|---|---|
| [`langchain-issue-triage`](examples/langchain-issue-triage/) | LangChain + OpenAI | Triages GitHub issues via GitHub MCP, step-level data flow for the discovered issue number |
| [`langgraph-multi-agent-supervisor`](examples/langgraph-multi-agent-supervisor/) | LangGraph + OpenAI | Research + math multi-agent handoff with running-total data flow |
| [`strands-weather-forecaster`](examples/strands-weather-forecaster/) | Strands Agents + Bedrock | Two-step ReAct that discovers a forecast URL, then calls it |
| [`openai-agents-llm-as-a-judge`](examples/openai-agents-llm-as-a-judge/) | OpenAI Agents SDK | Probabilistic generator ↔ evaluator refinement loop, frozen into a deterministic test with feedback-chain data flow |
| [`mcp-server`](examples/mcp-server/) | — | Mock GitHub MCP server for token-free testing |

See each example's README for setup and recording mode details.

## Roadmap

- Async support — first-class `asyncio` testing for async agents and tools
- Responses API cassette adapter — record/replay for OpenAI Agents SDK (Responses API) with end-to-end example
- Framework adapters for Google ADK and CrewAI — pending async support and stable tool-call APIs from these frameworks
- Cost estimation from tokens — auto-calculate `total_cost_usd` from token usage and model pricing

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

[MIT](LICENSE)

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change. See [CONTRIBUTING.md](CONTRIBUTING.md) for commit, CHANGELOG, and coverage conventions.

Development setup:

```bash
git clone https://github.com/simukappu/agentverify.git
cd agentverify
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
