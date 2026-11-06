# API Reference

Every public name, what it is for, and where it is described. Names are imported from `agentverify` unless another module is given.

## Index

| Area | Names | Described in |
|---|---|---|
| Results | `ExecutionResult`, `Step`, `ToolCall`, `TokenUsage`, `Handoff` | [Build an ExecutionResult](../README.md#build-an-executionresult), [Result Objects](#result-objects) |
| Matchers | `ANY`, `MATCHES`, `OrderMode` | [Assertion Modes](../README.md#assertion-modes) |
| Tool call assertions | `assert_tool_calls`, `assert_no_tool_call`, `assert_tool_calls_allowed`, `assert_all` | [Assertion Modes](../README.md#assertion-modes), [Allowed Tools](../README.md#allowed-tools) |
| Budgets and output | `assert_cost`, `assert_latency`, `assert_final_output` | [Budgets and Final Output](../README.md#budgets-and-final-output) |
| Step assertions | `assert_step`, `assert_step_output`, `assert_step_uses_result_from`, `step_probe`, `ProbeHandle` | [Step-Level Assertions](../README.md#step-level-assertions), [`step_probe`](#step_probe) |
| Handoffs | `assert_handoffs`, `assert_no_handoff` | [Multi-Agent Handoffs](../README.md#multi-agent-handoffs) |
| Tool results | `assert_tool_invocation_succeeded`, `assert_no_tool_errors`, `assert_tool_result_matches`, `assert_retry_count` | [Tool Result Assertions](../README.md#tool-result-assertions), [tool-results.md](tool-results.md) |
| What the model was sent | `iter_request_texts`, `assert_no_credential_urls` (`agentverify.credentials.CREDENTIAL_QUERY_NAMES`) | [README](../README.md#checking-what-the-model-was-sent), [cassettes.md](cassettes.md#checking-what-the-model-was-sent) |
| Cassettes | the `cassette` fixture, `LLMCassetteRecorder`, `CassetteMode`, `cassette_tool`, `SanitizePattern`, `DEFAULT_PATTERNS` (`agentverify.cassette.recorder.OnMissingRequest`) | [Record & Replay](../README.md#record--replay-with-cassettes), [cassettes.md](cassettes.md), [below](#llmcassetterecorder-without-the-fixture) |
| Mocking | `MockLLM`, `mock_response` | [LLM Mocking with MockLLM](../README.md#llm-mocking-with-mockllm) |
| Framework adapters | `agentverify.frameworks.<framework>.from_<framework>` | [Built-in Adapters](../README.md#built-in-adapters) |
| Traces | `agentverify.traces.from_otel` | [Production Traces](../README.md#production-traces) |
| pytest | the `agentverify` marker (`runs=`, `min_pass_rate=`), `--cassette-mode`, `--no-cassette-match-requests`, `--cassette-max-age`, `--cassette-expired`, `--cassette-drift`, `--cassette-report`, `--agentverify-live`, the `agentverify_cassette_max_age` and `agentverify_cassette_drift` settings | [Record & Replay](../README.md#record--replay-with-cassettes), [Live Runs](../README.md#live-runs-and-pass-rates), [cassettes.md](cassettes.md) |
| Errors | every `*Error` | [Errors](#errors) |

The sections below cover what the README only mentions.

## `LLMCassetteRecorder` Without the Fixture

The `cassette` fixture builds an `LLMCassetteRecorder`; outside pytest, or to place the file yourself, build one directly. It takes the same options, and the pytest settings and command-line options (mode, expiry, drift) do not apply to it:

```python
from agentverify import CassetteMode, LLMCassetteRecorder

with LLMCassetteRecorder("tests/cassettes/weather.yaml", mode=CassetteMode.REPLAY, provider="bedrock") as rec:
    run_my_agent("What's the weather in Seattle?")
result = rec.to_execution_result()
```

| Argument | Default | Meaning |
|---|---|---|
| `cassette_path` | | The cassette file (`.yaml` or `.json`) |
| `mode` | `CassetteMode.AUTO` | `AUTO`, `RECORD`, `REPLAY`, or `RECORD_MISSING` (see the [modes](../README.md#record--replay-with-cassettes)). The fixture also takes the values as strings (`mode="replay"`) |
| `provider` | `"openai"` | `"openai"`, `"anthropic"`, `"bedrock"`, `"gemini"`, `"litellm"`, or an `LLMProviderAdapter` instance |
| `on_missing` | `OnMissingRequest.ERROR` | What a replayed request or tool call with no recording does; see below |
| `match_requests` | `True` | [Request matching](cassettes.md#request-matching) |
| `match_ignore` | `None` | Regular expressions left out of request matching and drift |
| `sanitize` | `True` | [Sanitization](../README.md#cassette-sanitization): `True`, `False`, or a list of `SanitizePattern` |
| `replay_tools` | `True` | [Tool replay](cassettes.md#tool-replay) |
| `max_age` / `on_expired` | `None` / `"error"` | [Expiry](cassettes.md#expiry-and-the-cassette-report) |
| `drift` / `drift_ignore` | `"off"` / `None` | [Drift](cassettes.md#drift) |

The fixture adds `cassette_dir` (default: a `cassettes/` directory next to the test file).

### Requests With No Recording

By default, a replayed LLM request the cassette has no response for raises `CassetteMissingRequestError` (the message gives the request number and how many responses the cassette has), and a tool call with no recorded result raises `CassetteToolMismatchError`. With `on_missing=OnMissingRequest.FALLBACK` (`on_missing="fallback"` in the fixture), both go to the real API or tool instead, and nothing new is saved:

```python
with cassette("weather.yaml", provider="openai", on_missing="fallback") as rec:
    run_my_agent("What's the weather in Tokyo and Paris?")  # Paris was not recorded: it calls the API
```

## `@cassette_tool`

For a hand-written agent whose tools no built-in hook covers. Outside a cassette the function runs as usual; inside one, its calls are recorded and replayed like a framework's tools. It works on sync and async functions and on methods (`self` / `cls` are not recorded). `name=` sets the recorded tool name (default: the function's name). Put it under a framework's own `@tool` decorator; the signature is kept for schema generation:

```python
from agentverify import cassette_tool

@cassette_tool
def get_weather(city: str) -> str:
    return http_get(f"https://weather.example/{city}")

@cassette_tool(name="fetch")
async def fetch_page(url: str) -> str:
    ...
```

## `step_probe`

`step_probe(name, output=None)` marks a step of your own in the agent's code ([example](../README.md#workflow-style-agents-with-step_probe)). It yields a `ProbeHandle`:

- `handle.set_tool_result(result, is_error=False)`: attach a tool result to the step; `is_error=True` marks it a failed call for the [tool result assertions](tool-results.md)
- `handle.set_output(value)`: set the step's text output (the same as `output=`)

LLM calls made inside the block join the step. Outside a cassette or `MockLLM`, the probe does nothing.

## Result Objects

- `ExecutionResult.from_dict()`, `from_json()`, `to_dict()`, `to_json()`: see [Build an ExecutionResult](../README.md#build-an-executionresult)
- `ExecutionResult.from_flat_tool_calls(tool_calls, *, token_usage=None, total_cost_usd=None, duration_ms=None, final_output=None)`: a single-step result from a list of `ToolCall`s
- `ExecutionResult.tool_calls`: every step's tool calls in order; `steps`, `handoffs`, `final_output`, `token_usage`, `total_cost_usd`, `duration_ms`
- `Step.has_tool_error()`: whether any tool result at the step is a known error; `Step.tool_result_is_error(i)`: `True`, `False`, or `None` (unknown) for result `i`
- `ToolCall(name, arguments, result=None, call_id=None)`: `call_id` is not compared
- `TokenUsage(input_tokens, output_tokens)` and its `total_tokens`
- `Handoff(from_agent, to_agent, step=None)`

## Errors

Every error extends `AgentVerifyError`, which extends `AssertionError`, so pytest reports them as test failures. All are in `agentverify.errors`; the ones marked ✓ are also exported from `agentverify`.

| Error | Raised by | `agentverify` |
|---|---|---|
| `ToolCallSequenceError` | `assert_tool_calls`, and `assert_step` on the step's tool calls | |
| `SafetyRuleViolationError` | `assert_no_tool_call` | |
| `ToolPolicyError` | `assert_tool_calls_allowed` | ✓ |
| `CostBudgetError` | `assert_cost` | |
| `LatencyBudgetError` | `assert_latency` | |
| `FinalOutputError` | `assert_final_output` | |
| `MultipleAssertionError` | `assert_all`, with every failure | |
| `StepIndexError`, `StepNameNotFoundError`, `StepNameAmbiguousError` | step assertions given a step that does not exist or a name more than one step has | ✓ |
| `StepOutputError` | `assert_step_output` | ✓ |
| `StepAgentError` | `assert_step(agent=...)` | ✓ |
| `StepDependencyError` | `assert_step_uses_result_from` | ✓ |
| `HandoffError` | `assert_handoffs`, `assert_no_handoff` | ✓ |
| `ToolInvocationError` | `assert_tool_invocation_succeeded`, `assert_no_tool_errors` | ✓ |
| `ToolResultMatchError` | `assert_tool_result_matches` | ✓ |
| `RetryBudgetError` | `assert_retry_count` | ✓ |
| `CredentialURLError` | `assert_no_credential_urls` | ✓ |
| `CassetteRequestMismatchError` | replay, when the request changed | ✓ |
| `CassetteMissingRequestError` | replay or `MockLLM`, out of responses | ✓ |
| `CassetteToolMismatchError` | replay, a tool call with no recorded result | ✓ |
| `CassetteExpiredError` | replay, a cassette past its `max_age` | ✓ |
| `CassetteDriftError` | replay with `drift="fail"` | ✓ |
| `CassetteUnsupportedCallError` | an LLM call a cassette cannot record or replay | ✓ |
