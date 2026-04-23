# Changelog

## Unreleased

### Features

- **Step-level assertions** for agents that make multiple LLM calls per run (ReAct, Plan-and-Execute, workflow-style). New `assert_step`, `assert_step_output`, and `assert_step_uses_result_from` verify tool calls, intermediate outputs, and step-to-step data flow on the new `Step` data model. See README "Step-Level Assertions".
- **`step_probe` context manager** lets you mark logical step boundaries in agent code (including LLM-free steps like cache hits, state management, validation) so tests can assert on workflow logic. Zero-cost no-op outside of recorder/MockLLM contexts — safe to leave in production code.
- **Regex argument matcher**: `MATCHES(pattern)` for verifying string tool call arguments against a regex — the same `ANY` wildcard semantics, constrained to a regex match.
- **Tool mocking**: `MockLLM` + `mock_response(...)` replay predefined LLM responses in-memory — test agent routing without a cassette or real LLM call.
- **Latency assertion**: `assert_latency(result, max_ms=...)` enforces response time SLAs. `ExecutionResult.duration_ms` is captured automatically by the cassette fixture and `MockLLM`.
- **LangGraph multi-agent supervisor example** demonstrates step-level + cross-step data flow testing on a research + math agent handoff (see `examples/langgraph-multi-agent-supervisor/`).

### Improvements

- **Numeric-string data flow** in `assert_step_uses_result_from`: tool results returned as strings like `"231317.0"` now correctly match downstream steps that consume them as ints or floats. Uses digit-boundary matching to avoid false positives (e.g. `"231"` no longer matches inside `1231`).
- **langchain-openai v1.x compatibility** in the OpenAI cassette adapter: intercepts the `with_raw_response.create` code path (which langchain-openai v1.x routes through) and returns a `LegacyAPIResponse`-compatible wrapper on replay. Also unwraps `LegacyAPIResponse` during recording.

### Breaking Changes

- **`ExecutionResult` is now step-centric.** `steps: list[Step]` is the single source of truth; `result.tool_calls` is a derived read-only property. Existing read-side code (`assert_tool_calls`, `assert_no_tool_call`, etc.) works unchanged. `ExecutionResult(tool_calls=[...])` constructor kwarg still works for backward compatibility. `result.to_dict()` now emits `steps: [...]` instead of `tool_calls: [...]` — `from_dict()` still accepts the legacy `tool_calls` key on input.

## 0.2.0 (2026-04-18)

### Features

- **Built-in framework adapters** for Strands Agents, LangChain, LangGraph, and OpenAI Agents SDK. Extract `ExecutionResult` directly from agent framework outputs without writing a custom converter (`from_strands`, `from_langchain`, `from_langgraph`, `from_openai_agents`).
- **Cassette request matching** detects stale cassettes by verifying model name and tool names during replay. Enabled by default; raises `CassetteRequestMismatchError` with a clear diff on mismatch.
- **Cassette sanitization** automatically redacts API keys and sensitive data when recording. Built-in patterns cover OpenAI, Anthropic, AWS, and Bearer tokens; extendable with custom `SanitizePattern` objects.
- **Strands Weather Forecaster example**: End-to-end example testing the official Strands sample with a pre-recorded Bedrock cassette.

## 0.1.0 (2026-04-16)

Initial release.

### Features

- **Tool call assertions**: `assert_tool_calls` with EXACT, IN_ORDER, and ANY_ORDER modes; `ANY` wildcard and `partial_args` for flexible argument matching.
- **Cost budget assertions**: `assert_cost` enforces `max_tokens` and `max_cost_usd` limits.
- **Safety guardrails**: `assert_no_tool_call` detects forbidden tool invocations.
- **Final output assertions**: `assert_final_output` with `contains`, `equals`, and `matches` (regex).
- **Batch assertions**: `assert_all` collects all failures without stopping at the first.
- **LLM Cassette Record & Replay**: VCR-style recording of LLM API calls for deterministic CI testing. Human-readable YAML / JSON cassettes that you commit to git.
- **5 LLM provider adapters**: OpenAI, Amazon Bedrock, Google Gemini, Anthropic, LiteLLM.
- **pytest plugin**: Auto-registers on install, provides the `cassette` fixture and `@pytest.mark.agentverify` marker.
- **Structured error messages**: Clear diffs with mismatch position highlighting.
