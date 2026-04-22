# Changelog

## Unreleased

### Features

- **Step-level assertions** for agents that make multiple LLM calls per run (ReAct, Plan-and-Execute, workflow-style). New `assert_step`, `assert_step_output`, and `assert_step_uses_result_from` verify tool calls, intermediate outputs, and step-to-step data flow on the new `Step` data model. See README "Step-Level Assertions".
- **`step_probe` context manager** lets you mark logical step boundaries in agent code (including LLM-free steps like cache hits, state management, validation) so tests can assert on workflow logic. Zero-cost no-op outside of recorder/MockLLM contexts — safe to leave in production code.
- **Regex argument matcher**: `MATCHES(pattern)` for verifying string tool call arguments against a regex — the same `ANY` wildcard semantics, constrained to a regex match.
- **Tool mocking**: `MockLLM` + `mock_response(...)` replay predefined LLM responses in-memory — test agent routing without a cassette or real LLM call.
- **Latency assertion**: `assert_latency(result, max_ms=...)` enforces response time SLAs. `ExecutionResult.duration_ms` is captured automatically by the cassette fixture and `MockLLM`.

### Breaking Changes

- **`ExecutionResult` is now step-centric.** `steps: list[Step]` is the single source of truth; `result.tool_calls` is a derived read-only property. Existing read-side code (`assert_tool_calls`, `assert_no_tool_call`, etc.) works unchanged. `ExecutionResult(tool_calls=[...])` constructor kwarg still works for backward compatibility. `result.to_dict()` now emits `steps: [...]` instead of `tool_calls: [...]` — `from_dict()` still accepts the legacy `tool_calls` key on input.

## 0.2.0 (2026-04-18)

### Features

- **Built-in framework adapters**: Extract `ExecutionResult` directly from agent framework outputs without writing a custom converter
  - `from agentverify.frameworks.strands import from_strands` — Strands Agents `AgentResult` adapter
  - `from agentverify.frameworks.langchain import from_langchain` — LangChain `AgentExecutor` output adapter
  - `from agentverify.frameworks.langgraph import from_langgraph` — LangGraph `create_react_agent` result adapter
  - `from agentverify.frameworks.openai_agents import from_openai_agents` — OpenAI Agents SDK `RunResult` adapter
- **Cassette request matching**: Detect stale cassettes by verifying model name and tool names during replay
  - Enabled by default; disable with `--no-cassette-match-requests` CLI option or `match_requests=False` parameter
  - Raises `CassetteRequestMismatchError` with clear diff on mismatch
- **Cassette sanitization**: Automatic redaction of API keys and sensitive data when recording cassettes
  - Enabled by default with built-in patterns for OpenAI, Anthropic, AWS, and Bearer tokens
  - Customizable via `sanitize` parameter with `SanitizePattern` objects
  - Disable with `sanitize=False` when needed
- **Strands Weather Forecaster example**: End-to-end example testing the official Strands Weather Forecaster sample with pre-recorded Bedrock cassette

### Internal

- Cassette version stamp now derived dynamically from `importlib.metadata` (single version source in `pyproject.toml`)
- CI now runs on all branches (not just `main`)

## 0.1.0 (2026-04-16)

Initial release.

### Features

- **Tool call assertions**: `assert_tool_calls()` with ExactOrder, InOrder, and AnyOrder modes
- **Cost/token budget assertions**: `assert_cost()` with max_tokens and max_cost_usd
- **Safety guardrails**: `assert_no_tool_call()` for forbidden tool detection
- **Final output assertions**: `assert_final_output()` with contains, equals, and regex matching
- **Batch assertions**: `assert_all()` collects all failures without stopping at first
- **ANY matcher**: Wildcard for non-deterministic argument values
- **Partial argument matching**: Only check the keys you care about
- **LLM Cassette Record & Replay**: VCR-style recording of LLM API calls for deterministic CI testing
- **5 LLM provider adapters**: OpenAI, Amazon Bedrock, Google Gemini, Anthropic, LiteLLM
- **YAML and JSON cassette formats**: Auto-detected by file extension
- **pytest plugin**: Auto-registers on install, provides `cassette` fixture and `@pytest.mark.agentverify` marker
- **Structured error messages**: Clear diffs with mismatch position highlighting
- **Input validation**: `ExecutionResult.from_dict()` validates tool call entries with clear error messages