# Changelog

## 0.3.0 (2026-04-26)

### Features

- **Step-level assertions** for agents making multiple LLM calls per run. New `assert_step`, `assert_step_output`, and `assert_step_uses_result_from` verify tool calls, intermediate outputs, and step-to-step data flow. See README "Step-Level Assertions".
- **`step_probe` context manager** to mark logical step boundaries in agent code, including LLM-free steps like cache hits, state management, and validation. Zero-cost no-op outside of recorder/MockLLM contexts — safe to leave in production code.
- **Data-flow matching** in `assert_step_uses_result_from` tolerates common serialization differences — numeric tool results encoded as strings match their int/float consumers, and multi-line produced strings match consumers that hold them inside a container.
- **`MATCHES(pattern)` regex matcher** for verifying string tool-call arguments against a regex, with the same semantics as `ANY`.
- **`MockLLM` + `mock_response(...)`** replay predefined LLM responses in-memory — test agent routing without a cassette or real LLM call.
- **`assert_latency(result, max_ms=...)`** enforces response-time SLAs. `ExecutionResult.duration_ms` is captured automatically by the cassette fixture and `MockLLM`.
- **Three new example suites** exercising step-level and data-flow testing on different agent shapes — OpenAI Agents SDK LLM-as-a-Judge, LangGraph multi-agent supervisor, and custom-converter-python-agent. See `examples/`.

### Improvements

- **OpenAI cassette adapter** now also intercepts `AsyncCompletions.create`, so agent frameworks that drive the SDK through `AsyncOpenAI` internally — including the OpenAI Agents SDK — are recorded and replayed transparently.
- **OpenAI cassette adapter** strips `openai.omit` / `openai.NOT_GIVEN` sentinels from `tools`, per-message dicts, and extra parameters before they reach the cassette YAML.
- **OpenAI cassette adapter** handles the `with_raw_response.create` code path used by langchain-openai v1.x, unwrapping `LegacyAPIResponse` on record and re-wrapping the synthesised `ChatCompletion` on replay.
- **Anthropic cassette adapter** flattens SDK content-block objects to plain dicts at record time, so cassettes recorded from ReAct-style agents load cleanly regardless of the installed Anthropic SDK version.

### Dependency

- **Minimum `pytest>=7`** is now declared in the runtime dependency so pip refuses to install on older pytest. CI tests against pytest 7, 8, and 9 majors on Python 3.10-3.14.

### Breaking Changes

- **`ExecutionResult.to_dict()` now emits `steps: [...]`** instead of `tool_calls: [...]`. Read-side assertions and the `ExecutionResult(tool_calls=[...])` constructor are backward compatible; `from_dict()` still accepts the legacy `tool_calls` key on input.

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
