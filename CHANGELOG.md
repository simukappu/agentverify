# Changelog

## Unreleased

### Features

- **Streamed responses are recorded and replayed** for every provider, so frameworks that stream by default work without turning streaming off. A cassette recorded either way replays either way. See README "Streaming".

### Bug Fixes

- **Data-flow assertions work on Anthropic, Bedrock, and Gemini cassettes**: replayed steps carry the tool results from those providers' tool result messages, with their error status, as they did for OpenAI.
- **Replayed Anthropic tool calls keep their recorded ids**, and calls recorded without one get unique ids instead of reusing `toolu_0`.

## 0.6.0 (2026-10-21)

### Features

- **Async agents**: the async clients of Anthropic, Gemini, and LiteLLM are recorded and replayed, as OpenAI's already were. See README "Async Agents".
- **OpenAI Responses API** is recorded and replayed, so the OpenAI Agents SDK works on its default model. See README "OpenAI Responses API".
- **Multi-agent handoff assertions**: `assert_handoffs`, `assert_no_handoff`, and `assert_step(agent=...)`, via `ExecutionResult.handoffs` and `Step.agent`. See README "Multi-Agent Handoffs".

### Bug Fixes

- **Multi-turn tool runs replay**: replayed tool calls keep their recorded ids instead of reusing `call_0`, which the OpenAI Agents SDK rejected.
- **Replayed `tool_results` hold only the step's own results**, not every earlier result the next request resends.
- **LiteLLM replay returns `litellm.ModelResponse`**, which the OpenAI Agents SDK's `LitellmModel` requires.
- **Bedrock replay** omits `usage` when none was recorded, instead of returning `None`.

### Breaking Changes

- **`from_openai_agents` includes handoff items** and starts a new step when the acting agent changes, so multi-agent runs gain steps and tool calls.
- **Overlapping LLM calls while recording raise `CassetteUnsupportedCallError`**, since their recorded order varies between runs.

## 0.5.0 (2026-10-18)

### Improvements

- **`contains=` accepts a list** of substrings, and `case_sensitive=False` ignores case, in `assert_final_output`, `assert_step_output`, and `assert_tool_result_matches`. Thanks to @Neohu-ceo ([#4](https://github.com/simukappu/agentverify/pull/4)). See README "Final Output Assertions".

### Bug Fixes

- **Replay no longer reaches the real API.** Calls a cassette cannot replay, such as streaming, the Responses API, embeddings, or async Anthropic and Gemini clients, now raise `CassetteUnsupportedCallError` instead of calling the provider. See README "Calls Cassettes Cannot Record".
- **Credentials passed as call arguments are redacted from cassettes**, such as `api_key=` or an `x-api-key` header. Default patterns also cover Google, Bedrock, AWS temporary, and GitHub keys. See README "Cassette Sanitization".
- **OpenAI Agents SDK tool failures are detected**, so `assert_no_tool_errors` and `assert_tool_invocation_succeeded` no longer pass on them.

### Dependency

- **Python 3.15 and openai 3.x** are tested in CI.

## 0.4.0 (2026-06-19)

### Features

- **Tool result assertions** verify that tool calls actually succeeded, not just that they were made: `assert_tool_invocation_succeeded`, `assert_no_tool_errors`, `assert_tool_result_matches`, and `assert_retry_count`. See README "Tool Result Assertions".
- **Per-result error signal on `Step`** via the additive `tool_results_meta` field, populated by the built-in adapters, reconstructed on cassette replay, and settable from custom converters and `step_probe`.

### Bug Fixes

- **Example test suites on Python 3.14**: `pytest examples/<name>/tests` no longer fails with a duplicate `--cassette-mode` registration during collection.

## 0.3.0 (2026-04-26)

### Features

- **Step-level assertions** for agents making multiple LLM calls per run. New `assert_step`, `assert_step_output`, and `assert_step_uses_result_from` verify tool calls, intermediate outputs, and step-to-step data flow. See README "Step-Level Assertions".
- **`step_probe` context manager** to mark logical step boundaries in agent code, including LLM-free steps like cache hits, state management, and validation. Zero-cost no-op outside of recorder/MockLLM contexts, safe to leave in production code.
- **Data-flow matching** in `assert_step_uses_result_from` tolerates common serialization differences: numeric tool results encoded as strings match their int/float consumers, and multi-line produced strings match consumers that hold them inside a container.
- **`MATCHES(pattern)` regex matcher** for verifying string tool-call arguments against a regex, with the same semantics as `ANY`.
- **`MockLLM` + `mock_response(...)`** replay predefined LLM responses in-memory. Test agent routing without a cassette or real LLM call.
- **`assert_latency(result, max_ms=...)`** enforces response-time SLAs. `ExecutionResult.duration_ms` is captured automatically by the cassette fixture and `MockLLM`.
- **Three new example suites** exercising step-level and data-flow testing on different agent shapes: OpenAI Agents SDK LLM-as-a-Judge, LangGraph multi-agent supervisor, and custom-converter-python-agent. See `examples/`.

### Improvements

- **OpenAI cassette adapter** now also intercepts `AsyncCompletions.create`, so agent frameworks that drive the SDK through `AsyncOpenAI` internally (including the OpenAI Agents SDK) are recorded and replayed transparently.
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
