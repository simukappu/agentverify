# Changelog

## Unreleased

### Features

- **Cassettes can expire**: replaying a cassette older than `max_age` raises `CassetteExpiredError` (or warns). Set it in `agentverify_cassette_max_age`, `--cassette-max-age`, or `cassette(max_age=...)`. `pytest --cassette-report` lists each cassette's age and model. See [README "Keeping Cassettes Current"](README.md#keeping-cassettes-current).
- **`assert_tool_calls_allowed`**: fails on calls to tools outside an allow list, or with arguments outside a JSON Schema subset. See [README "Allowed Tools"](README.md#allowed-tools).
- **`ToolCall.call_id`**: the provider's or framework's id for each tool call, from cassettes, `MockLLM`, the adapters, and `from_otel`. Not compared in assertions.

### Bug Fixes

- **`from_strands` works on real Strands runs**: it takes the `Agent`, since the `AgentResult` has no conversation.
- **Strands structured output works on replay**: the tool behind `structured_output_model` now runs for real.

### Breaking Changes

- **`from_strands(agent_result)` raises `TypeError`**; pass the agent.
- **Generated tool call ids are unique across runs** and appear in `to_dict()`; set fixed ones with `mock_response(tool_calls=[{"id": ...}])`.

## 0.7.1 (2026-10-22)

### Bug Fixes

- **Cassettes read Google ADK handoffs**: `transfer_to_agent(agent_name=...)` names its target in the arguments, and was read as a handoff to an agent called `agent`. A step after a new user turn is also no longer credited to the agent of an earlier turn's handoff.

### Documentation

- **New example: [Google ADK agent team](examples/google-adk-agent-team/)**, the ADK tutorial's weather agent with greeting and farewell sub-agents, on Bedrock through LiteLLM.

## 0.7.0 (2026-10-22)

### Features

- **Tool calls are recorded and replayed**: replay returns each tool's recorded result instead of running the tool. Built in for LangChain / LangGraph, Strands Agents, OpenAI Agents SDK, Pydantic AI, Google ADK, CrewAI, and MCP; `@cassette_tool` covers hand-written tools. A call with no recorded match raises `CassetteToolMismatchError`. See [README "Tool Calls"](README.md#tool-calls).
- **Stale cassettes are caught when the prompt changes**: replay now also compares each tool's schema, the system prompt, and user message text, and shows a diff. pytest lists the stale cassettes and the command to re-record them. `match_ignore` leaves changing text out. See [README "Stale Cassette Detection"](README.md#stale-cassette-detection).
- **`from_otel` builds `ExecutionResult`s from OpenTelemetry traces**, so the assertions run on production and staging runs. Reads the GenAI conventions and OpenInference. See [README "Production Traces"](README.md#production-traces).
- **Framework adapters for Pydantic AI, Google ADK, and CrewAI**: `from_pydantic_ai`, `from_google_adk`, `from_crewai`.
- **Concurrent LLM calls are recorded and replayed** (`asyncio.gather`, threads): replay matches each request to its recording, so calls that finish in another order still get their own responses. Async Bedrock clients (`aiobotocore` / `aioboto3`) are recorded too.
- **Live runs judged by pass rate**: `@pytest.mark.agentverify(runs=10, min_pass_rate=0.9)` with `--agentverify-live` runs a test against the live model and passes at the required rate. See [README "Live Runs and Pass Rates"](README.md#live-runs-and-pass-rates).
- **Streamed responses are recorded and replayed** for every provider, so frameworks that stream by default work without turning streaming off. A cassette recorded either way replays either way.

### Bug Fixes

- **Data-flow assertions work on Anthropic, Bedrock, and Gemini cassettes**: replayed steps carry the tool results from those providers' tool result messages (with their error status on Anthropic and Bedrock), as they did for OpenAI.
- **`CassetteMissingRequestError` says how far the run got**: the request number that ran out and how many responses the cassette or `MockLLM` has ([#3](https://github.com/simukappu/agentverify/issues/3)).
- **Replayed Anthropic tool calls keep their recorded ids**, and calls recorded without one get unique ids instead of reusing `toolu_0`.

### Breaking Changes

- **Replay fails when the system prompt, user message text, or a tool schema differs from the recording.** Re-record stale cassettes, or pass `match_ignore` / `match_requests=False`.
- **Replay no longer runs the tools of cassettes recorded with this version.** Tests that check a tool's side effects need `replay_tools=False`. Cassettes recorded before keep running their tools.

## 0.6.0 (2026-10-21)

### Features

- **Async agents**: the async clients of Anthropic, Gemini, and LiteLLM are recorded and replayed, as OpenAI's already were.
- **OpenAI Responses API** is recorded and replayed, so the OpenAI Agents SDK works on its default model. See [README "OpenAI Responses API"](README.md#openai-responses-api).
- **Multi-agent handoff assertions**: `assert_handoffs`, `assert_no_handoff`, and `assert_step(agent=...)`, via `ExecutionResult.handoffs` and `Step.agent`. See [README "Multi-Agent Handoffs"](README.md#multi-agent-handoffs).

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

- **`contains=` accepts a list** of substrings, and `case_sensitive=False` ignores case, in `assert_final_output`, `assert_step_output`, and `assert_tool_result_matches`. Thanks to @Neohu-ceo ([#4](https://github.com/simukappu/agentverify/pull/4)).

### Bug Fixes

- **Replay no longer reaches the real API.** Calls a cassette cannot replay, such as streaming, the Responses API, embeddings, or async Anthropic and Gemini clients, now raise `CassetteUnsupportedCallError` instead of calling the provider. See [README "Recorded and Unsupported Calls"](README.md#recorded-and-unsupported-calls).
- **Credentials passed as call arguments are redacted from cassettes**, such as `api_key=` or an `x-api-key` header. Default patterns also cover Google, Bedrock, AWS temporary, and GitHub keys. See [README "Cassette Sanitization"](README.md#cassette-sanitization).
- **OpenAI Agents SDK tool failures are detected**, so `assert_no_tool_errors` and `assert_tool_invocation_succeeded` no longer pass on them.

### Dependency

- **Python 3.15 and openai 3.x** are tested in CI.

## 0.4.0 (2026-06-19)

### Features

- **Tool result assertions** verify that tool calls actually succeeded, not just that they were made: `assert_tool_invocation_succeeded`, `assert_no_tool_errors`, `assert_tool_result_matches`, and `assert_retry_count`. See [README "Tool Result Assertions"](README.md#tool-result-assertions).
- **Per-result error signal on `Step`** via the additive `tool_results_meta` field, populated by the built-in adapters, reconstructed on cassette replay, and settable from custom converters and `step_probe`.

### Bug Fixes

- **Example test suites on Python 3.14**: `pytest examples/<name>/tests` no longer fails with a duplicate `--cassette-mode` registration during collection.

## 0.3.0 (2026-04-26)

### Features

- **Step-level assertions** for agents making multiple LLM calls per run. New `assert_step`, `assert_step_output`, and `assert_step_uses_result_from` verify tool calls, intermediate outputs, and step-to-step data flow. See [README "Step-Level Assertions"](README.md#step-level-assertions).
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
