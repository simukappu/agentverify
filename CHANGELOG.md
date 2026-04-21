# Changelog

## Unreleased

### Features

- **Regex argument matcher**: New `MATCHES(pattern)` matcher for verifying string tool call arguments against a regex
  - Works like `ANY` but constrained to a `re.search(pattern, value)` match
  - Accepts raw pattern strings or pre-compiled `re.Pattern` objects
  - Non-string values never match
- **Tool mocking**: In-memory LLM response replay for testing agent routing without a cassette or a real LLM call
  - New `MockLLM(responses, provider=...)` context manager
  - New `mock_response(content=..., tool_calls=..., input_tokens=..., output_tokens=...)` builder
  - Raises `CassetteMissingRequestError` when the agent makes more LLM calls than predefined responses
- **Latency assertion**: `assert_latency(result, max_ms=3000)` to enforce response time SLAs on agent executions
  - New `duration_ms` field on `ExecutionResult` for wall-clock execution time
  - Wall-clock duration is captured automatically by the `cassette` fixture and by `MockLLM` on context exit
  - New `LatencyBudgetError` with formatted diff output
  - Supports `strict=True` to require that `duration_ms` is present

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