# Changelog

## Unreleased

### Features

- **Built-in framework adapters**: Extract `ExecutionResult` directly from agent framework outputs without writing a custom converter
  - `from agentverify.frameworks.strands import from_strands` — Strands Agents `AgentResult` adapter
  - `from agentverify.frameworks.langchain import from_langchain` — LangChain `AgentExecutor` output adapter
  - `from agentverify.frameworks.langgraph import from_langgraph` — LangGraph `create_react_agent` result adapter
  - `from agentverify.frameworks.openai_agents import from_openai_agents` — OpenAI Agents SDK `RunResult` adapter
- **Cassette request matching**: Detect stale cassettes by verifying model name and tool names during replay
  - Enable with `--cassette-match-requests` CLI option or `match_requests=True` parameter
  - Raises `CassetteRequestMismatchError` with clear diff on mismatch

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