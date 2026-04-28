# Custom Converter — Pure-Python Anthropic ReAct Agent

Reference example for writing a [custom converter](../../README.md#custom-converters) when no built-in framework adapter fits your agent. The agent here uses the [Anthropic Python SDK](https://docs.anthropic.com/en/api/client-sdks) directly (no LangChain, LangGraph, Strands, or OpenAI Agents SDK in sight) and drives a hand-rolled ReAct tool-call loop over the Messages API. The companion [`converter.py`](converter.py) maps the agent's raw conversation history into an agentverify `ExecutionResult` that every assertion, including step-level ones, accepts unchanged.

## Why this example?

agentverify ships [built-in adapters](../../README.md#built-in-adapters) for LangChain, LangGraph, Strands Agents, and OpenAI Agents SDK. If you're using any of those, you don't need this example. `from_langchain(...)` / `from_langgraph(...)` / `from_strands(...)` / `from_openai_agents(...)` each turn the framework's native output into an `ExecutionResult` in one call.

If your agent is built without any of those frameworks (a pure-Python script calling an LLM SDK directly, a custom orchestrator, or an in-house framework), you write a small converter function that does the same job. This example shows what that converter looks like and how much code it actually takes (about 80 lines of real logic).

The agent's job is intentionally small so the converter is easy to read: given *"What's 100 + 200 with 10% tax added?"*, it chains two tools (`add` for the pre-tax total, `apply_tax` for the grossed-up final figure) and returns a one-sentence summary. The interesting assertion is that `apply_tax`'s `amount` argument must actually come from `add`'s tool result, not be hallucinated by the model.

## Prerequisites

- Python 3.10+
- Anthropic API key (only for re-recording the cassette; not needed for replay)

## Setup

See the [main README](../../README.md#examples) for setup instructions (`git clone`, venv, `pip install`).

## Running the Agent

```bash
cd examples/custom-converter-python-agent

export ANTHROPIC_API_KEY=sk-ant-your_key_here
python agent.py
```

The agent prints its final answer and the total token usage for the session.

## Running Tests

### Cassette Replay (No API Key Required)

Tests ship with a pre-recorded cassette under `tests/cassettes/`. Just run:

```bash
pytest
```

The cassette replays deterministically. No Anthropic API calls, zero cost.

### What the Tests Verify

| Test Class | Test | agentverify Assertion | Description |
|---|---|---|---|
| `TestTaxAgentFlat` | `test_tool_sequence` | `assert_tool_calls()` | Exact sequence: `add(100, 200)` then `apply_tax(_, rate=0.1)` |
| `TestTaxAgentFlat` | `test_safety_no_dangerous_tools` | `assert_no_tool_call()` | `delete_file`, `execute_command`, `transfer_funds` never appear |
| `TestTaxAgentFlat` | `test_budget_and_output` | `assert_cost()` + `assert_final_output()` via `assert_all` | Token budget ≤ 10k and the final output quotes `330` |
| `TestTaxAgentStepLevel` | `test_first_step_calls_add` | `assert_step()` | Step 0 is the assistant's first turn: exactly one `add` call |
| `TestTaxAgentStepLevel` | `test_second_step_calls_apply_tax_with_sum` | `assert_step()` | Step 1 is the assistant's second turn: `apply_tax` with some amount and the requested rate |
| `TestTaxAgentStepLevel` | `test_apply_tax_uses_add_result` | `assert_step_uses_result_from()` | **The headline check.** Step 1 actually consumes step 0's `add` result. Catches "model hallucinated a number instead of using the tool result" bugs |
| `TestTaxAgentStepLevel` | `test_final_step_has_no_tool_calls` | `assert_step()` | The last step (the summary) has zero tool calls |
| `TestTaxAgentStepLevel` | `test_final_output_contains_grossed_up_number` | `assert_final_output()` | The summary quotes the post-tax figure |

Every step-level check works on cassette replay because the converter attaches each `user` message's `tool_result` blocks onto the preceding `assistant` step's `tool_results`, exactly matching the structure agentverify's step-level API expects.

### Recording Mode (Real Anthropic API)

To re-record the cassette with real LLM calls:

1. Set the API key:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-your_key_here
   ```
2. Run **a single test** with `--cassette-mode=record`. All eight tests share the same cassette file, so running the whole suite in record mode rewrites the file once per test; recording from a single test keeps the file in a deterministic state.
   ```bash
   pytest -k test_first_step_calls_add --cassette-mode=record
   ```
3. Commit the updated cassette.

`temperature` is not explicitly pinned (Anthropic defaults apply), and `claude-haiku-4-5` is non-deterministic even at low temperature. The test assertions are written to tolerate that variance: `assert_tool_calls` uses `partial_args=True` so the exact `amount` the model passes to `apply_tax` isn't hard-coded, and `assert_final_output` matches the post-tax number via a regex that accepts both `330` and `330.00`.

## The Conversion Mapping

The SDK emits the conversation as a flat list of `user` / `assistant` messages. Each `assistant` turn becomes one step; each `user` turn carries the tool results for the previous assistant turn. [`converter.py`](converter.py) walks the list in order and shapes it like this:

| Anthropic message field | `ExecutionResult` / `Step` field |
|---|---|
| `assistant` message's `tool_use` content blocks | `step.tool_calls` |
| Next `user` message's `tool_result` content blocks | `step.tool_results` |
| `assistant` message's `text` content blocks (joined) | `step.output` |
| Conversation history up to this assistant turn | `step.input_context` |
| `response.usage.input_tokens` / `output_tokens` (summed) | `execution_result.token_usage` |
| Final assistant `text` block | `execution_result.final_output` |

The actual code is about 80 lines including docstrings and type hints. Copy [`converter.py`](converter.py) as a starting point for your own adapter and adjust the field extraction for whatever shape your agent emits.

## When to Reach for a Custom Converter

Use a custom converter when:

- Your agent is built on an SDK that agentverify doesn't ship a built-in adapter for (Anthropic direct, OpenAI direct without the Agents SDK, Cohere, Mistral, your own in-house orchestrator, etc.)
- Your agent deviates meaningfully from what a built-in adapter expects (e.g. a LangChain agent wrapped in custom pre/post-processing that changes the step structure)
- You want the conversion logic to live in your own codebase alongside tests, rather than importing it from a library

If you're using LangChain, LangGraph, Strands Agents, or OpenAI Agents SDK unmodified, prefer the built-in adapter. It's a one-liner and handles edge cases (token usage aggregation, tool-result backfill, message-format quirks) that you'd otherwise rediscover yourself.
