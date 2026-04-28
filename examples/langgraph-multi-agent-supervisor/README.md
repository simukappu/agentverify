# LangGraph Multi-Agent Supervisor Example

A multi-agent supervisor workflow built with [LangGraph](https://github.com/langchain-ai/langgraph) and [langgraph-supervisor](https://github.com/langchain-ai/langgraph-supervisor-py). A central supervisor coordinates a research expert and a math expert to answer a compound question. Tested with [agentverify](https://github.com/simukappu/agentverify).

The example mirrors the official [langgraph-supervisor quickstart](https://github.com/langchain-ai/langgraph-supervisor-py#quickstart). Given the question *"What's the combined headcount of the FAANG companies in 2024?"* the supervisor delegates: the research expert fetches the per-company numbers via a `web_search` tool, then the math expert adds them up with an `add` tool. The cassette captures ~12 observable steps, which agentverify asserts against step-by-step.

## Why this example?

This is the one place in agentverify where the **step-level + data-flow testing** story shines the brightest:

- **Multi-agent handoff**: the supervisor's routing decisions are distinct steps, separate from the sub-agents' tool work.
- **Parallel tool calls**: the research expert issues multiple `web_search` calls in a single step; `assert_step` with `expected_tools` + `OrderMode.ANY_ORDER` handles this cleanly.
- **Cross-step numeric data flow**: each `add` call's running total feeds the next `add`'s input. `assert_step_uses_result_from` verifies the chain didn't drop a value.

## Prerequisites

- Python 3.10+
- OpenAI API key (only for re-recording cassettes; not needed for replay)

## Setup

See the [main README](../../README.md#examples) for setup instructions (`git clone`, venv, `pip install`).

## Running the Agent

```bash
cd examples/langgraph-multi-agent-supervisor

export OPENAI_API_KEY=sk-your_key_here
python agent.py
```

The agent prints each message in the conversation. You'll see the supervisor route to `research_expert`, the researcher call `web_search` and summarise, the supervisor hand off to `math_expert`, and finally the math expert chain several `add` calls into the final answer.

## Running Tests

### Cassette Replay (No API Key Required)

Tests ship with a pre-recorded cassette under `tests/cassettes/`. Just run:

```bash
pytest
```

The cassette replays deterministically. No OpenAI API calls, zero cost.

### What the Tests Verify

| Test Class | Test | agentverify Assertion | Description |
|---|---|---|---|
| `TestSupervisorFlat` | `test_tool_sequence_in_order` | `assert_tool_calls()` | End-to-end sequence: `transfer_to_research_expert` → `web_search` → `transfer_to_math_expert` → `add` (IN_ORDER) |
| `TestSupervisorFlat` | `test_safety_no_dangerous_tools` | `assert_no_tool_call()` | The supervisor never hands off to non-existent agents and never runs destructive tools |
| `TestSupervisorFlat` | `test_budget_and_output` | `assert_cost()` + `assert_final_output()` via `assert_all` | Token budget ≤ 20k and final answer contains a numeric total |
| `TestSupervisorStepLevel` | `test_supervisor_routes_to_research_first` | `assert_step()` | Step 0 is the supervisor handing off to the research expert, with no domain tool calls yet |
| `TestSupervisorStepLevel` | `test_research_agent_uses_web_search` | `assert_step(order=ANY_ORDER)` | The research step contains at least one `web_search` call whose query mentions FAANG / headcount / employees. Uses `ANY_ORDER` to handle parallel tool calls in the same step |
| `TestSupervisorStepLevel` | `test_math_agent_adds_with_research_derived_numbers` | `assert_step_uses_result_from()` | The second `add` step consumes the first `add` step's result. Catches the "agent dropped the running total" bug |
| `TestSupervisorStepLevel` | `test_final_answer_includes_a_number` | `assert_step_output()` | The final summary step produces text matching `\d[\d,]*` |

Step-level testing works on cassette replay because agentverify backfills tool results from the next step's LLM input. See [Step-Level Assertions](../../README.md#step-level-assertions) in the main README.

### Recording Mode (Real OpenAI API)

To re-record the cassette with real LLM calls:

1. Set the API key:
   ```bash
   export OPENAI_API_KEY=sk-your_key_here
   ```
2. Run the tests with `--cassette-mode=record`:
   ```bash
   pytest --cassette-mode=record
   ```
   The cassette file is overwritten with a fresh recording.
3. Commit the updated cassette.

Note: the supervisor's exact routing and the number of `web_search` calls can vary between recordings because `gpt-4o-mini` is non-deterministic even at `temperature=0`. The test assertions are written to tolerate that variance: `OrderMode.IN_ORDER` for the flat sequence, `ANY_ORDER` for in-step parallel tool calls, and helper functions that locate steps by tool name rather than hard-coded indices.

## Framework Adapter

This example uses the built-in adapter:

```python
from agentverify.frameworks.langgraph import from_langgraph

raw = app.invoke({"messages": [{"role": "user", "content": query}]})
execution_result = from_langgraph(raw)
```

The adapter walks the `messages` list, promoting each `AIMessage` to a `Step` and attaching subsequent `ToolMessage` content as the step's `tool_results`. See [Framework Integration](../../README.md#framework-integration) in the main README for details.
