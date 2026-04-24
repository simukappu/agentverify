# OpenAI Agents SDK — LLM as a Judge Example

An iterative refinement agent built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/): a generator writes a short story outline, an evaluator grades it with a structured verdict, and the loop re-runs the generator with the evaluator's feedback until it accepts the draft. Tested with [agentverify](https://github.com/simukappu/agentverify).

Adapted from the official [`agent_patterns/llm_as_a_judge.py`](https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/llm_as_a_judge.py) sample.

## Why this example?

This is where agentverify's value proposition shows the most contrast. The underlying agent is *probabilistic*:

- how many rounds the refinement loop takes depends on what the evaluator says
- the evaluator's feedback text is free-form — it differs run to run
- the generator's rewrite is only as good as its ability to incorporate that feedback

Recording a single concrete run into a cassette freezes all of that. Every assertion below then runs deterministically in CI with no API key, and `assert_step_uses_result_from` verifies that the second generator round actually consumed the evaluator's feedback — catching the classic "agent regenerated but ignored the judge" bug without any real LLM call.

## Prerequisites

- Python 3.10+
- OpenAI API key (only for re-recording the cassette — not needed for replay)

## Setup

See the [main README](../../README.md#examples) for setup instructions (`git clone`, venv, `pip install`).

## Running the Agent

```bash
cd examples/openai-agents-llm-as-a-judge

export OPENAI_API_KEY=sk-your_key_here
python agent.py
```

The agent prints the number of rounds the loop took, whether it reached a `pass` verdict, and the final story outline.

## Deviations from the official sample

Two small changes let the sample run under agentverify's cassette layer while preserving the original pattern:

1. `set_default_openai_api("chat_completions")` is called at module import so agentverify's existing `openai` cassette adapter (Chat Completions API) can record the interactions. A Responses API adapter is on the roadmap but not required for this example.
2. The evaluator's "when may I pass?" rule is tightened from *"after 5 attempts"* to *"on the second attempt once your earlier feedback has been incorporated"*. The intent — never pass on the first try, require feedback-driven revision — is preserved; the shorter horizon keeps the recording within a predictable round budget.

The loop wrapper is packaged as `run_judge_loop` / `run_judge_loop_sync` returning a `JudgeLoopResult` dataclass so tests can assert on the loop shape (rounds, pass_reached) without re-running the agent.

## Running Tests

### Cassette Replay (No API Key Required)

Tests ship with a pre-recorded cassette under `tests/cassettes/`. Just run:

```bash
pytest
```

The cassette replays deterministically — no OpenAI API calls, zero cost.

### What the Tests Verify

| Test Class | Test | agentverify Assertion | Description |
|---|---|---|---|
| `TestJudgeLoopFlat` | `test_loop_terminates_within_round_cap` | `JudgeLoopResult` shape | The loop reaches `pass` in ≥ 2 rounds and `len(result.steps) == 2 * rounds` (generator + evaluator per round) |
| `TestJudgeLoopFlat` | `test_safety_no_tool_calls` | `assert_no_tool_call()` | Neither agent calls any of the blacklisted tools |
| `TestJudgeLoopFlat` | `test_budget_and_output` | `assert_cost()` + `assert_final_output()` via `assert_all` | Token budget ≤ 20k and the final output is non-empty |
| `TestJudgeLoopStepLevel` | `test_first_generator_step_has_no_tool_calls` | `assert_step()` + `assert_step_output()` | Step 0 is the initial generator — no tool calls, non-empty outline text |
| `TestJudgeLoopStepLevel` | `test_first_evaluator_rejects_on_first_try` | `assert_step_output()` | Step 1 is the evaluator's first verdict — structured output carries `"needs_improvement"` or `"fail"`, never `"pass"` |
| `TestJudgeLoopStepLevel` | `test_regeneration_consumes_evaluator_feedback` | `assert_step_uses_result_from()` | **The headline check.** Step 2 (generator, round 2) must reference step 1's feedback — catches "agent ignored the judge" bugs |
| `TestJudgeLoopStepLevel` | `test_final_evaluator_step_emits_pass` | `assert_step_output()` | The last step is an evaluator verdict with `"score": "pass"` |
| (module level) | `test_no_tool_calls_anywhere` | direct assertion | Neither agent defines any tools — `result.tool_calls` is empty and every step's `tool_calls` list is empty |

Step-level checks work on cassette replay because agentverify reconstructs each step's `input_context` from the recorded LLM request, and the consumed-leaf walker in `assert_step_uses_result_from` tolerates the multi-line text formatting the evaluator produces.

### Recording Mode (Real OpenAI API)

To re-record the cassette with real LLM calls:

1. Set the API key:
   ```bash
   export OPENAI_API_KEY=sk-your_key_here
   ```
2. Run **a single test** with `--cassette-mode=record`. Every cassette-backed test shares the same file, so running the whole suite in record mode rewrites the cassette once per test; recording from a single test keeps the file in a deterministic state.
   ```bash
   pytest -k test_loop_terminates_within_round_cap --cassette-mode=record
   ```
3. Commit the updated cassette.

Note: `gpt-4o-mini` is non-deterministic even at `temperature=0`, so re-recording may change exact token counts and produce slightly different feedback text. The test assertions tolerate that variance — flat tests check budget ceilings and verdict shape, step-level tests match on structural patterns (tool counts, `"score"` regex) rather than exact text.

## Framework Adapter

This example relies on `LLMCassetteRecorder.to_execution_result()` directly rather than `from_openai_agents` — because the loop spans multiple `Runner.run` calls, the cassette-level recorder is the natural single source of truth for the full sequence of steps:

```python
with cassette("detective_in_space.yaml", provider="openai") as rec:
    run_judge_loop_sync(query, max_rounds=3)
result = rec.to_execution_result()
```

`from_openai_agents` remains the right adapter for single-`Runner.run` workflows; see the [main README's Framework Integration section](../../README.md#framework-integration).
