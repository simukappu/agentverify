# Assertion Execution Model Comparison Benchmark — Design

This document explains *why* the benchmark is shaped the way it is. The [README](./README.md) is the operating manual; this is the methodology.

Throughout this document, *assertion execution model* (or simply *execution model*) refers to where and how the assertion code runs: inside the test process, in a separate runner process, or in a managed remote runtime.

## Background: the question this benchmark answers

Three classes of tools can deterministically check an AI agent's **execution trajectory** (the actual sequence of tool calls a run produced):

1. An inline pytest plugin that intercepts the LLM SDK and runs assertions in the same Python process as the test (agentverify).
2. A trace-based testing framework that decorates the agent under test with `@observe` to capture a live trace, and runs deterministic metrics (`ToolCorrectnessMetric` plus user-supplied checks) over the captured trace (DeepEval).
3. A managed evaluation service that consumes recorded sessions through a data-plane API and runs a user-supplied function in a separate runtime (Amazon Bedrock AgentCore Evaluations - Custom code-based evaluator on Lambda).

All three can implement the same logical assertion. The user-visible difference is the **assertion execution model**: where the assertion runs, what it needs in addition to the assertion code, and what it costs per execution. This benchmark measures that difference on the same underlying agent and the same underlying assertion.

## Two scenarios: why the benchmark needs both

The benchmark runs each cell under two scenarios that correspond to two real-world entry points for trajectory testing.

### Scenario 1 - Local dev / first run

The agent has never been exercised against this assertion before. There is no cassette, no recorded trajectory, no fixture. The test must drive the agent end-to-end against a real LLM and assert against whatever the agent actually did. All three execution models pay the same agent runtime + LLM cost; the benchmark measures the **framework overhead** each one adds on top of that.

This is what every test author starts from when they write a new trajectory check, and what authoring engineers feel as "first iteration latency."

### Scenario 2 - CI / repeat run

The same assertion runs on every PR. The agent does not need to be re-driven against the LLM unless the test explicitly chooses to. The benchmark measures **what each execution model can avoid re-paying**. This is the question CI cost owners ask: "for the test we've already written, what does it cost me 1000 times a month?"

The execution-model behaviour under Scenario 2 differs sharply:

| Execution model | Mechanism for skipping LLM in CI | Result |
|---|---|---|
| A. agentverify | Cassette replay: a previously recorded LLM session is replayed deterministically inside the test process. | LLM cost = 0, agent runtime ~ cassette parse + assertion. |
| B. DeepEval | None: DeepEval has no first-class "trajectory cassette" or "trace fixture" mechanism. The documented CI pattern is `@observe` + `deepeval test run` against a live agent. | LLM cost = same as Scenario 1, agent runtime = same as Scenario 1. |
| C. AgentCore Evaluations | None: AgentCore Evaluations is positioned around production-recorded sessions. There is no documented CI pattern that skips agent runtime; tests typically deploy an agent (or run it under AgentCore Runtime), drive it with test prompts, and evaluate the resulting session. | LLM cost = same as Scenario 1, agent runtime = same as Scenario 1, plus AgentCore Evaluations + Lambda. |

The benchmark deliberately does **not** invent a "trajectory fixture" path for B or C. Doing so would compare a documented agentverify capability with a custom benchmark scaffold that no DeepEval or AgentCore Evaluations user is expected to write, which is the wrong axis.

The previous iteration of this benchmark did exactly that: a `trajectory_record.json` was hand-authored once and consumed by all three execution models. That made the comparison "assertion engine cost over a JSON fixture," which is a different question and which obscures the cassette mechanism that is agentverify's core capability. Scenario 2 in the current design restores the apples-to-apples comparison.

## Why two subjects, not one

Two existing examples are reused as agents under test, matching the framework set that Amazon Bedrock AgentCore Evaluations supports at the SDK level. Picking two serves three purposes simultaneously:

- **Framework support coverage**: Strands and LangGraph are the two frameworks AgentCore Evaluations supports at the SDK level. Pairing them exercises the AgentCore Evaluations code-based evaluator path on both supported frameworks instead of just one.
- **Single-agent vs multi-agent expressiveness**: DeepEval's `ToolCorrectnessMetric` treats a run as a flat tool-call list. Multi-agent handoff and `add` chaining surface whether that flat-list view actually loses information that the agentverify step model retains. Multi-agent is also where the AgentCore Evaluations Lambda-driven Custom code-based evaluator pattern shows its setup overhead more clearly.
- **Weight of evidence**: a single subject reads as a special case. Two subjects with materially different shapes (single-agent ReAct vs multi-agent supervisor) make the execution-model argument harder to dismiss as cherry-picking.

LangChain GitHub Issue Triage was considered as a third subject but dropped: AgentCore Evaluations does not have first-class LangChain SDK support, so a benchmark on that subject would test something the AgentCore Evaluations execution model is not designed to do.

## Why the assertion has to go beyond tool-name ordering

The "execution trajectory" is the actual sequence of tool calls a run produced. Several things are worth checking about it:

- **Tool selection and ordering**: which tools were called, and in what order.
- **Argument correctness**: were the arguments well-formed and matched the intent.
- **Tool invocation outcome**: did calls succeed, were errors handled.
- **Cross-step data flow**: did the next step actually consume the previous step's result, or did the agent hallucinate a value that does not trace back to anything earlier.
- **Efficiency**: did the run avoid redundant calls.

AgentCore Evaluations exposes two paths for code-based deterministic checking:

- A **built-in trajectory matcher** (`Builtin.TrajectoryExactOrderMatch` / `Builtin.TrajectoryInOrderMatch` / `Builtin.TrajectoryAnyOrderMatch`) that takes an `expectedTrajectory` field as a list of tool names. It handles tool selection and ordering only, and it does not require a Lambda.
- A **Custom code-based evaluator** that takes a user-supplied function and runs it on Lambda. It can express arbitrary deterministic checks, but it has the full Lambda + IAM + on-demand-invocation overhead.

If this benchmark's assertion stopped at tool-name ordering, the AgentCore Evaluations execution model would effectively become the built-in matcher, which is a different product from the Custom code-based evaluator. The benchmark would then be comparing apples (inline pytest) to a different apple (managed tool-name matcher), instead of apples to oranges (managed code-based evaluator). To compare like with like at the "arbitrary deterministic check" tier across all three execution models, the assertion must require something the built-in matcher cannot express. Both subject-level assertions therefore require **argument matching** and **cross-step data flow** in addition to tool ordering.

## The two assertions, in plain English

**Strands Weather Forecaster (single-agent, two-step ReAct):**

> Step 0 calls `http_request` with `method=GET` and a URL matching `/points/`. Step 1 calls `http_request` with `method=GET` and a URL matching `/forecast`. The URL used in step 1 contains data returned by step 0's response body.

The data-flow piece (third clause) reflects the agent's actual behaviour: the NWS API requires a "points" lookup first, whose response body contains the URL for the forecast endpoint. A correct run must thread that URL forward.

**LangGraph Multi-Agent Supervisor (multi-agent handoff with `add` chain):**

> Supervisor hands off to `research_expert` first. The research step issues at least one `web_search` whose query mentions FAANG / headcount / employees. Control returns to the supervisor, which hands off to `math_expert`. The second `add` step's `a` argument equals the first `add` step's result.

The data-flow piece (last clause) catches the "agent dropped the running total and hallucinated a fresh number" failure mode, which is one of the common bug shapes for multi-step arithmetic agents.

## Per-execution-model implementation

For each subject, the per-execution-model implementation is:

| Execution model | Tool ordering | Argument matching | Cross-step data flow |
|---|---|---|---|
| A. agentverify | `assert_step` with `expected_tool=ToolCall(...)` | `MATCHES(pattern)` inside `ToolCall` arguments | `assert_step_uses_result_from(result, step=N, depends_on=N-1)` |
| B. DeepEval deterministic mode | `ToolCorrectnessMetric(should_consider_ordering=True)` over the `LLMTestCase.tools_called` list captured by `@observe` | a custom deterministic check that walks `tools_called` and validates argument shape | a custom deterministic check that compares one tool call's `output` field to a substring/equality of another's `input_parameters` |
| C. AgentCore Evaluations Custom code-based evaluator | a user-supplied Lambda function that walks the OTLP span attributes carrying the recorded tool-call sequence | the same Lambda, walking the recorded tool-call arguments | the same Lambda, walking the inter-step data lineage |

DeepEval's `should_consider_ordering=True` and `should_exact_match=False` keep `ToolCorrectnessMetric` deterministic; if these are misconfigured, DeepEval can silently fall back to an LLM-as-a-Judge path, which would move the comparison out of the deterministic tier and break the benchmark's apples-to-apples property. Each `deepeval_test.py` pins these flags explicitly and the results file records the DeepEval version that was tested.

For execution model C, the test client drives the agent under test in-process, observes its tool calls (via the framework's callback or message-history walk), and packs them into `span.attributes["agentverify.tool_calls"]` as a JSON-encoded string before calling the AgentCore Evaluations data-plane `Evaluate` API. The Lambda walks that attribute. This shape is necessary because the `Evaluate` API validates the payload against the OTLP span schema and silently drops content placed on `span_events.body`, while `span.attributes` survive intact.

## Metrics and what each one captures

The README lists the primary metrics. The reasoning behind each:

- **Implementation LOC (test + wiring)** captures developer effort per assertion. The LOC counting rule (below) is what makes this comparable across execution models.
- **Test execution wall time (trimmed mean of middle 3 of 5)** captures the inner-loop cost. PR feedback latency lives here. We run 5 invocations per cell, drop the single highest and lowest, and average the remaining 3. This is more stable against transient outliers (a one-off slow Bedrock cold path on run 1, an unrelated GC pause on run 4) than a best-of-5, while still being robust enough to ignore stochastic spikes. Wall time is reported separately for Scenario 1 and Scenario 2.
- **API calls per test run** captures the network surface. Reported separately for Scenario 1 and Scenario 2 because A drops to zero LLM calls under Scenario 2 while B and C do not.
- **Dollar cost per test run** is derived from API call count and current public pricing. It is the bridge between per-test cost and CI scale-out cost. Reported separately for the two scenarios.
- **CI secrets required** is the credentials surface a CI job must hold. Each secret is an attack surface and an operational artifact. This is reported per scenario because A's secrets surface collapses under Scenario 2 (no LLM credentials needed during CI) while B and C still need their LLM and AWS credentials.
- **Cold start overhead** is the time from "I asked the test to run" to "the assertion is actually running." For A and B this is process startup and module import; for C it includes Lambda init time. Cold start is captured once (it is the same in either scenario for B and C; for A under Scenario 2 it is the time to first cassette parse + assertion).

### LOC counting rule

LOC counts only the **assertion-essential** lines: the `assert_*` calls (or metric definitions) plus the minimum wiring that makes them runnable (decorators, fixtures, expected-trajectory definitions). Excluded:

- Import statements.
- Module docstrings and comments.
- IAM / AWS resource creation scripts (counted under setup LOC, footnote).
- Reusable helpers that would be shared across multiple tests in a real project (e.g. the bench-local agent factory module).
- The per-scenario test wrapper bodies. Each `*_test.py` ships a `_dev` and a `_ci` wrapper that obtain the trajectory differently (live agent run vs cassette replay vs JSON-fixture-style payload) and then delegate to the shared `_assert_trajectory` helper. The wrappers are wiring; the helper is the assertion. The benchmark counts the helper, once per cell, and that count applies to both scenarios.

The same rule is applied to all three execution models. Each results file pins the rule it used and the exact files it counted.

For execution model C, the counted block is the sum of the test-side assertion block (in `agentcore_test.py`) and the Lambda-side assertion block (in `agentcore_evaluator_cdk/lambda_src/lambda_function.py`). The two move together: changing the assertion (e.g. tightening the URL match, adding a new step check) touches both the payload construction on the test side and the walker on the Lambda side. Counting only one would underreport the per-assertion development cost. The CDK app, IAM resources, deploy / destroy scripts, and the AgentCore evaluator registration are one-time setup and live in the [Setup LOC](#setup-loc-footnote-not-primary) footnote.

### Setup LOC (footnote, not primary)

Setup LOC is a separate, footnoted metric. It counts the one-time wiring: `@observe` plumbing and metric construction for B; CDK app, Lambda deployment, IAM policy, and AgentCore Evaluations evaluator registration for C; the existing `cassette` fixture for A. Setup LOC amortises across many tests in a real project and is therefore less informative than per-assertion LOC, but it is recorded so that "the assertion itself is short" claims for B and C can be checked against the full picture.

## Non-goals

The benchmark deliberately does not cover several things, in the interest of keeping the comparison clean:

- **LLM-as-a-Judge modes** of DeepEval or AgentCore Evaluations. Those are LLM-judged scoring, not code-based deterministic checks. Mixing them would compare an evaluator type with an execution model, which is two variables changing at once.
- **A trajectory-fixture path for B or C.** As discussed under "Two scenarios", this would lend agentverify's cassette capability to the other two execution models in a way that no DeepEval / AgentCore Evaluations user is expected to write. The previous iteration of this benchmark did this, and the resulting numbers obscured the cassette mechanism. The current design restores apples-to-apples by letting each execution model use its own documented CI pattern.
- **Load / scaling tests.** The question is per-assertion ergonomics, not throughput. A 10x scale-out study is a different benchmark.
- **Accuracy comparison on a correct run.** All three execution models are deterministic by design; they should agree on a correct run. If any disagrees, that is a finding worth flagging in the observation bullets, but it is not a primary metric.
- **CI integration on every commit.** The benchmark is a one-time run that refreshes when tooling versions change materially. Wiring it into per-commit CI would distort the AgentCore Evaluations execution model measurement (cold start dominates if invoked rarely; warm start dominates if invoked frequently) and would also incur ongoing AWS charges.

## Apples-to-apples scoping

For a given (subject, scenario) pair, the three execution models start from the same point and stop at the same point:

- **Same agent**: each subject has a single bench-local agent factory (`_strands_agent_factory.py` / `_langgraph_agent_factory.py` next to the tests) that pins the model id and the agent construction path. All three execution models import that factory.
- **Same assertion**: the three implementations encode the same logical clauses (tool ordering + argument matching + cross-step data flow). The README has the plain-English version per subject.
- **Same scenario semantics**: under Scenario 1 all three run a live agent. Under Scenario 2 all three run their own documented CI pattern (cassette replay for A, live agent for B and C). The benchmark does not bridge between them with a custom JSON fixture.

This is what lets the benchmark report differences in execution model honestly: each cell uses the assertion-execution-model that its product actually ships.

## Trajectory data — where it comes from

There is **no benchmark-only trajectory fixture file**. Each scenario gets its trajectory from the path that fits the scenario:

- **Scenario 1** (all three execution models, both subjects): the agent is driven against a real LLM. agentverify reads the resulting cassette (record mode); DeepEval reads the `@observe` trace; AgentCore Evaluations reads the in-process callback observations packed into an OTLP span.
- **Scenario 2 / agentverify**: the canonical example cassette under `examples/<subject>/tests/cassettes/` is replayed.
- **Scenario 2 / DeepEval and AgentCore Evaluations**: same as Scenario 1.

If the example cassettes are re-recorded (e.g., the agent's prompt changes), Scenario 2 / agentverify reflects the new cassette automatically. There is no separate JSON file to keep in sync.

## Reading the results

The results document under `results/` reports the metrics for each cell × scenario, plus three to five observation bullets per scenario. The bullets are the qualitative claims that the numbers support; the numbers are the evidence. When citing the benchmark elsewhere, cite both: the qualitative claim alone is too easy to over-read, and the numbers alone leave too much interpretation work for the reader.

The two scenarios tell different stories. Scenario 1 is the "is the framework I just installed in my way" story; small differences here are interpretable as overhead. Scenario 2 is the "what does this cost me 1000 times a month" story; orders-of-magnitude differences here are interpretable as a structural cost difference between execution models.
