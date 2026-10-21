# What each evaluation tool checks

Checked against public documentation in October 2026. These tools change quickly; follow the linked pages for their current state.

agentverify runs on every PR as deterministic regression tests. The tools below are mostly used on staging for quality evaluation. The two are meant to be used together, not chosen between. Amazon Bedrock AgentCore Evaluations stands in for managed services here because it offers all three evaluator kinds side by side.

| What to check | agentverify | AgentCore built-in evaluators | AgentCore third-party (DeepEval) evaluators | AgentCore code-based evaluator | DeepEval, self-run |
|---|---|---|---|---|---|
| Tool names and their order | Asserted | Asserted (`Builtin.Trajectory*Match`) | Scored by an LLM | Your Lambda code | Asserted (`ToolCorrectnessMetric`) |
| Tool arguments | Asserted (`ANY`, `MATCHES`, `partial_args`) | Not covered | Scored by an LLM (`ToolUse`) | Your Lambda code | Scored by an LLM (`ArgumentCorrectnessMetric`) |
| Data passed between steps | Asserted (`assert_step_uses_result_from`) | Not covered | Not covered | Your Lambda code | Not covered |
| Tool success and retries | Asserted | Not covered | Not covered | Your Lambda code | Not covered |
| Token, cost, and latency budgets | Asserted | Partly, as metrics | Not covered | Your Lambda code | Partly |
| Answer quality | Not covered, by design | Scored by an LLM | Scored by an LLM | — | Scored by an LLM |
| Multi-turn user simulation | Not covered, by design | Available | Available | — | Available |
| Runs in CI with no LLM call, secret, or cloud account | Yes, on cassette replay | No | No | No | No |

A code-based evaluator can check anything, but it runs as a deployed Lambda behind the `Evaluate` API. The [execution-model benchmark](../benchmarks/execution-model-trajectory/) measures that difference: lines of code, wall time, API calls, cost, and CI secrets for the same assertion written three ways.

Other tools in the same space: [Google Vertex AI Gen AI evaluation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview) and [Azure AI Foundry evaluators](https://learn.microsoft.com/azure/ai-foundry/concepts/evaluation-evaluators/agent-evaluators) are managed services like AgentCore; [LangSmith](https://www.langchain.com/langsmith) combines tracing with dataset evaluation for the LangChain stack; [Scenario](https://github.com/langwatch/scenario) simulates a user in multi-turn conversations.

## Sources

- AgentCore Evaluations: [overview](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations), [ground-truth (trajectory) evaluators](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/ground-truth-evaluations.html), [third-party evaluators](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/third-party-evaluators.html), [code-based evaluators](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-based-evaluators.html)
- DeepEval: [Tool Correctness](https://www.deepeval.com/docs/metrics-tool-correctness), [Argument Correctness](https://www.deepeval.com/docs/metrics-argument-correctness)
