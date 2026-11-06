# Tool Result Error Signals

The details behind [Tool Result Assertions](../README.md#tool-result-assertions) in the README.

agentverify reads a per-result error signal that the built-in adapters populate from each framework's native tool-error convention:

- LangGraph `ToolMessage.status == "error"`
- Strands `toolResult` status
- OpenAI Agents SDK default failure and timeout messages
- Pydantic AI retry prompts and tool return outcomes
- CrewAI `tool_failures`
- error-shaped results such as ADK's `{"status": "error"}`

Cassette replay reconstructs it from the recorded tool messages (OpenAI tool messages, Anthropic `is_error`, Bedrock `toolResult` status).

The signal is three-valued: a result is a **known error**, a **known success**, or **unknown** (for example, a plain-string tool output on cassette replay where no structured status survived). `assert_tool_invocation_succeeded` and `assert_no_tool_errors` fail only on a *known* error; unknown and absent results pass, so the same test gives the same verdict whether it runs against a live agent or a replayed cassette. To require that a result is actually present and well-formed, use `assert_tool_result_matches`, which fails when the step has no tool result at all.

If you build an `ExecutionResult` by hand or with a custom converter, set the signal yourself via the `tool_results_meta` field on each `Step`: a list positionally aligned with `tool_results`, where entry `i` is `{"is_error": True}`, `{"is_error": False}`, or `{}` (unknown). With `step_probe`, pass it through `handle.set_tool_result(result, is_error=True)`.
