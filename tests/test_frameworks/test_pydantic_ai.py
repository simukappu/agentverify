"""Tests for the Pydantic AI adapter (duck-typed messages)."""

from types import SimpleNamespace as NS

from agentverify.frameworks.pydantic_ai import from_pydantic_ai


def _call(name, args, as_dict=True):
    part = NS(part_kind="tool-call", tool_name=name, args=args)
    if as_dict:
        part.args_as_dict = lambda: args if isinstance(args, dict) else {}
    return part


def test_steps_tool_results_and_errors():
    def broken():
        raise ValueError("bad json")

    weird = _call("w", None)
    weird.args_as_dict = broken
    messages = [
        NS(kind="request", parts=[NS(part_kind="user-prompt", content="hi")]),  # before any response
        NS(
            kind="response",
            parts=[_call("add", {"a": 1}), _call("raw", '{"x": 2}', as_dict=False), _call("text", "plain", as_dict=False), weird, NS(part_kind="thinking")],
            usage=NS(input_tokens=10, output_tokens=3),
        ),
        NS(
            kind="request",
            parts=[
                NS(part_kind="tool-return", content=1, outcome="success"),
                NS(part_kind="tool-return", content="denied", outcome="denied"),
                NS(part_kind="retry-prompt", content="fix it", tool_name="text"),
                NS(part_kind="retry-prompt", content="output invalid", tool_name=None),
            ],
        ),
        NS(kind="request", parts=[NS(part_kind="user-prompt", content="more")]),  # no tool results
        NS(kind="response", parts=[NS(part_kind="text", content="done")], usage=NS(input_tokens=0, output_tokens=0)),
        NS(kind="response", parts=[], usage=None),
    ]
    result = from_pydantic_ai(NS(new_messages=lambda: messages, output=None), agent_name="calc")
    step = result.steps[0]
    assert [(c.name, c.arguments) for c in step.tool_calls] == [("add", {"a": 1}), ("raw", {"x": 2}), ("text", {"input": "plain"}), ("w", {})]
    assert step.tool_results == [1, "denied", "fix it"]
    assert step.tool_results_meta == [{"is_error": False}, {"is_error": True}, {"is_error": True}]
    assert step.agent == "calc" and step.token_usage.input_tokens == 10
    assert result.steps[1].token_usage is None and result.final_output == "done"
    assert result.token_usage.output_tokens == 3


def test_message_list_and_string_output():
    messages = [NS(kind="response", parts=[NS(part_kind="text", content="x")])]
    assert from_pydantic_ai(messages).final_output == "x"
    result = from_pydantic_ai(NS(new_messages=lambda: messages, output="final"))
    assert result.final_output == "final" and result.token_usage is None


def test_output_tool_calls_are_not_tool_calls_and_structured_output_is_text():
    messages = [
        NS(kind="response", parts=[_call("final_result", {"n": 5}), _call("fast", {})]),
        NS(kind="request", parts=[NS(part_kind="tool-return", tool_name="final_result", content="Final result processed.", outcome="success"), NS(part_kind="tool-return", tool_name="fast", content="F", outcome="success")]),
    ]
    result = from_pydantic_ai(NS(new_messages=lambda: messages, output={"n": 5}))
    assert [c.name for c in result.steps[0].tool_calls] == ["fast"] and result.steps[0].tool_results == ["F"]
    assert result.final_output == "{'n': 5}"
