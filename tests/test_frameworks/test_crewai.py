"""Tests for the CrewAI adapter (duck-typed CrewOutput)."""

from types import SimpleNamespace as NS

from agentverify.frameworks.crewai import from_crewai


def test_function_calling_path_and_failures():
    task = NS(
        agent="researcher",
        tool_failures=[{"tool_name": "fetch", "tool_args": '{"url": "bad"}'}],
        messages=[
            {"role": "tool", "content": "before any step"},
            {"role": "system", "content": "You are a researcher."},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "fetch", "arguments": '{"url": "bad"}'}}, {"function": {"name": "search", "arguments": {"q": "x"}}}]},
            {"role": "tool", "name": "fetch", "content": "Error executing tool: 500"},
            {"role": "tool", "name": "search", "content": "found"},
            {"role": "tool", "name": "unknown", "content": "?"},
            NS(role="assistant", content="Summary written.", tool_calls=None),
        ],
    )
    result = from_crewai(NS(tasks_output=[task], token_usage=NS(prompt_tokens=10, completion_tokens=4), raw="Summary written."))
    first = result.steps[0]
    assert [(c.name, c.arguments) for c in first.tool_calls] == [("fetch", {"url": "bad"}), ("search", {"q": "x"})]
    assert first.tool_results == ["Error executing tool: 500", "found", "?"]
    assert first.tool_results_meta[0] == {"is_error": True} and first.agent == "researcher"
    assert result.steps[1].output == "Summary written." and result.final_output == "Summary written."
    assert result.token_usage.input_tokens == 10


def test_react_path():
    task = NS(
        agent="mathematician",
        tool_failures=[NS(tool_name="boom", tool_args={"x": "y"})],
        messages=[
            {"role": "assistant", "content": 'Thought: add\nAction: add\nAction Input: {"a": 2, "b": 3}\nObservation: 5'},
            {"role": "assistant", "content": 'Thought: boom\nAction: boom\nAction Input: {"x": "y"}\nObservation: I encountered an error'},
            {"role": "assistant", "content": "Thought: next\nAction: lookup\nAction Input: not json"},
        ],
    )
    result = from_crewai(NS(tasks_output=[task], token_usage=None, raw=None))
    assert [(s.tool_calls[0].name, s.tool_calls[0].arguments) for s in result.steps] == [("add", {"a": 2, "b": 3}), ("boom", {"x": "y"}), ("lookup", {"input": "not json"})]
    assert result.steps[0].tool_results == ["5"] and result.steps[0].tool_results_meta == [{}]
    assert result.steps[1].tool_results_meta == [{"is_error": True}]
    assert result.steps[2].tool_results == [] and result.steps[0].output is None
    assert result.token_usage is None and result.final_output is None


def test_empty_usage_and_non_string_agent():
    task = NS(agent=None, messages=[{"role": "assistant", "content": "x"}])
    result = from_crewai(NS(tasks_output=[task], token_usage=NS(prompt_tokens=0, completion_tokens=0), raw="x"))
    assert result.token_usage is None and result.steps[0].agent is None


def test_parallel_calls_match_results_by_call_id_and_react_tasks_get_their_answer():
    task = NS(
        agent="calc",
        raw="2.0",
        tool_failures=[{"tool_name": "div", "tool_args": {"a": 1, "b": 0}}],
        messages=[
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "div", "arguments": {"a": 1, "b": 0}}},
                {"id": "c2", "function": {"name": "div", "arguments": {"a": 4, "b": 2}}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "div", "content": "Error executing tool: division by zero"},
            {"role": "tool", "tool_call_id": "c2", "name": "div", "content": "2.0"},
        ],
    )
    result = from_crewai(NS(tasks_output=[task], token_usage=None, raw="2.0"))
    assert result.steps[0].tool_results_meta == [{"is_error": True}, {}]
    assert result.steps[-1].output == "2.0" and result.steps[-1].agent == "calc" and len(result.steps) == 2
