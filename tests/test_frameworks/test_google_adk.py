"""Tests for the Google ADK adapter (duck-typed events)."""

from types import SimpleNamespace as NS

from agentverify import Handoff
from agentverify.frameworks.google_adk import from_google_adk


def _event(author, *parts, usage=None):
    return NS(author=author, content=NS(parts=list(parts)), usage_metadata=usage)


def _part(**kwargs):
    return NS(**{"function_call": None, "function_response": None, "text": None, "thought": False, **kwargs})


def test_steps_handoffs_and_tool_results():
    events = [
        _event("user", _part(text="2+3?")),  # the user's message, authored "user", is a step without an agent
        _event("triage", _part(function_call=NS(name="transfer_to_agent", args={"agent_name": "math"})), usage=NS(prompt_token_count=5, candidates_token_count=1)),
        _event("triage", _part(function_response=NS(response={"result": None}))),
        _event("math", _part(function_call=NS(name="add", args={"a": 2})), _part(function_call=NS(name="boom", args=None))),
        _event("math", _part(function_response=NS(response={"sum": 2})), _part(function_response=NS(response={"status": "error", "error_message": "x"}))),
        _event("math", _part(text="thinking...", thought=True)),  # thoughts only: not a step
        NS(author="math", content=None, usage_metadata=None),
        _event("math", _part(text="2"), usage=NS(prompt_token_count=0, candidates_token_count=0)),
    ]
    result = from_google_adk(events)
    assert [s.agent for s in result.steps] == [None, "triage", "math", "math"]
    assert result.handoffs == [Handoff("triage", "math", 1)]
    assert result.steps[2].tool_results == [{"sum": 2}, {"status": "error", "error_message": "x"}]
    assert result.steps[2].tool_results_meta == [{}, {"is_error": True}]
    assert result.steps[2].tool_calls[1].arguments == {}
    assert result.token_usage.input_tokens == 5 and result.final_output == "2"


def test_a_response_before_any_step_is_ignored():
    result = from_google_adk([_event("math", _part(function_response=NS(response={"x": 1})))])
    assert result.steps == [] and result.token_usage is None and result.final_output is None


def test_parallel_agents_partial_events_and_response_ids():
    def call(name, call_id):
        return _part(function_call=NS(name=name, args={}, id=call_id))

    def response(call_id, value):
        return _part(function_response=NS(id=call_id, response=value))

    events = [
        NS(author="A", branch="par.A", content=NS(parts=[call("slow", "a1")]), usage_metadata=None),
        NS(author="B", branch="par.B", content=NS(parts=[call("fast", "b1")]), usage_metadata=None),
        NS(author="B", branch="par.B", content=NS(parts=[response("b1", {"v": "FAST"})]), usage_metadata=None),
        NS(author="B", branch="par.B", partial=True, content=NS(parts=[_part(text="B do")]), usage_metadata=None),
        NS(author="B", branch="par.B", content=NS(parts=[_part(text="B done")]), usage_metadata=None),
        NS(author="A", branch="par.A", content=NS(parts=[response("a1", {"v": "SLOW"})]), usage_metadata=None),
        NS(author="A", branch="par.A", content=NS(parts=[response(None, {"late": True})]), usage_metadata=None),
        NS(author="C", branch="par", content=NS(parts=[_part(text="summary")]), usage_metadata=None),
    ]
    result = from_google_adk(events)
    assert [(s.agent, s.tool_results) for s in result.steps] == [
        ("A", [{"v": "SLOW"}, {"late": True}]),
        ("B", [{"v": "FAST"}]),
        ("B", []),
        ("C", []),
    ]
    # A and B are parallel branches; C's branch contains B's, so B -> C is a handoff.
    assert [(h.from_agent, h.to_agent) for h in result.handoffs] == [("B", "C")]
