"""How failures read: pytest points at the test's line, and assert_all indents each failure under its number."""

from __future__ import annotations

import pytest

from agentverify import ExecutionResult, ToolCall, assert_all, assert_cost, assert_tool_calls
from agentverify.errors import MultipleAssertionError

pytest_plugins = ["pytester"]


def test_tracebacks_stop_at_the_test(pytester):
    pytester.makepyfile(
        """
from agentverify import ExecutionResult, ToolCall, assert_step, assert_tool_calls

def test_x():
    result = ExecutionResult.from_dict({"steps": [{"index": 0, "tool_calls": [{"name": "a"}]}]})
    assert_step(result, step=0, expected_tool=ToolCall("b"))
"""
    )
    out = pytester.runpytest_inprocess("--tb=short").stdout.str()
    assert "ToolCallSequenceError" in out and "assertions.py" not in out


def test_assert_all_indents_each_failure():
    result = ExecutionResult.from_dict({"tool_calls": [{"name": "a"}], "token_usage": {"input_tokens": 10, "output_tokens": 0}})
    with pytest.raises(MultipleAssertionError) as error:
        assert_all(result, lambda r: assert_tool_calls(r, expected=[ToolCall("b")]), lambda r: assert_cost(r, max_tokens=1))
    lines = str(error.value).split("\n")
    assert lines[0] == "2 assertions failed:"
    first = lines.index(next(line for line in lines if line.startswith("  [1] ToolCallSequenceError")))
    assert lines[first + 2] == "      Expected:" and any(line.startswith("  [2] CostBudgetError") for line in lines)
    assert "        Exceeded by: 9 tokens (900.0%)" in lines


def test_cassette_errors_hide_agentverify_frames_but_other_errors_keep_them(pytester):
    pytester.makepyfile(
        """
from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.io import save_cassette
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder

def test_stale(tmp_path):
    path = tmp_path / "c.yaml"
    save_cassette(path, [(NormalizedRequest(messages=[{"role": "user", "content": "a"}], model="m"), NormalizedResponse(content="ok"))], provider="openai")
    with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY) as rec:
        rec.lookup(NormalizedRequest(messages=[{"role": "user", "content": "b"}], model="m"))

def test_bug(tmp_path):
    path = tmp_path / "c.yaml"
    save_cassette(path, [(NormalizedRequest(messages=[], model="m"), NormalizedResponse(content="ok"))], provider="openai")
    with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY) as rec:
        rec.lookup(None)  # not a request: an error from inside agentverify, whose frames stay
"""
    )
    out = pytester.runpytest_inprocess("--tb=short", "-p", "no:agentverify").stdout.str()
    stale, bug = out.split("_ test_bug _")
    assert "CassetteRequestMismatchError" in stale and "recorder.py" not in stale
    assert "recorder.py" in bug
