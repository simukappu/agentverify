"""Tool results are read back from the next request for every provider's message shape."""

from __future__ import annotations

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.recorder import _block_text, _build_execution_result, _tool_results_in


def _result(histories, calls):
    interactions = [
        (NormalizedRequest(messages=h, model="m"), NormalizedResponse(tool_calls=c)) for h, c in zip(histories, calls)
    ]
    return _build_execution_result(
        interactions=interactions,
        probe_stacks=[[] for _ in interactions],
        probe_events=[],
        probe_tool_results={},
        duration_ms=None,
    )


def test_bedrock_tool_results_with_status():
    call = {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "f", "input": {}}}]}
    answer = {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}], "status": "success"}},
            {"toolResult": {"toolUseId": "t2", "content": [{"json": {"temp": 3}}], "status": "error"}},
            {"toolResult": {"toolUseId": "t3", "content": [{"text": "a"}, {"image": {}}]}},
        ],
    }
    result = _result([[{"role": "user", "content": "q"}], [call, answer]], [[{"name": "f", "arguments": "{}"}], None])
    step = result.steps[0]
    assert step.tool_results == ["ok", {"temp": 3}, [{"text": "a"}, {"image": {}}]]
    assert [m.get("is_error") for m in step.tool_results_meta] == [False, True, None]


def test_anthropic_tool_results():
    call = {"role": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "f", "input": {}}]}
    answer = {
        "role": "user",
        "content": [
            "not-a-block",
            {"type": "tool_result", "tool_use_id": "u1", "content": [{"type": "text", "text": "done"}], "is_error": True},
            {"type": "tool_result", "tool_use_id": "u2", "content": "plain"},
        ],
    }
    step = _result([[], [call, answer]], [[{"name": "f", "arguments": "{}"}], None]).steps[0]
    assert step.tool_results == ["done", "plain"]
    assert step.tool_results_meta[0]["is_error"] is True


def test_gemini_function_responses_and_other_shapes():
    call = {"role": "model", "parts": [{"function_call": {"name": "f", "args": {}}}]}
    answer = {"role": "user", "parts": [{"text": "x"}, {"function_response": {"name": "f", "response": {"r": 1}}}]}
    step = _result([[], [call, answer]], [[{"name": "f", "arguments": "{}"}], None]).steps[0]
    assert step.tool_results == [{"r": 1}]
    assert _tool_results_in("not-a-message") == []
    assert _block_text("plain") == "plain"


def test_anthropic_sdk_blocks_mark_the_last_tool_call_while_recording():
    pytest.importorskip("anthropic")
    from anthropic.types import ToolUseBlock

    def turn(n):
        return [
            {"role": "assistant", "content": [ToolUseBlock(type="tool_use", id=f"u{n}", name="f", input={})]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"u{n}", "content": f"R{n}"}]},
        ]

    calls = [{"name": "f", "arguments": "{}"}]
    steps = _result([[], turn(1), turn(1) + turn(2)], [calls, calls, None]).steps
    assert steps[1].tool_results == ["R2"]
