"""ToolCall.call_id: the id the framework or provider gave a tool call, filled everywhere a result is built."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentverify import ExecutionResult, MockLLM, ToolCall, assert_tool_calls, mock_response
from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse, fallback_call_id, with_call_ids
from agentverify.cassette.io import save_cassette
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder
from agentverify.errors import ToolCallSequenceError


def test_call_id_is_not_compared_and_round_trips():
    assert ToolCall("add", {"a": 1}, call_id="c1") == ToolCall("add", {"a": 1})
    result = ExecutionResult.from_dict({"steps": [{"index": 0, "tool_calls": [{"name": "add", "call_id": "c1"}, {"name": "sub"}]}]})
    assert [c.call_id for c in result.tool_calls] == ["c1", None]
    calls = result.to_dict()["steps"][0]["tool_calls"]
    assert calls[0]["call_id"] == "c1" and "call_id" not in calls[1]
    legacy = ExecutionResult.from_dict({"tool_calls": [{"name": "add", "call_id": "c2"}]})
    assert legacy.tool_calls[0].call_id == "c2"


def test_failures_show_the_id():
    result = ExecutionResult.from_dict({"steps": [{"index": 0, "tool_calls": [{"name": "add", "arguments": {"a": 1}, "call_id": "c1"}, {"name": "sub", "call_id": "c2"}]}]})
    with pytest.raises(ToolCallSequenceError, match=r'add\(a=1\)  \[c1\]') as error:
        assert_tool_calls(result, expected=[ToolCall("mul")])
    assert "sub()  [c2]" in str(error.value)


def test_fallback_ids_carry_a_per_process_token():
    first, second = fallback_call_id("toolu"), fallback_call_id("toolu")
    assert first != second and first.split("_")[2] == second.split("_")[2]
    response = NormalizedResponse(tool_calls=[{"name": "a", "id": "kept"}, {"name": "b"}, "odd"])
    filled = with_call_ids(response, "call")
    assert filled.tool_calls[0]["id"] == "kept" and filled.tool_calls[1]["id"].startswith("call_replayed_") and filled.tool_calls[2] == "odd"
    assert response.tool_calls[1] == {"name": "b"}  # the given response is not changed
    assert with_call_ids(response, None) is response and with_call_ids(NormalizedResponse(), "call").tool_calls is None


def test_mock_llm_ids_match_what_the_sdk_saw():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(api_key="sk-test", max_retries=0)
    responses = [mock_response(tool_calls=[{"name": "add", "arguments": {"a": 1}}, {"name": "sub", "arguments": {}, "id": "fixed"}])]
    with MockLLM(responses, provider="openai") as mock:
        message = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}]).choices[0].message
    result = mock.to_execution_result()
    assert [c.call_id for c in result.tool_calls] == [tc.id for tc in message.tool_calls]
    assert result.tool_calls[1].call_id == "fixed" and result.tool_calls[0].call_id.startswith("call_replayed_")


def test_replayed_ids_are_the_recorded_ones_or_filled_in(tmp_path):
    openai = pytest.importorskip("openai")
    path = tmp_path / "c.yaml"
    request = NormalizedRequest(messages=[{"role": "user", "content": "hi"}], model="m")
    save_cassette(path, [(request, NormalizedResponse(tool_calls=[{"name": "add", "arguments": {}, "id": "call_rec"}, {"name": "sub", "arguments": {}}]))], provider="openai")
    client = openai.OpenAI(api_key="sk-test", max_retries=0)
    with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY) as rec:
        message = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}]).choices[0].message
    ids = [c.call_id for c in rec.to_execution_result().tool_calls]
    assert ids == [tc.id for tc in message.tool_calls] and ids[0] == "call_rec"


def test_framework_adapters():
    from agentverify.frameworks.crewai import from_crewai
    from agentverify.frameworks.google_adk import from_google_adk
    from agentverify.frameworks.langchain import from_langchain
    from agentverify.frameworks.langgraph import from_langgraph
    from agentverify.frameworks.openai_agents import from_openai_agents
    from agentverify.frameworks.pydantic_ai import from_pydantic_ai
    from agentverify.frameworks.strands import from_strands

    strands = SimpleNamespace(messages=[{"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "f", "input": {}}}]}])
    assert from_strands(strands).tool_calls[0].call_id == "t1"

    action = SimpleNamespace(tool="f", tool_input={}, tool_call_id="lc1")
    assert from_langchain({"intermediate_steps": [(action, "ok")], "output": ""}).tool_calls[0].call_id == "lc1"

    ai = SimpleNamespace(type="ai", content="", tool_calls=[{"name": "f", "args": {}, "id": "lg1"}, SimpleNamespace(name="g", args={}, id=None)], usage_metadata=None, name=None)
    assert [c.call_id for c in from_langgraph({"messages": [ai]}).tool_calls] == ["lg1", None]

    item = SimpleNamespace(type="tool_call_item", agent=SimpleNamespace(name="A"), raw_item={"name": "f", "arguments": "{}", "call_id": "oa1"})
    run = SimpleNamespace(new_items=[item], raw_responses=[], final_output=None, context_wrapper=None)
    assert from_openai_agents(run).tool_calls[0].call_id == "oa1"

    part = SimpleNamespace(part_kind="tool-call", tool_name="f", args={}, tool_call_id="pa1")
    pai = [SimpleNamespace(kind="response", parts=[part], usage=None)]
    assert from_pydantic_ai(pai).tool_calls[0].call_id == "pa1"

    call = SimpleNamespace(name="f", args={}, id="adk1")
    event = SimpleNamespace(author="a", partial=False, content=SimpleNamespace(parts=[SimpleNamespace(function_call=call, function_response=None, text=None)]), usage_metadata=None)
    assert from_google_adk([event]).tool_calls[0].call_id == "adk1"

    message = {"role": "assistant", "tool_calls": [{"id": "cr1", "function": {"name": "f", "arguments": json.dumps({})}}], "content": None}
    crew = SimpleNamespace(tasks_output=[SimpleNamespace(agent="A", messages=[message], tool_failures=[], raw="")], token_usage=None, raw="")
    assert from_crewai(crew).tool_calls[0].call_id == "cr1"


def test_traces():
    from agentverify.traces import from_otel

    def span(name, span_id, start, attributes):
        stamp = f"2026-10-22T00:00:{start:06.3f}+00:00"
        return {"name": name, "context": {"trace_id": "t", "span_id": span_id}, "parent_id": None, "start_time": stamp, "end_time": stamp, "status": {"status_code": "UNSET"}, "attributes": attributes, "events": []}

    out = json.dumps([{"parts": [{"type": "tool_call", "name": "f", "id": "ot1"}]}])
    spans = [
        span("chat", "c", 0, {"gen_ai.operation.name": "chat", "gen_ai.output.messages": out}),
        span("execute_tool g", "x", 1, {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "g", "gen_ai.tool.call.id": "ot2"}),
    ]
    assert [c.call_id for c in from_otel(spans)[0].tool_calls] == ["ot1", "ot2"]
