"""iter_request_texts: every text a request sends the model, with where it sits, across provider shapes."""

from __future__ import annotations

import pytest

from agentverify import MockLLM, iter_request_texts, mock_response
from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.io import save_cassette
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder


def _texts(request):
    return dict(iter_request_texts(request))


def test_openai_and_litellm_shapes():
    request = NormalizedRequest(
        model="m",
        messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 22, "url": "data:image/png;base64,AAAA"}'},
            {"role": "user", "content": [{"type": "text", "text": "and?"}, {"type": "image_url", "image_url": {"url": "https://x/a.png?sig=1"}}]},
        ],
        tools=[{"name": "get_weather", "description": "Weather.", "parameters": {"properties": {"unit": {"enum": ["C", 1]}}}}],
        parameters={"temperature": 0.2, "instructions": "be kind"},
    )
    texts = _texts(request)
    assert texts["request[0].message[0](system).text"] == "be brief"
    assert texts["request[0].message[2](assistant).tool_call[0](get_weather).arguments.city"] == "Tokyo"
    assert texts["request[0].message[3](tool).tool_result[0](get_weather).temp"] == "22"
    assert "request[0].message[3](tool).tool_result[0](get_weather).url" not in texts  # an embedded image
    assert texts["request[0].message[4](user).content[1].image_url.url"] == "https://x/a.png?sig=1"
    assert texts["request[0].tool[get_weather].description"] == "Weather." and texts["request[0].tool[get_weather].schema.properties.unit.enum[1]"] == "1"
    assert texts["request[0].parameters.temperature"] == "0.2" and texts["request[0].system"] == "be kind"
    assert not any(t == "c1" for t in texts.values())  # ids are not text the agent sent


def test_anthropic_bedrock_and_gemini_shapes():
    anthropic = NormalizedRequest(model="m", messages=[
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "SIG"}, {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "2"}]}]},
    ], parameters={"system": [{"type": "text", "text": "sys"}]})
    texts = _texts(anthropic)
    assert texts["request[0].message[0](assistant).tool_call[0](add).arguments.a"] == "1"
    assert texts["request[0].message[1](user).tool_result[0](add)[0].text"] == "2"
    assert texts["request[0].message[0](assistant).content[0].thinking"] == "hmm" and "SIG" not in texts.values()
    assert texts["request[0].system[0].text"] == "sys"

    bedrock = NormalizedRequest(model="m", messages=[
        {"role": "assistant", "content": [{"reasoningContent": {"reasoningText": {"text": "think", "signature": "S"}}}, {"toolUse": {"toolUseId": "u1", "name": "f", "input": {"q": "x"}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "u1", "content": [{"json": {"k": "v"}}], "status": "success"}}, {"image": {"format": "png", "source": {"bytes": "AAAA"}}}]},
    ])
    texts = _texts(bedrock)
    assert texts["request[0].message[0](assistant).tool_call[0](f).arguments.q"] == "x"
    assert texts["request[0].message[1](user).tool_result[0](f)[0].json.k"] == "v"
    assert "S" not in texts.values() and "AAAA" not in texts.values()

    gemini = NormalizedRequest(model="m", messages=[
        {"role": "model", "parts": [{"function_call": {"name": "f", "args": {"q": "y"}}}, "plain"]},
        {"role": "user", "parts": [{"function_response": {"name": "f", "response": {"r": "z"}}}, {"text": "go"}]},
    ], parameters={"system_instruction": {"parts": [{"text": "sys"}]}})
    texts = _texts(gemini)
    assert texts["request[0].message[0](model).tool_call[0](f).arguments.q"] == "y" and texts["request[0].message[0](model).content[1]"] == "plain"
    assert texts["request[0].message[1](user).tool_result[0](f).r"] == "z" and texts["request[0].message[1](user).content[1].text"] == "go"
    assert texts["request[0].system.parts[0].text"] == "sys"


def test_responses_api_items_and_odd_values():
    request = NormalizedRequest(model="m", messages=[
        {"type": "function_call", "call_id": "c9", "name": "lookup", "arguments": '{"k": "v"}'},
        {"type": "function_call_output", "call_id": "c9", "output": "found"},
        {"type": "reasoning", "summary": [{"text": "s"}], "encrypted_content": "ENC"},
        "a bare string",
        {"role": "tool", "tool_call_id": "unknown", "content": "x"},
    ], tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}, {"toolSpec": {"name": "b", "inputSchema": {"json": {}}}}, "odd", {"name": "c", "strict": True, "parameters": "{not json"}])
    texts = _texts(request)
    assert texts["request[0].message[0](function_call).tool_call[0](lookup).arguments.k"] == "v"
    assert texts["request[0].message[1](function_call_output).tool_result[0](lookup)"] == "found"
    assert texts["request[0].message[2](reasoning).summary[0].text"] == "s" and "ENC" not in texts.values()
    assert texts["request[0].message[3]"] == "a bare string" and texts["request[0].message[4](tool).tool_result[0](?)"] == "x"
    assert texts["request[0].tool[lookup].schema.type"] == "object" and texts["request[0].tool[b].name"] == "b"
    assert texts["request[0].tool"] == "odd" and texts["request[0].tool[c].strict"] == "True" and texts["request[0].tool[c].schema"] == "{not json"


def test_sources(tmp_path):
    request = NormalizedRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    path = tmp_path / "c.yaml"
    save_cassette(path, [(request, NormalizedResponse(content="ok"))], provider="openai")
    assert list(iter_request_texts(path)) == list(iter_request_texts(str(path))) == [("request[0].message[0](user).text", "hi")]
    with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY) as rec:
        rec.lookup(request)
    assert list(iter_request_texts(rec)) == [("request[0].message[0](user).text", "hi")]
    mock = MockLLM([mock_response(content="x")], provider="openai")
    mock.lookup(request)
    assert list(iter_request_texts(mock)) == [("request[0].message[0](user).text", "hi")]
    with pytest.raises(TypeError, match="cassette path"):
        list(iter_request_texts(object()))


def test_malformed_tool_calls_are_skipped():
    request = NormalizedRequest(model="m", messages=[{"role": "assistant", "tool_calls": ["bad", {"function": "x"}], "content": ["text", 3]}])
    assert _texts(request) == {"request[0].message[0](assistant).content[0]": "text", "request[0].message[0](assistant).content[1]": "3"}


def test_review_fixes():
    """Payload keys named like envelope fields, SDK objects, and OpenAI tool messages with parts."""
    from dataclasses import dataclass

    @dataclass
    class Part:
        type: str
        text: str

    request = NormalizedRequest(model="m", messages=[
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "dl", "arguments": '{"id": 7}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": '{"data": {"download": "https://s3/x?sig=1"}}'}]},
        {"role": "user", "content": [Part("text", "from an SDK object")]},
    ], tools=[{"name": "dl", "parameters": {"properties": {"id": {"type": "string"}}}}])
    texts = _texts(request)
    assert texts["request[0].message[0](assistant).tool_call[0](dl).arguments.id"] == "7"
    assert texts["request[0].message[1](tool).tool_result[0](dl)[0].text.data.download"] == "https://s3/x?sig=1"
    assert texts["request[0].message[2](user).content[0].text"] == "from an SDK object"
    assert texts["request[0].tool[dl].schema.properties.id.type"] == "string"
