"""Real agent frameworks: tool calls are recorded and replayed, and traces convert (run where the framework is installed)."""

from __future__ import annotations

import asyncio

import pytest
import yaml

from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder

CALLS: list = []


def _round_trip(path, run, provider="openai"):
    CALLS.clear()
    with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider=provider):
        recorded = run()
    assert CALLS, "the tool did not run while recording"
    CALLS.clear()
    with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider=provider):
        replayed = run()
    assert CALLS == [], "the tool ran on replay"
    assert recorded == replayed
    return yaml.safe_load(path.read_text())["tool_executions"]


def test_strands_agent_created_before_the_cassette(tmp_path):
    pytest.importorskip("strands")
    from strands import Agent, tool
    from strands.models.model import Model

    @tool
    def http_request(url: str) -> str:
        """Fetch a URL."""
        CALLS.append(url)
        return f"body of {url}"

    class Fake(Model):
        def __init__(self):
            self.turn = 0

        def update_config(self, **kwargs):
            pass

        def get_config(self):
            return {}

        async def structured_output(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            self.turn += 1
            yield {"messageStart": {"role": "assistant"}}
            if self.turn % 2:
                yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self.turn}", "name": "http_request"}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"url": "http://x"}'}}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "tool_use"}}
            else:
                seen = messages[-1]["content"][0]["toolResult"]["content"][0]["text"]
                yield {"contentBlockDelta": {"delta": {"text": f"saw {seen}"}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "end_turn"}}

    agent = Agent(model=Fake(), tools=[http_request], callback_handler=None)
    executions = _round_trip(tmp_path / "strands.yaml", lambda: str(agent("go")).strip(), provider="bedrock")
    assert executions[0]["result"]["content"] == [{"text": "body of http://x"}]


def test_langchain_create_agent_sync_and_async_tools(tmp_path):
    pytest.importorskip("langchain")
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    @tool
    def search(q: str) -> str:
        """Search."""
        CALLS.append(q)
        return f"found {q}"

    @tool
    async def asearch(q: str) -> str:
        """Search, async."""
        CALLS.append(q)
        return f"afound {q}"

    class Model(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def run():
        calls = [{"name": "search", "args": {"q": "a"}, "id": "c1"}, {"name": "asearch", "args": {"q": "b"}, "id": "c2"}]
        agent = create_agent(Model(messages=iter([AIMessage("", tool_calls=calls), AIMessage("done")])), tools=[search, asearch])
        out = asyncio.run(agent.ainvoke({"messages": [("user", "go")]}))
        return [m.content for m in out["messages"] if m.type == "tool"]

    executions = _round_trip(tmp_path / "lc.yaml", run)
    assert sorted(e["result"] for e in executions) == ["afound b", "found a"]  # the two run in parallel


def test_openai_agents_function_tools_including_a_failing_one(tmp_path, monkeypatch):
    pytest.importorskip("agents")
    from agents import Agent, ModelResponse, Runner, Usage, function_tool, set_tracing_disabled
    from agents.models.interface import Model
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    set_tracing_disabled(True)

    @function_tool
    def lookup(key: str) -> str:
        CALLS.append(key)
        return f"value of {key}"

    @function_tool
    def broken(key: str) -> str:
        CALLS.append(key)
        raise RuntimeError("down")

    class Fake(Model):
        def __init__(self):
            self.turn = 0

        async def get_response(self, system_instructions, input, *args, **kwargs):
            self.turn += 1
            if self.turn == 1:
                output = [
                    ResponseFunctionToolCall(type="function_call", call_id="c1", name="lookup", arguments='{"key": "k"}', id="f1"),
                    ResponseFunctionToolCall(type="function_call", call_id="c2", name="broken", arguments='{"key": "k"}', id="f2"),
                ]
            else:
                outputs = [i["output"] for i in input if isinstance(i, dict) and i.get("type") == "function_call_output"]
                text = ResponseOutputText(type="output_text", text=str(outputs), annotations=[])
                output = [ResponseOutputMessage(id="m", type="message", role="assistant", status="completed", content=[text])]
            return ModelResponse(output=output, usage=Usage(), response_id=None)

        def stream_response(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    run = lambda: Runner.run_sync(Agent(name="a", model=Fake(), tools=[lookup, broken]), "go").final_output  # noqa: E731
    executions = _round_trip(tmp_path / "agents.yaml", run)
    assert executions[1]["result"]["is_sdk_generated_error"] is True


def test_pydantic_ai_tool(tmp_path):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("add", {"a": 1, "b": 2}, tool_call_id="c1")])
        return ModelResponse(parts=[TextPart(f"got {messages[-1].parts[0].content}")])

    agent = Agent(FunctionModel(model))

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        CALLS.append((a, b))
        return a + b

    executions = _round_trip(tmp_path / "pai.yaml", lambda: agent.run_sync("go").output)
    assert executions[0]["result"] == 3


def test_pydantic_ai_trace_becomes_an_execution_result():
    pytest.importorskip("pydantic_ai")
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from pydantic_ai import Agent, ModelRetry
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models.instrumented import InstrumentationSettings

    from agentverify import ToolCall, assert_tool_calls
    from agentverify.traces import from_otel

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    turns = iter(
        [
            ModelResponse(parts=[ToolCallPart("add", {"a": 2, "b": 3}, tool_call_id="c1"), ToolCallPart("check", {"x": "bad"}, tool_call_id="c2")]),
            ModelResponse(parts=[TextPart("5")]),
        ]
    )
    agent = Agent(FunctionModel(lambda messages, info: next(turns)), name="calc")
    agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=True)

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    @agent.tool_plain(retries=0)
    def check(x: str) -> str:
        raise ModelRetry("bad x")

    try:
        agent.run_sync("2+3?")
    except Exception:
        pass  # retries=0 ends the run; the trace still has the calls
    [result] = from_otel(exporter.get_finished_spans())
    assert_tool_calls(result, expected=[ToolCall("add", {"a": 2, "b": 3}), ToolCall("check", {"x": "bad"})])
    assert result.steps[0].agent == "calc" and result.steps[0].tool_results[0] == 5
    assert result.steps[0].tool_results_meta[1] == {"is_error": True}

