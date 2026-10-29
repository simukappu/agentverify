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


def test_from_pydantic_ai_on_a_real_run():
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent, ModelRetry
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from agentverify import ToolCall, assert_tool_calls
    from agentverify.frameworks.pydantic_ai import from_pydantic_ai

    turns = iter(
        [
            ModelResponse(parts=[ToolCallPart("add", {"a": 2, "b": 3}, tool_call_id="c1"), ToolCallPart("check", {"x": "bad"}, tool_call_id="c2")]),
            ModelResponse(parts=[ToolCallPart("check", {"x": "ok"}, tool_call_id="c3")]),
            ModelResponse(parts=[TextPart("5")]),
        ]
    )
    agent = Agent(FunctionModel(lambda messages, info: next(turns)))

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    @agent.tool_plain
    def check(x: str) -> str:
        if x != "ok":
            raise ModelRetry("bad x")
        return "fine"

    result = from_pydantic_ai(agent.run_sync("2+3?"), agent_name="calc")
    assert_tool_calls(result, expected=[ToolCall("add", {"a": 2, "b": 3}), ToolCall("check", {"x": "bad"}), ToolCall("check", {"x": "ok"})])
    assert result.steps[0].tool_results == [5, "bad x"] and result.steps[0].tool_results_meta[1] == {"is_error": True}
    assert result.final_output == "5"


def test_google_adk_function_tool(tmp_path):
    pytest.importorskip("google.adk")
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    def http_request(url: str) -> dict:
        """Fetch a URL."""
        CALLS.append(url)
        return {"body": f"body of {url}"}

    class Fake(BaseLlm):
        model: str = "fake"

        async def generate_content_async(self, llm_request, stream=False):
            last = llm_request.contents[-1].parts[0]
            if last.function_response is None:
                call = types.FunctionCall(id="c1", name="http_request", args={"url": "http://x"})
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))
            else:
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=f"saw {last.function_response.response}")]))

    async def main():
        runner = InMemoryRunner(agent=LlmAgent(name="a", model=Fake(), tools=[http_request]), app_name="x")
        session = await runner.session_service.create_session(app_name="x", user_id="u")
        texts = []
        message = types.Content(role="user", parts=[types.Part(text="go")])
        async for event in runner.run_async(user_id="u", session_id=session.id, new_message=message):
            texts += [p.text for p in (event.content.parts if event.content else []) if p.text]
        return texts

    executions = _round_trip(tmp_path / "adk.yaml", lambda: asyncio.run(main()), provider="gemini")
    assert executions[0]["name"] == "http_request" and executions[0]["result"] == {"body": "body of http://x"}


def test_mcp_client_session_call_tool(tmp_path):
    pytest.importorskip("mcp")
    import sys
    from pathlib import Path

    import mcp.client.session as client_session
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).parents[1] / "examples" / "mcp-server" / "github_server.py")])
    send_request = client_session.ClientSession.send_request

    async def counting(self, request, *args, **kwargs):
        # Count the calls that reach the server.
        if getattr(getattr(request, "root", request), "method", "") == "tools/call":
            CALLS.append("tools/call")
        return await send_request(self, request, *args, **kwargs)

    async def main():
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_issue", {"repo": "simukappu/agentverify", "issue_number": 1})
                return result.content[0].text

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(client_session.ClientSession, "send_request", counting)
        executions = _round_trip(tmp_path / "mcp.yaml", lambda: asyncio.run(main()))
    assert [e["name"] for e in executions] == ["get_issue"]


def test_crewai_tool(tmp_path, monkeypatch):
    pytest.importorskip("crewai")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import tool

    @tool("http_request")
    def http_request(url: str) -> str:
        """Fetch a URL."""
        CALLS.append(url)
        return f"BODY {url}"

    class Fake(BaseLLM):
        turn: int = 0

        def call(self, messages, *args, **kwargs):
            self.turn += 1
            if self.turn % 2:
                return 'Thought: fetch\nAction: http_request\nAction Input: {"url": "http://x"}'
            seen = any("BODY http://x" in str(m.get("content")) for m in messages)
            return "Thought: done\nFinal Answer: " + ("saw the body" if seen else "no body")

        def supports_function_calling(self):
            return False

        def supports_stop_words(self):
            return False

        def get_context_window_size(self):
            return 8000

    def run():
        agent = Agent(role="r", goal="g", backstory="b", llm=Fake(model="fake"), tools=[http_request], verbose=False)
        return Crew(agents=[agent], tasks=[Task(description="d", expected_output="e", agent=agent)]).kickoff().raw

    executions = _round_trip(tmp_path / "crew.yaml", run)
    assert executions[0]["name"] == "http_request" and executions[0]["result"] == "BODY http://x"


def test_from_strands_on_a_real_agent():
    pytest.importorskip("strands")
    from strands import Agent, tool
    from strands.models.bedrock import BedrockModel

    from agentverify import MockLLM, mock_response
    from agentverify.frameworks.strands import from_strands

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = Agent(model=BedrockModel(model_id="m", region_name="us-east-1"), tools=[add], callback_handler=None)
    responses = [mock_response(tool_calls=[{"name": "add", "arguments": {"a": 1, "b": 2}}]), mock_response(content="It is 3.")]
    with MockLLM(responses, provider="bedrock"):
        agent_result = agent("1+2?")
    result = from_strands(agent)
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("add", {"a": 1, "b": 2})]
    assert result.steps[0].tool_results[0]["content"] == [{"text": "3"}] and result.final_output == "It is 3."
    with pytest.raises(TypeError, match="pass the agent"):
        from_strands(agent_result)


# ---------------------------------------------------------------------------
# Each framework adapter on a real run's output (the adapters' own tests use stand-ins)
# ---------------------------------------------------------------------------


def _lc_tool_model():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    replies = iter([AIMessage("", tool_calls=[{"name": "search", "args": {"q": "a"}, "id": "c1"}]), AIMessage("done")])

    class Model(BaseChatModel):
        """Answers with a tool call, then with text. No streaming, so every caller gets whole messages."""

        @property
        def _llm_type(self) -> str:
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=next(replies))])

        def bind_tools(self, tools, **kwargs):
            return self

    return Model()


def _lc_search_tool():
    from langchain_core.tools import tool

    @tool
    def search(q: str) -> str:
        """Search."""
        return f"found {q}"

    return search


def test_from_langgraph_on_a_create_agent_result():
    pytest.importorskip("langchain")
    from langchain.agents import create_agent

    from agentverify.frameworks.langgraph import from_langgraph

    result = from_langgraph(create_agent(_lc_tool_model(), tools=[_lc_search_tool()]).invoke({"messages": [("user", "go")]}))
    assert [(c.name, c.arguments, c.call_id) for c in result.tool_calls] == [("search", {"q": "a"}, "c1")]
    assert result.steps[0].tool_results == ["found a"] and result.final_output == "done"


def test_from_langchain_on_an_agent_executor_result():
    pytest.importorskip("langchain_classic")
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    from agentverify.frameworks.langchain import from_langchain

    prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("placeholder", "{agent_scratchpad}")])
    tools = [_lc_search_tool()]
    executor = AgentExecutor(agent=create_tool_calling_agent(_lc_tool_model(), tools, prompt), tools=tools, return_intermediate_steps=True)
    result = from_langchain(executor.invoke({"input": "go"}))
    assert [(c.name, c.arguments, c.call_id) for c in result.tool_calls] == [("search", {"q": "a"}, "c1")]
    assert result.steps[0].tool_results == ["found a"] and result.final_output == "done"


def test_from_openai_agents_on_a_run_result(monkeypatch):
    pytest.importorskip("agents")
    from agents import Agent, ModelResponse, Runner, Usage, function_tool, set_tracing_disabled
    from agents.models.interface import Model
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

    from agentverify.frameworks.openai_agents import from_openai_agents

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    set_tracing_disabled(True)

    @function_tool
    def lookup(key: str) -> str:
        return f"value of {key}"

    class Fake(Model):
        turn = 0

        async def get_response(self, system_instructions, input, *args, **kwargs):
            self.turn += 1
            if self.turn == 1:
                output = [ResponseFunctionToolCall(type="function_call", call_id="c1", name="lookup", arguments='{"key": "k"}', id="f1")]
            else:
                text = ResponseOutputText(type="output_text", text="done", annotations=[])
                output = [ResponseOutputMessage(id="m", type="message", role="assistant", status="completed", content=[text])]
            return ModelResponse(output=output, usage=Usage(), response_id=None)

        def stream_response(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    result = from_openai_agents(Runner.run_sync(Agent(name="a", model=Fake(), tools=[lookup]), "go"))
    assert [(c.name, c.arguments, c.call_id) for c in result.tool_calls] == [("lookup", {"key": "k"}, "c1")]
    assert result.steps[0].tool_results == ["value of k"] and result.final_output == "done" and result.steps[0].agent == "a"


def test_from_google_adk_on_runner_events():
    pytest.importorskip("google.adk")
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from agentverify.frameworks.google_adk import from_google_adk

    def lookup(key: str) -> dict:
        """Look a key up."""
        return {"value": key}

    class Fake(BaseLlm):
        model: str = "fake"

        async def generate_content_async(self, llm_request, stream=False):
            if llm_request.contents[-1].parts[0].function_response is None:
                call = types.FunctionCall(id="c1", name="lookup", args={"key": "k"})
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))
            else:
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="done")]))

    async def main():
        runner = InMemoryRunner(agent=LlmAgent(name="a", model=Fake(), tools=[lookup]), app_name="x")
        session = await runner.session_service.create_session(app_name="x", user_id="u")
        message = types.Content(role="user", parts=[types.Part(text="go")])
        return [event async for event in runner.run_async(user_id="u", session_id=session.id, new_message=message)]

    result = from_google_adk(asyncio.run(main()))
    assert [(c.name, c.arguments, c.call_id) for c in result.tool_calls] == [("lookup", {"key": "k"}, "c1")]
    assert result.steps[0].tool_results == [{"value": "k"}] and result.final_output == "done" and result.steps[0].agent == "a"


def test_from_crewai_on_a_crew_output(monkeypatch):
    pytest.importorskip("crewai")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import tool

    from agentverify.frameworks.crewai import from_crewai

    @tool("lookup")
    def lookup(key: str) -> str:
        """Look a key up."""
        return f"value of {key}"

    class Fake(BaseLLM):
        turn: int = 0

        def call(self, messages, *args, **kwargs):
            self.turn += 1
            if self.turn == 1:
                return 'Thought: look it up\nAction: lookup\nAction Input: {"key": "k"}'
            return "Thought: done\nFinal Answer: done"

        def supports_function_calling(self):
            return False

        def supports_stop_words(self):
            return False

        def get_context_window_size(self):
            return 8000

    agent = Agent(role="r", goal="g", backstory="b", llm=Fake(model="fake"), tools=[lookup], verbose=False)
    result = from_crewai(Crew(agents=[agent], tasks=[Task(description="d", expected_output="e", agent=agent)]).kickoff())
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("lookup", {"key": "k"})]
    assert result.final_output == "done"


def test_strands_structured_output_runs_for_real_on_replay(tmp_path):
    """Strands answers structured_output_model through a tool of its own; a replayed result would not set the answer."""
    pytest.importorskip("strands")
    import pydantic
    from strands import Agent
    from strands.models.model import Model

    class Answer(pydantic.BaseModel):
        value: int

    class Fake(Model):
        def update_config(self, **kwargs):
            pass

        def get_config(self):
            return {}

        async def structured_output(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "s1", "name": "Answer"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"value": 3}'}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}

    def run():
        return Agent(model=Fake(), callback_handler=None)("what is 1+2?", structured_output_model=Answer).structured_output

    path = tmp_path / "structured.yaml"
    for mode in (CassetteMode.RECORD, CassetteMode.REPLAY):
        with LLMCassetteRecorder(path, mode=mode, provider="bedrock"):
            assert run() == Answer(value=3)
    assert yaml.safe_load(path.read_text())["tool_executions"] == []
