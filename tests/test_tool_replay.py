"""Tests for recording and replaying the agent's tool calls."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from dataclasses import dataclass
from typing import Any

import pytest
import yaml

from agentverify import CassetteToolMismatchError, cassette_tool
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder, OnMissingRequest
from agentverify.cassette.tools import (
    ReplayedToolError,
    ToolExecution,
    ToolSession,
    acall_tool,
    activate,
    active_session,
    call_tool,
    plain,
    rebuild_error,
)

CALLS: list[Any] = []


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()


@cassette_tool
def get_weather(city: str, unit: str = "C") -> str:
    CALLS.append(city)
    return f"sunny in {city}, 22{unit}"


@cassette_tool(name="fetch")
async def fetch(url: str) -> dict:
    CALLS.append(url)
    return {"body": url}


@cassette_tool
def explode(reason: str) -> str:
    CALLS.append(reason)
    raise ValueError(reason)


def _record_then_replay(path, run, **kwargs):
    with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, **kwargs):
        recorded = run()
    calls = list(CALLS)
    CALLS.clear()
    with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, **kwargs):
        replayed = run()
    return recorded, replayed, calls


class TestCore:
    def test_record_then_replay_without_running_the_tool(self, tmp_path):
        path = tmp_path / "tools.yaml"
        recorded, replayed, calls = _record_then_replay(path, lambda: (get_weather("Tokyo"), get_weather(city="Osaka", unit="F")))
        assert recorded == replayed == ("sunny in Tokyo, 22C", "sunny in Osaka, 22F")
        assert calls == ["Tokyo", "Osaka"] and CALLS == []
        doc = yaml.safe_load(path.read_text())
        assert doc["tool_executions"][0] == {"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "C"}, "result": "sunny in Tokyo, 22C"}

    def test_async_tools_and_parallel_order(self, tmp_path):
        async def run():
            return await asyncio.gather(fetch("a"), fetch("b"))

        recorded, _, _ = _record_then_replay(tmp_path / "a.yaml", lambda: asyncio.run(run()))

        async def reversed_order():
            return await fetch("b"), await fetch("a")

        with LLMCassetteRecorder(tmp_path / "a.yaml", mode=CassetteMode.REPLAY):
            assert asyncio.run(reversed_order()) == ({"body": "b"}, {"body": "a"})
        assert recorded == [{"body": "a"}, {"body": "b"}] and CALLS == []

    def test_errors_replay_as_the_recorded_exception(self, tmp_path):
        path = tmp_path / "err.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            with pytest.raises(ValueError, match="bad city"):
                explode("bad city")
        CALLS.clear()
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            with pytest.raises(ValueError, match="bad city"):
                explode("bad city")
        assert CALLS == []
        assert yaml.safe_load(path.read_text())["tool_executions"][0]["error"] == {"type": "builtins.ValueError", "message": "bad city"}

        async def async_boom():
            raise KeyError("k")

        session = ToolSession(_Mode(CassetteMode.RECORD))
        with activate(session):
            with pytest.raises(KeyError):
                asyncio.run(acall_tool("t", {}, async_boom))
        session.replayable, session.recorder = session.recorded, _Mode(CassetteMode.REPLAY)
        with activate(session):
            with pytest.raises(KeyError):
                asyncio.run(acall_tool("t", {}, async_boom))

    def test_rebuild_error_falls_back(self):
        assert isinstance(rebuild_error({"type": "nowhere.Missing", "message": "m"}), ReplayedToolError)
        assert isinstance(rebuild_error({"type": "builtins.len", "message": "m"}), ReplayedToolError)

        class NeedsTwo(Exception):
            def __init__(self, a, b):
                super().__init__(a)

        sys.modules[__name__].NeedsTwo = NeedsTwo  # type: ignore[attr-defined]
        try:
            assert isinstance(rebuild_error({"type": f"{__name__}.NeedsTwo", "message": "m"}), ReplayedToolError)
        finally:
            del sys.modules[__name__].NeedsTwo  # type: ignore[attr-defined]

    def test_mismatch_raises_and_fails_the_test_when_the_cassette_closes(self, tmp_path):
        path = tmp_path / "m.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            get_weather("Tokyo")
        with pytest.raises(CassetteToolMismatchError, match=r"get_weather\(city='Paris', unit='C'\)") as exc:
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
                try:
                    get_weather("Paris")
                except CassetteToolMismatchError:
                    pass  # a framework turning the error into a message for the model
        assert "get_weather(city='Tokyo', unit='C')" in str(exc.value)
        with pytest.raises(CassetteToolMismatchError, match=r"\(none\)"):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
                get_weather("Tokyo")
                get_weather("Tokyo")
        assert "plain" in str(CassetteToolMismatchError("t", "plain", []))

    def test_an_error_inside_the_block_wins_over_a_mismatch(self, tmp_path):
        path = tmp_path / "m2.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            pass
        with pytest.raises(RuntimeError):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
                with pytest.raises(CassetteToolMismatchError):
                    get_weather("x")
                raise RuntimeError("test failure")

    def test_old_cassettes_and_replay_tools_false_run_the_real_tool(self, tmp_path):
        path = tmp_path / "old.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, replay_tools=False):
            get_weather("Tokyo")
        assert "tool_executions" not in yaml.safe_load(path.read_text())
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            get_weather("Tokyo")
        assert CALLS == ["Tokyo", "Tokyo"]
        assert get_weather("Kyoto") and CALLS[-1] == "Kyoto"  # outside any cassette
        assert asyncio.run(fetch("z")) == {"body": "z"}
        from agentverify.cassette.sanitize import sanitize_tool_executions

        assert sanitize_tool_executions([{"name": "t", "arguments": {"password": "p"}}]) == [{"name": "t", "arguments": {"password": "***REDACTED***"}}]

    def test_tool_results_are_sanitized(self, tmp_path):
        path = tmp_path / "s.yaml"

        @cassette_tool
        def login(api_key: str) -> str:
            return "token sk-abcdefghijklmnopqrstuvwxyz0123456789"

        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            login("sk-live-secret-value-0000000000000000")
        text = path.read_text()
        assert "secret-value" not in text and "abcdefghijklmnop" not in text
        with LLMCassetteRecorder(tmp_path / "raw.yaml", mode=CassetteMode.RECORD, sanitize=False):
            login("plain")
        assert "plain" in (tmp_path / "raw.yaml").read_text()

        with LLMCassetteRecorder(tmp_path / "e.yaml", mode=CassetteMode.RECORD):
            with pytest.raises(ValueError):
                explode("leaked sk-abcdefghijklmnopqrstuvwxyz0123456789")
        assert "abcdefghijklmnop" not in (tmp_path / "e.yaml").read_text()

    def test_steps_point_at_the_llm_call_that_asked_for_the_tool(self):
        session = ToolSession(_Mode(CassetteMode.RECORD))
        session.add("a", {})
        session.recorder._interactions = [object()]
        session.add("b", {})
        assert [e.step for e in session.recorded] == [None, 0]
        assert ToolExecution.from_dict(session.recorded[1].to_dict()).step == 0

    def test_nested_layers_and_threads(self):
        session = ToolSession(_Mode(CassetteMode.RECORD))
        seen: list = []

        def outer():
            seen.append(active_session())  # a nested layer sees no session
            return "x"

        with activate(session):
            call_tool("outer", {}, outer)

            def in_thread():
                seen.append(active_session())

            thread = threading.Thread(target=in_thread)
            thread.start()
            thread.join()
        assert seen == [None, session]
        assert [e.name for e in session.recorded] == ["outer"]

    def test_plain(self):
        @dataclass
        class Point:
            x: int

        assert plain({"p": Point(1), 2: {1, 2}, "b": b"hi", "o": object}) == {"p": {"x": 1}, "2": [1, 2], "b": "hi", "o": str(object)}


@dataclass
class _Mode:
    mode: CassetteMode
    _interactions: list = None  # type: ignore[assignment]

    def __post_init__(self):
        self._interactions = []


def test_mock_llm_runs_real_tools():
    from agentverify import MockLLM, mock_response

    with MockLLM([mock_response(content="hi")], provider="openai"):
        get_weather("Tokyo")
    assert CALLS == ["Tokyo"]


def test_mock_llm_inside_a_cassette_keeps_its_tools_out_of_it(tmp_path):
    from agentverify import MockLLM, mock_response

    with LLMCassetteRecorder(tmp_path / "outer.yaml", mode=CassetteMode.RECORD):
        with MockLLM([mock_response(content="hi")], provider="openai"):
            get_weather("Tokyo")
        get_weather("Paris")
    assert [e["arguments"]["city"] for e in yaml.safe_load((tmp_path / "outer.yaml").read_text())["tool_executions"]] == ["Paris"]


def test_cassette_fixture_passes_replay_tools(cassette, tmp_path):
    with cassette("x.yaml", cassette_dir=tmp_path, mode="record", replay_tools=False):
        get_weather("Tokyo")
    assert "tool_executions" not in (tmp_path / "x.yaml").read_text()


# ---------------------------------------------------------------------------
# Framework hooks, against stand-in framework modules shaped like the real ones
# ---------------------------------------------------------------------------


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def fake_modules(monkeypatch):
    def install(**modules: types.ModuleType) -> None:
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)

    return install


def _round_trip(tmp_path, run):
    path = tmp_path / "hooks.yaml"
    with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
        recorded = run()
    calls = list(CALLS)
    CALLS.clear()
    with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
        replayed = run()
    assert calls and CALLS == []
    return recorded, replayed, yaml.safe_load(path.read_text())["tool_executions"]


class TestLangChainHook:
    @pytest.fixture
    def lc(self, fake_modules):
        class ToolMessage:
            def __init__(self, content, tool_call_id, name=None, artifact=None, status="success"):
                self.content, self.tool_call_id, self.name, self.artifact, self.status = content, tool_call_id, name, artifact, status

        class Schema:
            model_fields = {"q": None}

        class BaseTool:
            name = "search"
            tool_call_schema = Schema

            def run(self, tool_input, verbose=None, *, tool_call_id=None, **kwargs):
                CALLS.append(tool_input)
                if isinstance(tool_input, str):
                    return f"found {tool_input}"
                if tool_input.get("q") == "fail":
                    return ToolMessage("failed", tool_call_id, self.name, status="error")
                if tool_call_id is None:
                    return f"found {tool_input['q']}"
                return ToolMessage(f"found {tool_input['q']}", tool_call_id, self.name, artifact={"n": 1} if tool_input.get("q") == "art" else None)

            async def arun(self, tool_input, *, tool_call_id=None, **kwargs):
                return self.run(tool_input, tool_call_id=tool_call_id)

        class Untyped(BaseTool):
            tool_call_schema = None

        class WithInjected(BaseTool):
            """langchain-core 1.x: the tool drops its own injected arguments (state, runtime)."""

            def _filter_injected_args(self, tool_input):
                return {k: v for k, v in tool_input.items() if k != "runtime"}

        fake_modules(
            langchain_core=_module("langchain_core"),
            **{
                "langchain_core.messages": _module("langchain_core.messages", ToolMessage=ToolMessage),
                "langchain_core.tools": _module("langchain_core.tools", BaseTool=BaseTool),
            },
        )
        return BaseTool, Untyped, WithInjected

    def test_run_and_arun(self, tmp_path, lc):
        BaseTool, Untyped, WithInjected = lc

        def run():
            tool = BaseTool()
            messages = [tool.run({"q": q, "state": "injected"}, tool_call_id="c1") for q in ("a", "art", "fail")]
            return (
                [(m.content, m.artifact, m.status, m.tool_call_id) for m in messages],
                tool.run({"q": "plain"}),
                asyncio.run(tool.arun({"q": "b"}, tool_call_id="c2")).content,
                Untyped().run("raw text"),
                Untyped().run({"q": "dict", "x": 1}),
                WithInjected().run({"q": "inj", "runtime": object()}),
            )

        recorded, replayed, executions = _round_trip(tmp_path, run)
        assert recorded == replayed
        assert executions[0] == {"name": "search", "arguments": {"q": "a"}, "result": "found a"}
        assert executions[1]["result"] == {"tool_message": {"content": "found art", "artifact": {"n": 1}}}
        assert executions[2]["result"] == {"tool_message": {"content": "failed", "status": "error"}}
        assert executions[-1]["arguments"] == {"q": "inj"}

    def test_a_plain_recording_replays_into_a_message(self, tmp_path, lc):
        BaseTool, _, _ = lc
        path = tmp_path / "p.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            BaseTool().run({"q": "z"})
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            message = BaseTool().run({"q": "z"}, tool_call_id="c9")
        assert (message.content, message.tool_call_id, message.status) == ("found z", "c9", "success")


class TestStrandsHook:
    def test_replaces_the_selected_tool_and_records_after(self, tmp_path, fake_modules):
        class BeforeToolCallEvent:
            def __init__(self, tool_use, selected_tool, cancel_tool=False):
                self.tool_use, self.selected_tool, self.cancel_tool = tool_use, selected_tool, cancel_tool

        class AfterToolCallEvent:
            def __init__(self, tool_use, result):
                self.tool_use, self.result = tool_use, result

        class PythonAgentTool:
            def __init__(self, name, spec, func):
                self.tool_name, self.tool_spec, self._func = name, spec, func

        class HookRegistry:
            async def invoke_callbacks_async(self, event):
                return event, []

        class RealTool:
            tool_spec = {"name": "http"}

            def _func(self, tool_use):
                CALLS.append(tool_use["input"])
                return {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"text": "body"}]}

        fake_modules(
            strands=_module("strands"),
            **{
                "strands.hooks": _module("strands.hooks", AfterToolCallEvent=AfterToolCallEvent, BeforeToolCallEvent=BeforeToolCallEvent),
                "strands.hooks.registry": _module("strands.hooks.registry", HookRegistry=HookRegistry),
                "strands.tools": _module("strands.tools"),
                "strands.tools.tools": _module("strands.tools.tools", PythonAgentTool=PythonAgentTool),
            },
        )

        def agent_runs_tool(tool_use_id):
            registry, tool_use = HookRegistry(), {"toolUseId": tool_use_id, "name": "http", "input": {"url": "u"}}

            async def run():
                before, _ = await registry.invoke_callbacks_async(BeforeToolCallEvent(tool_use, RealTool()))
                selected = before.selected_tool
                result = selected._func(tool_use) if isinstance(selected, RealTool) else selected._func(tool_use)
                await registry.invoke_callbacks_async(AfterToolCallEvent(tool_use, result))
                await registry.invoke_callbacks_async(BeforeToolCallEvent(tool_use, None))  # unknown tool
                await registry.invoke_callbacks_async(BeforeToolCallEvent(tool_use, RealTool(), cancel_tool="no"))
                await registry.invoke_callbacks_async(AfterToolCallEvent(tool_use, None))
                return result

            return asyncio.run(run())

        path = tmp_path / "strands.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            recorded = agent_runs_tool("t1")
        assert CALLS == [{"url": "u"}]
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            replayed = agent_runs_tool("t2")
        assert CALLS == [{"url": "u"}] and replayed == {**recorded, "toolUseId": "t2"}
        execution = yaml.safe_load(path.read_text())["tool_executions"][0]
        assert execution["result"] == {"status": "success", "content": [{"text": "body"}]}
        # A cassette edited to a plain value still replays as a tool result.
        doc = yaml.safe_load(path.read_text())
        doc["tool_executions"][0]["result"] = "plain"
        path.write_text(yaml.safe_dump(doc))
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            assert agent_runs_tool("t3")["content"] == [{"text": "plain"}]
        old = tmp_path / "old.yaml"
        with LLMCassetteRecorder(old, mode=CassetteMode.RECORD, replay_tools=False):
            pass
        with LLMCassetteRecorder(old, mode=CassetteMode.REPLAY):  # recorded without tool results
            assert asyncio.run(HookRegistry().invoke_callbacks_async("event")) == ("event", [])


class TestOpenAIAgentsHook:
    def test_invocation_is_recorded_and_replayed(self, tmp_path, fake_modules):
        @dataclass
        class _FunctionToolInvocationResult:
            output: Any
            is_sdk_generated_error: bool = False

        async def invoke(*, function_tool, context, arguments):
            CALLS.append(arguments)
            return _FunctionToolInvocationResult(f"ran {function_tool.name}", arguments == "{bad")

        execution = _module("agents.run_internal.tool_execution", _invoke_function_tool_with_metadata=invoke)
        fake_modules(
            agents=_module("agents"),
            **{
                "agents.run_internal": _module("agents.run_internal", tool_execution=execution),
                "agents.run_internal.tool_execution": execution,
                "agents.tool": _module("agents.tool", _FunctionToolInvocationResult=_FunctionToolInvocationResult),
            },
        )
        tool = types.SimpleNamespace(name="lookup")

        def run():
            call = lambda args: asyncio.run(execution._invoke_function_tool_with_metadata(function_tool=tool, context=None, arguments=args))  # noqa: E731
            return [call('{"id": 7}'), call("{bad"), call(""), call({"already": "parsed"})]

        recorded, replayed, executions = _round_trip(tmp_path, run)
        assert recorded == replayed
        assert [e["arguments"] for e in executions] == [{"id": 7}, "{bad", {}, {"already": "parsed"}]
        assert executions[1]["result"] == {"output": "ran lookup", "is_sdk_generated_error": True}


class TestMCPHook:
    def test_call_tool(self, tmp_path, fake_modules):
        class CallToolResult:
            def __init__(self, content, isError=False):
                self.content, self.isError = content, isError

            def model_dump(self, **kwargs):
                return {"content": self.content, "isError": self.isError}

            @classmethod
            def model_validate(cls, data):
                return cls(**data)

        class ClientSession:
            async def call_tool(self, name, arguments=None, read_timeout_seconds=None):
                CALLS.append((name, arguments))
                return CallToolResult([{"type": "text", "text": f"{name} ok"}])

        fake_modules(
            mcp=_module("mcp", ClientSession=ClientSession),
            **{"mcp.types": _module("mcp.types", CallToolResult=CallToolResult)},
        )

        def run():
            session = ClientSession()
            return [asyncio.run(session.call_tool("get_issue", {"n": 1})).content, asyncio.run(session.call_tool("ping")).content]

        recorded, replayed, executions = _round_trip(tmp_path, run)
        assert recorded == replayed and executions[1]["arguments"] == {}


class TestPydanticAIHook:
    def test_call_tool_of_every_toolset(self, tmp_path, fake_modules):
        class AbstractToolset:
            async def call_tool(self, name, tool_args, ctx, tool):  # pragma: no cover — abstract in Pydantic AI
                raise NotImplementedError

        class FunctionToolset(AbstractToolset):
            async def call_tool(self, name, tool_args, ctx, tool):
                CALLS.append(tool_args)
                return {"sum": sum(tool_args.values())}

        class WrapperToolset(AbstractToolset):
            def __init__(self, wrapped):
                self.wrapped = wrapped

            async def call_tool(self, name, tool_args, ctx, tool):
                return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        outputs: list = []

        class OutputToolset(AbstractToolset):
            async def call_tool(self, name, tool_args, ctx, tool):
                outputs.append(tool_args)
                return tool_args

        OutputToolset.__module__ = "pydantic_ai._output"
        fake_modules(
            pydantic_ai=_module("pydantic_ai"),
            **{"pydantic_ai.toolsets": _module("pydantic_ai.toolsets", AbstractToolset=AbstractToolset)},
        )

        def run():
            wrapped = asyncio.run(WrapperToolset(FunctionToolset()).call_tool("add", {"a": 1, "b": 2}, None, None))
            output = asyncio.run(OutputToolset().call_tool("final_result", {"answer": 3}, None, None))
            return wrapped, output

        recorded, replayed, executions = _round_trip(tmp_path, run)
        assert recorded == replayed == ({"sum": 3}, {"answer": 3}) and len(outputs) == 2
        assert [e["name"] for e in executions] == ["add"]  # recorded once, and the agent's answer is not a tool


class TestGoogleADKHook:
    def test_run_async_of_every_tool_class(self, tmp_path, fake_modules):
        class BaseTool:
            name = "base"

            async def run_async(self, *, args, tool_context):  # pragma: no cover — abstract in ADK
                raise NotImplementedError

        class FunctionTool(BaseTool):
            name = "http"

            async def run_async(self, *, args, tool_context):
                CALLS.append(args)
                if args.get("url") == "bad":
                    raise ConnectionError("down")
                return {"body": args["url"]}

        class Wrapped(FunctionTool):
            async def run_async(self, *, args, tool_context):
                return await super().run_async(args=args, tool_context=tool_context)

        class Plain(FunctionTool):
            pass

        fake_modules(
            google=sys.modules.get("google") or _module("google"),
            **{
                "google.adk": _module("google.adk"),
                "google.adk.tools": _module("google.adk.tools"),
                "google.adk.tools.base_tool": _module("google.adk.tools.base_tool", BaseTool=BaseTool),
            },
        )

        def run():
            results = [asyncio.run(cls().run_async(args={"url": "u"}, tool_context=None)) for cls in (FunctionTool, Wrapped, Plain)]
            with pytest.raises(ConnectionError):
                asyncio.run(FunctionTool().run_async(args={"url": "bad"}, tool_context=None))
            return results

        recorded, replayed, executions = _round_trip(tmp_path, run)
        assert recorded == replayed and len(executions) == 4  # the super() call is not recorded twice
        assert executions[3]["error"] == {"type": "builtins.ConnectionError", "message": "down"}


class TestCrewAIHook:
    def test_global_hooks(self, tmp_path, fake_modules):
        befores: list = []
        afters: list = []
        hooks = _module(
            "crewai.hooks",
            register_before_tool_call_hook=befores.append,
            register_after_tool_call_hook=afters.append,
            unregister_before_tool_call_hook=befores.remove,
            unregister_after_tool_call_hook=afters.remove,
        )
        fake_modules(crewai=_module("crewai", hooks=hooks), **{"crewai.hooks": hooks})

        def use_tool(tool_input, fresh_after_input=False):
            """CrewAI's ReAct path: a blocked call gets a new context for the after hook."""
            context = types.SimpleNamespace(tool_name="http", tool_input=tool_input, tool_result=None)
            if any(hook(context) is False for hook in befores):
                after_context = types.SimpleNamespace(
                    tool_name="http", tool_input={} if fresh_after_input else tool_input, tool_result="blocked"
                )
                results = [hook(after_context) for hook in afters]
                return next((r for r in results if r is not None), "blocked")
            CALLS.append(tool_input)
            context.tool_result = f"body {tool_input.get('url')}"
            for hook in afters:
                hook(context)
            return context.tool_result

        path = tmp_path / "crew.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            assert use_tool({"url": "u"}) == "body u"
            assert use_tool({}) == "body None"
        assert befores == [] and afters == []
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            assert use_tool({"url": "u"}) == "body u"
            assert use_tool({}, fresh_after_input=True) == "body None"
            assert all(hook(types.SimpleNamespace(tool_name="other", tool_input={}, tool_result="r")) is None for hook in afters)
        assert CALLS == [{"url": "u"}, {}]


class TestReviewFixes:
    def test_a_replayed_tool_skips_the_llm_calls_it_made(self, tmp_path):
        """An agent used as a tool: its own LLM calls are not replayed when the tool is."""
        from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse

        path = tmp_path / "nested.yaml"

        def outer_agent(rec):
            def llm(name):
                if rec.mode == CassetteMode.REPLAY:
                    return rec.lookup(NormalizedRequest(messages=[], model="m")).content
                rec.record(NormalizedRequest(messages=[], model="m"), NormalizedResponse(content=name))
                return name

            @cassette_tool
            def sub_agent(task: str) -> str:
                CALLS.append(task)
                return llm("inner")

            return llm("outer1"), sub_agent("t"), llm("outer2")

        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD) as rec:
            recorded = outer_agent(rec)
        executions = yaml.safe_load(path.read_text())["tool_executions"]
        assert executions == [{"name": "sub_agent", "arguments": {"task": "t"}, "step": 0, "result": "inner", "llm_calls": [1]}]
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY) as rec:
            assert outer_agent(rec) == recorded == ("outer1", "inner", "outer2")
            rec.skip_interactions([1, 9])  # used or out of range: ignored
        assert len(rec.to_execution_result().steps) == 3

    def test_results_that_do_not_survive_yaml_run_on_replay(self, tmp_path):
        class Command:
            def __init__(self, update):
                self.update = update

        @cassette_tool
        def set_city(city: str) -> Any:
            CALLS.append(city)
            return Command({"city": city})

        @cassette_tool
        async def aset_city(city: str) -> Any:
            CALLS.append(city)
            return Command({"city": city})

        path = tmp_path / "cmd.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            set_city("Tokyo")
            asyncio.run(aset_city("Osaka"))
        executions = yaml.safe_load(path.read_text())["tool_executions"]
        assert [e.get("run_on_replay") for e in executions] == [True, True] and "result" not in executions[0]
        CALLS.clear()
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            assert set_city("Tokyo").update == {"city": "Tokyo"}
            assert asyncio.run(aset_city("Osaka")).update == {"city": "Osaka"}
        assert CALLS == ["Tokyo", "Osaka"]
        from agentverify.cassette.tools import faithful

        assert not faithful({1: "non-string key"})

    def test_redacted_arguments_still_match(self, tmp_path):
        @cassette_tool
        def login(user: str, password: str) -> str:
            CALLS.append(user)
            return "ok"

        path = tmp_path / "login.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            login("bob", "hunter2")
        assert "hunter2" not in path.read_text()
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            assert login("bob", "hunter2") == "ok"
        assert CALLS == ["bob"]

    def test_key_errors_keep_their_message(self):
        from agentverify.cassette.tools import error_of

        assert error_of(KeyError("x")) == {"type": "builtins.KeyError", "message": "x"}
        assert error_of(OSError(2, "missing"))["message"] == "[Errno 2] missing"

    def test_sets_are_stored_in_a_stable_order(self):
        assert plain({"b", "a"}) == ["a", "b"] and plain([("x", 1)]) == [["x", 1]]

    def test_nested_cassettes(self, tmp_path):
        outer, inner = tmp_path / "outer.yaml", tmp_path / "inner.yaml"
        with LLMCassetteRecorder(outer, mode=CassetteMode.RECORD):
            with LLMCassetteRecorder(inner, mode=CassetteMode.RECORD):
                get_weather("Tokyo")
            with LLMCassetteRecorder(tmp_path / "off.yaml", mode=CassetteMode.RECORD, replay_tools=False):
                get_weather("Osaka")  # not recorded into the outer cassette
        assert len(yaml.safe_load(inner.read_text())["tool_executions"]) == 1
        assert yaml.safe_load(outer.read_text())["tool_executions"] == []

    def test_strands_runs_on_replay_and_skips_replay_recording(self, tmp_path, fake_modules):
        class BeforeToolCallEvent:
            def __init__(self, tool_use, selected_tool):
                self.tool_use, self.selected_tool, self.cancel_tool = tool_use, selected_tool, False

        class AfterToolCallEvent:
            def __init__(self, tool_use, result):
                self.tool_use, self.result = tool_use, result

        class HookRegistry:
            async def invoke_callbacks_async(self, event):
                return event, []

        fake_modules(
            strands=_module("strands"),
            **{
                "strands.hooks": _module("strands.hooks", AfterToolCallEvent=AfterToolCallEvent, BeforeToolCallEvent=BeforeToolCallEvent),
                "strands.hooks.registry": _module("strands.hooks.registry", HookRegistry=HookRegistry),
                "strands.tools": _module("strands.tools"),
                "strands.tools.tools": _module("strands.tools.tools", PythonAgentTool=object),
            },
        )
        path = tmp_path / "s.yaml"
        path.write_text(yaml.safe_dump({"interactions": [], "tool_executions": [{"name": "t", "arguments": {}, "run_on_replay": True}]}))
        tool_use = {"toolUseId": "u1", "name": "t", "input": {}}

        async def run():
            registry, real = HookRegistry(), object()
            before, _ = await registry.invoke_callbacks_async(BeforeToolCallEvent(tool_use, real))
            assert before.selected_tool is real  # runs for real
            await registry.invoke_callbacks_async(AfterToolCallEvent(tool_use, {"status": "success", "content": []}))

        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            asyncio.run(run())
        assert "result" not in path.read_text()

    def test_mcp_calls_inside_a_running_framework_tool_are_not_recorded(self, tmp_path, fake_modules):
        class ClientSession:
            async def call_tool(self, name, arguments=None):
                CALLS.append(name)
                return "raw"

        fake_modules(mcp=_module("mcp", ClientSession=ClientSession), **{"mcp.types": _module("mcp.types", CallToolResult=object)})
        path = tmp_path / "mcp.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD) as rec:
            rec._tool_session.running += 1
            assert asyncio.run(ClientSession().call_tool("x")) == "raw"
            rec._tool_session.running -= 1
        assert yaml.safe_load(path.read_text())["tool_executions"] == []

    def test_crewai_answer_tools_and_replays_run_for_real(self, tmp_path, fake_modules):
        befores: list = []
        afters: list = []
        hooks = _module(
            "crewai.hooks",
            register_before_tool_call_hook=befores.append,
            register_after_tool_call_hook=afters.append,
            unregister_before_tool_call_hook=befores.remove,
            unregister_after_tool_call_hook=afters.remove,
        )
        fake_modules(crewai=_module("crewai", hooks=hooks), **{"crewai.hooks": hooks})
        answer_tool = types.SimpleNamespace(result_as_answer=True)

        def use(tool=None):
            context = types.SimpleNamespace(tool_name="t", tool_input={"a": 1}, tool_result="r", tool=tool)
            blocked = any(hook(context) is False for hook in befores)
            return blocked, [hook(context) for hook in afters]

        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"interactions": [], "tool_executions": [{"name": "t", "arguments": {"a": 1}, "run_on_replay": True}]}))
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            assert use(answer_tool) == (False, [None])
            assert use() == (False, [None])  # run_on_replay: not blocked
        with LLMCassetteRecorder(tmp_path / "d.yaml", mode=CassetteMode.RECORD):
            use(answer_tool)
            with LLMCassetteRecorder(tmp_path / "off.yaml", mode=CassetteMode.RECORD, replay_tools=False):
                context = types.SimpleNamespace(tool_name="t", tool_input={}, tool_result="r", tool=None)
                assert [hook(context) for hook in befores] == [None] and [hook(context) for hook in afters] == [None]
        assert yaml.safe_load((tmp_path / "d.yaml").read_text())["tool_executions"] == []


class TestFinalReview:
    def test_fallback_runs_a_tool_the_cassette_does_not_have(self, tmp_path):
        path = tmp_path / "f.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            get_weather("Tokyo")
        CALLS.clear()
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, on_missing=OnMissingRequest.FALLBACK):
            assert get_weather("Tokyo") == "sunny in Tokyo, 22C"
            assert get_weather("Paris") == "sunny in Paris, 22C"
        assert CALLS == ["Paris"]

    def test_a_method_records_without_its_object(self, tmp_path):
        class Weather:
            @cassette_tool
            def lookup(self, city: str) -> str:
                CALLS.append(city)
                return f"rain in {city}"

        recorded, replayed, calls = _record_then_replay(tmp_path / "m.yaml", lambda: Weather().lookup("Oslo"))
        assert recorded == replayed and calls == ["Oslo"] and CALLS == []
        assert yaml.safe_load((tmp_path / "m.yaml").read_text())["tool_executions"][0]["arguments"] == {"city": "Oslo"}

    def test_control_flow_signals_run_again_on_replay(self, tmp_path):
        class GraphBubbleUp(Exception):
            pass

        class GraphInterrupt(GraphBubbleUp):
            pass

        @cassette_tool
        def ask_human(question: str) -> str:
            CALLS.append(question)
            raise GraphInterrupt(question)

        path = tmp_path / "i.yaml"
        for mode in (CassetteMode.RECORD, CassetteMode.REPLAY):
            with LLMCassetteRecorder(path, mode=mode):
                with pytest.raises(GraphInterrupt):
                    ask_human("ok?")
        assert CALLS == ["ok?", "ok?"]
        assert yaml.safe_load(path.read_text())["tool_executions"][0]["run_on_replay"] is True

    def test_framework_control_tools_run_for_real(self, tmp_path, fake_modules):
        class BaseTool:
            name = "base"

            async def run_async(self, *, args, tool_context):
                CALLS.append(self.name)
                return {"done": True}

        class ExitLoopTool(BaseTool):
            name = "exit_loop"

            async def run_async(self, *, args, tool_context):
                return await super().run_async(args=args, tool_context=tool_context)

        fake_modules(
            google=sys.modules.get("google") or _module("google"),
            **{
                "google.adk": _module("google.adk"),
                "google.adk.tools": _module("google.adk.tools"),
                "google.adk.tools.base_tool": _module("google.adk.tools.base_tool", BaseTool=BaseTool),
            },
        )
        path = tmp_path / "c.yaml"
        for mode in (CassetteMode.RECORD, CassetteMode.REPLAY):
            with LLMCassetteRecorder(path, mode=mode):
                asyncio.run(ExitLoopTool().run_async(args={}, tool_context=None))
        assert CALLS == ["exit_loop", "exit_loop"] and yaml.safe_load(path.read_text())["tool_executions"] == []

    def test_strands_handoff_runs_for_real(self, tmp_path, fake_modules):
        class BeforeToolCallEvent:
            def __init__(self, tool_use):
                self.tool_use, self.selected_tool, self.cancel_tool = tool_use, object(), False

        class HookRegistry:
            async def invoke_callbacks_async(self, event):
                return event, []

        fake_modules(
            strands=_module("strands"),
            **{
                "strands.hooks": _module("strands.hooks", AfterToolCallEvent=type("AfterToolCallEvent", (), {}), BeforeToolCallEvent=BeforeToolCallEvent),
                "strands.hooks.registry": _module("strands.hooks.registry", HookRegistry=HookRegistry),
                "strands.tools": _module("strands.tools"),
                "strands.tools.tools": _module("strands.tools.tools", PythonAgentTool=object),
            },
        )
        path = tmp_path / "s.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD):
            pass
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY):
            event = BeforeToolCallEvent({"name": "handoff_to_agent", "input": {}})
            before, _ = asyncio.run(HookRegistry().invoke_callbacks_async(event))
        assert before.selected_tool is event.selected_tool  # not swapped for a replayed result

    def test_a_broken_framework_install_warns_and_the_others_still_hook(self, tmp_path, monkeypatch):
        from agentverify.cassette import tool_hooks

        def _missing(stack):
            raise ImportError("not installed")

        def _broken(stack):
            raise AttributeError("moved")

        monkeypatch.setattr(tool_hooks, "_INSTALLERS", (_missing, _broken, *tool_hooks._INSTALLERS))
        with pytest.warns(UserWarning, match="could not hook broken tool calls: AttributeError"):
            with LLMCassetteRecorder(tmp_path / "b.yaml", mode=CassetteMode.RECORD):
                get_weather("Tokyo")
        assert "get_weather" in (tmp_path / "b.yaml").read_text()

    def test_discarding_a_call_moves_the_positions_tools_point_at(self, tmp_path):
        from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse

        rec = LLMCassetteRecorder(tmp_path / "d.yaml", mode=CassetteMode.RECORD)
        request = NormalizedRequest(messages=[], model="m")
        failed = NormalizedResponse(content="broken stream")

        @cassette_tool
        def helper_agent(task: str) -> str:
            rec.record(request, NormalizedResponse(content="inner"))
            return "done"

        with rec:
            get_weather("Kyoto")  # before any call
            rec.record(request, NormalizedResponse(content="first"))
            get_weather("Osaka")  # step 0
            rec.record(request, failed)
            helper_agent("a")  # step 1, makes call 2
            rec.record(request, NormalizedResponse(content="outer"))
            get_weather("Tokyo")  # step 3
            rec.discard(failed)
            positions = [(e.step, e.llm_calls) for e in rec._tool_session.recorded]
            assert positions == [(None, []), (0, []), (0, [1]), (2, [])]
