"""Framework hooks that send the agent's tool calls through the cassette (see :mod:`agentverify.cassette.tools`).

Each hook is installed only if its framework is importable, and removed when the cassette closes:

- LangChain / LangGraph: ``BaseTool.run`` / ``arun`` (``invoke``, ``ToolNode``, ``create_agent``, and langchain-mcp-adapters all go through them)
- Strands Agents: the agent's ``BeforeToolCallEvent`` / ``AfterToolCallEvent``, at the hook registry, so agents created before the cassette are covered
- OpenAI Agents SDK: the function tool invocation in its run loop
- MCP: ``ClientSession.call_tool``, for MCP tools of any framework without its own hook here
- Pydantic AI: ``call_tool`` of every toolset class except the output toolset, inside its retry and usage handling
- Google ADK: ``run_async`` of every tool class, which both of ADK's tool runners call
- CrewAI: its global before / after tool call hooks

The hooks are installed once, by the outermost cassette, and act on the innermost cassette's session.
"""

from __future__ import annotations

import copy
import json
import warnings
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Generator
from unittest.mock import patch

from agentverify.cassette.tools import ToolSession, acall_tool, activate, active_session, call_tool

_installed = 0


@contextmanager
def tool_hooks(session: ToolSession | None) -> Generator[ToolSession | None, None, None]:
    """Activate *session* (``None``: no tool recording in this block) and install the framework hooks if no outer cassette has."""
    global _installed
    with ExitStack() as stack:
        stack.enter_context(activate(session))
        if _installed == 0:
            for install in _INSTALLERS:
                try:
                    install(stack)
                except ImportError:
                    pass  # the framework is not installed
                except Exception as error:  # a broken install of a framework the test may not even use
                    warnings.warn(f"agentverify: could not hook {install.__name__.strip('_')} tool calls: {error!r}", stacklevel=3)
        _installed += 1
        try:
            yield session
        finally:
            _installed -= 1


def _json_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments) if arguments else {}
        except ValueError:
            return arguments
    return arguments


def _subclasses(cls: type) -> list[type]:
    found = [cls]
    for sub in cls.__subclasses__():
        found += _subclasses(sub)
    return found


# Framework tools that steer the run rather than fetch something; they run for real on replay.
_ADK_CONTROL_TOOLS = frozenset({"transfer_to_agent", "exit_loop"})
_STRANDS_CONTROL_TOOLS = frozenset({"handoff_to_agent"})


# ---------------------------------------------------------------------------
# LangChain / LangGraph
# ---------------------------------------------------------------------------


def _langchain(stack: ExitStack) -> None:
    try:
        from langchain_core.messages import ToolMessage
        from langchain_core.tools import BaseTool
    except ImportError:
        return
    run, arun = BaseTool.run, BaseTool.arun

    def arguments_of(tool: Any, tool_input: Any) -> Any:
        # Injected arguments (state, runtime, the call id) are not the model's arguments; keep the ones the model sees.
        if not isinstance(tool_input, dict):
            return tool_input
        drop_injected = getattr(tool, "_filter_injected_args", None)  # langchain-core 1.x
        if callable(drop_injected):
            try:
                return dict(drop_injected(dict(tool_input)))
            except Exception:  # pragma: no cover — a tool whose schema the helper cannot read
                pass
        fields = getattr(getattr(tool, "tool_call_schema", None), "model_fields", None)
        if isinstance(fields, dict) and fields:
            return {k: v for k, v in tool_input.items() if k in fields}
        return dict(tool_input)

    def encode(result: Any) -> Any:
        if isinstance(result, ToolMessage):
            if result.artifact is None and result.status == "success":
                return result.content
            message: dict[str, Any] = {"content": result.content}
            if result.artifact is not None:
                message["artifact"] = result.artifact
            if result.status != "success":
                message["status"] = result.status
            return {"tool_message": message}
        return result

    def decoder(tool: Any, tool_call_id: Any) -> Callable[[Any], Any]:
        def decode(recorded: Any) -> Any:
            message = recorded.get("tool_message") if isinstance(recorded, dict) and len(recorded) == 1 else None
            content = recorded if message is None else message.get("content")
            if tool_call_id is None:
                return content
            message = message or {}
            return ToolMessage(
                content=content,
                artifact=message.get("artifact"),
                tool_call_id=tool_call_id,
                name=tool.name,
                status=message.get("status", "success"),
            )

        return decode

    def patched_run(self: Any, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
        return call_tool(
            self.name,
            arguments_of(self, tool_input),
            lambda: run(self, tool_input, *args, **kwargs),
            encode,
            decoder(self, kwargs.get("tool_call_id")),
        )

    async def patched_arun(self: Any, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
        return await acall_tool(
            self.name,
            arguments_of(self, tool_input),
            lambda: arun(self, tool_input, *args, **kwargs),
            encode,
            decoder(self, kwargs.get("tool_call_id")),
        )

    stack.enter_context(patch.object(BaseTool, "run", new=patched_run))
    stack.enter_context(patch.object(BaseTool, "arun", new=patched_arun))


# ---------------------------------------------------------------------------
# Strands Agents
# ---------------------------------------------------------------------------


def _strands(stack: ExitStack) -> None:
    try:
        from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
        from strands.hooks.registry import HookRegistry
        from strands.tools.tools import PythonAgentTool
    except ImportError:
        return
    invoke = HookRegistry.invoke_callbacks_async
    # Tool calls running for real, by toolUseId.
    started: dict[Any, int] = {}

    def replayed_tool(selected: Any, tool_use: dict[str, Any], result: Any) -> Any:
        def run(tool_use: dict[str, Any], **_: Any) -> Any:
            return {**(result if isinstance(result, dict) else {"status": "success", "content": [{"text": str(result)}]}), "toolUseId": tool_use["toolUseId"]}

        return PythonAgentTool(tool_use["name"], selected.tool_spec, run)

    async def patched_invoke(self: Any, event: Any, *args: Any, **kwargs: Any) -> Any:
        session = active_session()
        if session is not None and isinstance(event, AfterToolCallEvent):
            # The result before the agent's own after-hooks change it; they run again on replay.
            raw = copy.deepcopy(event.result)
        outcome = await invoke(self, event, *args, **kwargs)
        if session is None:
            return outcome
        interrupted = isinstance(outcome, tuple) and len(outcome) > 1 and bool(outcome[1])
        tool_use = getattr(event, "tool_use", None) or {}
        if tool_use.get("name") in _STRANDS_CONTROL_TOOLS:
            return outcome  # a Swarm handoff acts on the run, which a replayed result would not
        if isinstance(event, BeforeToolCallEvent) and event.selected_tool is not None and not event.cancel_tool and not interrupted:
            if session.replaying:
                execution = session.take(tool_use["name"], tool_use.get("input"))
                if not execution.run_on_replay:
                    event.selected_tool = replayed_tool(event.selected_tool, tool_use, execution.result)
                    return outcome
            started[tool_use.get("toolUseId")] = session.begin()
            session.running += 1
        elif isinstance(event, AfterToolCallEvent) and tool_use.get("toolUseId") in started:
            frame, step = session.end(started.pop(tool_use.get("toolUseId")))
            session.running -= 1
            if session.recording and raw is not None:
                result = {k: v for k, v in dict(raw).items() if k != "toolUseId"}
                session.add(tool_use["name"], tool_use.get("input"), result, llm_calls=frame, step=step)
        return outcome

    stack.enter_context(patch.object(HookRegistry, "invoke_callbacks_async", new=patched_invoke))
    stack.callback(started.clear)


# ---------------------------------------------------------------------------
# OpenAI Agents SDK
# ---------------------------------------------------------------------------


def _openai_agents(stack: ExitStack) -> None:
    try:
        import agents.run_internal.tool_execution as execution_module
        from agents.tool import _FunctionToolInvocationResult
    except ImportError:
        return
    invoke = getattr(execution_module, "_invoke_function_tool_with_metadata", None)
    if invoke is None:  # pragma: no cover — an SDK version without this entry point
        return

    def encode(result: Any) -> Any:
        return {"output": result.output, "is_sdk_generated_error": result.is_sdk_generated_error}

    def decode(recorded: Any) -> Any:
        return _FunctionToolInvocationResult(recorded.get("output"), bool(recorded.get("is_sdk_generated_error")))

    async def patched(*args: Any, **kwargs: Any) -> Any:
        function_tool = kwargs.get("function_tool")
        if function_tool is None:  # pragma: no cover — the SDK passes keywords
            return await invoke(*args, **kwargs)
        return await acall_tool(
            function_tool.name, _json_arguments(kwargs.get("arguments")), lambda: invoke(*args, **kwargs), encode, decode
        )

    stack.enter_context(patch.object(execution_module, "_invoke_function_tool_with_metadata", new=patched))


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


def _mcp(stack: ExitStack) -> None:
    try:
        from mcp import ClientSession
        from mcp.types import CallToolResult
    except ImportError:
        return
    call = ClientSession.call_tool

    def encode(result: Any) -> Any:
        return result.model_dump(by_alias=True, mode="json", exclude_none=True)

    def decode(recorded: Any) -> Any:
        return CallToolResult.model_validate(recorded)

    async def patched(self: Any, name: str, arguments: Any = None, *args: Any, **kwargs: Any) -> Any:
        session = active_session()
        if session is not None and session.running:
            # Part of a framework tool call that is being recorded or run (Strands' MCP tools).
            return await call(self, name, arguments, *args, **kwargs)
        return await acall_tool(name, arguments or {}, lambda: call(self, name, arguments, *args, **kwargs), encode, decode)

    stack.enter_context(patch.object(ClientSession, "call_tool", new=patched))


# ---------------------------------------------------------------------------
# Pydantic AI
# ---------------------------------------------------------------------------


def _pydantic_ai(stack: ExitStack) -> None:
    try:
        from pydantic_ai.toolsets import AbstractToolset
    except ImportError:
        return

    def wrap(cls: type) -> Any:
        call = cls.__dict__["call_tool"]

        async def patched(self: Any, name: str, tool_args: Any, *args: Any, **kwargs: Any) -> Any:
            # A wrapper toolset delegating to the one it wraps is inside the outer call and passes through.
            return await acall_tool(name, tool_args, lambda: call(self, name, tool_args, *args, **kwargs))

        return patched

    # Below ToolManager, so a ModelRetry from the tool replays through Pydantic AI's own retry
    # and usage handling. The output toolset (final_result) is the agent's answer, not a tool.
    for cls in dict.fromkeys(_subclasses(AbstractToolset)):
        if "call_tool" in cls.__dict__ and not cls.__module__.startswith("pydantic_ai._output"):
            stack.enter_context(patch.object(cls, "call_tool", new=wrap(cls)))


# ---------------------------------------------------------------------------
# Google ADK
# ---------------------------------------------------------------------------


def _google_adk(stack: ExitStack) -> None:
    try:
        from google.adk.tools.base_tool import BaseTool
    except ImportError:
        return

    def wrap(cls: type) -> Any:
        run_async = cls.__dict__["run_async"]

        async def patched(self: Any, *, args: Any, tool_context: Any) -> Any:
            if self.name in _ADK_CONTROL_TOOLS:
                # They act on the run itself (transfer, escalate), which a replayed result would not.
                return await run_async(self, args=args, tool_context=tool_context)
            # A subclass calling super().run_async() is inside the outer call and passes through.
            return await acall_tool(self.name, args, lambda: run_async(self, args=args, tool_context=tool_context))

        return patched

    # Every class that defines run_async, so ADK's callbacks, confirmation gate, and error
    # handling see a replayed call as they see a real one.
    for cls in dict.fromkeys(_subclasses(BaseTool)):
        if "run_async" in cls.__dict__:
            stack.enter_context(patch.object(cls, "run_async", new=wrap(cls)))


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


def _crewai(stack: ExitStack) -> None:
    try:
        from crewai import hooks
    except ImportError:
        return
    # CrewAI builds a new hook context for the after hook of a blocked call. It usually shares the
    # tool_input dict; a call without arguments may get a fresh empty one, so fall back to the name.
    replayed: dict[int, tuple[str, Any]] = {}
    started: dict[int, int] = {}

    def answers_directly(context: Any) -> bool:
        # A blocked call loses result_as_answer, so such tools run for real on replay.
        return bool(getattr(getattr(context, "tool", None), "result_as_answer", False))

    def before(context: Any) -> Any:
        session = active_session()
        if session is None or answers_directly(context):
            return None
        if session.replaying:
            execution = session.take(context.tool_name, dict(context.tool_input))
            if not execution.run_on_replay:
                replayed[id(context.tool_input)] = (context.tool_name, execution.result)
                return False  # block the real tool; the after hook supplies the result
        started[id(context.tool_input)] = session.begin()
        session.running += 1
        return None

    def after(context: Any) -> Any:
        key = id(context.tool_input)
        if key not in replayed and key not in started:
            key = next((k for k, (name, _) in replayed.items() if name == context.tool_name), key)
        if key in replayed:
            return replayed.pop(key)[1]
        session = active_session()
        if key in started and session is not None:
            frame, step = session.end(started.pop(key))
            session.running -= 1
            if session.recording:
                session.add(context.tool_name, dict(context.tool_input), context.tool_result, llm_calls=frame, step=step)
        return None

    hooks.register_before_tool_call_hook(before)
    hooks.register_after_tool_call_hook(after)
    stack.callback(hooks.unregister_after_tool_call_hook, after)
    stack.callback(hooks.unregister_before_tool_call_hook, before)


_INSTALLERS = (_langchain, _strands, _openai_agents, _mcp, _pydantic_ai, _google_adk, _crewai)
