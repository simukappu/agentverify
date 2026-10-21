"""Record the agent's tool calls into the cassette, and replay their results.

In record mode each tool call the agent makes is run for real and its result saved next to the LLM interactions. In replay mode the recorded result comes back and the real tool does not run, so a replayed test needs no network, no tool credentials, and leaves no side effects. A cassette recorded before tool results were saved runs the real tools as before.

Calls are matched on the tool name and its arguments, in order among calls with the same name and arguments, since frameworks that run tools in parallel finish them in any order.

Some calls run for real on replay too, and are marked so in the cassette (``run_on_replay``): a result that does not survive the round trip through YAML (a LangGraph ``Command``, binary content, an SDK object), and an error a framework hook cannot raise again where the framework expects it. A replayed tool that called the LLM itself while recording (an agent used as a tool) skips those LLM calls in the cassette, so the outer agent's calls stay in step.

The framework hooks in :mod:`agentverify.cassette.tool_hooks` call :func:`call_tool` / :func:`acall_tool`; :func:`cassette_tool` does the same for a hand-written tool function.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

from agentverify.errors import CassetteToolMismatchError


@dataclass
class ToolExecution:
    """One tool call the agent made: its arguments and its result, or the error it raised."""

    name: str
    arguments: Any
    result: Any = None
    error: dict[str, str] | None = None
    step: int | None = None
    # Positions in the cassette of the LLM calls the tool made while recording (an agent used as a tool).
    llm_calls: list[int] = field(default_factory=list)
    run_on_replay: bool = False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "arguments": self.arguments}
        if self.step is not None:
            data["step"] = self.step
        if self.run_on_replay:
            data["run_on_replay"] = True
        elif self.error is not None:
            data["error"] = self.error
        else:
            data["result"] = self.result
        if self.llm_calls:
            data["llm_calls"] = self.llm_calls
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolExecution:
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments"),
            result=data.get("result"),
            error=data.get("error"),
            step=data.get("step"),
            llm_calls=[int(i) for i in data.get("llm_calls") or []],
            run_on_replay=bool(data.get("run_on_replay")),
        )


def plain(value: Any) -> Any:
    """A YAML- and JSON-safe copy of a tool's arguments or result."""
    from agentverify.cassette.sanitize import _as_plain

    value = _as_plain(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((plain(v) for v in value), key=repr)
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def faithful(value: Any) -> bool:
    """True when *value* survives the round trip through the cassette unchanged (JSON-shaped data)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(faithful(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and faithful(v) for k, v in value.items())
    return False


def _key(name: str, arguments: Any) -> str:
    return json.dumps([name, plain(arguments)], sort_keys=True, ensure_ascii=False)


@dataclass
class ToolSession:
    """The tool calls of one cassette: those recorded in this run, and those available to replay."""

    recorder: Any
    replayable: list[ToolExecution] | None = None
    # Applied to live arguments before matching, as recording applied it before saving (sanitization).
    normalize: Callable[[Any], Any] = lambda arguments: arguments
    recorded: list[ToolExecution] = field(default_factory=list)
    errors: list[CassetteToolMismatchError] = field(default_factory=list)
    # Framework-level tool calls running for real. An MCP call made meanwhile is part of one
    # (Strands runs MCP on a background thread the context does not reach).
    running: int = 0
    _used: set[int] = field(default_factory=set)

    @property
    def replaying(self) -> bool:
        from agentverify.cassette.recorder import CassetteMode

        return self.recorder.mode == CassetteMode.REPLAY and self.replayable is not None

    @property
    def recording(self) -> bool:
        from agentverify.cassette.recorder import CassetteMode

        return self.recorder.mode != CassetteMode.REPLAY

    def llm_count(self) -> int:
        return len(getattr(self.recorder, "_interactions", []))

    def take(self, name: str, arguments: Any) -> ToolExecution:
        """The next unreplayed recording of this call; raise (and remember) a mismatch when there is none.

        A replayed call skips the LLM calls the tool made while recording.
        """
        key = _key(name, self.normalize(plain(arguments)))
        for index, execution in enumerate(self.replayable or []):
            if index not in self._used and _key(execution.name, execution.arguments) == key:
                self._used.add(index)
                if execution.llm_calls and not execution.run_on_replay:
                    self.recorder.skip_interactions(execution.llm_calls)
                return execution
        left = [(e.name, e.arguments) for i, e in enumerate(self.replayable or []) if i not in self._used]
        error = CassetteToolMismatchError(name, plain(arguments), left)
        # Frameworks often turn a tool's exception into a message for the model, so the
        # cassette raises it again when it closes.
        self.errors.append(error)
        raise error

    def begin(self) -> Any:
        """Start a tool call running for real: collect the LLM calls it makes, in this context only."""
        frame: list[int] = []
        return _llm_frame.set(frame), frame, self.llm_count() - 1

    def end(self, started: Any) -> tuple[list[int], int]:
        token, frame, step = started
        try:
            _llm_frame.reset(token)
        except ValueError:  # pragma: no cover — ended in another context than it began
            pass
        return frame, step

    def add(
        self,
        name: str,
        arguments: Any,
        result: Any = None,
        error: BaseException | None = None,
        llm_calls: list[int] | None = None,
        run_on_replay: bool = False,
        step: int | None = None,
    ) -> None:
        if step is None:
            step = self.llm_count() - 1
        if error is None and not faithful(result):
            run_on_replay, result = True, None
        self.recorded.append(
            ToolExecution(
                name=name,
                arguments=plain(arguments),
                result=None if error is not None or run_on_replay else result,
                error=None if error is None else error_of(error),
                step=step if step >= 0 else None,
                llm_calls=list(llm_calls or []),
                run_on_replay=run_on_replay,
            )
        )


_active: ContextVar[Any] = ContextVar("agentverify_tool_session", default=None)
# The LLM calls made inside the tool call running in this context, if any.
_llm_frame: ContextVar[list[int] | None] = ContextVar("agentverify_llm_frame", default=None)


def note_llm_call(index: int) -> None:
    """Called by the recorder for each LLM call it records."""
    frame = _llm_frame.get()
    if frame is not None:
        frame.append(index)
# Set while a recorded tool runs, so a nested layer (an MCP call inside a LangChain tool) is not recorded twice.
_inside: ContextVar[bool] = ContextVar("agentverify_inside_tool", default=False)
# The sessions of the open cassettes, innermost last, for tools a framework runs on a thread that does
# not inherit the caller's context (a background event loop for MCP, a plain thread pool).
_open_sessions: list[Any] = []
# Pushed by a cassette with replay_tools=False, so its tool calls do not reach an outer cassette.
_DISABLED: Any = object()


@contextmanager
def activate(session: ToolSession | None) -> Generator[ToolSession | None, None, None]:
    """Make *session* the active one; ``None`` turns tool recording off inside the block."""
    entry = _DISABLED if session is None else session
    token = _active.set(entry)
    _open_sessions.append(entry)
    try:
        yield session
    finally:
        _open_sessions.remove(entry)
        _active.reset(token)


def active_session() -> ToolSession | None:
    """The tool session to record into or replay from, or None when tool calls should just run."""
    session = _active.get() or (_open_sessions[-1] if _open_sessions else None)
    if session is None or session is _DISABLED or _inside.get():
        return None
    if session.replaying or session.recording:
        return session
    return None  # replaying a cassette recorded without tool results


class ReplayedToolError(Exception):
    """A recorded tool error whose exception type cannot be rebuilt on replay."""


def error_of(error: BaseException) -> dict[str, str]:
    cls = type(error)
    # The constructor argument, not str(): str(KeyError("x")) is "'x'".
    message = error.args[0] if len(error.args) == 1 and isinstance(error.args[0], str) else str(error)
    return {"type": f"{cls.__module__}.{cls.__qualname__}", "message": message}


def rebuild_error(error: dict[str, str]) -> BaseException:
    """The recorded exception, rebuilt from its type and message when the type takes a message."""
    module_name, _, qualname = error.get("type", "").rpartition(".")
    message = error.get("message", "")
    try:
        cls: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        if isinstance(cls, type) and issubclass(cls, Exception):
            return cls(message)
    except Exception:
        pass
    return ReplayedToolError(f"{error.get('type')}: {message}")


@contextmanager
def running_inside() -> Generator[None, None, None]:
    token = _inside.set(True)
    try:
        yield
    finally:
        _inside.reset(token)


def call_tool(
    name: str,
    arguments: Any,
    run: Callable[[], Any],
    encode: Callable[[Any], Any] = lambda result: result,
    decode: Callable[[Any], Any] = lambda result: result,
) -> Any:
    """Run a tool call through the active cassette: record its result, or replay it without running the tool."""
    session = active_session()
    if session is None:
        return run()
    if session.replaying:
        execution = session.take(name, arguments)
        if execution.run_on_replay:
            with running_inside():
                return run()
        if execution.error is not None:
            raise rebuild_error(execution.error)
        return decode(execution.result)
    started = session.begin()
    try:
        with running_inside():
            result = run()
    except Exception as error:
        frame, step = session.end(started)
        session.add(name, arguments, error=error, llm_calls=frame, step=step)
        raise
    frame, step = session.end(started)
    session.add(name, arguments, encode(result), llm_calls=frame, step=step)
    return result


async def acall_tool(
    name: str,
    arguments: Any,
    run: Callable[[], Any],
    encode: Callable[[Any], Any] = lambda result: result,
    decode: Callable[[Any], Any] = lambda result: result,
) -> Any:
    """Async counterpart of :func:`call_tool`; *run* returns an awaitable."""
    session = active_session()
    if session is None:
        return await run()
    if session.replaying:
        execution = session.take(name, arguments)
        if execution.run_on_replay:
            with running_inside():
                return await run()
        if execution.error is not None:
            raise rebuild_error(execution.error)
        return decode(execution.result)
    started = session.begin()
    try:
        with running_inside():
            result = await run()
    except Exception as error:
        frame, step = session.end(started)
        session.add(name, arguments, error=error, llm_calls=frame, step=step)
        raise
    frame, step = session.end(started)
    session.add(name, arguments, encode(result), llm_calls=frame, step=step)
    return result


def cassette_tool(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Record a hand-written tool function's results into the cassette, and replay them.

    Use it on tools of an agent the built-in hooks do not cover (a hand-written ReAct loop). Outside a cassette the function runs as usual. Put it under a framework's own ``@tool`` decorator; the signature is kept for schema generation::

        @cassette_tool
        def get_weather(city: str) -> str:
            ...
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        signature = inspect.signature(func)

        def arguments_of(args: Any, kwargs: Any) -> dict[str, Any]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await acall_tool(tool_name, arguments_of(args, kwargs), lambda: func(*args, **kwargs))

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return call_tool(tool_name, arguments_of(args, kwargs), lambda: func(*args, **kwargs))

        return wrapper

    return decorate(fn) if fn is not None else decorate
