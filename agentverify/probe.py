"""User-side step boundary injection for agentverify.

``step_probe`` is a context manager that marks a logical step boundary
in user agent code.  It is a **zero-cost no-op outside of test recording
contexts** (when no :class:`LLMCassetteRecorder` or :class:`MockLLM` is
active), so it is safe to leave in production code.

Example::

    from agentverify import step_probe

    def run_agent(query: str) -> str:
        with step_probe("fetch_cache"):
            cached = cache.get(query)
            if cached:
                return cached
        with step_probe("call_llm"):
            response = llm.invoke(query)
        with step_probe("postprocess"):
            return format_output(response)

In tests::

    with MockLLM([mock_response(content="42")]) as rec:
        run_agent("what's the answer?")
    result = rec.to_execution_result()
    assert_step(result, name="fetch_cache", expected_tools=[])
    assert_step(result, name="call_llm", ...)
    assert_step_output(result, name="postprocess", contains="42")
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


# Module-level ContextVar holds the currently active recording session,
# if any.  ``None`` means we're running outside any agentverify test
# context — ``step_probe`` becomes a no-op in that case.
_active_session: ContextVar["ProbeSession | None"] = ContextVar(
    "agentverify_active_session", default=None
)


class ProbeSession(Protocol):
    """Interface that recorder/MockLLM expose to the probe machinery."""

    def probe_enter(self, name: str) -> int:
        """Record a probe-enter event; return a handle id."""

    def probe_exit(self, handle_id: int, output: str | None) -> None:
        """Record a probe-exit event."""

    def probe_attach_tool_result(self, handle_id: int, result: Any) -> None:
        """Attach a tool result to the step produced by this probe."""


@dataclass
class ProbeHandle:
    """Handle yielded from ``step_probe`` for tests to attach data.

    ``set_output`` and ``set_tool_result`` are no-ops outside of a
    recording session, matching the overall zero-cost-in-production
    promise.
    """

    name: str
    _session: ProbeSession | None = None
    _handle_id: int | None = None
    _output: str | None = None
    _tool_results: list[Any] = field(default_factory=list)

    def set_output(self, value: str | None) -> None:
        """Set the step's text output (for pure compute/cache steps)."""
        self._output = value

    def set_tool_result(self, result: Any) -> None:
        """Attach a tool execution result to this step."""
        self._tool_results.append(result)
        if self._session is not None and self._handle_id is not None:
            self._session.probe_attach_tool_result(self._handle_id, result)


@contextmanager
def step_probe(name: str, output: str | None = None) -> Iterator[ProbeHandle]:
    """Mark a logical step boundary in user agent code.

    **No-op outside of recorder/MockLLM contexts** — safe to leave in
    production.

    Args:
        name: Human-readable step name.  Used for
            ``assert_step(result, name=...)``.
        output: Optional pre-computed text output for the step (for
            pure compute/cache steps that don't call an LLM).

    Yields:
        A :class:`ProbeHandle` for attaching additional data to the step.
    """
    session = _active_session.get()
    handle = ProbeHandle(name=name, _session=session)
    if output is not None:
        handle._output = output

    if session is None:
        # No-op path.  Yield the handle so user code works unchanged.
        yield handle
        return

    handle_id = session.probe_enter(name)
    handle._handle_id = handle_id
    try:
        yield handle
    finally:
        session.probe_exit(handle_id, handle._output)


@contextmanager
def _activate_session(session: ProbeSession) -> Iterator[None]:
    """Install ``session`` as the active probe session (recorder use only).

    Nested activation is supported — the inner session shadows the outer
    for the duration of the with-block.
    """
    token = _active_session.set(session)
    try:
        yield
    finally:
        _active_session.reset(token)
