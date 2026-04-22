"""In-memory LLM response mocking — zero-cost agent routing tests.

Where :class:`LLMCassetteRecorder` replays responses from a recorded
cassette file, :class:`MockLLM` replays responses from an in-memory
list you define in code. This is useful for testing:

* Agent routing logic without any prior LLM recording.
* Hypothetical scenarios you couldn't easily record (errors, edge cases).
* Behavior-driven tests where the LLM response is the specification.

Usage::

    from agentverify import MockLLM, mock_response, assert_tool_calls, ToolCall

    with MockLLM([
        mock_response(tool_calls=[("get_weather", {"city": "Tokyo"})]),
        mock_response(content="Tokyo is sunny, 22°C."),
    ], provider="openai") as rec:
        my_agent("What's the weather in Tokyo?")

    result = rec.to_execution_result()
    assert_tool_calls(result, expected=[ToolCall("get_weather", {"city": "Tokyo"})])
"""

from __future__ import annotations

import time
from typing import Any

from agentverify.cassette.adapters.base import (
    LLMProviderAdapter,
    NormalizedRequest,
    NormalizedResponse,
)
from agentverify.cassette.recorder import _resolve_provider
from agentverify.errors import CassetteMissingRequestError
from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall


def mock_response(
    content: str | None = None,
    tool_calls: list[tuple[str, dict[str, Any]]] | list[dict[str, Any]] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> NormalizedResponse:
    """Build a :class:`NormalizedResponse` for use with :class:`MockLLM`.

    Args:
        content: Final text content the mock LLM should return.
        tool_calls: Tool calls the mock LLM should emit. Each entry may be
            either a ``(name, arguments)`` tuple or a dict with ``"name"``
            and ``"arguments"`` keys.
        input_tokens: Input token count to report (default 0).
        output_tokens: Output token count to report (default 0).

    Returns:
        A :class:`NormalizedResponse` ready to hand to :class:`MockLLM`.
    """
    normalized_tool_calls: list[dict[str, Any]] | None = None
    if tool_calls is not None:
        normalized_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, tuple):
                name, arguments = tc
                normalized_tool_calls.append(
                    {"name": name, "arguments": arguments}
                )
            elif isinstance(tc, dict):
                normalized_tool_calls.append(
                    {
                        "name": tc["name"],
                        "arguments": tc.get("arguments", {}),
                    }
                )
            else:
                raise TypeError(
                    f"tool_calls entries must be tuple or dict, got {type(tc).__name__}"
                )

    token_usage: TokenUsage | None = None
    if input_tokens or output_tokens:
        token_usage = TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens
        )

    return NormalizedResponse(
        content=content,
        tool_calls=normalized_tool_calls,
        token_usage=token_usage,
    )


class MockLLM:
    """In-memory LLM response replayer.

    Behaves like :class:`LLMCassetteRecorder` in REPLAY mode but takes
    its response sequence from a list you pass in code. No cassette
    file is read or written. No real LLM API is ever called.

    Args:
        responses: Ordered list of responses to return, one per LLM call.
        provider: Provider name (``"openai"``, ``"anthropic"``, ...) or
            an :class:`LLMProviderAdapter` instance.

    Example::

        with MockLLM([
            mock_response(tool_calls=[("search", {"q": "Tokyo"})]),
            mock_response(content="Done"),
        ], provider="openai") as rec:
            my_agent("Find Tokyo")
        result = rec.to_execution_result()
    """

    def __init__(
        self,
        responses: list[NormalizedResponse],
        provider: str | LLMProviderAdapter = "openai",
    ) -> None:
        # Import here to avoid a circular import at module load time.
        from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

        self._adapter = _resolve_provider(provider)
        self._responses = list(responses)
        # Mimic the subset of the LLMCassetteRecorder interface that
        # provider adapters inspect during ``patch()``.
        self.mode = CassetteMode.REPLAY
        self.on_missing = OnMissingRequest.ERROR
        # (request, response) pairs observed during the session — populated
        # on each ``lookup()`` so that ``to_execution_result`` can inspect
        # which requests actually ran.
        self._interactions: list[tuple[NormalizedRequest, NormalizedResponse]] = []
        # Probe event tracking — mirrors LLMCassetteRecorder.
        self._probe_events: list[tuple[str, int, str | None, Any]] = []
        self._probe_tool_results: dict[int, list[Any]] = {}
        self._interaction_probe_stack: list[list[int]] = []
        self._next_probe_handle_id: int = 0
        self._current_probe_stack: list[int] = []
        self._replay_index: int = 0
        self._patch_ctx: Any = None
        self._start_time: float | None = None
        self._duration_ms: float | None = None

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> MockLLM:
        from agentverify.probe import _activate_session

        self._patch_ctx = self._adapter.patch(self)
        self._patch_ctx.__enter__()
        self._probe_ctx = _activate_session(self)
        self._probe_ctx.__enter__()
        self._start_time = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._start_time is not None:
            self._duration_ms = (time.monotonic() - self._start_time) * 1000
            self._start_time = None
        if getattr(self, "_probe_ctx", None) is not None:
            self._probe_ctx.__exit__(*args)
            self._probe_ctx = None
        if self._patch_ctx is not None:
            self._patch_ctx.__exit__(*args)
            self._patch_ctx = None

    # -- ProbeSession protocol ----------------------------------------------

    def probe_enter(self, name: str) -> int:
        handle_id = self._next_probe_handle_id
        self._next_probe_handle_id += 1
        self._probe_events.append(("enter", handle_id, name, None))
        self._current_probe_stack.append(handle_id)
        return handle_id

    def probe_exit(self, handle_id: int, output: Any) -> None:
        self._probe_events.append(("exit", handle_id, None, output))
        try:
            self._current_probe_stack.remove(handle_id)
        except ValueError:
            pass

    def probe_attach_tool_result(self, handle_id: int, result: Any) -> None:
        self._probe_tool_results.setdefault(handle_id, []).append(result)

    # -- Protocol expected by adapter.patch() -------------------------------

    def lookup(self, request: NormalizedRequest) -> NormalizedResponse | None:
        """Return the next predefined response.

        Raises :class:`CassetteMissingRequestError` when the predefined
        response list is exhausted, to make under-specified tests fail
        loudly rather than hang.
        """
        if self._replay_index >= len(self._responses):
            raise CassetteMissingRequestError(
                f"MockLLM ran out of predefined responses after "
                f"{self._replay_index} call(s). "
                f"Add more mock_response(...) entries to the MockLLM list."
            )

        response = self._responses[self._replay_index]
        self._interactions.append((request, response))
        self._interaction_probe_stack.append(list(self._current_probe_stack))
        self._replay_index += 1
        return response

    # -- Result construction -------------------------------------------------

    def to_execution_result(self) -> ExecutionResult:
        """Build an :class:`ExecutionResult` from the mocked interactions.

        Behaves identically to :class:`LLMCassetteRecorder` with respect
        to ``step_probe`` handling.
        """
        from agentverify.cassette.recorder import _build_execution_result

        return _build_execution_result(
            interactions=self._interactions,
            probe_stacks=self._interaction_probe_stack,
            probe_events=self._probe_events,
            probe_tool_results=self._probe_tool_results,
            duration_ms=self._duration_ms,
        )
