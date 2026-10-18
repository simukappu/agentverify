"""Base classes for LLM provider adapters.

Defines the provider-agnostic normalized request/response formats and the abstract base class that all provider adapters must implement.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import threading
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator

from agentverify.models import TokenUsage

if TYPE_CHECKING:
    from agentverify.cassette.recorder import LLMCassetteRecorder


@dataclass
class NormalizedRequest:
    """Provider-agnostic request format."""

    messages: list[dict[str, Any]]
    model: str
    tools: list[dict[str, Any]] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedResponse:
    """Provider-agnostic response format."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    token_usage: TokenUsage | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


# Fallback ids for tool calls recorded without one (cassettes from before ids were kept).
# Unique per process, so replayed turns never reuse an id; frameworks such as the OpenAI
# Agents SDK reject a run that does.
_fallback_call_ids = itertools.count()


def fallback_call_id(prefix: str = "call") -> str:
    return f"{prefix}_replayed_{next(_fallback_call_ids)}"


class LLMProviderAdapter(ABC):
    """Abstract base class for LLM SDK chat completion adapters."""

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover — abstract method
        """Provider name (e.g. "openai", "anthropic")."""
        ...

    @abstractmethod
    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:  # pragma: no cover — abstract method
        """Monkey-patch the LLM SDK's chat completion method.

        Records or replays requests/responses through the recorder. The patch is automatically removed when the context manager exits.
        """
        ...

    @abstractmethod
    def normalize_request(self, raw_request: dict[str, Any]) -> NormalizedRequest:  # pragma: no cover — abstract method
        """Normalize an SDK-specific request into the common format."""
        ...

    @abstractmethod
    def normalize_response(self, raw_response: Any) -> NormalizedResponse:  # pragma: no cover — abstract method
        """Normalize an SDK-specific response into the common format."""
        ...

    @abstractmethod
    def denormalize_response(self, normalized: NormalizedResponse) -> Any:  # pragma: no cover — abstract method
        """Restore a normalized response back to the SDK-specific format."""
        ...


# ---------------------------------------------------------------------------
# Guards for calls the adapters cannot record or replay
# ---------------------------------------------------------------------------

# Set while an adapter itself calls the real API: the recorded call in record
# mode, or a replay miss with ``on_missing=FALLBACK``. The guards below stay
# out of the way of anything that call does internally.
_live_call_allowed: ContextVar[bool] = ContextVar("agentverify_live_call_allowed", default=False)


@contextmanager
def allow_live_call() -> Generator[None, None, None]:
    """Mark a deliberate real API call made by an adapter, so the guards let it through."""
    token = _live_call_allowed.set(True)
    try:
        yield
    finally:
        _live_call_allowed.reset(token)


class PatchScope:
    """Tracks whether an adapter's patch is still in effect.

    SDKs cache bound methods (for example ``with_raw_response``), so a wrapper can outlive the cassette that installed it. Once :attr:`active` is False, wrappers hand the call to whatever is installed now (another cassette's wrapper, or the SDK's own method).
    """

    def __init__(self) -> None:
        self.active = True
        # Real LLM calls running under this patch while recording, guarded by
        # ``lock`` because threads can record too.
        self.in_flight = 0
        self.lock = threading.Lock()
        # litellm only: completion() calls running, so its internal acompletion() is not recorded twice.
        self.litellm_sync_depth = 0


@contextmanager
def patch_scope() -> Generator[PatchScope, None, None]:
    scope = PatchScope()
    try:
        yield scope
    finally:
        scope.active = False


def redispatch(
    current: Callable[[], Callable[..., Any]], wrapper: Callable[..., Any], original: Callable[..., Any], args: Any, kwargs: Any
) -> Any:
    """Call whatever is installed now in place of a stale *wrapper*."""
    fn = current()
    if fn is wrapper:  # pragma: no cover — defensive; the patch is removed before the scope closes
        fn = original
    return fn(*args, **kwargs)


def _guards_off(recorder: LLMCassetteRecorder) -> bool:
    from agentverify.cassette.recorder import CassetteMode

    # AUTO without a cassette file calls the real API and saves nothing, so
    # there is nothing to protect.
    return recorder.mode == CassetteMode.AUTO or _live_call_allowed.get()


def unsupported_call_error(provider: str, call: str, reason: str):
    """Build the error for an LLM call the cassette recorder cannot handle."""
    from agentverify.errors import CassetteUnsupportedCallError

    return CassetteUnsupportedCallError(
        f"{provider}: {call} is not supported inside a cassette or MockLLM ({reason}). "
        "Use the non-streaming call the adapter records, or make this call outside the cassette."
    )


def reject_stream_kwarg(provider: str, call: str, kwargs: dict[str, Any]) -> None:
    """Raise if a call asks for a streamed response."""
    if kwargs.get("stream") is True:
        raise unsupported_call_error(provider, f"{call}(stream=True)", "streaming is not recorded")


def check_unrecorded(recorder: LLMCassetteRecorder, provider: str, call: str, kwargs: dict[str, Any], stacklevel: int) -> None:
    """Apply the guard for an LLM call the adapter does not record.

    Streaming requests raise. Otherwise replay mode raises instead of reaching the real API, and record mode warns that the cassette will not contain the call.
    """
    from agentverify.cassette.recorder import CassetteMode

    if _guards_off(recorder):
        return
    reject_stream_kwarg(provider, call, kwargs)
    if recorder.mode == CassetteMode.REPLAY:
        raise unsupported_call_error(provider, call, "the adapter does not record it, so replay would call the real API")
    warnings.warn(
        f"{provider}: {call} is not recorded into the cassette; replaying this test will fail at this call.",
        UserWarning,
        stacklevel=stacklevel + 1,
    )


def _is_async(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(inspect.unwrap(fn))


def reject_streaming(
    scope: PatchScope,
    recorder: LLMCassetteRecorder,
    provider: str,
    call: str,
    original: Callable[..., Any],
    current: Callable[[], Callable[..., Any]],
) -> Callable[..., Any]:
    """Wrap a streaming-only entry point so it raises while the cassette is active."""
    if _is_async(original):

        @functools.wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not scope.active:
                return await redispatch(current, async_wrapper, original, args, kwargs)
            if _guards_off(recorder):
                return await original(*args, **kwargs)
            raise unsupported_call_error(provider, call, "streaming is not recorded")

        return async_wrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not scope.active:
            return redispatch(current, wrapper, original, args, kwargs)
        if _guards_off(recorder):
            return original(*args, **kwargs)
        raise unsupported_call_error(provider, call, "streaming is not recorded")

    return wrapper


def guard_unintercepted(
    scope: PatchScope,
    recorder: LLMCassetteRecorder,
    provider: str,
    call: str,
    original: Callable[..., Any],
    current: Callable[[], Callable[..., Any]],
) -> Callable[..., Any]:
    """Wrap an LLM entry point the adapter does not record (see :func:`check_unrecorded`)."""
    if _is_async(original):

        @functools.wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not scope.active:
                return await redispatch(current, async_wrapper, original, args, kwargs)
            check_unrecorded(recorder, provider, call, kwargs, stacklevel=2)
            return await original(*args, **kwargs)

        return async_wrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not scope.active:
            return redispatch(current, wrapper, original, args, kwargs)
        check_unrecorded(recorder, provider, call, kwargs, stacklevel=2)
        return original(*args, **kwargs)

    return wrapper


def block_transport_in_replay(
    recorder: LLMCassetteRecorder, provider: str, original: Callable[..., Any], is_async: bool
) -> Callable[..., Any]:
    """Wrap an SDK's low-level request method so replay mode cannot reach the network.

    Recorded calls are answered before they reach this layer, so any request that does is one the cassette cannot replay: a different endpoint (``parse``, beta resources, embeddings, ...) or a cached wrapper the adapter could not patch.
    """
    from agentverify.cassette.recorder import CassetteMode

    def _check() -> None:
        if recorder.mode == CassetteMode.REPLAY and not _live_call_allowed.get():
            raise unsupported_call_error(
                provider, "this request", "replay mode does not reach the network; record it, or make it outside the cassette"
            )

    if is_async:

        @functools.wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _check()
            return await original(*args, **kwargs)

        return async_wrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _check()
        return original(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Shared record / replay flow for the intercepted call
# ---------------------------------------------------------------------------

_REPLAY_MISS: Any = object()


@contextmanager
def _live_call(scope: PatchScope, recorder: LLMCassetteRecorder, provider: str) -> Generator[None, None, None]:
    """Run one real call, rejecting a second one that overlaps it while recording.

    A cassette replays responses in the order they were recorded. Overlapping calls (``asyncio.gather``, threads) finish in an order that changes from run to run, so recording them would make a cassette that replays the wrong response. Nothing is saved in AUTO passthrough or in replay, so the check applies to record mode only.
    """
    from agentverify.cassette.recorder import CassetteMode

    counted = recorder.mode == CassetteMode.RECORD
    if counted:
        with scope.lock:
            if scope.in_flight > 0:
                raise unsupported_call_error(
                    provider,
                    "a concurrent LLM call",
                    "calls that overlap are recorded in a nondeterministic order; await them one at a time",
                )
            scope.in_flight += 1
    try:
        with allow_live_call():
            yield
    finally:
        if counted:
            with scope.lock:
                scope.in_flight -= 1


def _replay_or_miss(
    recorder: LLMCassetteRecorder,
    adapter: LLMProviderAdapter,
    norm_req: NormalizedRequest,
    replayed: Callable[[NormalizedResponse], Any] | None,
) -> Any:
    """Return the replayed response, or ``_REPLAY_MISS`` when the real API should be called."""
    from agentverify.cassette.recorder import CassetteMode, OnMissingRequest
    from agentverify.errors import CassetteMissingRequestError

    if recorder.mode != CassetteMode.REPLAY:
        return _REPLAY_MISS
    norm_resp = recorder.lookup(norm_req)
    if norm_resp is not None:
        return (replayed or adapter.denormalize_response)(norm_resp)
    if recorder.on_missing == OnMissingRequest.FALLBACK:
        return _REPLAY_MISS
    raise CassetteMissingRequestError(f"No matching cassette entry for request: model={norm_req.model}")


def _stream_passthrough(recorder: LLMCassetteRecorder, provider: str, call: str, kwargs: dict[str, Any]) -> bool:
    """True when a streaming request should go straight to the real API (AUTO passthrough); raise if it cannot."""
    from agentverify.cassette.recorder import CassetteMode

    if kwargs.get("stream") is True and recorder.mode == CassetteMode.AUTO:
        return True
    reject_stream_kwarg(provider, call, kwargs)
    return False


def call_sync(
    scope: PatchScope,
    recorder: LLMCassetteRecorder,
    adapter: LLMProviderAdapter,
    call: str,
    kwargs: dict[str, Any],
    call_real: Callable[[], Any],
    replayed: Callable[[NormalizedResponse], Any] | None = None,
) -> Any:
    """Record or replay one intercepted call.

    Streaming requests raise (or pass straight through in AUTO passthrough). Replay mode returns the recorded response, built by *replayed* (default: ``adapter.denormalize_response``); a miss calls the real API with ``on_missing=FALLBACK`` and raises otherwise. Record mode and AUTO passthrough call the real API and record the normalized response.
    A call made while another intercepted call is reaching the real API in the same task or thread (litellm fallbacks, a tool that calls the model during Gemini automatic function calling) is part of that outer call: it runs unrecorded, because replaying the outer call never makes it.
    """
    if _live_call_allowed.get():
        return call_real()
    if _stream_passthrough(recorder, adapter.name, call, kwargs):
        return call_real()
    norm_req = adapter.normalize_request(kwargs)
    result = _replay_or_miss(recorder, adapter, norm_req, replayed)
    if result is not _REPLAY_MISS:
        return result
    with _live_call(scope, recorder, adapter.name):
        response = call_real()
    recorder.record(norm_req, adapter.normalize_response(response))
    return response


async def call_async(
    scope: PatchScope,
    recorder: LLMCassetteRecorder,
    adapter: LLMProviderAdapter,
    call: str,
    kwargs: dict[str, Any],
    call_real: Callable[[], Any],
    replayed: Callable[[NormalizedResponse], Any] | None = None,
) -> Any:
    """Async counterpart of :func:`call_sync`; *call_real* returns an awaitable."""
    if _live_call_allowed.get():
        return await call_real()
    if _stream_passthrough(recorder, adapter.name, call, kwargs):
        return await call_real()
    norm_req = adapter.normalize_request(kwargs)
    result = _replay_or_miss(recorder, adapter, norm_req, replayed)
    if result is not _REPLAY_MISS:
        return result
    with _live_call(scope, recorder, adapter.name):
        response = await call_real()
    recorder.record(norm_req, adapter.normalize_response(response))
    return response
