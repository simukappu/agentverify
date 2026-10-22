"""Base classes for LLM provider adapters.

Defines the provider-agnostic normalized request/response formats and the abstract base class that all provider adapters must implement.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
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
        "Use a call the adapter records, or make this call outside the cassette."
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
# Streaming
# ---------------------------------------------------------------------------


class StreamAccumulator(ABC):
    """Collects a streamed response, item by item, into a :class:`NormalizedResponse`."""

    @abstractmethod
    def add(self, item: Any) -> None:  # pragma: no cover — abstract method
        ...

    @abstractmethod
    def result(self) -> NormalizedResponse:  # pragma: no cover — abstract method
        ...


class StreamingWire(ABC):
    """The streaming half of a wire format: read a live stream, and rebuild one on replay.

    A cassette stores a streamed call as the same normalized response as a non-streamed one, so either kind of call replays either kind of recording. Replay yields the response in as few items as the SDK's stream shape allows, not in the recorded chunk sizes.
    """

    @abstractmethod
    def stream_accumulator(self) -> StreamAccumulator:  # pragma: no cover — abstract method
        ...

    @abstractmethod
    def replay_stream(self, normalized: NormalizedResponse, kwargs: dict[str, Any], is_async: bool) -> Any:  # pragma: no cover
        """Build the SDK's stream object for a recorded response."""
        ...

    def tap_response(self, response: Any, tap: Callable[[Any], Any]) -> Any:
        """Apply *tap* to the stream in a live response. Override when the stream sits inside the response."""
        return tap(response)


class _StreamTap:
    """Feeds each item a caller reads from a live stream into an accumulator, and reports the outcome once.

    Reading the stream counts as part of the live call: a lazy stream (Gemini, LiteLLM) sends its request, and runs any automatic tool calls, only as it is read.
    """

    def __init__(
        self, accumulator: StreamAccumulator, finish: Callable[[NormalizedResponse], None], fail: Callable[[], None]
    ) -> None:
        self.accumulator = accumulator
        self._finish = finish
        self._fail = fail
        self._done = False

    def end(self) -> None:
        if not self._done:
            self._done = True
            self._finish(self.accumulator.result())

    def failed(self) -> None:
        """The stream raised before its end: record nothing, as for a call that raises."""
        if not self._done:
            self._done = True
            self._fail()

    def step(self, read: Callable[[], Any], end: type[BaseException]) -> Any:
        try:
            with allow_live_call():
                item = read()
        except end:
            self.end()
            raise
        except Exception:
            self.failed()
            raise
        self.accumulator.add(item)
        return item

    async def astep(self, read: Callable[[], Any]) -> Any:
        try:
            with allow_live_call():
                item = await read()
        except StopAsyncIteration:
            self.end()
            raise
        except Exception:
            self.failed()
            raise
        self.accumulator.add(item)
        return item


_tapped_classes: dict[type, type] = {}


def _tapped_class(cls: type) -> type:
    """A subclass of a stream class whose iteration feeds the instance's tap.

    Swapping the instance's class keeps ``isinstance`` checks and every other attribute of the SDK's stream intact.
    """
    if cls in _tapped_classes:
        return _tapped_classes[cls]
    namespace: dict[str, Any] = {}
    if hasattr(cls, "__next__"):

        def __next__(self: Any) -> Any:
            return self._agentverify_tap.step(super(tapped, self).__next__, StopIteration)

        def __iter__(self: Any) -> Any:
            return self

        namespace.update(__next__=__next__, __iter__=__iter__)
    elif hasattr(cls, "__iter__"):

        def __iter__(self: Any) -> Any:  # type: ignore[no-redef]
            return _tap_generator(super(tapped, self).__iter__(), self._agentverify_tap)

        namespace["__iter__"] = __iter__
    if inspect.isasyncgenfunction(getattr(cls, "__anext__", None)):
        # aiobotocore's AioEventStream: __aiter__ returns the async generator __anext__ makes.

        def __aiter__(self: Any) -> Any:  # type: ignore[no-redef]
            return _tap_async_generator(super(tapped, self).__aiter__(), self._agentverify_tap)

        namespace["__aiter__"] = __aiter__
    elif hasattr(cls, "__anext__"):

        async def __anext__(self: Any) -> Any:
            return await self._agentverify_tap.astep(super(tapped, self).__anext__)

        def __aiter__(self: Any) -> Any:
            return self

        namespace.update(__anext__=__anext__, __aiter__=__aiter__)
    close = getattr(cls, "close", None)
    if close is not None and _is_async(close):

        async def aclose_(self: Any) -> Any:
            self._agentverify_tap.end()
            return await super(tapped, self).close()

        namespace["close"] = aclose_
    elif close is not None:

        def close_(self: Any) -> Any:
            self._agentverify_tap.end()
            return super(tapped, self).close()

        namespace["close"] = close_
    tapped = type(cls.__name__, (cls,), namespace)
    tapped.__module__ = cls.__module__
    _tapped_classes[cls] = tapped
    return tapped


def tap_stream(
    stream: Any,
    accumulator: StreamAccumulator,
    finish: Callable[[NormalizedResponse], None],
    fail: Callable[[], None] = lambda: None,
) -> Any:
    """Return *stream* so that reading it records the response through *finish*.

    The response is recorded when the caller reads the stream to the end or closes it, with whatever it read by then. A stream that raises calls *fail* instead. SDK stream objects keep their class (a tapped subclass); plain generators are wrapped.
    """
    tap = _StreamTap(accumulator, finish, fail)
    if inspect.isgenerator(stream):
        return _tap_generator(stream, tap)
    if inspect.isasyncgen(stream):
        return _tap_async_generator(stream, tap)
    try:
        stream.__class__ = _tapped_class(type(stream))
    except TypeError:
        return stream  # not a stream object (a test double); nothing to record
    stream._agentverify_tap = tap
    return stream


class _ReplayedHTTPResponse:
    """Stands in for the HTTP response behind a replayed SDK stream."""

    status_code = 200
    request = None

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


def replayed_sdk_stream(stream_cls: type, async_stream_cls: type, items: list[Any], is_async: bool) -> Any:
    """Build an OpenAI / Anthropic SDK ``Stream`` (or ``AsyncStream``) that yields *items*.

    Both SDKs share one stream implementation: iteration reads ``_iterator`` and closing closes ``response``. Building the object without ``__init__`` keeps the SDK's own class, so the SDK's stream helpers and ``isinstance`` checks accept it.
    """
    cls = async_stream_cls if is_async else stream_cls
    stream = cls.__new__(cls)
    stream.response = _ReplayedHTTPResponse()
    stream._options = None
    if is_async:

        async def iterate() -> Any:
            for item in items:
                yield item

        stream._iterator = iterate()
    else:
        stream._iterator = iter(items)
    return stream


def _tap_generator(stream: Any, tap: _StreamTap) -> Any:
    # Stepped by hand so the live-call marker covers each read, not the caller's code between them.
    iterator = iter(stream)
    try:
        while True:
            try:
                item = tap.step(lambda: next(iterator), StopIteration)
            except StopIteration:
                return
            yield item
    finally:
        tap.end()


async def _tap_async_generator(stream: Any, tap: _StreamTap) -> Any:
    iterator = stream.__aiter__()
    try:
        while True:
            try:
                item = await tap.astep(iterator.__anext__)
            except StopAsyncIteration:
                return
            yield item
    finally:
        tap.end()


# ---------------------------------------------------------------------------
# Shared record / replay flow for the intercepted call
# ---------------------------------------------------------------------------

_REPLAY_MISS: Any = object()


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


def _streaming(adapter: Any, kwargs: dict[str, Any]) -> bool:
    """True for a ``stream=True`` request the wire format can record."""
    return kwargs.get("stream") is True and isinstance(adapter, StreamingWire)


def _stream_replayer(adapter: Any, kwargs: dict[str, Any], is_async: bool) -> Callable[[NormalizedResponse], Any]:
    return lambda normalized: adapter.replay_stream(normalized, kwargs, is_async)


def _record(
    recorder: LLMCassetteRecorder, adapter: Any, norm_req: NormalizedRequest, response: Any, streaming: bool
) -> Any:
    """Record a live response and return it to the caller.

    A stream is recorded in request order, as an empty slot filled in when the caller finishes reading it (see :func:`record_stream`).
    """
    if streaming:
        return record_stream(recorder, adapter, norm_req, response)
    recorder.record(norm_req, adapter.normalize_response(response))
    return response


def record_stream(recorder: LLMCassetteRecorder, wire: StreamingWire, norm_req: NormalizedRequest, stream: Any) -> Any:
    """Record a live stream: reserve its slot now, fill it in when the caller has read it.

    A stream the caller never finishes is filled in, with what was read, when the cassette closes. A stream that raises partway is dropped, as a call that raises records nothing: the framework's retry then takes the slot.
    """
    slot = NormalizedResponse()
    recorder.record(norm_req, slot)
    accumulator = wire.stream_accumulator()
    filled = False

    def finish(normalized: NormalizedResponse) -> None:
        nonlocal filled
        if not filled:
            filled = True
            for f in fields(NormalizedResponse):
                setattr(slot, f.name, getattr(normalized, f.name))

    def fail() -> None:
        nonlocal filled
        filled = True
        discard = getattr(recorder, "discard", None)
        if discard is not None:
            discard(slot)

    pending = getattr(recorder, "_pending_streams", None)
    if pending is not None:
        pending.append(lambda: finish(accumulator.result()))
    return wire.tap_response(stream, lambda s: tap_stream(s, accumulator, finish, fail))


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

    Streaming requests are recorded and replayed when the wire format is a :class:`StreamingWire`, and raise otherwise (or pass straight through in AUTO passthrough). Replay mode returns the recorded response, built by *replayed* (default: ``adapter.denormalize_response``); a miss calls the real API with ``on_missing=FALLBACK`` and raises otherwise. Record mode and AUTO passthrough call the real API and record the normalized response.
    A call made while another intercepted call is reaching the real API in the same task or thread (litellm fallbacks, a tool that calls the model during Gemini automatic function calling) is part of that outer call: it runs unrecorded, because replaying the outer call never makes it.
    """
    if _live_call_allowed.get():
        return call_real()
    streaming = _streaming(adapter, kwargs)
    if not streaming and _stream_passthrough(recorder, adapter.name, call, kwargs):
        return call_real()
    norm_req = adapter.normalize_request(kwargs)
    result = _replay_or_miss(recorder, adapter, norm_req, _stream_replayer(adapter, kwargs, False) if streaming else replayed)
    if result is not _REPLAY_MISS:
        return result
    with allow_live_call():  # the guards stay out of the way of what the real call does internally
        response = call_real()
    return _record(recorder, adapter, norm_req, response, streaming)


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
    streaming = _streaming(adapter, kwargs)
    if not streaming and _stream_passthrough(recorder, adapter.name, call, kwargs):
        return await call_real()
    norm_req = adapter.normalize_request(kwargs)
    result = _replay_or_miss(recorder, adapter, norm_req, _stream_replayer(adapter, kwargs, True) if streaming else replayed)
    if result is not _REPLAY_MISS:
        return result
    with allow_live_call():  # the guards stay out of the way of what the real call does internally
        response = await call_real()
    return _record(recorder, adapter, norm_req, response, streaming)
