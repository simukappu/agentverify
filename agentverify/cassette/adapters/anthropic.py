"""Anthropic SDK adapter for LLM cassette recording and replay.

Monkey-patches ``anthropic.resources.messages.Messages.create`` to intercept message creation calls for recording and replaying via the cassette recorder.

anthropic is an **optional** dependency.  Import this module only when the anthropic extra is installed (``pip install agentverify[anthropic]``).
"""

from __future__ import annotations

import importlib
import json
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import patch

from agentverify.cassette.adapters.base import (
    block_transport_in_replay,
    call_async,
    call_sync,
    check_unrecorded,
    fallback_call_id,
    guard_unintercepted,
    patch_scope,
    redispatch,
    replayed_sdk_stream,
    LLMProviderAdapter,
    NormalizedRequest,
    NormalizedResponse,
    StreamAccumulator,
    StreamingWire,
)
from agentverify.models import TokenUsage

if TYPE_CHECKING:
    from agentverify.cassette.recorder import LLMCassetteRecorder

try:
    import anthropic as _anthropic_module  # noqa: F401 – presence check
except ImportError:  # pragma: no cover
    _anthropic_module = None  # type: ignore[assignment]

_PATCH_TARGET = "anthropic.resources.messages.Messages.create"


def _is_given(value: Any) -> bool:
    if value is None:
        return False
    return not type(value).__name__ in ("NotGiven", "Omit")


def _not_given() -> Any:
    return getattr(_anthropic_module, "NOT_GIVEN", None)


def _ensure_anthropic_installed() -> None:
    """Raise a clear error when the anthropic package is missing."""
    if _anthropic_module is None:  # pragma: no cover
        raise ImportError(
            "The anthropic package is required to use AnthropicAdapter. "
            "Install it with: pip install agentverify[anthropic]"
        )


# ---------------------------------------------------------------------------
# Lightweight response objects used by denormalize_response so that callers
# can access ``response.content``, ``response.usage``, etc. without needing a
# real ``anthropic.types.Message`` instance.
# ---------------------------------------------------------------------------


class _Usage:
    """Minimal stand-in for ``anthropic.types.Usage``."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _TextBlock:
    """Minimal stand-in for ``anthropic.types.TextBlock``."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    """Minimal stand-in for ``anthropic.types.ToolUseBlock``."""

    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class _Message:
    """Minimal stand-in for ``anthropic.types.Message``."""

    def __init__(
        self,
        content: list[_TextBlock | _ToolUseBlock],
        usage: _Usage | None,
        stop_reason: str = "end_turn",
        model: str = "",
        id: str = "",
    ) -> None:
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason
        self.model = model
        self.id = id


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


def _content_block_to_dict(block: Any) -> Any:
    """Convert an Anthropic content block (SDK object or dict) to a plain dict.

    Keeps raw strings and plain dicts as-is; reshapes SDK objects by reading the fields we care about for every supported block type (``text`` / ``tool_use`` / ``tool_result``) and falls back to ``model_dump`` for anything else.
    """
    if isinstance(block, (str, int, float, bool)) or block is None:
        return block
    if isinstance(block, dict):
        return block

    block_type = getattr(block, "type", None)
    if block_type == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}) or {},
        }
    if block_type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": getattr(block, "tool_use_id", ""),
            "content": getattr(block, "content", ""),
        }

    # Unknown block type — try pydantic's ``model_dump`` if available,
    # otherwise stringify so the YAML stays parseable.
    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:  # pragma: no cover — defensive
            pass
    return str(block)


def _normalise_anthropic_message(message: Any) -> Any:
    """Flatten SDK content-block objects inside a single message dict."""
    if not isinstance(message, dict):
        return message
    content = message.get("content")
    if isinstance(content, list):
        message = dict(message)
        message["content"] = [_content_block_to_dict(b) for b in content]
    return message


class MessageEventAccumulator(StreamAccumulator):
    """Collects Anthropic ``RawMessageStreamEvent`` objects into one response."""

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._input_tokens: int | None = None
        self._output_tokens = 0
        self._model = ""

    def add(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if kind == "message_start":
            message = event.message
            self._model = getattr(message, "model", "") or ""
            usage = getattr(message, "usage", None)
            if usage is not None:
                self._input_tokens = usage.input_tokens
                self._output_tokens = getattr(usage, "output_tokens", 0) or 0
        elif kind == "content_block_start":
            block = event.content_block
            entry: dict[str, Any] = {"type": block.type, "text": getattr(block, "text", "") or "", "json": ""}
            if block.type == "tool_use":
                entry.update(id=block.id, name=block.name)
            self._blocks[event.index] = entry
        elif kind == "content_block_delta":
            entry = self._blocks.setdefault(event.index, {"type": "text", "text": "", "json": ""})
            delta = event.delta
            if delta.type == "text_delta":
                entry["text"] += delta.text
            elif delta.type == "input_json_delta":
                entry["json"] += delta.partial_json
        elif kind == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None and getattr(usage, "output_tokens", None) is not None:
                self._output_tokens = usage.output_tokens

    def result(self) -> NormalizedResponse:
        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for index in sorted(self._blocks):
            block = self._blocks[index]
            if block["type"] == "text":
                texts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append({"name": block["name"], "arguments": block["json"] or "{}", "id": block["id"]})
        usage = None
        if self._input_tokens is not None:
            usage = TokenUsage(input_tokens=self._input_tokens, output_tokens=self._output_tokens)
        return NormalizedResponse(
            content="".join(texts) if texts else None,
            tool_calls=tool_calls or None,
            token_usage=usage,
            raw_metadata={"model": self._model} if self._model else {},
        )


def message_events(normalized: NormalizedResponse) -> list[dict[str, Any]]:
    """The events of a replayed Messages stream, as JSON-shaped dicts: one delta per content block."""
    tokens = normalized.token_usage
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": normalized.raw_metadata.get("id", "") or "msg_replayed",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": normalized.raw_metadata.get("model", "") or "claude-replayed",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": tokens.input_tokens if tokens else 0, "output_tokens": 0},
            },
        }
    ]
    blocks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if normalized.content is not None:
        blocks.append(({"type": "text", "text": ""}, {"type": "text_delta", "text": normalized.content}))
    for tc in normalized.tool_calls or []:
        args = tc["arguments"]
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        start = {"type": "tool_use", "id": tc.get("id") or fallback_call_id("toolu"), "name": tc["name"], "input": {}}
        blocks.append((start, {"type": "input_json_delta", "partial_json": args}))
    for index, (start, delta) in enumerate(blocks):
        events.append({"type": "content_block_start", "index": index, "content_block": start})
        events.append({"type": "content_block_delta", "index": index, "delta": delta})
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use" if normalized.tool_calls else "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": tokens.output_tokens if tokens else 0},
        }
    )
    events.append({"type": "message_stop"})
    return events


class AnthropicAdapter(LLMProviderAdapter, StreamingWire):
    """Adapter for the Anthropic Python SDK (``anthropic>=0.20``)."""

    @property
    def name(self) -> str:
        return "anthropic"

    # -- normalisation -------------------------------------------------------

    def normalize_request(self, raw_request: dict[str, Any]) -> NormalizedRequest:
        """Convert Anthropic ``messages.create()`` kwargs to *NormalizedRequest*."""
        messages = raw_request.get("messages", [])
        model = raw_request.get("model", "")

        # Normalise tools: Anthropic format uses
        # {"name": ..., "input_schema": {...}}
        raw_tools = raw_request.get("tools")
        tools: list[dict[str, Any]] | None = None
        if raw_tools is not None:
            tools = []
            for t in raw_tools:
                tools.append(
                    {
                        "name": t.get("name", ""),
                        "parameters": t.get("input_schema", {}),
                    }
                )

        # Everything else goes into parameters
        _reserved = {"messages", "model", "tools"}
        parameters = {k: v for k, v in raw_request.items() if k not in _reserved}

        # Normalise messages: ReAct-style agents append the SDK's own
        # content-block objects (``TextBlock`` / ``ToolUseBlock``) back
        # into the conversation history so the next turn's request
        # carries them verbatim.  Dumping those pydantic objects into
        # the cassette YAML would embed Python-class tags
        # (``!!python/object:anthropic.types...``) that can't be loaded
        # on a machine with a different anthropic SDK version.  Flatten
        # them to plain dicts on the way in.
        if isinstance(messages, list):
            messages = [_normalise_anthropic_message(m) for m in messages]

        return NormalizedRequest(
            messages=messages,
            model=model,
            tools=tools,
            parameters=parameters,
        )

    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        """Convert an Anthropic ``Message`` response to *NormalizedResponse*."""
        content: str | None = None
        tool_calls: list[dict[str, Any]] | None = None

        for block in raw_response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                if content is None:
                    content = block.text
                else:
                    content += block.text
            elif block_type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                call = {"name": block.name, "arguments": json.dumps(block.input)}
                if isinstance(getattr(block, "id", None), str):
                    call["id"] = block.id  # replayed turns keep distinct ids
                tool_calls.append(call)

        token_usage: TokenUsage | None = None
        if raw_response.usage is not None:
            token_usage = TokenUsage(
                input_tokens=raw_response.usage.input_tokens,
                output_tokens=raw_response.usage.output_tokens,
            )

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
        )

    def denormalize_response(self, normalized: NormalizedResponse) -> Any:
        """Build a lightweight object that mimics an Anthropic ``Message``."""
        content_blocks: list[_TextBlock | _ToolUseBlock] = []

        if normalized.content is not None:
            content_blocks.append(_TextBlock(text=normalized.content))

        if normalized.tool_calls:
            for i, tc in enumerate(normalized.tool_calls):
                arguments = tc["arguments"]
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, ValueError):
                        arguments = {}
                content_blocks.append(
                    _ToolUseBlock(
                        id=tc.get("id") or fallback_call_id("toolu"),
                        name=tc["name"],
                        input=arguments,
                    )
                )

        stop_reason = "tool_use" if normalized.tool_calls else "end_turn"

        usage: _Usage | None = None
        if normalized.token_usage is not None:
            usage = _Usage(
                input_tokens=normalized.token_usage.input_tokens,
                output_tokens=normalized.token_usage.output_tokens,
            )

        model = normalized.raw_metadata.get("model", "")
        message_id = normalized.raw_metadata.get("id", "")

        return _Message(
            content=content_blocks,
            usage=usage,
            stop_reason=stop_reason,
            model=model,
            id=message_id,
        )

    # -- streaming -------------------------------------------------------------

    def stream_accumulator(self) -> StreamAccumulator:
        return MessageEventAccumulator()

    def replay_stream(self, normalized: NormalizedResponse, kwargs: dict[str, Any], is_async: bool) -> Any:
        """Build an ``anthropic.Stream[RawMessageStreamEvent]`` (or ``AsyncStream``) for a recorded response."""
        from anthropic import AsyncStream, Stream
        from anthropic._models import construct_type
        from anthropic.types import RawMessageStreamEvent

        items = [construct_type(type_=RawMessageStreamEvent, value=e) for e in message_events(normalized)]
        return replayed_sdk_stream(Stream, AsyncStream, items, is_async)

    # -- monkey-patching -----------------------------------------------------

    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:
        """Monkey-patch ``anthropic.resources.messages.Messages.create``.

        In **record** mode the real API is called, the response is normalised, recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the cassette, denormalised, and returned without calling the real API.
        """
        _ensure_anthropic_installed()

        adapter = self

        def _is_raw_response_call(kwargs: dict[str, Any]) -> bool:
            extra = kwargs.get("extra_headers") or {}
            return bool(extra.get("X-Stainless-Raw-Response"))

        def _patched_create(original_fn, scope, current):  # type: ignore[no-untyped-def]
            """Return a wrapper that intercepts ``Messages.create()`` calls."""

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not scope.active:  # a cached bound method outliving the cassette
                    return redispatch(current, wrapper, original_fn, args, kwargs)
                if _is_raw_response_call(kwargs):
                    # with_raw_response: the adapter cannot rebuild the raw wrapper on replay.
                    check_unrecorded(recorder, "anthropic", "messages.with_raw_response.create", kwargs, stacklevel=2)
                    return original_fn(*args, **kwargs)
                return call_sync(
                    scope, recorder, adapter, "messages.create", kwargs, lambda: original_fn(*args, **kwargs)
                )

            return wrapper

        def _patched_async_create(original_fn, scope, current):  # type: ignore[no-untyped-def]
            """Return a wrapper that intercepts ``AsyncMessages.create()`` calls."""

            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not scope.active:  # a cached bound method outliving the cassette
                    return await redispatch(current, wrapper, original_fn, args, kwargs)
                if _is_raw_response_call(kwargs):
                    # with_raw_response: the adapter cannot rebuild the raw wrapper on replay.
                    check_unrecorded(recorder, "anthropic", "AsyncMessages.with_raw_response.create", kwargs, stacklevel=2)
                    return await original_fn(*args, **kwargs)
                return await call_async(
                    scope, recorder, adapter, "AsyncMessages.create", kwargs, lambda: original_fn(*args, **kwargs)
                )

            return wrapper

        def _stream_via_create(original_fn, scope, is_async):  # type: ignore[no-untyped-def]
            """Wrap ``messages.stream()`` to stream through the recorded ``create(stream=True)``."""
            from anthropic.lib.streaming import AsyncMessageStreamManager, MessageStreamManager

            def wrapper(self_: Any, **kwargs: Any) -> Any:
                if not scope.active:  # a cached bound method outliving the cassette
                    return original_fn(self_, **kwargs)
                if _is_given(kwargs.get("output_format")):
                    # Structured output: create() does not take output_format, so this stream is not recorded.
                    check_unrecorded(recorder, "anthropic", "messages.stream(output_format=...)", {"stream": True}, stacklevel=2)
                    return original_fn(self_, **kwargs)
                if is_async:
                    return AsyncMessageStreamManager(self_.create(**kwargs, stream=True))
                return MessageStreamManager(lambda: self_.create(**kwargs, stream=True), output_format=_not_given())

            return wrapper

        anthropic_mod = __import__("anthropic")
        messages_mod = anthropic_mod.resources.messages
        base_client = importlib.import_module("anthropic._base_client")
        Messages, AsyncMessages = messages_mod.Messages, messages_mod.AsyncMessages

        def _guard(stack, scope, cls, method, call, guard):  # type: ignore[no-untyped-def]
            original = getattr(cls, method)
            current = lambda: getattr(cls, method)  # noqa: E731
            stack.enter_context(patch.object(cls, method, new=guard(scope, recorder, "anthropic", call, original, current)))

        # Use unittest.mock.patch to ensure clean teardown
        with ExitStack() as stack:
            scope = stack.enter_context(patch_scope())
            stack.enter_context(
                patch(_PATCH_TARGET, new=_patched_create(Messages.create, scope, lambda: Messages.create))
            )
            stack.enter_context(
                patch.object(
                    AsyncMessages,
                    "create",
                    new=_patched_async_create(AsyncMessages.create, scope, lambda: AsyncMessages.create),
                )
            )
            # messages.stream() posts directly, bypassing create(); send it through create(stream=True).
            stack.enter_context(patch.object(Messages, "stream", new=_stream_via_create(Messages.stream, scope, False)))
            stack.enter_context(patch.object(AsyncMessages, "stream", new=_stream_via_create(AsyncMessages.stream, scope, True)))
            for cls in (Messages, AsyncMessages):
                if hasattr(cls, "parse"):  # pragma: no branch — older SDKs lack parse
                    _guard(stack, scope, cls, "parse", "messages.parse", guard_unintercepted)
            # Anything else that reaches the HTTP layer in replay mode raises
            # (beta.messages, count_tokens, cached with_raw_response wrappers, ...).
            for cls, is_async in ((base_client.SyncAPIClient, False), (base_client.AsyncAPIClient, True)):
                stack.enter_context(
                    patch.object(cls, "request", new=block_transport_in_replay(recorder, "anthropic", cls.request, is_async))
                )
            yield
