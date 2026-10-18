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
    guard_unintercepted,
    patch_scope,
    redispatch,
    reject_stream_kwarg,
    reject_streaming,
    LLMProviderAdapter,
    NormalizedRequest,
    NormalizedResponse,
)
from agentverify.models import TokenUsage

if TYPE_CHECKING:
    from agentverify.cassette.recorder import LLMCassetteRecorder

try:
    import anthropic as _anthropic_module  # noqa: F401 – presence check
except ImportError:  # pragma: no cover
    _anthropic_module = None  # type: ignore[assignment]

_PATCH_TARGET = "anthropic.resources.messages.Messages.create"


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


class AnthropicAdapter(LLMProviderAdapter):
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
                tool_calls.append(
                    {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    }
                )

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
                        id=f"toolu_{i}",
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
            # messages.stream() posts directly, bypassing create().
            _guard(stack, scope, Messages, "stream", "messages.stream", reject_streaming)
            _guard(stack, scope, AsyncMessages, "stream", "AsyncMessages.stream", reject_streaming)
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
