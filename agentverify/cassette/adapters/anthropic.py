"""Anthropic SDK adapter for LLM cassette recording and replay.

Monkey-patches ``anthropic.resources.messages.Messages.create``
to intercept message creation calls for recording and replaying via the
cassette recorder.

anthropic is an **optional** dependency.  Import this module only when the
anthropic extra is installed (``pip install agentverify[anthropic]``).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import patch

from agentverify.cassette.adapters.base import (
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

        In **record** mode the real API is called, the response is normalised,
        recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the
        cassette, denormalised, and returned without calling the real API.
        """
        _ensure_anthropic_installed()

        adapter = self

        def _patched_create(original_fn):  # type: ignore[no-untyped-def]
            """Return a wrapper that intercepts ``create()`` calls."""

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

                norm_req = adapter.normalize_request(kwargs)

                if recorder.mode in (CassetteMode.RECORD, CassetteMode.AUTO):
                    # Call the real API and record the interaction
                    response = original_fn(*args, **kwargs)
                    norm_resp = adapter.normalize_response(response)
                    recorder.record(norm_req, norm_resp)
                    return response

                # REPLAY mode
                norm_resp = recorder.lookup(norm_req)
                if norm_resp is None:
                    if recorder.on_missing == OnMissingRequest.FALLBACK:
                        response = original_fn(*args, **kwargs)
                        norm_resp = adapter.normalize_response(response)
                        recorder.record(norm_req, norm_resp)
                        return response
                    from agentverify.errors import CassetteMissingRequestError

                    raise CassetteMissingRequestError(
                        f"No matching cassette entry for request: model={norm_req.model}"
                    )
                return adapter.denormalize_response(norm_resp)

            return wrapper

        # Use unittest.mock.patch to ensure clean teardown
        with patch(
            _PATCH_TARGET,
            new=_patched_create(
                # Grab the real function *before* patching
                __import__("anthropic").resources.messages.Messages.create
            ),
        ):
            yield
