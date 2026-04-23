"""OpenAI SDK adapter for LLM cassette recording and replay.

Monkey-patches ``openai.resources.chat.completions.Completions.create``
to intercept chat completion calls for recording and replaying via the
cassette recorder.

openai is an **optional** dependency.  Import this module only when the
openai extra is installed (``pip install agentverify[openai]``).
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
    import openai as _openai_module  # noqa: F401 – presence check
except ImportError:  # pragma: no cover
    _openai_module = None  # type: ignore[assignment]

_PATCH_TARGET = "openai.resources.chat.completions.Completions.create"
_ASYNC_PATCH_TARGET = "openai.resources.chat.completions.AsyncCompletions.create"


def _ensure_openai_installed() -> None:
    """Raise a clear error when the openai package is missing."""
    if _openai_module is None:  # pragma: no cover
        raise ImportError(
            "The openai package is required to use OpenAIAdapter. "
            "Install it with: pip install agentverify[openai]"
        )


# ---------------------------------------------------------------------------
# Lightweight wrappers used for replay responses.
# ---------------------------------------------------------------------------


class _RawResponseWrapper:
    """Minimal stand-in for ``openai._legacy_response.LegacyAPIResponse``.

    langchain-openai v1.x calls ``client.with_raw_response.create(...)``
    which returns an object that exposes ``.parse()`` and ``.headers``.
    When the underlying ``create`` is intercepted via our monkey-patch we
    detect the raw-response code path (via the ``X-Stainless-Raw-Response``
    extra header) and wrap the synthesized ``ChatCompletion`` so the
    caller can ``.parse()`` it.
    """

    def __init__(self, parsed: Any) -> None:
        self._parsed = parsed
        self.headers: dict[str, str] = {}
        self.http_response = None

    def parse(self) -> Any:  # noqa: D401 — mimics openai SDK
        return self._parsed


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class OpenAIAdapter(LLMProviderAdapter):
    """Adapter for the OpenAI Python SDK (``openai>=1.0``)."""

    @property
    def name(self) -> str:
        return "openai"

    # -- normalisation -------------------------------------------------------

    def normalize_request(self, raw_request: dict[str, Any]) -> NormalizedRequest:
        """Convert OpenAI ``chat.completions.create`` kwargs to *NormalizedRequest*."""
        messages = raw_request.get("messages", [])
        model = raw_request.get("model", "")

        # Normalise tools: OpenAI format wraps each tool in
        # {"type": "function", "function": {"name": ..., "parameters": ...}}
        raw_tools = raw_request.get("tools")
        tools: list[dict[str, Any]] | None = None
        if raw_tools is not None:
            tools = []
            for t in raw_tools:
                func = t.get("function", {})
                tools.append(
                    {
                        "name": func.get("name", ""),
                        "parameters": func.get("parameters", {}),
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
        """Convert an OpenAI ``ChatCompletion`` object to *NormalizedResponse*.

        Also handles ``LegacyAPIResponse`` objects returned by
        ``with_raw_response.create()`` (used internally by
        langchain-openai v1.x); these are unwrapped via ``.parse()``.
        """
        # langchain-openai v1.x calls ``client.with_raw_response.create(...)``
        # which returns a ``LegacyAPIResponse`` wrapping the real
        # ``ChatCompletion``. Unwrap it here so recording works across
        # both the plain and raw-response paths.
        if not hasattr(raw_response, "choices") and hasattr(raw_response, "parse"):
            raw_response = raw_response.parse()

        choice = raw_response.choices[0]
        message = choice.message

        content = message.content

        tool_calls: list[dict[str, Any]] | None = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )

        token_usage: TokenUsage | None = None
        if raw_response.usage is not None:
            token_usage = TokenUsage(
                input_tokens=raw_response.usage.prompt_tokens,
                output_tokens=raw_response.usage.completion_tokens,
            )

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
        )

    def denormalize_response(self, normalized: NormalizedResponse) -> Any:
        """Build an ``openai.types.chat.ChatCompletion`` instance from a normalized response.

        Returns a genuine SDK object (constructed via ``model_validate``) so
        that callers which rely on ``model_dump``, ``.parse()``, or other
        pydantic features — e.g. langchain-openai v1.x — interoperate
        correctly.
        """
        from openai.types.chat import ChatCompletion as _ChatCompletionType

        tool_calls_data: list[dict[str, Any]] | None = None
        if normalized.tool_calls:
            tool_calls_data = []
            for i, tc in enumerate(normalized.tool_calls):
                # OpenAI's ChatCompletion schema requires ``arguments`` to be
                # a JSON-encoded string. Normalized representations sometimes
                # carry a dict — serialize it to match the SDK shape.
                args = tc["arguments"]
                if isinstance(args, (dict, list)):
                    args = json.dumps(args, ensure_ascii=False)
                tool_calls_data.append(
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": args,
                        },
                    }
                )

        finish_reason = "tool_calls" if tool_calls_data else "stop"

        usage_data: dict[str, int] | None = None
        if normalized.token_usage is not None:
            usage_data = {
                "prompt_tokens": normalized.token_usage.input_tokens,
                "completion_tokens": normalized.token_usage.output_tokens,
                "total_tokens": (
                    normalized.token_usage.input_tokens
                    + normalized.token_usage.output_tokens
                ),
            }

        model = normalized.raw_metadata.get("model", "") or "gpt-4o-mini"
        completion_id = normalized.raw_metadata.get("id", "") or "chatcmpl-replayed"

        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": normalized.content,
                        "tool_calls": tool_calls_data,
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage_data is not None:
            payload["usage"] = usage_data

        return _ChatCompletionType.model_validate(payload)

    # -- monkey-patching -----------------------------------------------------

    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:
        """Monkey-patch OpenAI's sync and async chat completion entry points.

        Patches both ``openai.resources.chat.completions.Completions.create``
        and ``...AsyncCompletions.create`` so that agents built on either
        the sync ``OpenAI`` client or the ``AsyncOpenAI`` client (used
        by, e.g., OpenAI Agents SDK internals even when driven via
        ``Runner.run_sync``) are recorded and replayed transparently.

        In **record** mode the real API is called, the response is normalised,
        recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the
        cassette, denormalised, and returned without calling the real API.
        """
        _ensure_openai_installed()

        adapter = self

        def _is_raw_response_call(kwargs: dict[str, Any]) -> bool:
            """Detect whether the caller is the ``with_raw_response``
            wrapper from openai SDK. langchain-openai v1.x routes
            through this path and expects a ``LegacyAPIResponse``-like
            object back.

            The openai SDK sets ``X-Stainless-Raw-Response`` to a
            truthy string when routing through ``with_raw_response``;
            the exact value has varied across SDK versions
            (``"raw"`` / ``"true"``), so we accept any non-empty value.
            """
            extra = kwargs.get("extra_headers") or {}
            return bool(extra.get("X-Stainless-Raw-Response"))

        def _handle_replay(
            norm_req: NormalizedRequest, original_fn, args, kwargs, raw_mode: bool
        ):
            """Common replay path for sync and async wrappers."""
            from agentverify.cassette.recorder import OnMissingRequest

            norm_resp = recorder.lookup(norm_req)
            if norm_resp is None:
                if recorder.on_missing == OnMissingRequest.FALLBACK:
                    # Caller handles the fallback call itself; signal miss.
                    return _REPLAY_MISS
                from agentverify.errors import CassetteMissingRequestError

                raise CassetteMissingRequestError(
                    f"No matching cassette entry for request: model={norm_req.model}"
                )
            denormalized = adapter.denormalize_response(norm_resp)
            if raw_mode:
                return _RawResponseWrapper(denormalized)
            return denormalized

        def _patched_sync_create(original_fn):  # type: ignore[no-untyped-def]
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                from agentverify.cassette.recorder import CassetteMode

                norm_req = adapter.normalize_request(kwargs)
                raw_mode = _is_raw_response_call(kwargs)

                if recorder.mode in (CassetteMode.RECORD, CassetteMode.AUTO):
                    response = original_fn(*args, **kwargs)
                    norm_resp = adapter.normalize_response(response)
                    recorder.record(norm_req, norm_resp)
                    return response

                # REPLAY
                result = _handle_replay(norm_req, original_fn, args, kwargs, raw_mode)
                if result is _REPLAY_MISS:  # FALLBACK
                    response = original_fn(*args, **kwargs)
                    norm_resp = adapter.normalize_response(response)
                    recorder.record(norm_req, norm_resp)
                    return response
                return result

            return wrapper

        def _patched_async_create(original_fn):  # type: ignore[no-untyped-def]
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                from agentverify.cassette.recorder import CassetteMode

                norm_req = adapter.normalize_request(kwargs)
                raw_mode = _is_raw_response_call(kwargs)

                if recorder.mode in (CassetteMode.RECORD, CassetteMode.AUTO):
                    response = await original_fn(*args, **kwargs)
                    norm_resp = adapter.normalize_response(response)
                    recorder.record(norm_req, norm_resp)
                    return response

                # REPLAY
                result = _handle_replay(norm_req, original_fn, args, kwargs, raw_mode)
                if result is _REPLAY_MISS:  # FALLBACK
                    response = await original_fn(*args, **kwargs)
                    norm_resp = adapter.normalize_response(response)
                    recorder.record(norm_req, norm_resp)
                    return response
                return result

            return wrapper

        openai_mod = __import__("openai")
        sync_original = openai_mod.resources.chat.completions.Completions.create
        async_original = openai_mod.resources.chat.completions.AsyncCompletions.create

        # Patch both sync and async entry points. ``unittest.mock.patch``
        # restores the originals on context exit even if the body raises.
        with patch(_PATCH_TARGET, new=_patched_sync_create(sync_original)), patch(
            _ASYNC_PATCH_TARGET, new=_patched_async_create(async_original)
        ):
            yield


# Sentinel used by the shared replay helper to signal a cassette miss
# when ``on_missing=FALLBACK`` so the (sync or async) caller can drive
# the real API call itself.
_REPLAY_MISS: Any = object()
