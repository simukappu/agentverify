"""OpenAI SDK adapter for LLM cassette recording and replay.

Monkey-patches ``openai.resources.chat.completions.Completions.create`` to intercept chat completion calls for recording and replaying via the cassette recorder.

openai is an **optional** dependency.  Import this module only when the openai extra is installed (``pip install agentverify[openai]``).
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
    guard_unintercepted,
    patch_scope,
    redispatch,
    reject_streaming,
    LLMProviderAdapter,
    NormalizedRequest,
    NormalizedResponse,
)
from agentverify.cassette.adapters.openai_responses import ResponsesWire, fallback_call_id
from agentverify.models import TokenUsage

if TYPE_CHECKING:
    from agentverify.cassette.recorder import LLMCassetteRecorder

try:
    import openai as _openai_module  # noqa: F401 – presence check
except ImportError:  # pragma: no cover
    _openai_module = None  # type: ignore[assignment]

_PATCH_TARGET = "openai.resources.chat.completions.Completions.create"
_ASYNC_PATCH_TARGET = "openai.resources.chat.completions.AsyncCompletions.create"

# Responses API entry points the adapter does not record yet. Each is patched
# only if the installed SDK has it.
_RESPONSES_UNINTERCEPTED = ("parse",)
_RESPONSES_STREAMING = ("stream", "connect")


def _is_openai_sentinel(value: Any) -> bool:
    """Return ``True`` if ``value`` is an openai ``Omit`` / ``NotGiven`` sentinel.

    The openai SDK passes these in place of absent keyword arguments so they propagate through to provider calls.  We drop them on the way into the cassette so YAML stays clean and deterministic.
    """
    if _openai_module is None:  # pragma: no cover — skipped when openai missing
        return False
    for cls_name in ("Omit", "NotGiven"):
        cls = getattr(_openai_module, cls_name, None)
        if cls is not None and isinstance(value, cls):
            return True
    return False


def _strip_message_sentinels(message: Any) -> Any:
    """Return ``message`` with openai sentinels removed from top-level keys."""
    if not isinstance(message, dict):
        return message
    return {k: v for k, v in message.items() if not _is_openai_sentinel(v)}


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

    langchain-openai v1.x calls ``client.with_raw_response.create(...)`` which returns an object that exposes ``.parse()`` and ``.headers``. When the underlying ``create`` is intercepted via our monkey-patch we detect the raw-response code path (via the ``X-Stainless-Raw-Response`` extra header) and wrap the synthesized ``ChatCompletion`` so the caller can ``.parse()`` it.
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
        # OpenAI Agents SDK (and any native ``openai`` caller) may pass
        # ``openai.omit`` / ``openai.NOT_GIVEN`` sentinels for absent
        # arguments instead of ``None`` — treat those as "no tools".
        raw_tools = raw_request.get("tools")
        tools: list[dict[str, Any]] | None = None
        if isinstance(raw_tools, list):
            tools = []
            for t in raw_tools:
                if not isinstance(t, dict):
                    continue
                func = t.get("function", {}) or {}
                tools.append(
                    {
                        "name": func.get("name", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )

        # Everything else goes into parameters.  Filter out openai
        # sentinel values so the cassette YAML stays clean.
        _reserved = {"messages", "model", "tools"}
        parameters = {
            k: v
            for k, v in raw_request.items()
            if k not in _reserved and not _is_openai_sentinel(v)
        }

        # Normalise messages so ``role="user"`` / ``"system"`` dicts
        # without a ``content`` key (sent by the Agents SDK with
        # ``content=omit``) don't leak sentinels into the cassette.
        if isinstance(messages, list):
            messages = [_strip_message_sentinels(m) for m in messages]

        return NormalizedRequest(
            messages=messages,
            model=model,
            tools=tools,
            parameters=parameters,
        )

    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        """Convert an OpenAI ``ChatCompletion`` object to *NormalizedResponse*.

        Also handles ``LegacyAPIResponse`` objects returned by ``with_raw_response.create()`` (used internally by langchain-openai v1.x); these are unwrapped via ``.parse()``.
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
                call = {"name": tc.function.name, "arguments": tc.function.arguments}
                # Kept so replayed turns carry the recorded ids; frameworks such as the
                # OpenAI Agents SDK reject a run that reuses a tool call id.
                if isinstance(getattr(tc, "id", None), str):
                    call["id"] = tc.id
                tool_calls.append(call)

        token_usage: TokenUsage | None = None
        usage = getattr(raw_response, "usage", None)  # litellm.ModelResponse drops a None usage
        if usage is not None:
            token_usage = TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            )

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
        )

    def denormalize_response(self, normalized: NormalizedResponse) -> Any:
        """Build an ``openai.types.chat.ChatCompletion`` instance from a normalized response.

        Returns a genuine SDK object (constructed via ``model_validate``) so that callers which rely on ``model_dump``, ``.parse()``, or other pydantic features — e.g. langchain-openai v1.x — interoperate correctly.
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
                        "id": tc.get("id") or fallback_call_id(),
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

        Patches both ``openai.resources.chat.completions.Completions.create`` and ``...AsyncCompletions.create`` so that agents built on either the sync ``OpenAI`` client or the ``AsyncOpenAI`` client (used by, e.g., OpenAI Agents SDK internals even when driven via ``Runner.run_sync``) are recorded and replayed transparently.

        In **record** mode the real API is called, the response is normalised, recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the cassette, denormalised, and returned without calling the real API.
        """
        _ensure_openai_installed()

        adapter = self

        def _is_raw_response_call(kwargs: dict[str, Any]) -> bool:
            """Detect whether the caller is the ``with_raw_response`` wrapper from openai SDK. langchain-openai v1.x routes through this path and expects a ``LegacyAPIResponse``-like object back.

            The openai SDK sets ``X-Stainless-Raw-Response`` to a truthy string when routing through ``with_raw_response``; the exact value has varied across SDK versions (``"raw"`` / ``"true"``), so we accept any non-empty value.
            """
            extra = kwargs.get("extra_headers") or {}
            return bool(extra.get("X-Stainless-Raw-Response"))

        def _replayed(wire, kwargs: dict[str, Any]):  # type: ignore[no-untyped-def]
            """Build the replayed response; ``with_raw_response`` callers get a ``.parse()``-able wrapper."""
            raw_mode = _is_raw_response_call(kwargs)

            def build(norm_resp: NormalizedResponse) -> Any:
                denormalized = wire.denormalize_response(norm_resp)
                return _RawResponseWrapper(denormalized) if raw_mode else denormalized

            return build

        def _patched_create(wire, call, original_fn, scope, cls, is_async):  # type: ignore[no-untyped-def]
            """Wrap ``create`` for one wire format (Chat Completions or Responses)."""

            def run(args: Any, kwargs: dict[str, Any], helper):  # type: ignore[no-untyped-def]
                return helper(
                    scope, recorder, wire, call, kwargs, lambda: original_fn(*args, **kwargs), _replayed(wire, kwargs)
                )

            if is_async:

                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    if not scope.active:  # a cached bound method outliving the cassette
                        return await redispatch(lambda: cls.create, async_wrapper, original_fn, args, kwargs)
                    return await run(args, kwargs, call_async)

                return async_wrapper

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not scope.active:  # a cached bound method outliving the cassette
                    return redispatch(lambda: cls.create, wrapper, original_fn, args, kwargs)
                return run(args, kwargs, call_sync)

            return wrapper

        responses_wire = ResponsesWire()

        openai_mod = __import__("openai")
        completions_mod = openai_mod.resources.chat.completions
        Completions, AsyncCompletions = completions_mod.Completions, completions_mod.AsyncCompletions
        base_client = importlib.import_module("openai._base_client")

        def _guard(stack, scope, cls, method, call, guard):  # type: ignore[no-untyped-def]
            original = getattr(cls, method)
            current = lambda: getattr(cls, method)  # noqa: E731
            stack.enter_context(patch.object(cls, method, new=guard(scope, recorder, "openai", call, original, current)))

        # ``unittest.mock.patch`` restores the originals on context exit even if
        # the body raises. The scope closes after the patches are removed.
        with ExitStack() as stack:
            scope = stack.enter_context(patch_scope())
            for cls, is_async in ((Completions, False), (AsyncCompletions, True)):
                stack.enter_context(
                    patch.object(
                        cls,
                        "create",
                        new=_patched_create(adapter, "chat.completions.create", cls.create, scope, cls, is_async),
                    )
                )
            for cls in (Completions, AsyncCompletions):
                if hasattr(cls, "parse"):  # pragma: no branch — older SDKs lack parse
                    _guard(stack, scope, cls, "parse", "chat.completions.parse", guard_unintercepted)
            try:
                # Imported explicitly: the SDK loads resource submodules lazily.
                responses_mod = importlib.import_module("openai.resources.responses")
            except ImportError:  # pragma: no cover — openai without the Responses API
                responses_mod = None
            for cls_name in ("Responses", "AsyncResponses"):
                cls = getattr(responses_mod, cls_name, None)
                if cls is None:  # pragma: no cover — openai without the Responses API
                    continue
                stack.enter_context(
                    patch.object(
                        cls,
                        "create",
                        new=_patched_create(
                            responses_wire, "responses.create", cls.create, scope, cls, cls_name == "AsyncResponses"
                        ),
                    )
                )
                for method in _RESPONSES_UNINTERCEPTED:
                    if hasattr(cls, method):  # pragma: no branch — older SDKs lack some methods
                        _guard(stack, scope, cls, method, f"responses.{method}", guard_unintercepted)
                for method in _RESPONSES_STREAMING:
                    if hasattr(cls, method):  # pragma: no branch — older SDKs lack some methods
                        _guard(stack, scope, cls, method, f"responses.{method}", reject_streaming)
            # Anything else that reaches the HTTP layer in replay mode raises.
            for cls, is_async in ((base_client.SyncAPIClient, False), (base_client.AsyncAPIClient, True)):
                stack.enter_context(
                    patch.object(cls, "request", new=block_transport_in_replay(recorder, "openai", cls.request, is_async))
                )
            yield
