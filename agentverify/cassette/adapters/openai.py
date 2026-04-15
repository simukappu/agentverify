"""OpenAI SDK adapter for LLM cassette recording and replay.

Monkey-patches ``openai.resources.chat.completions.Completions.create``
to intercept chat completion calls for recording and replaying via the
cassette recorder.

openai is an **optional** dependency.  Import this module only when the
openai extra is installed (``pip install agentverify[openai]``).
"""

from __future__ import annotations

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


def _ensure_openai_installed() -> None:
    """Raise a clear error when the openai package is missing."""
    if _openai_module is None:  # pragma: no cover
        raise ImportError(
            "The openai package is required to use OpenAIAdapter. "
            "Install it with: pip install agentverify[openai]"
        )


# ---------------------------------------------------------------------------
# Lightweight response objects used by denormalize_response so that callers
# can access ``response.choices[0].message.content`` etc. without needing a
# real ``openai.types.chat.ChatCompletion`` instance.
# ---------------------------------------------------------------------------


class _Usage:
    """Minimal stand-in for ``openai.types.CompletionUsage``."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FunctionCall:
    """Minimal stand-in for ``openai.types.chat.ChatCompletionMessageToolCall.Function``."""

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """Minimal stand-in for ``openai.types.chat.ChatCompletionMessageToolCall``."""

    def __init__(self, id: str, type: str, function: _FunctionCall) -> None:
        self.id = id
        self.type = type
        self.function = function


class _Message:
    """Minimal stand-in for ``openai.types.chat.ChatCompletionMessage``."""

    def __init__(
        self,
        content: str | None,
        tool_calls: list[_ToolCall] | None,
        role: str = "assistant",
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.role = role


class _Choice:
    """Minimal stand-in for ``openai.types.chat.chat_completion.Choice``."""

    def __init__(self, index: int, message: _Message, finish_reason: str) -> None:
        self.index = index
        self.message = message
        self.finish_reason = finish_reason


class _ChatCompletion:
    """Minimal stand-in for ``openai.types.chat.ChatCompletion``."""

    def __init__(
        self,
        choices: list[_Choice],
        usage: _Usage | None,
        model: str = "",
        id: str = "",
    ) -> None:
        self.choices = choices
        self.usage = usage
        self.model = model
        self.id = id


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
        """Convert an OpenAI ``ChatCompletion`` object to *NormalizedResponse*."""
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
        """Build a lightweight object that mimics ``ChatCompletion`` attribute access."""
        tool_calls: list[_ToolCall] | None = None
        if normalized.tool_calls:
            tool_calls = []
            for i, tc in enumerate(normalized.tool_calls):
                tool_calls.append(
                    _ToolCall(
                        id=f"call_{i}",
                        type="function",
                        function=_FunctionCall(
                            name=tc["name"],
                            arguments=tc["arguments"],
                        ),
                    )
                )

        finish_reason = "tool_calls" if tool_calls else "stop"

        message = _Message(
            content=normalized.content,
            tool_calls=tool_calls,
        )
        choice = _Choice(index=0, message=message, finish_reason=finish_reason)

        usage: _Usage | None = None
        if normalized.token_usage is not None:
            usage = _Usage(
                prompt_tokens=normalized.token_usage.input_tokens,
                completion_tokens=normalized.token_usage.output_tokens,
            )

        model = normalized.raw_metadata.get("model", "")
        completion_id = normalized.raw_metadata.get("id", "")

        return _ChatCompletion(
            choices=[choice],
            usage=usage,
            model=model,
            id=completion_id,
        )

    # -- monkey-patching -----------------------------------------------------

    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:
        """Monkey-patch ``openai.resources.chat.completions.Completions.create``.

        In **record** mode the real API is called, the response is normalised,
        recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the
        cassette, denormalised, and returned without calling the real API.
        """
        _ensure_openai_installed()

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
                    # FALLBACK: call real API when cassette is exhausted
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
                __import__("openai").resources.chat.completions.Completions.create
            ),
        ):
            yield
