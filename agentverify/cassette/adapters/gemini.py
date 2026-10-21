"""Google Gemini SDK adapter for LLM cassette recording and replay.

Monkey-patches ``google.genai.models.Models.generate_content`` to intercept content generation calls for recording and replaying via the cassette recorder.

google-genai is an **optional** dependency.  Import this module only when the google-genai extra is installed (``pip install agentverify[gemini]``).
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
    patch_scope,
    redispatch,
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
    import google.genai  # noqa: F401 – presence check

    _genai_available = True
except ImportError:  # pragma: no cover
    _genai_available = False

_PATCH_TARGET = "google.genai.models.Models.generate_content"


def _ensure_genai_installed() -> None:
    """Raise a clear error when the google-genai package is missing."""
    if not _genai_available:  # pragma: no cover
        raise ImportError(
            "The google-genai package is required to use GeminiAdapter. "
            "Install it with: pip install agentverify[gemini]"
        )


# ---------------------------------------------------------------------------
# Lightweight response objects used by denormalize_response so that callers
# can access ``response.candidates[0].content.parts`` etc. without needing a
# real Gemini SDK response instance.
# ---------------------------------------------------------------------------


class _UsageMetadata:
    """Minimal stand-in for Gemini ``UsageMetadata``."""

    def __init__(self, prompt_token_count: int, candidates_token_count: int) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _FunctionCall:
    """Minimal stand-in for Gemini ``FunctionCall``."""

    def __init__(self, name: str, args: dict[str, Any]) -> None:
        self.name = name
        self.args = args


class _Part:
    """Minimal stand-in for Gemini ``Part``.

    A part has either ``text`` or ``function_call``, but not both.
    """

    def __init__(
        self,
        text: str | None = None,
        function_call: _FunctionCall | None = None,
    ) -> None:
        self.text = text
        self.function_call = function_call


class _Content:
    """Minimal stand-in for Gemini ``Content``."""

    def __init__(self, parts: list[_Part], role: str = "model") -> None:
        self.parts = parts
        self.role = role


class _Candidate:
    """Minimal stand-in for Gemini ``Candidate``."""

    def __init__(self, content: _Content, finish_reason: str = "STOP") -> None:
        self.content = content
        self.finish_reason = finish_reason


class _GenerateContentResponse:
    """Minimal stand-in for Gemini ``GenerateContentResponse``."""

    def __init__(
        self,
        candidates: list[_Candidate],
        usage_metadata: _UsageMetadata | None = None,
    ) -> None:
        self.candidates = candidates
        self.usage_metadata = usage_metadata

    @property
    def text(self) -> str | None:
        """Return concatenated text from all text parts, or None."""
        texts: list[str] = []
        if self.candidates:
            for part in self.candidates[0].content.parts:
                if part.text is not None:
                    texts.append(part.text)
        return "".join(texts) if texts else None


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class GeminiChunkAccumulator(StreamAccumulator):
    """Collects the ``GenerateContentResponse`` chunks of a Gemini stream into one response."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._calls: list[dict[str, Any]] = []
        self._usage: TokenUsage | None = None

    def add(self, chunk: Any) -> None:
        part = GeminiAdapter().normalize_response(chunk)
        if part.content is not None:
            self._text.append(part.content)
        self._calls.extend(part.tool_calls or [])
        self._usage = part.token_usage or self._usage  # each chunk reports the running total

    def result(self) -> NormalizedResponse:
        return NormalizedResponse(
            content="".join(self._text) if self._text else None,
            tool_calls=self._calls or None,
            token_usage=self._usage,
        )


class GeminiAdapter(LLMProviderAdapter, StreamingWire):
    """Adapter for the Google Gemini SDK (``google-genai``)."""

    @property
    def name(self) -> str:
        return "gemini"

    # -- normalisation -------------------------------------------------------

    def normalize_request(self, raw_request: dict[str, Any]) -> NormalizedRequest:
        """Convert Gemini ``generate_content()`` kwargs to *NormalizedRequest*.

        Gemini uses ``contents`` (str or list) for messages, ``model`` for the model name, and ``config`` (a ``GenerateContentConfig``) for tools and other parameters.
        """
        # Messages: contents can be a plain string or a list of messages
        raw_contents = raw_request.get("contents", [])
        if isinstance(raw_contents, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": raw_contents}]
        elif isinstance(raw_contents, list):
            messages = []
            for item in raw_contents:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    messages.append(item)
                else:
                    # SDK Content objects: their fields, so the parts stay readable and comparable
                    dump = getattr(item, "model_dump", None)
                    if callable(dump):
                        messages.append(dump(mode="json", exclude_none=True))
                    else:
                        messages.append({"role": getattr(item, "role", "user"), "content": str(item)})
        else:
            messages = []

        model = raw_request.get("model", "")

        # Extract tools and parameters from config
        config = raw_request.get("config")
        tools: list[dict[str, Any]] | None = None
        parameters: dict[str, Any] = {}

        if config is not None:
            # config can be a dict or a GenerateContentConfig object
            if isinstance(config, dict):
                config_dict = config
            else:
                # SDK object — extract attributes
                config_dict = {}
                for attr in ("temperature", "top_p", "top_k", "max_output_tokens",
                             "stop_sequences", "candidate_count", "response_mime_type", "system_instruction"):
                    val = getattr(config, attr, None)
                    if val is not None:
                        config_dict[attr] = val
                raw_tools = getattr(config, "tools", None)
                if raw_tools is not None:
                    config_dict["tools"] = raw_tools

            # Extract tools from config
            raw_tools = config_dict.pop("tools", None)
            if raw_tools is not None:
                tools = []
                for t in raw_tools:
                    if isinstance(t, dict):
                        # Dict-based tool definition
                        func_decls = t.get("function_declarations", [])
                        for fd in func_decls:
                            tools.append({
                                "name": fd.get("name", ""),
                                "parameters": fd.get("parameters", {}),
                            })
                    else:
                        # SDK Tool object
                        func_decls = getattr(t, "function_declarations", []) or []
                        for fd in func_decls:
                            if isinstance(fd, dict):
                                tools.append({
                                    "name": fd.get("name", ""),
                                    "parameters": fd.get("parameters", {}),
                                })
                            else:
                                tools.append({
                                    "name": getattr(fd, "name", ""),
                                    "parameters": getattr(fd, "parameters", {}),
                                })

            # Remaining config fields go into parameters
            if config_dict:
                parameters = config_dict

        # Extra top-level keys (excluding reserved ones)
        _reserved = {"contents", "model", "config"}
        for k, v in raw_request.items():
            if k not in _reserved:
                parameters[k] = v

        return NormalizedRequest(
            messages=messages,
            model=model,
            tools=tools if tools else None,
            parameters=parameters,
        )

    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        """Convert a Gemini ``GenerateContentResponse`` to *NormalizedResponse*.

        Gemini responses have ``candidates[0].content.parts`` which can contain ``Part(text=...)`` or ``Part(function_call=FunctionCall(name=..., args=...))``.
        """
        content: str | None = None
        tool_calls: list[dict[str, Any]] | None = None

        if raw_response.candidates:
            parts = raw_response.candidates[0].content.parts
            for part in parts:
                # Text part
                if getattr(part, "text", None) is not None:
                    if content is None:
                        content = part.text
                    else:
                        content += part.text

                # Function call part
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    if tool_calls is None:
                        tool_calls = []
                    args = getattr(fc, "args", {}) or {}
                    # Convert args dict to JSON string for consistency
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    else:
                        args_str = json.dumps(dict(args))
                    call = {"name": getattr(fc, "name", ""), "arguments": args_str}
                    if isinstance(getattr(fc, "id", None), str):
                        call["id"] = fc.id
                    tool_calls.append(call)

        # Token usage
        token_usage: TokenUsage | None = None
        usage = getattr(raw_response, "usage_metadata", None)
        if usage is not None:
            token_usage = TokenUsage(
                input_tokens=getattr(usage, "prompt_token_count", 0),
                output_tokens=getattr(usage, "candidates_token_count", 0),
            )

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
        )

    def denormalize_response(self, normalized: NormalizedResponse) -> Any:
        """Build a lightweight object that mimics a Gemini ``GenerateContentResponse``."""
        parts: list[_Part] = []

        if normalized.content is not None:
            parts.append(_Part(text=normalized.content))

        if normalized.tool_calls:
            for tc in normalized.tool_calls:
                arguments = tc["arguments"]
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, ValueError):
                        arguments = {}
                parts.append(
                    _Part(function_call=_FunctionCall(name=tc["name"], args=arguments))
                )

        finish_reason = "FUNCTION_CALL" if normalized.tool_calls else "STOP"
        candidate = _Candidate(
            content=_Content(parts=parts, role="model"),
            finish_reason=finish_reason,
        )

        usage: _UsageMetadata | None = None
        if normalized.token_usage is not None:
            usage = _UsageMetadata(
                prompt_token_count=normalized.token_usage.input_tokens,
                candidates_token_count=normalized.token_usage.output_tokens,
            )

        return _GenerateContentResponse(
            candidates=[candidate],
            usage_metadata=usage,
        )

    # -- streaming -------------------------------------------------------------

    def stream_accumulator(self) -> StreamAccumulator:
        return GeminiChunkAccumulator()

    def replay_stream(self, normalized: NormalizedResponse, kwargs: dict[str, Any], is_async: bool) -> Any:
        """Yield the recorded response as a single ``GenerateContentResponse`` chunk, from a generator (or async generator).

        Streaming callers such as Strands' ``GeminiModel`` read SDK-only fields (``thought``, ``function_call.id``), so the chunk is a real SDK object.
        """
        from google.genai import types

        parts = [] if normalized.content is None else [types.Part(text=normalized.content)]
        for tc in normalized.tool_calls or []:
            arguments = tc["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {}
            parts.append(types.Part(function_call=types.FunctionCall(name=tc["name"], args=arguments, id=tc.get("id"))))
        usage = None
        if normalized.token_usage is not None:
            tokens = normalized.token_usage
            usage = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=tokens.input_tokens,
                candidates_token_count=tokens.output_tokens,
                total_token_count=tokens.input_tokens + tokens.output_tokens,
            )
        candidate = types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=types.FinishReason.STOP)
        response = types.GenerateContentResponse(candidates=[candidate], usage_metadata=usage)
        if is_async:

            async def chunks() -> Any:
                yield response

            return chunks()
        return iter([response])

    # -- monkey-patching -----------------------------------------------------

    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:
        """Monkey-patch ``google.genai.models.Models.generate_content``.

        In **record** mode the real API is called, the response is normalised, recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the cassette, denormalised, and returned without calling the real API.
        """
        _ensure_genai_installed()

        adapter = self

        import google.genai.models

        original_fn = google.genai.models.Models.generate_content

        def _patched_generate_content(self_client: Any, **kwargs: Any) -> Any:
            """Intercept generate_content calls."""
            if not scope.active:  # a cached bound method outliving the cassette
                return redispatch(
                    lambda: google.genai.models.Models.generate_content,
                    _patched_generate_content,
                    original_fn,
                    (self_client,),
                    kwargs,
                )
            return call_sync(
                scope, recorder, adapter, "models.generate_content", kwargs, lambda: original_fn(self_client, **kwargs)
            )

        async_original = google.genai.models.AsyncModels.generate_content

        async def _patched_async_generate_content(self_client: Any, **kwargs: Any) -> Any:
            """Intercept aio.models.generate_content calls."""
            if not scope.active:  # a cached bound method outliving the cassette
                return await redispatch(
                    lambda: google.genai.models.AsyncModels.generate_content,
                    _patched_async_generate_content,
                    async_original,
                    (self_client,),
                    kwargs,
                )
            return await call_async(
                scope, recorder, adapter, "aio.models.generate_content", kwargs, lambda: async_original(self_client, **kwargs)
            )

        stream_original = google.genai.models.Models.generate_content_stream

        def _patched_stream(self_client: Any, **kwargs: Any) -> Any:
            """Intercept generate_content_stream calls (``stream`` marks the request as streamed)."""
            if not scope.active:  # a cached bound method outliving the cassette
                return redispatch(
                    lambda: google.genai.models.Models.generate_content_stream,
                    _patched_stream,
                    stream_original,
                    (self_client,),
                    kwargs,
                )
            return call_sync(
                scope,
                recorder,
                adapter,
                "models.generate_content_stream",
                {**kwargs, "stream": True},
                lambda: stream_original(self_client, **kwargs),
            )

        async_stream_original = google.genai.models.AsyncModels.generate_content_stream

        async def _patched_async_stream(self_client: Any, **kwargs: Any) -> Any:
            """Intercept aio.models.generate_content_stream calls."""
            if not scope.active:  # a cached bound method outliving the cassette
                return await redispatch(
                    lambda: google.genai.models.AsyncModels.generate_content_stream,
                    _patched_async_stream,
                    async_stream_original,
                    (self_client,),
                    kwargs,
                )
            return await call_async(
                scope,
                recorder,
                adapter,
                "aio.models.generate_content_stream",
                {**kwargs, "stream": True},
                lambda: async_stream_original(self_client, **kwargs),
            )

        models = google.genai.models
        api_client = importlib.import_module("google.genai._api_client").BaseApiClient
        with ExitStack() as stack:
            scope = stack.enter_context(patch_scope())
            stack.enter_context(patch(_PATCH_TARGET, new=_patched_generate_content))
            stack.enter_context(patch.object(models.AsyncModels, "generate_content", new=_patched_async_generate_content))
            stack.enter_context(patch.object(models.Models, "generate_content_stream", new=_patched_stream))
            stack.enter_context(patch.object(models.AsyncModels, "generate_content_stream", new=_patched_async_stream))
            # Anything else that reaches the HTTP layer in replay mode raises.
            for method, is_async in (
                ("request", False),
                ("request_streamed", False),
                ("async_request", True),
                ("async_request_streamed", True),
            ):
                if hasattr(api_client, method):  # pragma: no branch — older SDKs lack some methods
                    stack.enter_context(
                        patch.object(
                            api_client,
                            method,
                            new=block_transport_in_replay(recorder, "gemini", getattr(api_client, method), is_async),
                        )
                    )
            yield
