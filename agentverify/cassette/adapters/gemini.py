"""Google Gemini SDK adapter for LLM cassette recording and replay.

Monkey-patches ``google.genai.models.Models.generate_content`` to intercept
content generation calls for recording and replaying via the cassette recorder.

google-genai is an **optional** dependency.  Import this module only when the
google-genai extra is installed (``pip install agentverify[gemini]``).
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


class GeminiAdapter(LLMProviderAdapter):
    """Adapter for the Google Gemini SDK (``google-genai``)."""

    @property
    def name(self) -> str:
        return "gemini"

    # -- normalisation -------------------------------------------------------

    def normalize_request(self, raw_request: dict[str, Any]) -> NormalizedRequest:
        """Convert Gemini ``generate_content()`` kwargs to *NormalizedRequest*.

        Gemini uses ``contents`` (str or list) for messages, ``model`` for the
        model name, and ``config`` (a ``GenerateContentConfig``) for tools and
        other parameters.
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
                    # SDK Content objects — convert to dict representation
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
                             "stop_sequences", "candidate_count", "response_mime_type"):
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

        Gemini responses have ``candidates[0].content.parts`` which can contain
        ``Part(text=...)`` or ``Part(function_call=FunctionCall(name=..., args=...))``.
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
                    tool_calls.append({
                        "name": getattr(fc, "name", ""),
                        "arguments": args_str,
                    })

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

    # -- monkey-patching -----------------------------------------------------

    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:
        """Monkey-patch ``google.genai.models.Models.generate_content``.

        In **record** mode the real API is called, the response is normalised,
        recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the
        cassette, denormalised, and returned without calling the real API.
        """
        _ensure_genai_installed()

        adapter = self

        import google.genai.models

        original_fn = google.genai.models.Models.generate_content

        def _patched_generate_content(self_client: Any, **kwargs: Any) -> Any:
            """Intercept generate_content calls."""
            from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

            norm_req = adapter.normalize_request(kwargs)

            if recorder.mode in (CassetteMode.RECORD, CassetteMode.AUTO):
                response = original_fn(self_client, **kwargs)
                norm_resp = adapter.normalize_response(response)
                recorder.record(norm_req, norm_resp)
                return response

            # REPLAY mode
            norm_resp = recorder.lookup(norm_req)
            if norm_resp is None:
                if recorder.on_missing == OnMissingRequest.FALLBACK:
                    response = original_fn(self_client, **kwargs)
                    norm_resp = adapter.normalize_response(response)
                    recorder.record(norm_req, norm_resp)
                    return response
                from agentverify.errors import CassetteMissingRequestError

                raise CassetteMissingRequestError(
                    f"No matching cassette entry for request: model={norm_req.model}"
                )
            return adapter.denormalize_response(norm_resp)

        with patch(_PATCH_TARGET, new=_patched_generate_content):
            yield
