"""LiteLLM adapter for LLM cassette recording and replay.

Monkey-patches ``litellm.completion`` to intercept chat completion calls
for recording and replaying via the cassette recorder.

Since LiteLLM uses an OpenAI-compatible request/response format, this
adapter delegates normalize_request, normalize_response, and
denormalize_response to the :class:`OpenAIAdapter`.

litellm is an **optional** dependency.  Import this module only when the
litellm extra is installed (``pip install agentverify[litellm]``).
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
from agentverify.cassette.adapters.openai import OpenAIAdapter

if TYPE_CHECKING:
    from agentverify.cassette.recorder import LLMCassetteRecorder

try:
    import litellm as _litellm_module  # noqa: F401 – presence check
except ImportError:  # pragma: no cover
    _litellm_module = None  # type: ignore[assignment]

_PATCH_TARGET = "litellm.completion"

# Shared OpenAI adapter instance for normalize/denormalize delegation.
_openai_adapter = OpenAIAdapter()


def _ensure_litellm_installed() -> None:
    """Raise a clear error when the litellm package is missing."""
    if _litellm_module is None:  # pragma: no cover
        raise ImportError(
            "The litellm package is required to use LiteLLMAdapter. "
            "Install it with: pip install agentverify[litellm]"
        )


class LiteLLMAdapter(LLMProviderAdapter):
    """Adapter for LiteLLM (OpenAI-compatible interface to 140+ providers)."""

    @property
    def name(self) -> str:
        return "litellm"

    # -- normalisation (delegated to OpenAI adapter) -------------------------

    def normalize_request(self, raw_request: dict[str, Any]) -> NormalizedRequest:
        """Convert LiteLLM ``completion()`` kwargs to *NormalizedRequest*.

        LiteLLM uses the same request format as OpenAI.
        """
        return _openai_adapter.normalize_request(raw_request)

    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        """Convert a LiteLLM response to *NormalizedResponse*.

        LiteLLM returns OpenAI-compatible response objects.
        """
        return _openai_adapter.normalize_response(raw_response)

    def denormalize_response(self, normalized: NormalizedResponse) -> Any:
        """Build a lightweight object that mimics an OpenAI-compatible response.

        Reuses the OpenAI adapter's denormalize since LiteLLM consumers
        expect the same attribute-access pattern.
        """
        return _openai_adapter.denormalize_response(normalized)

    # -- monkey-patching -----------------------------------------------------

    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:
        """Monkey-patch ``litellm.completion``.

        In **record** mode the real API is called, the response is normalised,
        recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the
        cassette, denormalised, and returned without calling the real API.
        """
        _ensure_litellm_installed()

        adapter = self

        import litellm

        original_fn = litellm.completion

        def _patched_completion(*args: Any, **kwargs: Any) -> Any:
            """Intercept litellm.completion() calls."""
            from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

            norm_req = adapter.normalize_request(kwargs)

            if recorder.mode in (CassetteMode.RECORD, CassetteMode.AUTO):
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

        with patch(_PATCH_TARGET, new=_patched_completion):
            yield
