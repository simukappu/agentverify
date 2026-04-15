"""LLM Cassette Recorder — VCR-style record/replay for LLM API calls.

Provides :class:`LLMCassetteRecorder` which intercepts LLM SDK chat
completion calls, records request/response pairs to a YAML cassette file,
and replays them for deterministic testing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from agentverify.cassette.adapters.base import (
    LLMProviderAdapter,
    NormalizedRequest,
    NormalizedResponse,
)
from agentverify.cassette.io import load_cassette, save_cassette
from agentverify.models import ExecutionResult, TokenUsage, ToolCall


class CassetteMode(Enum):
    """Operating mode for the cassette recorder."""

    RECORD = "record"
    REPLAY = "replay"
    AUTO = "auto"


class OnMissingRequest(Enum):
    """Behaviour when a request has no matching cassette entry during replay."""

    ERROR = "error"
    FALLBACK = "fallback"


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, type[LLMProviderAdapter]] = {}


def _resolve_provider(provider: str | LLMProviderAdapter) -> LLMProviderAdapter:
    """Resolve a provider string or instance to an adapter instance."""
    if isinstance(provider, LLMProviderAdapter):
        return provider

    # Lazy-populate the registry on first call.
    if not _PROVIDER_REGISTRY:
        try:
            from agentverify.cassette.adapters.openai import OpenAIAdapter
            _PROVIDER_REGISTRY["openai"] = OpenAIAdapter
        except Exception:  # pragma: no cover
            pass
        try:
            from agentverify.cassette.adapters.bedrock import BedrockAdapter
            _PROVIDER_REGISTRY["bedrock"] = BedrockAdapter
        except Exception:  # pragma: no cover
            pass
        try:
            from agentverify.cassette.adapters.gemini import GeminiAdapter
            _PROVIDER_REGISTRY["gemini"] = GeminiAdapter
        except Exception:  # pragma: no cover
            pass
        try:
            from agentverify.cassette.adapters.anthropic import AnthropicAdapter
            _PROVIDER_REGISTRY["anthropic"] = AnthropicAdapter
        except Exception:  # pragma: no cover
            pass
        try:
            from agentverify.cassette.adapters.litellm import LiteLLMAdapter
            _PROVIDER_REGISTRY["litellm"] = LiteLLMAdapter
        except Exception:  # pragma: no cover
            pass

    adapter_cls = _PROVIDER_REGISTRY.get(provider)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available providers: {sorted(_PROVIDER_REGISTRY.keys())}"
        )
    return adapter_cls()


# ---------------------------------------------------------------------------
# LLMCassetteRecorder
# ---------------------------------------------------------------------------


class LLMCassetteRecorder:
    """VCR-style recorder/replayer for LLM API calls.

    Usage::

        with LLMCassetteRecorder("tests/cassettes/my_test.yaml", provider="openai") as rec:
            # Your agent code that calls the LLM goes here.
            ...
        result = rec.to_execution_result()
    """

    def __init__(
        self,
        cassette_path: Path | str,
        mode: CassetteMode = CassetteMode.AUTO,
        on_missing: OnMissingRequest = OnMissingRequest.ERROR,
        provider: str | LLMProviderAdapter = "openai",
    ) -> None:
        self.cassette_path = Path(cassette_path)
        self._requested_mode = mode
        self.on_missing = on_missing
        self._adapter = _resolve_provider(provider)

        # Resolve AUTO mode based on file existence.
        if mode == CassetteMode.AUTO:
            if self.cassette_path.exists():
                self.mode = CassetteMode.REPLAY
            else:
                # No cassette file: call real LLM API but don't save.
                # Use RECORD mode explicitly to create cassette files.
                self._auto_passthrough = True
                self.mode = CassetteMode.AUTO
        else:
            self._auto_passthrough = False
            self.mode = mode

        # Interactions recorded/loaded for this session.
        self._interactions: list[tuple[NormalizedRequest, NormalizedResponse]] = []

        # Replay cursor — index of the next interaction to return.
        self._replay_index: int = 0

        # Load existing cassette when replaying.
        if self.mode == CassetteMode.REPLAY:
            self._metadata, self._interactions = load_cassette(self.cassette_path)
        else:
            self._metadata: dict[str, Any] = {}

        # Track whether the patch context manager is active.
        self._patch_ctx: Any = None

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> LLMCassetteRecorder:
        """Apply the adapter's monkey-patch."""
        self._patch_ctx = self._adapter.patch(self)
        self._patch_ctx.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Remove the monkey-patch and save the cassette if recording."""
        if self._patch_ctx is not None:
            self._patch_ctx.__exit__(*args)
            self._patch_ctx = None

        if self.mode == CassetteMode.RECORD:
            save_cassette(
                self.cassette_path,
                self._interactions,
                provider=self._adapter.name,
            )
        # AUTO with no cassette file (passthrough): interactions are
        # collected in memory but never persisted to disk.

    # -- Core operations -----------------------------------------------------

    def lookup(self, request: NormalizedRequest) -> NormalizedResponse | None:
        """Return the next unplayed interaction response.

        v1 uses sequential matching — interactions are consumed in recorded
        order.  The *request* parameter is accepted for API compatibility
        but is not used for matching.

        .. note::

            Because request content is not verified, a stale cassette will
            still replay successfully even if the prompt or model has changed.
            Delete and re-record cassettes after significant agent changes.

        Returns:
            The next :class:`NormalizedResponse`, or ``None`` if all
            interactions have been consumed.  When ``on_missing`` is
            :attr:`OnMissingRequest.FALLBACK`, the caller (adapter) should
            invoke the real API and record the result.
        """
        if self._replay_index < len(self._interactions):
            _, response = self._interactions[self._replay_index]
            self._replay_index += 1
            return response

        # No more recorded interactions.
        return None

    def record(
        self, request: NormalizedRequest, response: NormalizedResponse
    ) -> None:
        """Append a request/response pair to the session interactions."""
        self._interactions.append((request, response))

    # -- Result construction -------------------------------------------------

    def to_execution_result(self) -> ExecutionResult:
        """Build an :class:`ExecutionResult` from recorded/replayed interactions.

        * Collects all ``tool_calls`` from responses that have them.
        * Sums ``token_usage`` across all responses.
        * Uses the last text-content response as ``final_output``.
        """
        tool_calls: list[ToolCall] = []
        total_input = 0
        total_output = 0
        has_token_usage = False
        final_output: str | None = None

        for _, resp in self._interactions:
            # Collect tool calls.
            if resp.tool_calls:
                for tc in resp.tool_calls:
                    arguments = tc.get("arguments", {})
                    # arguments may be a JSON string from some adapters.
                    if isinstance(arguments, str):
                        import json

                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, ValueError):
                            arguments = {}
                    tool_calls.append(
                        ToolCall(name=tc["name"], arguments=arguments)
                    )

            # Accumulate token usage.
            if resp.token_usage is not None:
                has_token_usage = True
                total_input += resp.token_usage.input_tokens
                total_output += resp.token_usage.output_tokens

            # Track last text content as final output.
            if resp.content is not None:
                final_output = resp.content

        token_usage = (
            TokenUsage(input_tokens=total_input, output_tokens=total_output)
            if has_token_usage
            else None
        )

        return ExecutionResult(
            tool_calls=tool_calls,
            token_usage=token_usage,
            final_output=final_output,
        )
