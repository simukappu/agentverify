"""LLM Cassette Recorder — VCR-style record/replay for LLM API calls.

Provides :class:`LLMCassetteRecorder` which intercepts LLM SDK chat
completion calls, records request/response pairs to a YAML cassette file,
and replays them for deterministic testing.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any

from agentverify.cassette.adapters.base import (
    LLMProviderAdapter,
    NormalizedRequest,
    NormalizedResponse,
)
from agentverify.cassette.io import load_cassette, save_cassette
from agentverify.cassette.sanitize import DEFAULT_PATTERNS, SanitizePattern, sanitize_interactions
from agentverify.errors import CassetteRequestMismatchError
from agentverify.models import ExecutionResult, TokenUsage, ToolCall


class CassetteMode(Enum):
    """Operating mode for the cassette recorder."""

    RECORD = "record"
    REPLAY = "replay"
    AUTO = "auto"


def _request_to_context(request: NormalizedRequest | None) -> dict[str, Any] | None:
    """Snapshot a NormalizedRequest into a Step.input_context dict.

    The snapshot contains the ``messages`` list (immutable copy) and the
    model name.  Used by :func:`assert_step_uses_result_from` to detect
    data flow between steps.
    """
    if request is None:
        return None
    return {
        "messages": list(request.messages),
        "model": request.model,
    }


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
        match_requests: bool = True,
        sanitize: bool | list[SanitizePattern] = True,
    ) -> None:
        self.cassette_path = Path(cassette_path)
        self._requested_mode = mode
        self.on_missing = on_missing
        self.match_requests = match_requests
        self._adapter = _resolve_provider(provider)

        # Resolve sanitize patterns.
        if sanitize is True:
            self._sanitize_patterns: list[SanitizePattern] | None = DEFAULT_PATTERNS
        elif sanitize is False:
            self._sanitize_patterns = None
        else:
            self._sanitize_patterns = sanitize

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

        # Probe events observed during the session, as a flat list of
        # ``("enter"|"exit", handle_id, name_or_None, output_or_None)``.
        # Interleaved with interaction indices to reconstruct step
        # boundaries in ``to_execution_result``.
        self._probe_events: list[tuple[str, int, str | None, Any]] = []
        # Tool results attached directly to probe handles.
        self._probe_tool_results: dict[int, list[Any]] = {}
        # For each interaction, the probe-handle stack at lookup() time.
        self._interaction_probe_stack: list[list[int]] = []
        self._next_probe_handle_id: int = 0
        self._current_probe_stack: list[int] = []

        # Replay cursor — index of the next interaction to return.
        self._replay_index: int = 0

        # Load existing cassette when replaying.
        if self.mode == CassetteMode.REPLAY:
            self._metadata, self._interactions = load_cassette(self.cassette_path)
        else:
            self._metadata: dict[str, Any] = {}

        # Track whether the patch context manager is active.
        self._patch_ctx: Any = None

        # Timing for latency assertion.
        self._start_time: float | None = None
        self._duration_ms: float | None = None

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> LLMCassetteRecorder:
        """Apply the adapter's monkey-patch and activate probe session."""
        from agentverify.probe import _activate_session

        self._patch_ctx = self._adapter.patch(self)
        self._patch_ctx.__enter__()
        self._probe_ctx = _activate_session(self)
        self._probe_ctx.__enter__()
        self._start_time = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        """Remove the monkey-patch and save the cassette if recording."""
        if self._start_time is not None:
            self._duration_ms = (time.monotonic() - self._start_time) * 1000
            self._start_time = None
        if getattr(self, "_probe_ctx", None) is not None:
            self._probe_ctx.__exit__(*args)
            self._probe_ctx = None
        if self._patch_ctx is not None:
            self._patch_ctx.__exit__(*args)
            self._patch_ctx = None

        if self.mode == CassetteMode.RECORD:
            interactions = self._interactions
            if self._sanitize_patterns:
                interactions = sanitize_interactions(interactions, self._sanitize_patterns)
            save_cassette(
                self.cassette_path,
                interactions,
                provider=self._adapter.name,
            )
        # AUTO with no cassette file (passthrough): interactions are
        # collected in memory but never persisted to disk.

    # -- ProbeSession protocol ----------------------------------------------

    def probe_enter(self, name: str) -> int:
        handle_id = self._next_probe_handle_id
        self._next_probe_handle_id += 1
        self._probe_events.append(("enter", handle_id, name, None))
        self._current_probe_stack.append(handle_id)
        return handle_id

    def probe_exit(self, handle_id: int, output: Any) -> None:
        self._probe_events.append(("exit", handle_id, None, output))
        # Pop from the stack (may not be top of stack if non-LIFO usage).
        try:
            self._current_probe_stack.remove(handle_id)
        except ValueError:
            pass

    def probe_attach_tool_result(self, handle_id: int, result: Any) -> None:
        self._probe_tool_results.setdefault(handle_id, []).append(result)

    # -- Core operations -----------------------------------------------------

    def lookup(self, request: NormalizedRequest) -> NormalizedResponse | None:
        """Return the next unplayed interaction response.

        When ``match_requests`` is enabled, the recorded request's model
        and tool names are compared against the incoming request.  A
        :class:`CassetteRequestMismatchError` is raised on mismatch,
        signalling that the cassette is stale.

        When ``match_requests`` is disabled (default), interactions are
        consumed in recorded order without verifying request content
        (v1 sequential matching).

        Returns:
            The next :class:`NormalizedResponse`, or ``None`` if all
            interactions have been consumed.
        """
        if self._replay_index < len(self._interactions):
            recorded_req, response = self._interactions[self._replay_index]

            if self.match_requests:
                self._verify_request(self._replay_index, recorded_req, request)

            # Record the probe stack active at this lookup for step
            # reconstruction in to_execution_result.
            self._interaction_probe_stack.append(list(self._current_probe_stack))
            self._replay_index += 1
            return response

        # No more recorded interactions.
        return None

    @staticmethod
    def _extract_tool_names(req: NormalizedRequest) -> list[str]:
        """Extract sorted tool names from a NormalizedRequest."""
        if not req.tools:
            return []
        names: list[str] = []
        for tool in req.tools:
            # OpenAI format: {"type": "function", "function": {"name": ...}}
            func = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = func.get("name", "") if isinstance(func, dict) else ""
            if not name:
                # Fallback: top-level "name" key (Bedrock, Anthropic, etc.)
                name = tool.get("name", "") if isinstance(tool, dict) else ""
            if name:
                names.append(name)
        return sorted(names)

    def _verify_request(
        self,
        index: int,
        recorded: NormalizedRequest,
        actual: NormalizedRequest,
    ) -> None:
        """Compare recorded and actual requests, raising on mismatch."""
        # Check model name
        if recorded.model and actual.model and recorded.model != actual.model:
            raise CassetteRequestMismatchError(
                index=index,
                field="model",
                recorded=recorded.model,
                actual=actual.model,
            )

        # Check tool names (sorted for order-independent comparison)
        recorded_tools = self._extract_tool_names(recorded)
        actual_tools = self._extract_tool_names(actual)
        if recorded_tools and actual_tools and recorded_tools != actual_tools:
            raise CassetteRequestMismatchError(
                index=index,
                field="tools",
                recorded=recorded_tools,
                actual=actual_tools,
            )

    def record(
        self, request: NormalizedRequest, response: NormalizedResponse
    ) -> None:
        """Append a request/response pair to the session interactions."""
        self._interactions.append((request, response))
        # Record the probe stack active at record time for step
        # reconstruction in to_execution_result.
        self._interaction_probe_stack.append(list(self._current_probe_stack))

    # -- Result construction -------------------------------------------------

    def to_execution_result(self) -> ExecutionResult:
        """Build an :class:`ExecutionResult` from recorded/replayed interactions.

        Step boundary rules:

        * Each LLM interaction produces one step (``source="llm"``).
        * When an interaction happens inside a ``step_probe(name)``
          block, the innermost probe's name is assigned as the step's
          ``name`` and the step source becomes ``"probe"``.  Multiple
          LLM calls under the same probe handle are merged into a
          single step whose ``tool_calls`` / ``tool_results`` are the
          concatenation.
        * ``step_probe`` blocks that contain no LLM call become
          standalone ``source="probe"`` steps using the pre-computed
          ``output`` passed to ``step_probe(name, output=...)``.
        """
        return _build_execution_result(
            interactions=self._interactions,
            probe_stacks=self._interaction_probe_stack,
            probe_events=self._probe_events,
            probe_tool_results=self._probe_tool_results,
            duration_ms=self._duration_ms,
        )


def _build_execution_result(
    interactions: list[tuple[NormalizedRequest, NormalizedResponse]],
    probe_stacks: list[list[int]],
    probe_events: list[tuple[str, int, str | None, Any]],
    probe_tool_results: dict[int, list[Any]],
    duration_ms: float | None,
) -> ExecutionResult:
    """Reduce interaction + probe events into an ExecutionResult.

    Shared by :class:`LLMCassetteRecorder` and :class:`MockLLM`.
    """
    from agentverify._step_builder import (
        aggregate_token_usage,
        final_output_from_steps,
        tool_calls_from_response,
    )
    from agentverify.models import Step

    # Build probe metadata: handle_id -> (name, output).
    probe_meta: dict[int, dict[str, Any]] = {}
    for kind, handle_id, name, output in probe_events:
        if kind == "enter":
            probe_meta.setdefault(
                handle_id, {"name": name, "output": None, "seen_exit": False}
            )
        elif kind == "exit":
            if handle_id in probe_meta:
                probe_meta[handle_id]["output"] = output
                probe_meta[handle_id]["seen_exit"] = True

    # Group interactions by the innermost probe handle id (or None).
    # Preserve interaction order; a new step is emitted whenever the
    # innermost probe changes.
    steps: list[Step] = []
    current_probe_id: int | None = None
    current_bucket: dict[str, Any] | None = None

    def flush_bucket() -> None:
        nonlocal current_bucket, current_probe_id
        if current_bucket is None:
            return
        index = len(steps)
        probe_id = current_probe_id
        name = None
        source: Any = "llm"
        extra_tool_results: list[Any] = []
        if probe_id is not None and probe_id in probe_meta:
            name = probe_meta[probe_id]["name"]
            source = "probe"
            extra_tool_results = probe_tool_results.get(probe_id, [])
        steps.append(
            Step(
                index=index,
                name=name,
                source=source,
                tool_calls=current_bucket["tool_calls"],
                tool_results=current_bucket["tool_results"] + extra_tool_results,
                output=current_bucket["output"],
                duration_ms=None,
                token_usage=current_bucket["token_usage"],
                input_context=current_bucket["input_context"],
            )
        )
        current_bucket = None
        current_probe_id = None

    for i, (request, response) in enumerate(interactions):
        stack = probe_stacks[i] if i < len(probe_stacks) else []
        innermost = stack[-1] if stack else None

        if current_bucket is None:
            current_probe_id = innermost
            current_bucket = {
                "tool_calls": list(tool_calls_from_response(response.tool_calls)),
                "tool_results": [],
                "output": response.content,
                "token_usage": response.token_usage,
                "input_context": _request_to_context(request),
            }
        elif innermost == current_probe_id and innermost is not None:
            # Same probe, merge interactions.
            current_bucket["tool_calls"].extend(
                tool_calls_from_response(response.tool_calls)
            )
            if response.content is not None:
                current_bucket["output"] = response.content
            if response.token_usage is not None:
                existing = current_bucket["token_usage"]
                if existing is None:
                    current_bucket["token_usage"] = response.token_usage
                else:
                    current_bucket["token_usage"] = TokenUsage(
                        input_tokens=existing.input_tokens + response.token_usage.input_tokens,
                        output_tokens=existing.output_tokens + response.token_usage.output_tokens,
                    )
        else:
            # Probe changed (or current has no probe): flush and start new.
            flush_bucket()
            current_probe_id = innermost
            current_bucket = {
                "tool_calls": list(tool_calls_from_response(response.tool_calls)),
                "tool_results": [],
                "output": response.content,
                "token_usage": response.token_usage,
                "input_context": _request_to_context(request),
            }

    flush_bucket()

    # Emit standalone probe steps for probes that had no LLM interaction.
    seen_probe_ids = {s.index for s in steps}  # placeholder — not by index
    used_probe_handles: set[int] = set()
    # Actually we need to track handle_id usage — reconstruct from stacks.
    for stack in probe_stacks:
        for h in stack:
            used_probe_handles.add(h)

    for handle_id, meta in probe_meta.items():
        if not meta["seen_exit"]:
            continue
        if handle_id in used_probe_handles:
            continue
        # Standalone probe step (no LLM call inside).
        steps.append(
            Step(
                index=len(steps),
                name=meta["name"],
                source="probe",
                tool_calls=[],
                tool_results=list(probe_tool_results.get(handle_id, [])),
                output=meta["output"],
            )
        )

    # Backfill tool_results from the NEXT step's input_context.  When
    # the cassette layer doesn't have direct access to tool execution
    # results (unlike framework adapters), the next LLM call's input
    # messages carry the previous step's tool output as ``role="tool"``
    # messages.  Lifting those onto the producing step makes data-flow
    # assertions (``assert_step_uses_result_from``) work on cassette
    # replay.
    for i in range(len(steps) - 1):
        producer = steps[i]
        consumer_next = steps[i + 1]
        if producer.tool_results:
            continue  # already populated (e.g. by framework adapter)
        if not producer.tool_calls:
            continue  # nothing to correlate with
        if consumer_next.input_context is None:
            continue
        messages = consumer_next.input_context.get("messages")
        if not isinstance(messages, list):  # pragma: no cover — defensive
            continue
        extracted: list[Any] = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                extracted.append(msg.get("content"))
        if extracted:
            steps[i] = Step(
                index=producer.index,
                name=producer.name,
                source=producer.source,
                tool_calls=producer.tool_calls,
                tool_results=extracted,
                output=producer.output,
                duration_ms=producer.duration_ms,
                token_usage=producer.token_usage,
                input_context=producer.input_context,
            )

    # Re-index steps to be contiguous 0..N-1 in case of out-of-order probe inserts.
    steps = [
        Step(
            index=i,
            name=s.name,
            source=s.source,
            tool_calls=s.tool_calls,
            tool_results=s.tool_results,
            output=s.output,
            duration_ms=s.duration_ms,
            token_usage=s.token_usage,
            input_context=s.input_context,
        )
        for i, s in enumerate(steps)
    ]

    return ExecutionResult(
        steps=steps,
        token_usage=aggregate_token_usage(steps),
        final_output=final_output_from_steps(steps),
        duration_ms=duration_ms,
    )
