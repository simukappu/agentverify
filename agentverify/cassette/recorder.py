"""LLM Cassette Recorder — VCR-style record/replay for LLM API calls.

Provides :class:`LLMCassetteRecorder` which intercepts LLM SDK chat completion calls, records request/response pairs to a YAML cassette file, and replays them for deterministic testing.
"""

from __future__ import annotations

from dataclasses import replace

import json
import re

import time
from enum import Enum
from pathlib import Path
from typing import Any

from agentverify.cassette.adapters.base import (
    LLMProviderAdapter,
    NormalizedRequest,
    NormalizedResponse,
)
from agentverify.cassette.io import _parse_interactions, load_document, save_cassette
from agentverify.cassette.sanitize import (
    DEFAULT_PATTERNS,
    SanitizePattern,
    sanitize_interactions,
    sanitize_tool_arguments,
    sanitize_tool_executions,
)
from agentverify.cassette.tools import ToolExecution, ToolSession
from agentverify.errors import CassetteRequestMismatchError
from agentverify.models import ExecutionResult, TokenUsage, ToolCall, attribute_agents, handoff_target


class CassetteMode(Enum):
    """Operating mode for the cassette recorder."""

    RECORD = "record"
    REPLAY = "replay"
    AUTO = "auto"


def _request_to_context(request: NormalizedRequest | None) -> dict[str, Any] | None:
    """Snapshot a NormalizedRequest into a Step.input_context dict.

    The snapshot contains the ``messages`` list (immutable copy) and the model name.  Used by :func:`assert_step_uses_result_from` to detect data flow between steps.
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
        replay_tools: bool = True,
        match_ignore: list[str] | None = None,
    ) -> None:
        self.cassette_path = Path(cassette_path)
        self.replay_tools = replay_tools
        self._match_ignore = [re.compile(p) for p in match_ignore or []]
        # Set by the cassette fixture, for the stale-cassette summary at the end of the run.
        self.test_id: str | None = None
        self._request_errors: list[CassetteRequestMismatchError] = []
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
        # Error flags aligned with _probe_tool_results entries.
        self._probe_tool_results_meta: dict[int, list[bool]] = {}
        # For each interaction, the probe-handle stack at lookup() time.
        self._interaction_probe_stack: list[list[int]] = []
        self._next_probe_handle_id: int = 0
        self._current_probe_stack: list[int] = []

        # Replay cursor — index of the next interaction to return.
        self._replay_index: int = 0

        # Fill-ins for recorded streams the caller has not finished reading.
        self._pending_streams: list[Any] = []

        # Load existing cassette when replaying. A cassette recorded without
        # tool results has no tool_executions key, and its tools run for real.
        self._recorded_tools: list[ToolExecution] | None = None
        if self.mode == CassetteMode.REPLAY:
            document = load_document(self.cassette_path)
            self._metadata, self._interactions = _parse_interactions(document)
            if isinstance(document.get("tool_executions"), list):
                self._recorded_tools = [ToolExecution.from_dict(e) for e in document["tool_executions"]]
        else:
            self._metadata: dict[str, Any] = {}
        self._tool_session: ToolSession | None = None
        self._tool_ctx: Any = None

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
        from agentverify.cassette.tool_hooks import tool_hooks

        if self.replay_tools:
            patterns = self._sanitize_patterns
            self._tool_session = ToolSession(
                self,
                replayable=self._recorded_tools,
                normalize=(lambda a: a) if patterns is None else (lambda a: sanitize_tool_arguments(a, patterns)),
            )
        # With replay_tools=False this still shields the block from an outer cassette's tool session.
        self._tool_ctx = tool_hooks(self._tool_session)
        self._tool_ctx.__enter__()
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
        if self._tool_ctx is not None:
            self._tool_ctx.__exit__(*args)
            self._tool_ctx = None
        if self._patch_ctx is not None:
            self._patch_ctx.__exit__(*args)
            self._patch_ctx = None
        for fill_in in self._pending_streams:
            fill_in()
        self._pending_streams = []

        if self.mode == CassetteMode.RECORD:
            interactions = self._interactions
            tools = None if self._tool_session is None else [e.to_dict() for e in self._tool_session.recorded]
            if self._sanitize_patterns is not None:
                interactions = sanitize_interactions(interactions, self._sanitize_patterns)
                tools = None if tools is None else sanitize_tool_executions(tools, self._sanitize_patterns)
            save_cassette(
                self.cassette_path,
                interactions,
                provider=self._adapter.name,
                tool_executions=tools,
            )
        # A mismatch raised inside the agent may have been caught by the framework
        # (and a tool error turned into a message for the model); fail the test here instead.
        if args[0] is None and self._request_errors:
            raise self._request_errors[0]
        if self._tool_session is not None and self._tool_session.errors and args[0] is None:
            raise self._tool_session.errors[0]
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

    def probe_attach_tool_result(self, handle_id: int, result: Any, is_error: bool = False) -> None:
        self._probe_tool_results.setdefault(handle_id, []).append(result)
        self._probe_tool_results_meta.setdefault(handle_id, []).append(is_error)

    # -- Core operations -----------------------------------------------------

    def lookup(self, request: NormalizedRequest) -> NormalizedResponse | None:
        """Return the next unplayed interaction response.

        When ``match_requests`` is enabled, the incoming request is compared with the recorded one: model, tool names and schemas, the system prompt, and the text of user messages (not tool results, which carry the tools' output). A :class:`CassetteRequestMismatchError` is raised on a difference, signalling that the cassette is stale.

        When ``match_requests`` is disabled, interactions are consumed in recorded order without verifying request content.

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
        if self._sanitize_patterns is not None:
            # The recording was sanitized when saved; compare like with like.
            actual = sanitize_interactions([(actual, NormalizedResponse())], self._sanitize_patterns)[0][0]
        try:
            self._compare(index, recorded, actual)
        except CassetteRequestMismatchError as error:
            self._request_errors.append(error)
            _stale_cassettes.append((str(self.cassette_path), self.test_id, error.field))
            raise

    def _compare(self, index: int, recorded: NormalizedRequest, actual: NormalizedRequest) -> None:
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

        # Check each tool's parameter schema
        recorded_schemas, actual_schemas = _tool_schemas(recorded), _tool_schemas(actual)
        for name in sorted(set(recorded_schemas) & set(actual_schemas)):
            # An empty schema on either side (a cassette that did not record it) skips the check.
            if recorded_schemas[name] != actual_schemas[name] and _EMPTY_SCHEMA not in (recorded_schemas[name], actual_schemas[name]):
                raise CassetteRequestMismatchError(
                    index=index,
                    field=f"tool schema {name!r}",
                    recorded=recorded_schemas[name],
                    actual=actual_schemas[name],
                )

        # Check the prompt the developer wrote: system prompt and user message text
        for field, view in (("system prompt", _system_texts), ("user messages", _user_texts)):
            recorded_texts = [self._ignore(s) for s in view(recorded)]
            actual_texts = [self._ignore(s) for s in view(actual)]
            # A cassette that recorded no system prompt (or SDK object reprs, from older versions) skips the check.
            if not recorded_texts or any(_SDK_REPR.match(s) for s in recorded_texts):
                continue
            if recorded_texts != actual_texts:
                raise CassetteRequestMismatchError(
                    index=index,
                    field=field,
                    recorded="\n".join(recorded_texts),
                    actual="\n".join(actual_texts),
                )

    def _ignore(self, text: str) -> str:
        for pattern in self._match_ignore:
            text = pattern.sub("<ignored>", text)
        return text

    def record(
        self, request: NormalizedRequest, response: NormalizedResponse
    ) -> None:
        """Append a request/response pair to the session interactions."""
        self._interactions.append((request, response))
        # Record the probe stack active at record time for step
        # reconstruction in to_execution_result.
        self._interaction_probe_stack.append(list(self._current_probe_stack))

    def skip_interactions(self, count: int) -> None:
        """Pass over the LLM calls a replayed tool made while recording (an agent used as a tool)."""
        for _ in range(count):
            if self._replay_index < len(self._interactions):
                self._interaction_probe_stack.append(list(self._current_probe_stack))
                self._replay_index += 1

    def discard(self, response: NormalizedResponse) -> None:
        """Remove a recorded interaction (a stream that raised partway) by its response object."""
        for index, (_, recorded) in enumerate(self._interactions):
            if recorded is response:
                del self._interactions[index]
                del self._interaction_probe_stack[index]
                return

    # -- Result construction -------------------------------------------------

    def to_execution_result(self) -> ExecutionResult:
        """Build an :class:`ExecutionResult` from recorded/replayed interactions.

        Step boundary rules:

        * Each LLM interaction produces one step (``source="llm"``).
        * When an interaction happens inside a ``step_probe(name)``
          block, the innermost probe's name is assigned as the step's ``name`` and the step source becomes ``"probe"``.  Multiple LLM calls under the same probe handle are merged into a single step whose ``tool_calls`` / ``tool_results`` are the concatenation.
        * ``step_probe`` blocks that contain no LLM call become
          standalone ``source="probe"`` steps using the pre-computed ``output`` passed to ``step_probe(name, output=...)``.
        """
        return _build_execution_result(
            interactions=self._interactions,
            probe_stacks=self._interaction_probe_stack,
            probe_events=self._probe_events,
            probe_tool_results=self._probe_tool_results,
            probe_tool_results_meta=self._probe_tool_results_meta,
            duration_ms=self._duration_ms,
        )


def _build_execution_result(
    interactions: list[tuple[NormalizedRequest, NormalizedResponse]],
    probe_stacks: list[list[int]],
    probe_events: list[tuple[str, int, str | None, Any]],
    probe_tool_results: dict[int, list[Any]],
    duration_ms: float | None,
    probe_tool_results_meta: dict[int, list[bool]] | None = None,
) -> ExecutionResult:
    """Reduce interaction + probe events into an ExecutionResult.

    Shared by :class:`LLMCassetteRecorder` and :class:`MockLLM`.
    """
    from agentverify._step_builder import (
        aggregate_token_usage,
        build_tool_results_meta,
        classify_tool_result_error,
        final_output_from_steps,
        tool_calls_from_response,
    )
    from agentverify.models import Step

    probe_tool_results_meta = probe_tool_results_meta or {}

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
        extra_meta_flags: list[bool] = []
        if probe_id is not None and probe_id in probe_meta:
            name = probe_meta[probe_id]["name"]
            source = "probe"
            extra_tool_results = probe_tool_results.get(probe_id, [])
            extra_meta_flags = probe_tool_results_meta.get(probe_id, [])
        combined_results = current_bucket["tool_results"] + extra_tool_results
        # The bucket's own (interaction-derived) tool_results are always
        # empty here; probe-attached results carry explicit error flags.
        explicit: list[bool | None] = (
            [None] * len(current_bucket["tool_results"]) + list(extra_meta_flags)
        )
        tool_results_meta = build_tool_results_meta(combined_results, explicit)
        steps.append(
            Step(
                index=index,
                name=name,
                source=source,
                tool_calls=current_bucket["tool_calls"],
                tool_results=combined_results,
                output=current_bucket["output"],
                duration_ms=None,
                token_usage=current_bucket["token_usage"],
                input_context=current_bucket["input_context"],
                tool_results_meta=tool_results_meta,
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
        standalone_results = list(probe_tool_results.get(handle_id, []))
        standalone_flags = list(probe_tool_results_meta.get(handle_id, []))
        steps.append(
            Step(
                index=len(steps),
                name=meta["name"],
                source="probe",
                tool_calls=[],
                tool_results=standalone_results,
                output=meta["output"],
                tool_results_meta=build_tool_results_meta(
                    standalone_results, standalone_flags
                ),
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
        # The request resends the whole conversation, so take only the tool
        # results that answer the latest tool call: the ones after the last
        # model message that made tool calls. Without such a message (a
        # provider shape without one), fall back to all of them.
        start = 0
        for pos, msg in enumerate(messages):
            if _tool_names_in(msg):
                start = pos + 1
        extracted: list[Any] = []
        flags: list[bool | None] = []
        for msg in messages[start:]:
            for content, is_error in _tool_results_in(msg):
                extracted.append(content)
                flags.append(is_error)
        if extracted:
            steps[i] = replace(
                producer, tool_results=extracted, tool_results_meta=build_tool_results_meta(extracted, flags)
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
            tool_results_meta=s.tool_results_meta,
            agent=s.agent,
        )
        for i, s in enumerate(steps)
    ]
    # Cassettes carry no agent names; infer them from handoff tool calls.
    steps = attribute_agents([_agent_from_history(s) for s in steps])

    return ExecutionResult(
        steps=steps,
        token_usage=aggregate_token_usage(steps),
        final_output=final_output_from_steps(steps),
        duration_ms=duration_ms,
    )


def _tool_names_in(message: Any) -> list[str]:
    """Names of the tool calls a model message makes: OpenAI, Anthropic, Bedrock, and Gemini shapes."""
    from agentverify.cassette.sanitize import _as_plain

    # Recording keeps the request's SDK objects; replay loads dicts. Read both the same way.
    message = _as_plain(message)
    if not isinstance(message, dict) or message.get("role") not in ("assistant", "model"):
        return []
    names: list[str] = []
    for tc in message.get("tool_calls") or []:  # OpenAI / LiteLLM
        if isinstance(tc, dict):
            fn = tc.get("function")
            name = fn.get("name") if isinstance(fn, dict) else tc.get("name")
            if isinstance(name, str):
                names.append(name)
    content = message.get("content")
    for block in content if isinstance(content, list) else []:
        block = _as_plain(block)  # SDK content blocks while recording
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and isinstance(block.get("name"), str):  # Anthropic
            names.append(block["name"])
        tool_use = block.get("toolUse")  # Bedrock Converse
        if isinstance(tool_use, dict) and isinstance(tool_use.get("name"), str):
            names.append(tool_use["name"])
    for part in message.get("parts") or []:  # Gemini
        part = _as_plain(part)
        call = (part.get("function_call") or part.get("functionCall")) if isinstance(part, dict) else None
        call = _as_plain(call)
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            names.append(call["name"])
    return names


# (cassette path, test id, field) for each stale cassette found in this run, for the pytest summary.
_stale_cassettes: list[tuple[str, str | None, str]] = []


# How agentverify before 0.7.0 recorded google-genai Content objects: their repr.
_SDK_REPR = re.compile(r"^\s*(parts|role)=")
_EMPTY_SCHEMA = "{}"


def _tool_schemas(request: NormalizedRequest) -> dict[str, str]:
    schemas: dict[str, str] = {}
    for tool in request.tools or []:
        if isinstance(tool, dict) and tool.get("name"):
            # Indented, so a mismatch shows the changed lines of the schema.
            schemas[tool["name"]] = json.dumps(_plain_json(tool.get("parameters")), sort_keys=True, indent=2, default=str)
    return schemas


def _plain_json(value: Any) -> Any:
    from agentverify.cassette.sanitize import _as_plain

    value = _as_plain(value)
    if isinstance(value, dict):
        # Unset fields of SDK schema objects (google-genai Schema) dump as None; a new SDK field must not count as a change.
        return {str(k): _plain_json(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_plain_json(v) for v in value]
    return value


def _texts(content: Any) -> list[str]:
    """The text of a message's content: a string, or the text blocks of a list (no tool results, no images)."""
    from agentverify.cassette.sanitize import _as_plain

    content = _as_plain(content)
    if isinstance(content, str):
        return [content]
    texts: list[str] = []
    for block in content if isinstance(content, list) else []:
        block = _as_plain(block)
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str) and block.get("type") in (None, "text", "input_text"):
            texts.append(block["text"])
    return texts


def _message_texts(request: NormalizedRequest, roles: tuple[str, ...]) -> list[str]:
    from agentverify.cassette.sanitize import _as_plain

    texts: list[str] = []
    for message in request.messages or []:
        message = _as_plain(message)
        if isinstance(message, dict) and message.get("role") in roles:
            texts += _texts(message.get("content")) + _texts(message.get("parts"))
    return texts


def _system_texts(request: NormalizedRequest) -> list[str]:
    """The system prompt: system / developer messages, and the providers' system parameters."""
    from agentverify.cassette.sanitize import _as_plain

    texts = _message_texts(request, ("system", "developer"))
    for key in ("system", "system_instruction", "instructions"):
        value = _as_plain(request.parameters.get(key))
        texts += _texts(value.get("parts")) if isinstance(value, dict) else _texts(value)
    return texts


def _user_texts(request: NormalizedRequest) -> list[str]:
    return _message_texts(request, ("user",))


def _block_text(content: Any) -> Any:
    """The text of a list of text blocks (Anthropic, Bedrock); a lone JSON block's value; anything else as it is."""
    from agentverify.cassette.sanitize import _as_plain

    if not isinstance(content, list):
        return content
    blocks = [_as_plain(b) for b in content]
    if blocks and all(isinstance(b, dict) and isinstance(b.get("text"), str) for b in blocks):
        return "".join(b["text"] for b in blocks)
    if len(blocks) == 1 and isinstance(blocks[0], dict) and "json" in blocks[0]:
        return blocks[0]["json"]
    return blocks


def _tool_results_in(message: Any) -> list[tuple[Any, bool | None]]:
    """The tool results a request message carries, with their error flags: OpenAI, Anthropic, Bedrock, and Gemini shapes."""
    from agentverify.cassette.sanitize import _as_plain

    message = _as_plain(message)
    if not isinstance(message, dict):
        return []
    if message.get("role") == "tool":  # OpenAI / LiteLLM
        return [(message.get("content"), None)]
    results: list[tuple[Any, bool | None]] = []
    content = message.get("content")
    for block in content if isinstance(content, list) else []:
        block = _as_plain(block)
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":  # Anthropic
            is_error = block.get("is_error")
            results.append((_block_text(block.get("content")), is_error if isinstance(is_error, bool) else None))
        tool_result = _as_plain(block.get("toolResult"))  # Bedrock Converse
        if isinstance(tool_result, dict):
            status = tool_result.get("status")
            results.append((_block_text(tool_result.get("content")), None if status is None else status == "error"))
    for part in message.get("parts") or []:  # Gemini
        part = _as_plain(part)
        response = (part.get("function_response") or part.get("functionResponse")) if isinstance(part, dict) else None
        response = _as_plain(response)
        if isinstance(response, dict):
            results.append((response.get("response"), None))
    return results


def _agent_from_history(step: Step) -> Step:
    """Name the step's agent after the last handoff in its input history.

    The request resends the conversation, including handoffs the framework writes without an LLM call (langgraph-supervisor's ``transfer_back_to_supervisor``), so the last handoff there names the agent making this call.
    """
    if step.agent is not None or not step.input_context:
        return step
    messages = step.input_context.get("messages")
    agent = None
    for message in messages if isinstance(messages, list) else []:
        # Only the first handoff in a message takes effect (see first_handoff).
        target = next((t for t in map(handoff_target, _tool_names_in(message)) if t is not None), None)
        if target is not None:
            agent = target
    return step if agent is None else replace(step, agent=agent)
