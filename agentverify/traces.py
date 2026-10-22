"""Build :class:`ExecutionResult` objects from OpenTelemetry traces of agent runs.

A production or staging trace becomes something the agentverify assertions can check: the tool calls and their arguments, the data passed between steps, tool errors, handoffs, and token usage, without a judge model::

    from agentverify import assert_no_tool_call
    from agentverify.traces import from_otel

    for result in from_otel("traces.jsonl"):
        assert_no_tool_call(result, forbidden_tools=["delete_user"])

Spans following the OpenTelemetry GenAI semantic conventions are read, in the current form (``gen_ai.input.messages`` / ``gen_ai.output.messages``, as span attributes or as attributes of the ``gen_ai.client.inference.operation.details`` event) and the earlier one (``gen_ai.choice`` and per-message events, as Strands Agents emits by default), as well as OpenInference spans (``llm.*`` / ``tool.*`` attributes). Message content is only in a trace when the instrumentation was told to capture it; without it, the steps still carry tool names, arguments from tool spans, errors, agents, and token usage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from agentverify._step_builder import classify_tool_result_error
from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall, handoffs_from_agents

_INFERENCE_OPERATIONS = {"chat", "generate_content", "text_completion"}


@dataclass
class _Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start: float
    end: float
    error: bool
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def operation(self) -> str | None:
        return self.attributes.get("gen_ai.operation.name")

    @property
    def is_inference(self) -> bool:
        return self.operation in _INFERENCE_OPERATIONS or self.attributes.get("openinference.span.kind") == "LLM"

    @property
    def is_tool(self) -> bool:
        if "(merged)" in self.name or self.name.startswith("handoff to "):
            return False  # Google ADK's summary of parallel calls; OpenInference's handoff span
        return self.operation == "execute_tool" or self.attributes.get("openinference.span.kind") == "TOOL"

    @property
    def is_agent(self) -> bool:
        return self.operation == "invoke_agent" or self.attributes.get("openinference.span.kind") == "AGENT"


def from_otel(source: Any) -> list[ExecutionResult]:
    """One :class:`ExecutionResult` per trace in *source*, in the order the traces started.

    Args:
        source: A path to an OTLP JSON file (one document, or one per line as the collector's file exporter writes), an OTLP JSON document (``{"resourceSpans": [...]}``), a list of OpenTelemetry SDK spans (``ReadableSpan``, e.g. from ``InMemorySpanExporter.get_finished_spans()``), or a list of span dicts as ``ReadableSpan.to_json()`` writes them.
    """
    spans = list(_load(source))
    traces: dict[str, list[_Span]] = {}
    for span in spans:
        traces.setdefault(span.trace_id, []).append(span)
    ordered = sorted(traces.values(), key=lambda group: min(s.start for s in group))
    return [_trace_to_result(sorted(group, key=lambda s: s.start)) for group in ordered]


# ---------------------------------------------------------------------------
# Loading the supported span formats into _Span
# ---------------------------------------------------------------------------


def _load(source: Any) -> Iterable[_Span]:
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8")
        try:
            documents = [json.loads(text)]
        except ValueError:
            documents = [json.loads(line) for line in text.splitlines() if line.strip()]
        for document in documents:
            yield from _load(document)
        return
    if isinstance(source, dict):
        if "resourceSpans" in source:
            for resource in source["resourceSpans"]:
                for scope in resource.get("scopeSpans") or resource.get("instrumentationLibrarySpans") or []:
                    for span in scope.get("spans", []):
                        yield _from_otlp(span)
        elif "context" in source:
            yield _from_dict(source)
        # Other documents in the same file (resourceLogs, resourceMetrics) carry no spans.
        return
    for item in source:
        if isinstance(item, dict):
            yield from _load(item)
        else:
            yield _from_sdk(item)


def _otlp_value(value: dict[str, Any]) -> Any:
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    if "intValue" in value:
        return int(value["intValue"])
    if "arrayValue" in value:
        return [_otlp_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _otlp_attributes(value["kvlistValue"].get("values", []))
    return None


def _otlp_attributes(attributes: list[dict[str, Any]]) -> dict[str, Any]:
    return {a["key"]: _otlp_value(a.get("value", {})) for a in attributes or []}


def _from_otlp(span: dict[str, Any]) -> _Span:
    status = span.get("status") or {}
    return _Span(
        trace_id=span.get("traceId", ""),
        span_id=span.get("spanId", ""),
        parent_id=span.get("parentSpanId") or None,
        name=span.get("name", ""),
        start=int(span.get("startTimeUnixNano", 0)) / 1e6,
        end=int(span.get("endTimeUnixNano") or span.get("startTimeUnixNano", 0)) / 1e6,
        error=status.get("code") in (2, "STATUS_CODE_ERROR"),
        attributes=_otlp_attributes(span.get("attributes", [])),
        events=[(e.get("name", ""), _otlp_attributes(e.get("attributes", []))) for e in span.get("events", [])],
    )


def _iso_ms(value: Any) -> float:
    if isinstance(value, str):
        # Python 3.10 reads only 3 or 6 fractional digits; exporters write up to 9.
        value = re.sub(r"\.(\d+)", lambda m: "." + m.group(1)[:6].ljust(6, "0"), value.replace("Z", "+00:00"), count=1)
        return datetime.fromisoformat(value).timestamp() * 1000
    return 0.0


def _from_dict(span: dict[str, Any]) -> _Span:
    """A span as ``ReadableSpan.to_json()`` writes it."""
    context = span.get("context") or {}
    status = span.get("status") or {}
    return _Span(
        trace_id=str(context.get("trace_id", "")),
        span_id=str(context.get("span_id", "")),
        parent_id=span.get("parent_id") or None,
        name=span.get("name", ""),
        start=_iso_ms(span.get("start_time")),
        end=_iso_ms(span.get("end_time") or span.get("start_time")),
        error=status.get("status_code") == "ERROR",
        attributes=dict(span.get("attributes") or {}),
        events=[(e.get("name", ""), dict(e.get("attributes") or {})) for e in span.get("events") or []],
    )


def _from_sdk(span: Any) -> _Span:
    parent = getattr(span, "parent", None)
    status = getattr(span, "status", None)
    code = getattr(getattr(status, "status_code", None), "name", None)
    return _Span(
        trace_id=format(span.context.trace_id, "032x"),
        span_id=format(span.context.span_id, "016x"),
        parent_id=format(parent.span_id, "016x") if parent is not None else None,
        name=span.name,
        start=(span.start_time or 0) / 1e6,
        end=(span.end_time or span.start_time or 0) / 1e6,
        error=code == "ERROR",
        attributes=dict(span.attributes or {}),
        events=[(e.name, dict(e.attributes or {})) for e in span.events or []],
    )


# ---------------------------------------------------------------------------
# One trace into an ExecutionResult
# ---------------------------------------------------------------------------


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _attribute(span: _Span, key: str) -> Any:
    """An attribute of the span, or of one of its events (the current GenAI conventions allow both)."""
    if key in span.attributes:
        return span.attributes[key]
    for _, attributes in span.events:
        if key in attributes:
            return attributes[key]
    return None


def _ancestors(span: _Span, by_id: dict[str, _Span]) -> Iterable[_Span]:
    seen: set[str] = {span.span_id}
    current = by_id.get(span.parent_id) if span.parent_id else None
    while current is not None and current.span_id not in seen:
        seen.add(current.span_id)
        yield current
        current = by_id.get(current.parent_id) if current.parent_id else None


def _usage(span: _Span) -> TokenUsage | None:
    def first(*keys: str) -> Any:
        return next((span.attributes[k] for k in keys if span.attributes.get(k) is not None), None)

    inputs = first("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens", "llm.token_count.prompt")
    outputs = first("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens", "llm.token_count.completion")
    if inputs is None and outputs is None:
        return None
    return TokenUsage(input_tokens=int(inputs or 0), output_tokens=int(outputs or 0))


def _arguments(value: Any) -> dict[str, Any]:
    value = _json(value)
    return value if isinstance(value, dict) else ({} if value in (None, "") else {"input": value})


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _output(span: _Span, parent: _Span | None = None) -> tuple[str | None, list[tuple[ToolCall, Any]]]:
    """The text and tool calls (with their ids) an inference span produced."""
    texts: list[str] = []
    calls: list[tuple[ToolCall, Any]] = []
    for message in _dicts(_json(_attribute(span, "gen_ai.output.messages"))):  # current GenAI conventions
        for part in _dicts(message.get("parts")):
            if part.get("type") == "text" and isinstance(part.get("content"), str):
                texts.append(part["content"])
            elif part.get("type") == "tool_call":
                calls.append((ToolCall(str(part.get("name", "")), _arguments(part.get("arguments"))), part.get("id")))
    for name, attributes in span.events:  # earlier GenAI conventions (Strands' default)
        if name == "gen_ai.choice":
            message = _json(attributes.get("message"))
            for block in _dicts(message if isinstance(message, list) else [message]):
                if isinstance(block.get("text"), str):
                    texts.append(block["text"])
                tool_use = block.get("toolUse")
                if isinstance(tool_use, dict):
                    calls.append((ToolCall(str(tool_use.get("name", "")), _arguments(tool_use.get("input"))), tool_use.get("toolUseId")))
    # Google ADK's default attributes, on its call_llm span around the model call.
    response = _json(span.attributes.get("gcp.vertex.agent.llm_response") or (parent.attributes.get("gcp.vertex.agent.llm_response") if parent else None))
    if not texts and not calls and isinstance(response, dict):
        for part in _dicts((response.get("content") or {}).get("parts") if isinstance(response.get("content"), dict) else None):
            if isinstance(part.get("text"), str) and not part.get("thought"):
                texts.append(part["text"])
            call = part.get("function_call")
            if isinstance(call, dict):
                calls.append((ToolCall(str(call.get("name", "")), _arguments(call.get("args"))), call.get("id")))
    index = 0
    while f"llm.output_messages.{index}.message.role" in span.attributes:  # OpenInference
        prefix = f"llm.output_messages.{index}.message"
        content = span.attributes.get(f"{prefix}.content")
        if isinstance(content, str) and content:
            texts.append(content)
        call_index = 0
        while f"{prefix}.tool_calls.{call_index}.tool_call.function.name" in span.attributes:
            base = f"{prefix}.tool_calls.{call_index}.tool_call"
            tool_call = ToolCall(span.attributes[f"{base}.function.name"], _arguments(span.attributes.get(f"{base}.function.arguments")))
            calls.append((tool_call, span.attributes.get(f"{base}.id")))
            call_index += 1
        index += 1
    return ("".join(texts) if texts else None), calls


def _input_context(span: _Span) -> dict[str, Any] | None:
    messages = _json(_attribute(span, "gen_ai.input.messages"))
    if isinstance(messages, list):
        return {"messages": messages}
    index, collected = 0, []
    while f"llm.input_messages.{index}.message.role" in span.attributes:
        collected.append(
            {
                "role": span.attributes[f"llm.input_messages.{index}.message.role"],
                "content": span.attributes.get(f"llm.input_messages.{index}.message.content"),
            }
        )
        index += 1
    legacy = [_json(a.get("content")) for n, a in span.events if n in ("gen_ai.user.message", "gen_ai.tool.message")]
    collected += [{"content": c} for c in legacy if c is not None]
    return {"messages": collected} if collected else None


def _tool_result(span: _Span) -> Any:
    for key in ("gen_ai.tool.call.result", "output.value", "gcp.vertex.agent.tool_response"):
        value = _attribute(span, key)
        if value is not None:
            return _json(value)
    for message in _dicts(_json(_attribute(span, "gen_ai.output.messages"))):  # a failed call's response (Strands)
        for part in _dicts(message.get("parts")):
            if part.get("type") == "tool_call_response":
                return _json(part.get("response", part.get("result")))
    for name, attributes in span.events:  # earlier conventions: the result is the tool span's choice
        if name == "gen_ai.choice" and "message" in attributes:
            return _json(attributes["message"])
    return None


def _tool_meta(span: _Span) -> dict[str, Any]:
    if span.error or span.attributes.get("gen_ai.tool.status") == "error" or "error.type" in span.attributes:
        return {"is_error": True}
    # An error the tool returned rather than raised ({"status": "error", ...}), as the adapters read it.
    error = classify_tool_result_error(_tool_result(span))
    return {} if error is None else {"is_error": error}


def _agent(span: _Span, by_id: dict[str, _Span]) -> str | None:
    for candidate in (span, *_ancestors(span, by_id)):
        if candidate is span or candidate.is_agent:
            name = candidate.attributes.get("gen_ai.agent.name") or candidate.attributes.get("agent.name")
            if name:
                return str(name)
    return None


def _agent_span(span: _Span, by_id: dict[str, _Span]) -> _Span | None:
    return next((a for a in _ancestors(span, by_id) if a.is_agent), None)


def _trace_to_result(spans: list[_Span]) -> ExecutionResult:
    by_id = {s.span_id: s for s in spans}
    # A model call wrapped in another inference span (Google ADK's call_llm around generate_content) is one call.
    inference = [s for s in spans if s.is_inference and not any(a.is_inference for a in _ancestors(s, by_id))]
    tools = [s for s in spans if s.is_tool]
    agent_spans = [_agent_span(s, by_id) for s in inference]
    outputs = [_output(s, by_id.get(s.parent_id or "")) for s in inference]

    # Each tool span belongs to one model call: the one whose tool call has its id, or else the
    # latest call before it in the same agent (parallel agents interleave in time).
    owner_by_id = {call_id: index for index, (_, calls) in enumerate(outputs) for _, call_id in calls if call_id}
    assigned: dict[int, list[_Span]] = {i: [] for i in range(len(inference))}
    for tool_span in tools:
        index = owner_by_id.get(tool_span.attributes.get("gen_ai.tool.call.id"))
        if index is None and inference:
            agent = _agent_span(tool_span, by_id)
            earlier = [i for i, s in enumerate(inference) if s.start <= tool_span.start]
            same_agent = [i for i in earlier if agent_spans[i] is agent]
            index = (same_agent or earlier or [0])[-1]
        if index is not None:
            assigned[index].append(tool_span)

    steps: list[Step] = []
    for index, span in enumerate(inference):
        output, calls = outputs[index]
        tool_calls = [call for call, _ in calls]
        results: list[Any] = []
        meta: list[dict[str, Any]] = []
        unmatched = list(assigned[index])
        for (call, call_id) in calls:
            tool_span = next((t for t in unmatched if call_id and t.attributes.get("gen_ai.tool.call.id") == call_id), None)
            tool_span = tool_span or next((t for t in unmatched if _tool_name(t) == call.name), None)
            if tool_span is None:
                # A call without a tool span (a handoff, a structured-output call) keeps its place.
                results.append(None)
                meta.append({})
                continue
            unmatched.remove(tool_span)
            results.append(_tool_result(tool_span))
            meta.append(_tool_meta(tool_span))
        # Without message content, tool spans are all there is to know about the calls.
        for tool_span in unmatched:
            arguments = _attribute(tool_span, "gen_ai.tool.call.arguments") or _attribute(tool_span, "gcp.vertex.agent.tool_call_args") or _attribute(tool_span, "input.value")
            tool_calls.append(ToolCall(_tool_name(tool_span), _arguments(arguments)))
            results.append(_tool_result(tool_span))
            meta.append(_tool_meta(tool_span))
        steps.append(
            Step(
                index=index,
                source="llm",
                tool_calls=tool_calls,
                tool_results=results,
                tool_results_meta=meta or None,
                output=output,
                duration_ms=span.end - span.start,
                token_usage=_usage(span),
                input_context=_input_context(span),
                agent=_agent(span, by_id),
            )
        )

    nested = [agent is not None and any(a.is_tool for a in _ancestors(agent, by_id)) for agent in agent_spans]
    usages = [s.token_usage for s in steps if s.token_usage is not None]
    token_usage = (
        TokenUsage(input_tokens=sum(u.input_tokens for u in usages), output_tokens=sum(u.output_tokens for u in usages)) if usages else None
    )
    roots = [s for s in spans if s.parent_id is None or s.parent_id not in by_id]
    duration = max(r.end for r in roots) - min(r.start for r in roots) if roots else None
    # The answer is the outer agent's: an agent called as a tool answers its caller, not the user.
    final_output = next((s.output for s, inner in zip(reversed(steps), reversed(nested)) if s.output and not inner), None)
    return ExecutionResult(
        steps=steps,
        token_usage=token_usage,
        duration_ms=duration,
        final_output=final_output,
        handoffs=_handoffs(steps, agent_spans, nested, by_id),
    )


def _handoffs(steps: list[Step], agent_spans: list[_Span | None], nested: list[bool], by_id: dict[str, _Span]) -> list[Any]:
    """Agent changes between consecutive steps, except an agent called as a tool, and agents running in parallel."""
    handoffs = []
    for i, change in ((h.step, h) for h in handoffs_from_agents(steps)):
        j = next(k for k in range(i + 1, len(steps)) if steps[k].agent == change.to_agent)
        a, b = agent_spans[i], agent_spans[j]
        if nested[i] or nested[j]:
            continue
        if a is not None and b is not None and a is not b:
            related = a in _ancestors(b, by_id) or b in _ancestors(a, by_id)
            if not related and a.start < b.end and b.start < a.end:
                continue  # overlapping sibling agents: a ParallelAgent, not a handoff
        handoffs.append(change)
    return handoffs


def _tool_name(span: _Span) -> str:
    name = span.attributes.get("gen_ai.tool.name") or span.attributes.get("tool.name")
    if name:
        return str(name)
    return span.name.removeprefix("execute_tool ").strip()
