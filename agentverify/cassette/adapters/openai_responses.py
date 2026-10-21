"""OpenAI Responses API wire format for the OpenAI cassette adapter.

The Responses API (``client.responses.create``) carries the same conversation as Chat Completions in a different shape: ``instructions`` plus ``input`` items instead of ``messages``, flat tool definitions, and typed ``output`` items instead of ``choices``. This module maps it onto the provider-agnostic :class:`NormalizedRequest` / :class:`NormalizedResponse`, using Chat Completions conventions where the two overlap (``role: "tool"`` messages for tool results, ``tool_calls`` on assistant messages), so step building and data-flow assertions work unchanged.

It is used by :class:`~agentverify.cassette.adapters.openai.OpenAIAdapter`, not registered as a provider of its own, so one ``provider="openai"`` cassette can hold both wire formats. Which format a replayed response is rebuilt in follows from the method the agent called.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

from types import SimpleNamespace

from agentverify.cassette.adapters._openai_raw import tap_raw_or_stream, wrap_raw
from agentverify.cassette.adapters.base import (
    NormalizedRequest,
    NormalizedResponse,
    StreamAccumulator,
    StreamingWire,
    fallback_call_id,
    replayed_sdk_stream,
)
from agentverify.models import TokenUsage

def _plain(value: Any) -> Any:
    """Return a dict for SDK objects (input items can be pydantic models)."""
    dump = getattr(value, "model_dump", None)
    if callable(dump) and not isinstance(value, type):
        return dump(exclude_none=True)
    return value


def _is_sentinel(value: Any) -> bool:
    from agentverify.cassette.adapters.openai import _is_openai_sentinel

    return _is_openai_sentinel(value)


# Tool calls the client executes, other than function calls. Replay cannot rebuild them yet.
_UNSUPPORTED_CALL_ITEMS = frozenset(
    {
        "custom_tool_call",
        "computer_call",
        "local_shell_call",
        "shell_call",
        "apply_patch_call",
    }
)

# Request parameters that change on every run without changing behavior (the Agents SDK
# sets a per-run prompt_cache_key). Dropping them keeps re-recorded cassettes diffable.
_VOLATILE_PARAMETERS = frozenset({"prompt_cache_key"})



def _text_of(content: Any) -> Any:
    """Collapse a list of text parts into one string; leave anything else as it is."""
    if isinstance(content, list):
        parts = [_plain(p) for p in content]
        if parts and all(isinstance(p, dict) and isinstance(p.get("text"), str) for p in parts):
            return "".join(p["text"] for p in parts)
        return parts
    return content


def _input_to_messages(items: Any) -> list[dict[str, Any]]:
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    messages: list[dict[str, Any]] = []
    for raw in items or []:
        item = _plain(raw)
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": item})
            continue
        kind = item.get("type")
        if kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id", ""),
                            "type": "function",
                            "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "")},
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            messages.append(
                {"role": "tool", "tool_call_id": item.get("call_id", ""), "content": _text_of(item.get("output"))}
            )
        elif "role" in item:
            messages.append({"role": item["role"], "content": _text_of(item.get("content"))})
        elif kind == "reasoning":
            # Encrypted reasoning is resent with every later request and never replayed;
            # keeping it would grow the cassette quadratically.
            messages.append({k: v for k, v in item.items() if k != "encrypted_content"})
        else:
            # Hosted-tool calls and other typed items are kept as they are.
            messages.append(item)
    return messages


def _tools(raw_tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_tools, list):
        return None
    tools = []
    for raw in raw_tools:
        tool = _plain(raw)
        if not isinstance(tool, dict):
            continue
        kind = tool.get("type", "function")
        if kind == "function":
            tools.append({"name": tool.get("name", ""), "parameters": tool.get("parameters", {}) or {}})
        elif kind == "namespace":
            # The Agents SDK groups function tools under a namespace; match on the tools inside.
            prefix = tool.get("name", "")
            for inner in _tools(tool.get("tools")) or []:
                tools.append({**inner, "name": f"{prefix}.{inner['name']}"})
        else:
            # Hosted and typed tools: use their own name or server label when they have one,
            # so two MCP servers or a renamed custom tool still change the request.
            name = tool.get("name") or tool.get("server_label") or kind
            tools.append({"name": name if name == kind else f"{kind}:{name}", "parameters": {}})
    return tools


def normalize_request(raw_request: dict[str, Any]) -> NormalizedRequest:
    messages: list[dict[str, Any]] = []
    instructions = raw_request.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    raw_input = raw_request.get("input")
    if raw_input is not None and not _is_sentinel(raw_input):
        messages.extend(_input_to_messages(raw_input))

    reserved = {"model", "input", "instructions", "tools"}
    parameters = {
        k: v
        for k, v in raw_request.items()
        if k not in reserved and k not in _VOLATILE_PARAMETERS and not _is_sentinel(v) and not (k.startswith("extra_") and v is None)
    }
    headers = parameters.get("extra_headers")
    if isinstance(headers, dict):
        # The SDK's own markers (helper method, raw response) say nothing about the request.
        headers = {h: v for h, v in headers.items() if not h.startswith("X-Stainless-")}
        if headers:
            parameters["extra_headers"] = headers
        else:
            del parameters["extra_headers"]
    if parameters.get("previous_response_id") or parameters.get("conversation"):
        warnings.warn(
            "openai: responses.create continues a stored conversation (previous_response_id / conversation), so the "
            "request carries only the new input. Earlier turns are missing from the cassette, and data-flow "
            "assertions that look back past this request cannot see them. Pass store=False and the full input instead.",
            UserWarning,
            stacklevel=4,  # normalize_request, call_sync / call_async, the adapter wrapper, the caller
        )

    raw_tools = raw_request.get("tools")
    return NormalizedRequest(
        messages=messages,
        model=raw_request.get("model", "") or "",
        tools=None if _is_sentinel(raw_tools) else _tools(raw_tools),
        parameters=parameters,
    )


def normalize_response(raw_response: Any) -> NormalizedResponse:
    from agentverify.cassette.adapters.base import unsupported_call_error

    # with_raw_response returns a LegacyAPIResponse; record the parsed Response.
    if not hasattr(raw_response, "output") and hasattr(raw_response, "parse"):
        raw_response = raw_response.parse()
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in raw_response.output or []:
        kind = getattr(item, "type", None)
        if kind == "message":
            for part in getattr(item, "content", None) or []:
                part_type = getattr(part, "type", None)
                if part_type == "output_text":
                    texts.append(part.text)
                elif part_type == "refusal":
                    texts.append(part.refusal)  # replayed as text
        elif kind == "function_call":
            call = {"name": item.name, "arguments": item.arguments}
            if isinstance(getattr(item, "call_id", None), str):
                call["id"] = item.call_id
            tool_calls.append(call)
        elif kind in _UNSUPPORTED_CALL_ITEMS:
            raise unsupported_call_error(
                "openai", f"a responses.create output with {kind}", "only function tool calls are recorded"
            )

    token_usage = None
    usage = getattr(raw_response, "usage", None)
    if usage is not None:
        token_usage = TokenUsage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)

    return NormalizedResponse(
        content="".join(texts) if texts else None,
        tool_calls=tool_calls or None,
        token_usage=token_usage,
        raw_metadata={"model": getattr(raw_response, "model", "") or ""},
    )


def _zeroed(model_cls: Any, **values: Any) -> Any:
    """Build a usage model with every int field present.

    The usage detail models gain required fields between SDK versions (``cache_write_tokens``, ...), so the fields the cassette does not know are filled with 0.
    """
    for name, field in model_cls.model_fields.items():
        if name not in values and field.annotation in (int, "int"):
            values[name] = 0
    return model_cls.model_construct(**values)


def _output_items(normalized: NormalizedResponse) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if normalized.content is not None:
        output.append(
            {
                "type": "message",
                "id": "msg_0",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": normalized.content, "annotations": []}],
            }
        )
    for i, tc in enumerate(normalized.tool_calls or []):
        args = tc["arguments"]
        if isinstance(args, (dict, list)):
            args = json.dumps(args, ensure_ascii=False)
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{i}",
                "call_id": tc.get("id") or fallback_call_id(),
                "name": tc["name"],
                "arguments": args,
                "status": "completed",
            }
        )
    return output


def _response_dict(normalized: NormalizedResponse, output: list[dict[str, Any]], status: str = "completed") -> dict[str, Any]:
    return {
        "id": "resp_replayed",
        "object": "response",
        "created_at": 0,
        "model": normalized.raw_metadata.get("model", "") or "gpt-4o-mini",
        "status": status,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def denormalize_response(normalized: NormalizedResponse) -> Any:
    """Build an ``openai.types.responses.Response`` from a normalized response."""
    return _build_response(normalized, _output_items(normalized))


def _build_response(normalized: NormalizedResponse, output: list[dict[str, Any]]) -> Any:
    from openai.types.responses import Response, ResponseUsage

    response = Response.model_validate(_response_dict(normalized, output))
    if normalized.token_usage is not None:
        details = ResponseUsage.model_fields
        input_details_cls = details["input_tokens_details"].annotation
        output_details_cls = details["output_tokens_details"].annotation
        tokens = normalized.token_usage
        response.usage = _zeroed(
            ResponseUsage,
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            total_tokens=tokens.input_tokens + tokens.output_tokens,
            input_tokens_details=_zeroed(input_details_cls),
            output_tokens_details=_zeroed(output_details_cls),
        )
    return response


class ResponseEventAccumulator(StreamAccumulator):
    """Collects Responses API stream events into one response.

    The final ``response.completed`` (or ``response.incomplete`` / ``response.failed``) event carries the whole response. A stream read only part of the way is rebuilt from its output items: the finished ones, and the text and arguments of the ones still arriving.
    """

    def __init__(self) -> None:
        self._final: Any = None
        self._items: dict[int, Any] = {}

    def add(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        index = getattr(event, "output_index", None)
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            self._final = event.response
        elif kind == "response.output_item.done":
            self._items[index] = event.item
        elif kind == "response.output_item.added":
            item = event.item
            if item.type == "message":
                self._items[index] = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="")])
            elif item.type == "function_call":
                self._items[index] = SimpleNamespace(
                    type="function_call", name=item.name, call_id=getattr(item, "call_id", None), arguments=""
                )
        elif kind == "response.output_text.delta" and isinstance(self._items.get(index), SimpleNamespace):
            self._items[index].content[0].text += event.delta
        elif kind == "response.function_call_arguments.delta" and isinstance(self._items.get(index), SimpleNamespace):
            self._items[index].arguments += event.delta

    def result(self) -> NormalizedResponse:
        if self._final is not None:
            return normalize_response(self._final)
        output = [self._items[i] for i in sorted(self._items)]
        return normalize_response(SimpleNamespace(output=output, usage=None, model=""))


def response_events(normalized: NormalizedResponse) -> list[dict[str, Any]]:
    """The events of a replayed Responses API stream, as JSON-shaped dicts.

    Each output item is added, filled in one delta, and finished, between ``response.created`` and ``response.completed``, which carries the full response.
    """
    output = _output_items(normalized)
    events: list[dict[str, Any]] = []

    def emit(kind: str, **fields: Any) -> None:
        events.append({"type": kind, "sequence_number": len(events), **fields})

    emit("response.created", response=_response_dict(normalized, [], status="in_progress"))
    for index, item in enumerate(output):
        if item["type"] == "message":
            part = item["content"][0]
            emit("response.output_item.added", output_index=index, item={**item, "status": "in_progress", "content": []})
            emit("response.content_part.added", item_id=item["id"], output_index=index, content_index=0, part={**part, "text": ""})
            emit("response.output_text.delta", item_id=item["id"], output_index=index, content_index=0, delta=part["text"], logprobs=[])
            emit("response.output_text.done", item_id=item["id"], output_index=index, content_index=0, text=part["text"], logprobs=[])
            emit("response.content_part.done", item_id=item["id"], output_index=index, content_index=0, part=part)
        else:
            emit("response.output_item.added", output_index=index, item={**item, "status": "in_progress", "arguments": ""})
            emit("response.function_call_arguments.delta", item_id=item["id"], output_index=index, delta=item["arguments"])
            emit("response.function_call_arguments.done", item_id=item["id"], output_index=index, name=item["name"], arguments=item["arguments"])
        emit("response.output_item.done", output_index=index, item=item)
    completed = _build_response(normalized, output)
    emit("response.completed", response=completed.model_dump(mode="json", exclude_none=True))
    return events


class ResponsesWire(StreamingWire):
    """The conversions :func:`~agentverify.cassette.adapters.base.call_sync` needs, for the Responses API."""

    name = "openai"
    normalize_request = staticmethod(normalize_request)
    normalize_response = staticmethod(normalize_response)
    denormalize_response = staticmethod(denormalize_response)

    def stream_accumulator(self) -> StreamAccumulator:
        return ResponseEventAccumulator()

    def replay_stream(self, normalized: NormalizedResponse, kwargs: dict[str, Any], is_async: bool) -> Any:
        """Build an ``openai.Stream[ResponseStreamEvent]`` (or ``AsyncStream``) for a recorded response."""
        from openai import AsyncStream, Stream
        from openai._models import construct_type
        from openai.types.responses import ResponseStreamEvent

        items = [construct_type(type_=ResponseStreamEvent, value=e) for e in response_events(normalized)]
        return wrap_raw(replayed_sdk_stream(Stream, AsyncStream, items, is_async), kwargs, is_async)

    def tap_response(self, response: Any, tap: Any) -> Any:
        """``with_raw_response`` / ``with_streaming_response`` return a raw response; the stream is what it parses to."""
        return tap_raw_or_stream(response, tap)
