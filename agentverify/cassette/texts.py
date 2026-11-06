"""Every piece of text a request sends the model, with where it sits: ``iter_request_texts``.

For checks that must see all of it (personal data, secrets, credential URLs): the system prompt, each message, tool call arguments and tool results, the tool definitions (including ``enum`` values), and the provider's other parameters. The providers' message shapes (OpenAI, the Responses API as recorded, Anthropic, Bedrock Converse, Gemini, LiteLLM) are read alike.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable, Iterator, Union

from agentverify.cassette.adapters.base import NormalizedRequest

# Not text the agent sent: provider ids, reasoning signatures, encrypted reasoning, and binary payloads.
_SKIP_KEYS = frozenset({"id", "toolUseId", "tool_use_id", "tool_call_id", "call_id", "signature", "encrypted_content", "redactedContent", "data", "bytes"})
_SYSTEM_KEYS = ("system", "system_instruction", "instructions")

Source = Union[str, Path, NormalizedRequest, Any]
# Drift compares nulls too (a new "description": null is a change); text checks have no use for them.
_NULLS: ContextVar[bool] = ContextVar("agentverify_texts_nulls", default=False)


def _requests(source: Source) -> list[NormalizedRequest]:
    if isinstance(source, NormalizedRequest):
        return [source]
    if isinstance(source, (str, Path)):
        from agentverify.cassette.io import _parse_interactions, load_document

        return [request for request, _ in _parse_interactions(load_document(Path(source)))[1]]
    interactions = getattr(source, "_interactions", None)  # LLMCassetteRecorder / MockLLM
    if isinstance(interactions, list):
        return [request for request, _ in interactions]
    raise TypeError(f"iter_request_texts() takes a cassette path, an LLMCassetteRecorder or MockLLM, or a NormalizedRequest; got {type(source).__name__}")


def _parsed(value: Any) -> Any:
    """A JSON string as the value it holds (tool arguments and results often arrive encoded)."""
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _leaves(value: Any, where: str) -> Iterator[tuple[str, str]]:
    """Each text, number, and boolean inside *value*, with its path."""
    value = _parsed(value)
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in _SKIP_KEYS:
                yield from _leaves(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{where}[{index}]")
    elif isinstance(value, str):
        if value and not value.startswith("data:"):  # an embedded image or file, not text
            yield where, value
    elif isinstance(value, (bool, int, float)):
        yield where, str(value)
    elif value is None and _NULLS.get():
        yield where, "null"


def _tool_names(request: NormalizedRequest) -> dict[str, str]:
    """Tool call id to tool name, from the calls in the request's history."""
    names: dict[str, str] = {}
    for message in request.messages or []:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:  # OpenAI / LiteLLM
            if isinstance(call, dict) and isinstance(call.get("function"), dict):
                names[str(call.get("id"))] = str(call["function"].get("name"))
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":  # Anthropic
                names[str(block.get("id"))] = str(block.get("name"))
            if isinstance(block.get("toolUse"), dict):  # Bedrock
                names[str(block["toolUse"].get("toolUseId"))] = str(block["toolUse"].get("name"))
        if message.get("type") == "function_call":  # Responses API items
            names[str(message.get("call_id"))] = str(message.get("name"))
    return names


def _message(message: Any, where: str, names: dict[str, str]) -> Iterator[tuple[str, str]]:
    if not isinstance(message, dict):
        yield from _leaves(message, where)
        return
    role = message.get("role") or message.get("type") or "message"
    where = f"{where}({role})"
    calls = 0
    results = 0
    for key, value in message.items():
        if key in ("role", "type") or key in _SKIP_KEYS:
            continue
        if key == "tool_calls" and isinstance(value, list):  # OpenAI / LiteLLM
            for call in value:
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict):
                    yield from _leaves(function.get("arguments"), f"{where}.tool_call[{calls}]({function.get('name')}).arguments")
                    calls += 1
        elif key in ("content", "parts") and isinstance(value, list):
            for index, block in enumerate(value):
                if not isinstance(block, dict):
                    yield from _leaves(block, f"{where}.content[{index}]")
                    continue
                if block.get("type") == "tool_use" or isinstance(block.get("toolUse"), dict):
                    use = block.get("toolUse") or block
                    yield from _leaves(use.get("input"), f"{where}.tool_call[{calls}]({use.get('name')}).arguments")
                    calls += 1
                elif isinstance(block.get("function_call") or block.get("functionCall"), dict):  # Gemini
                    call = block.get("function_call") or block.get("functionCall")
                    yield from _leaves(call.get("args"), f"{where}.tool_call[{calls}]({call.get('name')}).arguments")
                    calls += 1
                elif block.get("type") == "tool_result" or isinstance(block.get("toolResult"), dict):
                    result = block.get("toolResult") or block
                    name = names.get(str(result.get("toolUseId") or result.get("tool_use_id")), "?")
                    yield from _leaves(result.get("content"), f"{where}.tool_result[{results}]({name})")
                    results += 1
                elif isinstance(block.get("function_response") or block.get("functionResponse"), dict):  # Gemini
                    response = block.get("function_response") or block.get("functionResponse")
                    yield from _leaves(response.get("response"), f"{where}.tool_result[{results}]({response.get('name')})")
                    results += 1
                else:
                    yield from _leaves({k: v for k, v in block.items() if k != "type"}, f"{where}.content[{index}]")
        elif key == "content" and role == "tool":  # OpenAI tool message
            yield from _leaves(value, f"{where}.tool_result[0]({names.get(str(message.get('tool_call_id')), '?')})")
        elif key == "output" and role == "function_call_output":  # Responses API
            yield from _leaves(value, f"{where}.tool_result[0]({names.get(str(message.get('call_id')), '?')})")
        elif key == "arguments" and role == "function_call":
            yield from _leaves(value, f"{where}.tool_call[0]({message.get('name')}).arguments")
        elif key == "content" and isinstance(value, str):
            yield f"{where}.text", value
        elif not (key == "name" and role == "function_call"):
            yield from _leaves(value, f"{where}.{key}")


def _tool(tool: Any, where: str) -> Iterator[tuple[str, str]]:
    if not isinstance(tool, dict):
        yield from _leaves(tool, where)
        return
    spec = tool.get("function") if isinstance(tool.get("function"), dict) else tool.get("toolSpec") if isinstance(tool.get("toolSpec"), dict) else tool
    name = spec.get("name", "?")
    for key, value in spec.items():
        if key == "name":
            yield f"{where}[{name}].name", str(value)
        elif key in ("parameters", "input_schema", "inputSchema"):
            yield from _leaves(value, f"{where}[{name}].schema")
        else:
            yield from _leaves(value, f"{where}[{name}].{key}")


def iter_request_texts(source: Source) -> Iterable[tuple[str, str]]:
    """Every text a request sends the model, as ``(where, text)`` pairs.

    Args:
        source: A cassette path, an ``LLMCassetteRecorder`` or ``MockLLM`` after the run (all its requests: in replay, the recorded ones), or one ``NormalizedRequest``. An ``ExecutionResult`` is not accepted: it does not keep the tool definitions or the system prompt.

    ``where`` reads like ``request[0].system``, ``request[0].message[2](assistant).tool_call[0](get_weather).arguments.city``, ``request[0].message[3](user).tool_result[0](get_weather).temp``, ``request[0].tool[get_weather].schema.properties.unit.enum[1]``, or ``request[0].parameters.temperature``. Messages carry their role, so ``(assistant)`` picks out the model's own answers; tool results carry the tool's name. JSON strings (tool arguments, tool results) are read as the values they hold. Numbers and booleans come as text. Provider ids (``toolUseId``, ``tool_call_id``), reasoning signatures, and embedded images or files (``data:`` URLs, ``bytes``) are left out. A recorded cassette holds the sanitized request; while recording, the recorder keeps the request as sent.
    """
    for i, request in enumerate(_requests(source)):
        names = _tool_names(request)
        for key, value in (request.parameters or {}).items():
            yield from _leaves(value, f"request[{i}].system" if key in _SYSTEM_KEYS else f"request[{i}].parameters.{key}")
        for j, message in enumerate(request.messages or []):
            yield from _message(message, f"request[{i}].message[{j}]", names)
        for tool in request.tools or []:
            yield from _tool(tool, f"request[{i}].tool")
