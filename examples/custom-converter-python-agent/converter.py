"""Map an :class:`AgentRun` into an agentverify :class:`ExecutionResult`.

This file is the reference for what a custom converter looks like when
no built-in framework adapter fits. The shape is small — about 60 lines
of real logic — and the mapping to agentverify's data model is
mechanical once you know what shape your agent emits.

Anthropic's Messages API records the conversation as a flat list of
messages alternating between ``assistant`` (with ``text`` and/or
``tool_use`` content blocks) and ``user`` (with ``tool_result`` content
blocks feeding the previous turn's tool outputs back in). One step in
agentverify terms corresponds to one ``assistant`` turn, with its tool
calls, the tool results produced for it by the next ``user`` turn, and
the text output of the assistant turn itself.

Conversion mapping:

    assistant tool_use blocks[i]                      -> tool_calls[i]
    next user's tool_result blocks[i].content          -> tool_results[i]
    assistant text blocks (concatenated)               -> step.output
    full conversation up to this point                 -> step.input_context
    sum of response.usage.{input,output}_tokens        -> token_usage

The converter returns a fully populated :class:`ExecutionResult` that
supports every agentverify assertion, including
:func:`assert_step_uses_result_from` for step-to-step data flow.
"""

from __future__ import annotations

import json
from typing import Any

from agentverify import ExecutionResult, Step, TokenUsage, ToolCall

from agent import AgentRun


def _coerce_block(block: Any) -> dict[str, Any] | None:
    """Normalise Anthropic content blocks (SDK objects or dicts) into
    plain dicts so the converter can treat both uniformly."""
    if isinstance(block, dict):
        return block
    block_type = getattr(block, "type", None)
    if block_type is None:
        return None
    normalised: dict[str, Any] = {"type": block_type}
    if block_type == "text":
        normalised["text"] = getattr(block, "text", "")
    elif block_type == "tool_use":
        normalised["id"] = getattr(block, "id", "")
        normalised["name"] = getattr(block, "name", "")
        normalised["input"] = getattr(block, "input", {}) or {}
    elif block_type == "tool_result":
        normalised["tool_use_id"] = getattr(block, "tool_use_id", "")
        normalised["content"] = getattr(block, "content", "")
    return normalised


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    """Coerce an assistant or user message's ``content`` field into a
    list of plain-dict blocks. Handles both the raw-string form (user
    messages) and the mixed-object form (assistant messages)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    out: list[dict[str, Any]] = []
    for block in content:
        coerced = _coerce_block(block)
        if coerced is not None:
            out.append(coerced)
    return out


def _parse_tool_result(raw: Any) -> Any:
    """Decode a tool_result ``content`` value.

    The agent emits JSON-encoded primitives (``"300"`` / ``"330.0"``);
    falling back to the raw string is fine for non-JSON tool results.
    """
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def run_to_execution_result(run: AgentRun) -> ExecutionResult:
    """Convert an :class:`AgentRun` into an :class:`ExecutionResult`.

    One step per assistant turn. Tool results from the next user turn
    are attached to the step that produced the corresponding tool calls,
    which lets :func:`assert_step_uses_result_from` verify data flow
    across the chain.
    """
    messages = run.messages
    steps: list[Step] = []

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        blocks = _as_blocks(msg.get("content"))
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        for block in blocks:
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(name=block["name"], arguments=dict(block["input"]))
                )

        # Look one message ahead; if it's a user message with
        # tool_result blocks, lift them onto this step's tool_results.
        tool_results: list[Any] = []
        if i + 1 < len(messages):
            next_msg = messages[i + 1]
            if next_msg.get("role") == "user":
                for block in _as_blocks(next_msg.get("content")):
                    if block["type"] == "tool_result":
                        tool_results.append(_parse_tool_result(block.get("content")))

        # Snapshot the conversation up to (but not including) this
        # assistant turn as the step's input_context, so
        # ``assert_step_uses_result_from`` can see what was in play when
        # the assistant made its decisions.
        input_context = {"messages": _normalise_history(messages[:i])}

        steps.append(
            Step(
                index=len(steps),
                source="llm",
                tool_calls=tool_calls,
                tool_results=tool_results,
                output="\n".join(text_parts) or None,
                input_context=input_context,
            )
        )

    token_usage = None
    if run.input_tokens or run.output_tokens:
        token_usage = TokenUsage(
            input_tokens=run.input_tokens, output_tokens=run.output_tokens
        )

    return ExecutionResult(
        steps=steps,
        token_usage=token_usage,
        final_output=run.final_output,
    )


def _normalise_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialise a message history into plain dicts for
    :attr:`Step.input_context`. Replacing SDK content objects with dicts
    keeps the snapshot JSON-friendly and makes leaf walking inside
    :func:`assert_step_uses_result_from` deterministic."""
    out: list[dict[str, Any]] = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        else:
            out.append({"role": role, "content": _as_blocks(content)})
    return out
