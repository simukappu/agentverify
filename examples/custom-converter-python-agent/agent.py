"""Pure-Python ReAct agent using the Anthropic SDK directly.

No framework — just the Anthropic Messages API driven in a hand-rolled
tool-call loop. The point of this example is to show that agentverify
can test any agent architecture via a small ``converter.py`` when no
built-in framework adapter fits.

The agent's task: given a pre-tax total and a tax rate embedded in the
user query, chain two tool calls (``add`` to compute the pre-tax total,
``apply_tax`` to gross it up) and return the post-tax figure.

The converter that maps this agent's conversation history into an
agentverify :class:`ExecutionResult` lives next to this file in
``converter.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic


DEFAULT_MODEL = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Tool implementations — plain Python functions, no framework required
# ---------------------------------------------------------------------------


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def apply_tax(amount: float, rate: float) -> float:
    """Apply a tax rate (e.g. ``0.1`` for 10%) to ``amount`` and return the
    grossed-up total."""
    return round(amount * (1 + rate), 2)


_TOOL_IMPLEMENTATIONS = {
    "add": add,
    "apply_tax": apply_tax,
}


_TOOL_SCHEMAS = [
    {
        "name": "add",
        "description": "Add two numbers and return their sum.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "apply_tax",
        "description": (
            "Apply a tax rate to an amount and return the grossed-up total. "
            "The rate is expressed as a decimal (0.1 means 10%)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "rate": {"type": "number"},
            },
            "required": ["amount", "rate"],
        },
    },
]


SYSTEM_PROMPT = (
    "You are a precise arithmetic assistant. When the user asks a question "
    "that requires a calculation, use the provided tools step by step rather "
    "than computing the answer yourself. Always call ``add`` first to compute "
    "the pre-tax total, then call ``apply_tax`` with that total and the tax "
    "rate from the query. Finish with a one-sentence summary that includes "
    "the final post-tax amount."
)


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------


@dataclass
class AgentRun:
    """Raw transcript from :func:`run_tax_agent`.

    The converter consumes ``messages`` and ``input_tokens`` /
    ``output_tokens`` to build an :class:`ExecutionResult`.
    """

    messages: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    final_output: str | None


def run_tax_agent(query: str, *, max_turns: int = 6) -> AgentRun:
    """Drive the ReAct loop until the model stops calling tools.

    Args:
        query: The user's question.
        max_turns: Hard cap on tool-use iterations. Prevents runaway
            loops from burning tokens if the model misbehaves.
    """
    client = Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)

    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
    total_in = 0
    total_out = 0
    final_output: str | None = None

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=_TOOL_SCHEMAS,
            messages=messages,
        )
        total_in += response.usage.input_tokens
        total_out += response.usage.output_tokens

        # Persist the assistant's response as-is so the conversation
        # stays coherent for the next turn.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Terminal turn — extract any text block as the final output.
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    final_output = block.text
                    break
            break

        # Tool-use turn: run every tool block the model requested and
        # feed the results back as a ``user`` message of tool_result
        # content blocks.
        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            impl = _TOOL_IMPLEMENTATIONS.get(block.name)
            if impl is None:
                # Unknown tool — surface the error to the model so it
                # can recover.
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"error: unknown tool {block.name!r}",
                        "is_error": True,
                    }
                )
                continue
            try:
                result = impl(**block.input)
            except Exception as exc:  # pragma: no cover — defensive
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"error: {exc}",
                        "is_error": True,
                    }
                )
                continue
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return AgentRun(
        messages=messages,
        input_tokens=total_in,
        output_tokens=total_out,
        final_output=final_output,
    )


if __name__ == "__main__":
    run = run_tax_agent("What's 100 + 200 with 10% tax added?")
    print(f"Final output: {run.final_output}")
    print(f"Tokens: in={run.input_tokens} out={run.output_tokens}")
