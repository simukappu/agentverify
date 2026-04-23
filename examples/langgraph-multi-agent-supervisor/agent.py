"""LangGraph multi-agent supervisor example.

Based on the official langgraph-supervisor quickstart:
https://github.com/langchain-ai/langgraph-supervisor-py

A supervisor agent orchestrates a research expert and a math expert.
The supervisor routes the user query to the appropriate sub-agent based
on the task, and combines their outputs into a final answer.

This example demonstrates agentverify's step-level assertions for
multi-agent handoff patterns — each agent invocation becomes a step,
and ``assert_step_uses_result_from`` verifies that data flows correctly
from the researcher's findings to the math expert's inputs.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

try:
    # LangGraph v1.0+: create_react_agent was moved to langchain.agents.create_agent
    from langchain.agents import create_agent as _create_agent

    def create_react_agent(model, tools, name=None, prompt=None):
        return _create_agent(model, tools, name=name, system_prompt=prompt)
except ImportError:
    from langgraph.prebuilt import create_react_agent


DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def web_search(query: str) -> str:  # noqa: ARG001 — query unused in this mock
    """Search the web for information.

    Returns a hardcoded response about FAANG headcounts (matches the
    official langgraph-supervisor quickstart). The mock keeps the cassette
    deterministic with no external HTTP calls.
    """
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. Facebook (Meta): 67,317 employees.\n"
        "2. Apple: 164,000 employees.\n"
        "3. Amazon: 1,551,000 employees.\n"
        "4. Netflix: 14,000 employees.\n"
        "5. Google (Alphabet): 181,269 employees."
    )


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def build_supervisor_app(model_name: str | None = None):
    """Build the supervisor workflow and return the compiled graph.

    Args:
        model_name: OpenAI chat model name. Defaults to ``OPENAI_MODEL``
            env var if set, otherwise ``gpt-4o-mini``.
    """
    model_name = model_name or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    # temperature=0 keeps the conversation deterministic enough to replay
    # from a cassette reliably.
    model = ChatOpenAI(model=model_name, temperature=0)

    math_agent = create_react_agent(
        model=model,
        tools=[add, multiply],
        name="math_expert",
        prompt=(
            "You are a math expert. Always use one tool at a time. "
            "When adding multiple numbers, call the add tool repeatedly "
            "to accumulate the sum."
        ),
    )

    research_agent = create_react_agent(
        model=model,
        tools=[web_search],
        name="research_expert",
        prompt=(
            "You are a world class researcher with access to web search. "
            "Your job is to find raw data only. Do not do any math and do "
            "not summarize numerical totals — leave all arithmetic to the "
            "math expert."
        ),
    )

    workflow = create_supervisor(
        [research_agent, math_agent],
        model=model,
        prompt=(
            "You are a team supervisor managing a research expert and a "
            "math expert. Always delegate: use research_expert to gather "
            "raw data and use math_expert to perform any arithmetic. "
            "Never compute sums or totals yourself — route to math_expert."
        ),
        # Keep handoff messages in the state so agentverify can observe
        # the supervisor's routing decisions as their own steps.
        add_handoff_messages=True,
        output_mode="full_history",
    )

    return workflow.compile()


def run_supervisor(query: str, *, model_name: str | None = None) -> dict:
    """Invoke the supervisor with a single user query and return the result dict."""
    app = build_supervisor_app(model_name=model_name)
    return app.invoke({"messages": [{"role": "user", "content": query}]})


if __name__ == "__main__":
    result = run_supervisor(
        "What's the combined headcount of the FAANG companies in 2024?"
    )
    for msg in result.get("messages", []):
        pretty = getattr(msg, "pretty_print", None)
        if pretty:
            pretty()
        else:
            print(msg)
