"""agentverify integration tests for the LangGraph multi-agent supervisor.

Tests use cassette replay mode — no LLM API key is required.
The cassette file ``cassettes/supervisor_faang.yaml`` contains pre-recorded
OpenAI LLM interactions that are replayed deterministically.

The supervisor orchestrates a research_expert (web_search tool) and a
math_expert (add/multiply tools) to answer "What's the combined headcount
of the FAANG companies in 2024?" The resulting step structure demonstrates
multi-agent handoff and cross-step data flow — the math_expert's inputs
must come from the research_expert's findings.
"""

import sys
from pathlib import Path

import pytest

from agentverify import (
    ANY,
    MATCHES,
    OrderMode,
    ToolCall,
    assert_all,
    assert_cost,
    assert_final_output,
    assert_no_tool_call,
    assert_step,
    assert_step_output,
    assert_step_uses_result_from,
    assert_tool_calls,
)
from agentverify.frameworks.langgraph import from_langgraph

# Allow importing agent.py from the parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run_supervisor  # noqa: E402


CASSETTE = "supervisor_faang.yaml"
QUERY = "What's the combined headcount of the FAANG companies in 2024?"


def _run_and_convert(cassette_fixture):
    """Run the supervisor under the cassette context and convert the result."""
    with cassette_fixture(CASSETTE, provider="openai") as rec:
        raw = run_supervisor(QUERY)
    return rec, from_langgraph(raw)


# ---------------------------------------------------------------------------
# Flat tool-call + safety + budget assertions
# ---------------------------------------------------------------------------


class TestSupervisorFlat:
    """Tool-call sequence and safety checks across the whole run."""

    @pytest.mark.agentverify
    def test_tool_sequence_in_order(self, cassette):
        """Verify the end-to-end tool sequence: handoff → web_search → handoff → add.

        Uses IN_ORDER so intermediate supervisor reasoning steps or extra
        ``add`` calls don't break the test.
        """
        _, result = _run_and_convert(cassette)
        assert_tool_calls(
            result,
            expected=[
                ToolCall("transfer_to_research_expert", {}),
                ToolCall("web_search", {"query": ANY}),
                ToolCall("transfer_to_math_expert", {}),
                ToolCall("add", {"a": ANY, "b": ANY}),
            ],
            order=OrderMode.IN_ORDER,
            partial_args=True,
        )

    @pytest.mark.agentverify
    def test_safety_no_dangerous_tools(self, cassette):
        """The supervisor must never hand off to non-existent agents or run
        destructive tools.
        """
        _, result = _run_and_convert(cassette)
        assert_no_tool_call(
            result,
            forbidden_tools=[
                "transfer_to_admin_expert",
                "delete_user",
                "execute_command",
                "write_file",
            ],
        )

    @pytest.mark.agentverify
    def test_budget_and_output(self, cassette):
        """Collect cost/final-output failures together via ``assert_all``."""
        _, result = _run_and_convert(cassette)
        assert_all(
            result,
            lambda r: assert_cost(r, max_tokens=20_000),
            lambda r: assert_final_output(r, matches=r"\d[\d,]*"),
        )


# ---------------------------------------------------------------------------
# Step-level assertions (the reason this example exists)
# ---------------------------------------------------------------------------


class TestSupervisorStepLevel:
    """Step-level verification of the multi-agent handoff flow.

    The supervisor first delegates to ``research_expert`` which runs
    ``web_search``. Control returns to the supervisor, which then
    delegates to ``math_expert`` which runs ``add``. Each of these is a
    distinct step in ``result.steps`` — and the math_expert's inputs
    must trace back to the research_expert's findings.
    """

    @pytest.mark.agentverify
    def test_supervisor_routes_to_research_first(self, cassette):
        """Step 0 (the very first LLM call) is the supervisor handing off
        to the research expert. This is the routing decision, with no
        domain tool calls yet.
        """
        _, result = _run_and_convert(cassette)
        assert_step(
            result,
            step=0,
            expected_tool=ToolCall("transfer_to_research_expert", {}),
            partial_args=True,
        )

    @pytest.mark.agentverify
    def test_research_agent_uses_web_search(self, cassette):
        """Somewhere after the handoff, the research_expert must call
        ``web_search`` with a FAANG-related query. The agent may issue
        several parallel ``web_search`` calls in a single step — we assert
        that *at least one* of them is on-topic.
        """
        _, result = _run_and_convert(cassette)
        research_step = _find_step_with_tool(result, "web_search")
        assert research_step is not None, (
            "Expected a step that calls web_search; none found"
        )
        assert_step(
            result,
            step=research_step.index,
            expected_tools=[
                ToolCall(
                    "web_search", {"query": MATCHES(r"(?i)FAANG|headcount|employees")},
                ),
            ],
            order=OrderMode.ANY_ORDER,
            partial_args=True,
        )

    @pytest.mark.agentverify
    def test_math_agent_adds_with_research_derived_numbers(self, cassette):
        """Each ``add`` step in the math chain consumes the previous
        ``add`` step's result — the running total flows forward through
        the agent's tool calls. This is the clearest deterministic data
        flow in the whole run.

        Example from the recorded cassette:
        - step N:   ``add(67317, 164000)`` → produces ``231317.0``
        - step N+1: ``add(231317, 1551000)`` — ``a`` traces back to N

        ``assert_step_uses_result_from`` catches the "agent dropped the
        running total and hallucinated a fresh number" bug.
        """
        _, result = _run_and_convert(cassette)
        add_steps = [
            s for s in result.steps if any(tc.name == "add" for tc in s.tool_calls)
        ]
        assert len(add_steps) >= 2, (
            f"Expected at least 2 consecutive add steps; found {len(add_steps)}"
        )

        # The second add step must consume the result of the first
        assert_step_uses_result_from(
            result,
            step=add_steps[1].index,
            depends_on=add_steps[0].index,
        )

    @pytest.mark.agentverify
    def test_final_answer_includes_a_number(self, cassette):
        """The final step (after everyone has handed back to the supervisor)
        produces a summary text that contains a numeric answer.
        """
        _, result = _run_and_convert(cassette)
        # Use the last step with text output as the "final answer" step
        final_step = next(
            (s for s in reversed(result.steps) if s.output),
            None,
        )
        assert final_step is not None, "Expected at least one step with text output"
        assert_step_output(result, step=final_step.index, matches=r"\d[\d,]*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_step_with_tool(result, tool_name):
    """Return the first step whose tool_calls include ``tool_name``."""
    for step in result.steps:
        for tc in step.tool_calls:
            if tc.name == tool_name:
                return step
    return None
