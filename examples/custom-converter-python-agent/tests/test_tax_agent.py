"""agentverify integration tests for the custom-converter tax agent.

Tests use cassette replay mode — no LLM API key is required.
The cassette file ``cassettes/tax_calculation.yaml`` contains
pre-recorded Anthropic LLM interactions that are replayed
deterministically.

These tests exercise both the flat assertions and the step-level
assertions on an agent that does not use any built-in framework
adapter — instead, ``converter.run_to_execution_result`` maps the
raw Anthropic message history into an :class:`ExecutionResult`.
"""

import sys
from pathlib import Path

import pytest

from agentverify import (
    ANY,
    ToolCall,
    assert_all,
    assert_cost,
    assert_final_output,
    assert_no_tool_call,
    assert_step,
    assert_step_uses_result_from,
    assert_tool_calls,
)

# Allow importing agent.py / converter.py from the parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run_tax_agent  # noqa: E402
from converter import run_to_execution_result  # noqa: E402


CASSETTE = "tax_calculation.yaml"
QUERY = "What's 100 + 200 with 10% tax added?"


def _run_and_convert(cassette_fixture):
    """Drive the tax agent inside the cassette context and convert the result."""
    with cassette_fixture(CASSETTE, provider="anthropic") as rec:
        run = run_tax_agent(QUERY)
    return rec, run_to_execution_result(run)


# ---------------------------------------------------------------------------
# Flat assertions — independent of step structure
# ---------------------------------------------------------------------------


class TestTaxAgentFlat:
    """Cross-cutting assertions that don't depend on step indices."""

    @pytest.mark.agentverify
    def test_tool_sequence(self, cassette):
        """The agent must call ``add`` first, then ``apply_tax``."""
        _, result = _run_and_convert(cassette)
        assert_tool_calls(
            result,
            expected=[
                ToolCall("add", {"a": 100, "b": 200}),
                ToolCall("apply_tax", {"amount": ANY, "rate": 0.1}),
            ],
            partial_args=True,
        )

    @pytest.mark.agentverify
    def test_safety_no_dangerous_tools(self, cassette):
        """Neither tool is destructive; the agent must not call anything else."""
        _, result = _run_and_convert(cassette)
        assert_no_tool_call(
            result,
            forbidden_tools=["delete_file", "execute_command", "transfer_funds"],
        )

    @pytest.mark.agentverify
    def test_budget_and_output(self, cassette):
        """Run budget + final output assertions together via ``assert_all``."""
        _, result = _run_and_convert(cassette)
        assert_all(
            result,
            lambda r: assert_cost(r, max_tokens=10_000),
            lambda r: assert_final_output(r, matches=r"330(\.00?)?"),
        )


# ---------------------------------------------------------------------------
# Step-level assertions — exercise the converter's Step population
# ---------------------------------------------------------------------------


class TestTaxAgentStepLevel:
    """Step-level verification of the tool-chain and data flow.

    Anthropic's Messages API emits two assistant turns before the
    final summary: step 0 calls ``add``, step 1 calls ``apply_tax``
    with the ``add`` result as its ``amount``, and step 2 is the
    textual summary with no tool calls.  The converter places the
    tool result from each ``user`` follow-up message onto the
    preceding assistant step's ``tool_results``, so
    ``assert_step_uses_result_from`` can verify that the ``amount``
    passed to ``apply_tax`` actually came from ``add`` and wasn't
    hallucinated by the model.
    """

    @pytest.mark.agentverify
    def test_first_step_calls_add(self, cassette):
        """Step 0 is the assistant's first turn — one ``add`` call with the
        user's pre-tax numbers."""
        _, result = _run_and_convert(cassette)
        assert_step(
            result, step=0,
            expected_tool=ToolCall("add", {"a": 100, "b": 200}),
            partial_args=True,
        )

    @pytest.mark.agentverify
    def test_second_step_calls_apply_tax_with_sum(self, cassette):
        """Step 1 is the assistant's second turn — ``apply_tax`` with the
        rate the user requested and some ``amount`` value (exact value
        checked in the data-flow test below)."""
        _, result = _run_and_convert(cassette)
        assert_step(
            result, step=1,
            expected_tool=ToolCall("apply_tax", {"amount": ANY, "rate": 0.1}),
            partial_args=True,
        )

    @pytest.mark.agentverify
    def test_apply_tax_uses_add_result(self, cassette):
        """The headline check: step 1's input must trace back to step 0.

        Catches the "model hallucinated a number instead of using the
        tool result" bug.  The ``add`` tool returns ``300``; that
        value must appear in the ``apply_tax`` arguments as
        ``amount``.
        """
        _, result = _run_and_convert(cassette)
        assert_step_uses_result_from(result, step=1, depends_on=0)

    @pytest.mark.agentverify
    def test_final_step_has_no_tool_calls(self, cassette):
        """The last step is a plain text summary — no further tool use."""
        _, result = _run_and_convert(cassette)
        final = result.steps[-1]
        assert_step(result, step=final.index, expected_tools=[])

    @pytest.mark.agentverify
    def test_final_output_contains_grossed_up_number(self, cassette):
        """The final summary must quote the post-tax total (330)."""
        _, result = _run_and_convert(cassette)
        assert_final_output(result, matches=r"330(\.00?)?")
