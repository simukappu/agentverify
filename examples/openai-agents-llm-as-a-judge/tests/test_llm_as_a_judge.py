"""agentverify integration tests for the LLM-as-a-judge example.

Tests use cassette replay mode — no LLM API key is required.
The cassette file ``cassettes/detective_in_space.yaml`` contains
pre-recorded OpenAI LLM interactions that are replayed deterministically.

This is the example where agentverify's value shines the brightest:
the underlying agent is *probabilistic* (the evaluator's "pass" decision
depends on the LLM, and the refinement loop may take 2 or 3 rounds),
but the cassette freezes a single concrete run so every assertion
below runs deterministically in CI.
"""

import sys
from pathlib import Path

import pytest

from agentverify import (
    assert_all,
    assert_cost,
    assert_final_output,
    assert_no_tool_call,
    assert_step,
    assert_step_output,
    assert_step_uses_result_from,
)

# Allow importing agent.py from the parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run_judge_loop_sync  # noqa: E402


CASSETTE = "detective_in_space.yaml"
QUERY = "A detective story in space."


def _run_and_get_result(cassette_fixture):
    """Run the judge loop inside the cassette context, returning (loop_result, execution_result)."""
    with cassette_fixture(CASSETTE, provider="openai") as rec:
        loop_result = run_judge_loop_sync(QUERY, max_rounds=3)
    return loop_result, rec.to_execution_result()


# ---------------------------------------------------------------------------
# Flat assertions: safety, budget, final output
# ---------------------------------------------------------------------------


class TestJudgeLoopFlat:
    """Cross-cutting assertions that don't care about step structure."""

    @pytest.mark.agentverify
    def test_loop_terminates_within_round_cap(self, cassette):
        """The loop must reach a ``pass`` verdict within ``max_rounds``.

        The evaluator is instructed to reject the first attempt, so
        a correct run always goes through at least two (generator,
        evaluator) rounds — i.e. four steps minimum.
        """
        loop_result, result = _run_and_get_result(cassette)
        assert loop_result.pass_reached, (
            "Judge loop exited without reaching pass verdict; "
            "the generator failed to incorporate the evaluator feedback."
        )
        assert loop_result.rounds >= 2, (
            f"Expected at least 2 rounds (evaluator must reject first try); "
            f"got {loop_result.rounds}"
        )
        # 2 agents * rounds interactions become steps.
        assert len(result.steps) == 2 * loop_result.rounds

    @pytest.mark.agentverify
    def test_safety_no_tool_calls(self, cassette):
        """Neither agent uses any tools — both are pure text generators."""
        _, result = _run_and_get_result(cassette)
        assert_no_tool_call(
            result,
            forbidden_tools=[
                "web_search",
                "write_file",
                "execute_command",
                "delete_file",
            ],
        )

    @pytest.mark.agentverify
    def test_budget_and_output(self, cassette):
        """Collect budget + final output assertions together."""
        _, result = _run_and_get_result(cassette)
        assert_all(
            result,
            lambda r: assert_cost(r, max_tokens=20_000),
            lambda r: assert_final_output(r, matches=r".+"),
        )


# ---------------------------------------------------------------------------
# Step-level assertions: the reason this example exists
# ---------------------------------------------------------------------------


class TestJudgeLoopStepLevel:
    """Step-level verification of the probabilistic refinement loop.

    Steps alternate generator/evaluator:
      - step 0: generator — initial outline
      - step 1: evaluator — rejects with feedback (``needs_improvement``)
      - step 2: generator — regenerated outline (MUST consume step 1's feedback)
      - step 3: evaluator — ``pass`` (or another rejection + another round)
      - ...

    ``assert_step_uses_result_from(step=2, depends_on=1)`` is the key
    assertion: it catches the "agent regenerated but ignored the judge's
    feedback" bug, which is the most common failure mode of this
    pattern.
    """

    @pytest.mark.agentverify
    def test_first_generator_step_has_no_tool_calls(self, cassette):
        """Step 0 is the generator producing raw outline text — no tools."""
        _, result = _run_and_get_result(cassette)
        assert_step(result, step=0, expected_tools=[])
        assert_step_output(result, step=0, matches=r"\w+")

    @pytest.mark.agentverify
    def test_first_evaluator_rejects_on_first_try(self, cassette):
        """The evaluator is instructed to never pass on the first try,
        so step 1's structured output must carry a ``needs_improvement``
        or ``fail`` score — never ``pass``.
        """
        _, result = _run_and_get_result(cassette)
        assert_step(result, step=1, expected_tools=[])
        assert_step_output(
            result,
            step=1,
            matches=r'"score"\s*:\s*"(needs_improvement|fail)"',
        )

    @pytest.mark.agentverify
    def test_regeneration_consumes_evaluator_feedback(self, cassette):
        """Step 2 (generator, round 2) must reference step 1's feedback.

        The harness re-injects the evaluator's feedback as a user
        message before the next generator round, and the cassette
        recorder backfills that message into step 2's ``input_context``.
        ``assert_step_uses_result_from`` catches the case where the
        generator dropped the feedback and silently regenerated
        identical output.
        """
        _, result = _run_and_get_result(cassette)
        # Guard: this test is only meaningful when we actually had a
        # second round (pass_reached after >=2 rounds).
        assert len(result.steps) >= 3, (
            "Need at least 3 steps for regeneration-consumes-feedback check"
        )
        assert_step_uses_result_from(result, step=2, depends_on=1)

    @pytest.mark.agentverify
    def test_final_evaluator_step_emits_pass(self, cassette):
        """The last step is always the evaluator; its structured output
        must carry ``"score": "pass"`` for the loop to have terminated.
        """
        loop_result, result = _run_and_get_result(cassette)
        assert loop_result.pass_reached, (
            "Loop exited without reaching pass verdict — see "
            "test_loop_terminates_within_round_cap"
        )
        final_step = result.steps[-1]
        assert_step_output(
            result,
            step=final_step.index,
            matches=r'"score"\s*:\s*"pass"',
        )


# ---------------------------------------------------------------------------
# Sanity: there should never be any ToolCall (neither agent defines tools)
# ---------------------------------------------------------------------------


@pytest.mark.agentverify
def test_no_tool_calls_anywhere(cassette):
    """Neither agent has tools defined, so ``result.tool_calls`` is empty."""
    _, result = _run_and_get_result(cassette)
    assert result.tool_calls == []
    # Every step has an empty ``tool_calls`` list.
    for step in result.steps:
        assert step.tool_calls == [], (
            f"step {step.index} has unexpected tool calls: {step.tool_calls}"
        )
