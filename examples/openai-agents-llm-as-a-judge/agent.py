"""OpenAI Agents SDK — LLM as a Judge example.

Adapted from the official `agent_patterns/llm_as_a_judge.py` sample:
https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/llm_as_a_judge.py

A ``story_outline_generator`` agent writes a short story outline for a
given user request. An ``evaluator`` agent grades it with a structured
``{feedback, score}`` output where ``score`` is one of ``"pass"`` /
``"needs_improvement"`` / ``"fail"``. The loop re-runs the generator
with the evaluator's feedback appended to the input until the evaluator
returns ``pass`` (or a safety cap of ``max_rounds`` is hit).

This example demonstrates agentverify's value for *probabilistic*
agents: record a single run of the refinement loop once, replay it in
CI deterministically, and assert that each regeneration step actually
consumed the previous round's feedback — ``assert_step_uses_result_from``
catches the "agent ignored the judge" bug without any real LLM calls.

The only change from the official sample is that we switch the SDK's
default OpenAI API to Chat Completions so that agentverify's existing
``openai`` cassette adapter can record the interactions, and we tighten
the evaluator's "when may I pass?" rule from "after 5 attempts" to
"on the second attempt once feedback is incorporated" so the loop
terminates within a cassette-friendly number of rounds while still
exercising the core refinement pattern.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Literal

from agents import (
    Agent,
    ItemHelpers,
    Runner,
    TResponseInputItem,
    set_default_openai_api,
)

# agentverify's OpenAI cassette adapter patches
# ``openai.resources.chat.completions.Completions.create``. Switch the
# Agents SDK default from Responses API to Chat Completions so cassette
# recording and replay work out of the box.
set_default_openai_api("chat_completions")


DEFAULT_MODEL = "gpt-4o-mini"


story_outline_generator = Agent(
    name="story_outline_generator",
    instructions=(
        "You generate a very short story outline based on the user's input. "
        "If there is any feedback provided, use it to improve the outline."
    ),
    model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
)


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


evaluator = Agent[None](
    name="evaluator",
    instructions=(
        "You evaluate a story outline and decide if it's good enough. "
        "If it's not good enough, you provide feedback on what needs to be improved. "
        "Never give it a pass on the first try; return ``needs_improvement`` with "
        "concrete, actionable feedback. "
        "On the second attempt, if the outline has clearly incorporated your "
        "earlier feedback, return ``pass`` — do not go for perfection."
    ),
    output_type=EvaluationFeedback,
    model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
)


@dataclass
class JudgeLoopResult:
    """Return value from :func:`run_judge_loop`.

    Attributes:
        final_outline: The last outline produced by the generator.
        rounds: Number of (generator, evaluator) round trips executed.
        pass_reached: ``True`` if the evaluator returned ``score="pass"``
            before the round cap. ``False`` if the cap was hit first.
        generator_results: The :class:`~agents.RunResult` from each
            generator invocation, in order.
        evaluator_results: The :class:`~agents.RunResult` from each
            evaluator invocation, in order. Always the same length as
            ``generator_results``.
    """

    final_outline: str | None
    rounds: int
    pass_reached: bool
    generator_results: list
    evaluator_results: list


async def run_judge_loop(
    query: str = "A detective story in space.",
    *,
    max_rounds: int = 3,
) -> JudgeLoopResult:
    """Run the generator/evaluator refinement loop.

    Args:
        query: User's story request. The default mirrors the official
            sample for cassette stability.
        max_rounds: Safety cap on total rounds so the loop can't spin
            forever on an over-strict evaluator.
    """
    input_items: list[TResponseInputItem] = [{"content": query, "role": "user"}]
    latest_outline: str | None = None
    generator_results: list = []
    evaluator_results: list = []
    pass_reached = False
    rounds = 0

    while rounds < max_rounds:
        story_outline_result = await Runner.run(
            story_outline_generator, input_items
        )
        generator_results.append(story_outline_result)

        input_items = story_outline_result.to_input_list()
        latest_outline = ItemHelpers.text_message_outputs(
            story_outline_result.new_items
        )

        evaluator_result = await Runner.run(evaluator, input_items)
        evaluator_results.append(evaluator_result)
        feedback: EvaluationFeedback = evaluator_result.final_output

        rounds += 1
        if feedback.score == "pass":
            pass_reached = True
            break

        # Append the evaluator's feedback as a new user message so the
        # next generator round can see what needs to be fixed.
        input_items.append(
            {"content": f"Feedback: {feedback.feedback}", "role": "user"}
        )

    return JudgeLoopResult(
        final_outline=latest_outline,
        rounds=rounds,
        pass_reached=pass_reached,
        generator_results=generator_results,
        evaluator_results=evaluator_results,
    )


def run_judge_loop_sync(
    query: str = "A detective story in space.",
    *,
    max_rounds: int = 3,
) -> JudgeLoopResult:
    """Synchronous wrapper around :func:`run_judge_loop` for tests."""
    return asyncio.run(run_judge_loop(query, max_rounds=max_rounds))


if __name__ == "__main__":
    result = run_judge_loop_sync()
    print(f"Rounds: {result.rounds}  pass_reached: {result.pass_reached}")
    print(f"Final outline:\n{result.final_outline}")
