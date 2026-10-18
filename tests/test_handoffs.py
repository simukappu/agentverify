"""Tests for multi-agent handoffs: Step.agent, ExecutionResult.handoffs, and the handoff assertions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentverify import (
    ANY,
    MATCHES,
    ExecutionResult,
    Handoff,
    HandoffError,
    OrderMode,
    Step,
    StepAgentError,
    ToolCall,
    assert_handoffs,
    assert_no_handoff,
    assert_step,
)
from agentverify.models import attribute_agents, handoff_target


def _supervisor_run(with_agents: bool = True) -> ExecutionResult:
    """supervisor -> research_expert -> supervisor -> math_expert, as from_langgraph sees it."""
    name = (lambda a: a) if with_agents else (lambda a: None)
    return ExecutionResult(
        steps=[
            Step(index=0, tool_calls=[ToolCall("transfer_to_research_expert")], agent=name("supervisor")),
            Step(index=1, tool_calls=[ToolCall("web_search", {"q": "faang"})], agent=name("research_expert")),
            Step(index=2, tool_calls=[ToolCall("transfer_back_to_supervisor")], agent=name("research_expert")),
            Step(index=3, tool_calls=[ToolCall("transfer_to_math_expert")], agent=name("supervisor")),
            Step(index=4, tool_calls=[ToolCall("add", {"a": 1, "b": 2})], agent=name("math_expert")),
        ]
    )


class TestModel:
    def test_handoff_target(self):
        assert handoff_target("transfer_to_math_expert") == "math_expert"
        assert handoff_target("transfer_back_to_supervisor") == "supervisor"
        assert handoff_target("web_search") is None

    def test_handoffs_property(self):
        assert _supervisor_run().handoffs == [
            Handoff("supervisor", "research_expert", 0),
            Handoff("research_expert", "supervisor", 2),
            Handoff("supervisor", "math_expert", 3),
        ]

    def test_agent_change_without_a_call_is_a_framework_handoff(self):
        result = ExecutionResult(
            steps=[
                Step(index=0, tool_calls=[ToolCall("transfer_to_research_expert")]),
                Step(index=1, agent="research_expert"),
                Step(index=2, agent="supervisor"),
            ]
        )
        assert result.handoffs == [Handoff(None, "research_expert", 0), Handoff("research_expert", "supervisor", None)]

    def test_attribute_agents(self):
        steps = attribute_agents(
            [
                Step(index=0, tool_calls=[ToolCall("transfer_to_a")]),
                Step(index=1),
                Step(index=2, agent="b"),
                Step(index=3),
            ]
        )
        assert [s.agent for s in steps] == [None, "a", "b", "b"]

    def test_agent_round_trips_through_dicts(self):
        step = Step(index=0, agent="supervisor")
        assert Step.from_dict(step.to_dict()).agent == "supervisor"


class TestAssertHandoffs:
    def test_exact(self):
        assert_handoffs(
            _supervisor_run(),
            [Handoff("supervisor", "research_expert"), Handoff("research_expert", "supervisor"), Handoff("supervisor", "math_expert")],
        )

    def test_exact_mismatch_message(self):
        with pytest.raises(HandoffError) as exc:
            assert_handoffs(_supervisor_run(), [Handoff("supervisor", "math_expert")])
        message = str(exc.value)
        assert "Handoffs do not match (exact)" in message
        assert "'supervisor' -> 'math_expert'" in message and "(step 3)" in message

    def test_in_order_and_any_order(self):
        result = _supervisor_run()
        assert_handoffs(result, [Handoff("supervisor", "research_expert"), Handoff("supervisor", "math_expert")], order=OrderMode.IN_ORDER)
        with pytest.raises(HandoffError):
            assert_handoffs(result, [Handoff("supervisor", "math_expert"), Handoff("supervisor", "research_expert")], order=OrderMode.IN_ORDER)
        assert_handoffs(result, [Handoff("supervisor", "math_expert"), Handoff("supervisor", "research_expert")], order=OrderMode.ANY_ORDER)
        with pytest.raises(HandoffError):
            assert_handoffs(result, [Handoff("supervisor", "admin_expert")], order=OrderMode.ANY_ORDER)

    def test_matchers_and_unknown_first_agent(self):
        result = _supervisor_run(with_agents=False)
        # No agent names: the first handoff's from_agent is unknown, later ones are inferred.
        assert result.handoffs[0] == Handoff(None, "research_expert", 0)
        assert_handoffs(result, [Handoff(ANY, "research_expert"), Handoff(ANY, "supervisor"), Handoff("supervisor", MATCHES(r"^math"))])

    def test_empty_actual_is_reported(self):
        with pytest.raises(HandoffError, match="no handoffs"):
            assert_handoffs(ExecutionResult(steps=[Step(index=0)]), [Handoff(ANY, "x")])


class TestAssertNoHandoff:
    def test_passes_and_fails(self):
        result = _supervisor_run()
        assert_no_handoff(result, to_agent="admin_expert")
        assert_no_handoff(result, from_agent="math_expert")
        assert_no_handoff(result, to_agent="math_expert", from_agent="research_expert")
        with pytest.raises(HandoffError, match="Forbidden handoff"):
            assert_no_handoff(result, to_agent=MATCHES("expert$"))
        with pytest.raises(HandoffError):
            assert_no_handoff(result, from_agent="supervisor", to_agent="math_expert")

    def test_requires_a_filter(self):
        with pytest.raises(ValueError, match="at least one"):
            assert_no_handoff(_supervisor_run())


class TestAssertStepAgent:
    def test_agent_only(self):
        assert_step(_supervisor_run(), step=1, agent="research_expert")
        with pytest.raises(StepAgentError, match="step 1 agent does not match"):
            assert_step(_supervisor_run(), step=1, agent="math_expert")

    def test_agent_with_tools(self):
        assert_step(_supervisor_run(), step=1, agent=MATCHES("research"), expected_tool=ToolCall("web_search", {"q": ANY}))
        with pytest.raises(StepAgentError):
            assert_step(_supervisor_run(), step=4, agent="research_expert", expected_tool=ToolCall("add", ANY))


class TestCassetteInference:
    """Agents are inferred from the handoffs in each request's history, across provider message shapes."""

    def _result(self, histories, calls):
        from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
        from agentverify.cassette.recorder import _build_execution_result

        interactions = [
            (NormalizedRequest(messages=h, model="m"), NormalizedResponse(content=None, tool_calls=c))
            for h, c in zip(histories, calls)
        ]
        return _build_execution_result(
            interactions=interactions,
            probe_stacks=[[] for _ in interactions],
            probe_events=[],
            probe_tool_results={},
            duration_ms=None,
        )

    @pytest.mark.parametrize(
        "handoff_back",
        [
            {"role": "assistant", "tool_calls": [{"function": {"name": "transfer_back_to_supervisor"}}]},  # OpenAI
            {"role": "assistant", "tool_calls": [{"name": "transfer_back_to_supervisor"}]},  # flat
            {"role": "assistant", "content": [{"type": "tool_use", "name": "transfer_back_to_supervisor"}]},  # Anthropic
            {"role": "assistant", "content": [{"toolUse": {"name": "transfer_back_to_supervisor"}}, "text"]},  # Bedrock
        ],
        ids=["openai", "flat", "anthropic", "bedrock"],
    )
    def test_return_written_by_the_framework(self, handoff_back):
        to_research = {"role": "assistant", "tool_calls": [{"function": {"name": "transfer_to_research_expert"}}]}
        result = self._result(
            histories=[
                [{"role": "user", "content": "q"}],
                [{"role": "user", "content": "q"}, to_research, "not-a-dict"],
                [{"role": "user", "content": "q"}, to_research, handoff_back],
            ],
            calls=[
                [{"name": "transfer_to_research_expert", "arguments": {}}],
                [{"name": "web_search", "arguments": {}}],
                [{"name": "transfer_to_math_expert", "arguments": {}}],
            ],
        )
        assert [s.agent for s in result.steps] == [None, "research_expert", "supervisor"]
        assert result.handoffs == [
            Handoff(None, "research_expert", 0),
            Handoff("research_expert", "supervisor", None),
            Handoff("supervisor", "math_expert", 2),
        ]


class TestOpenAIAgentsAdapter:
    def _item(self, kind, agent, **kw):
        return SimpleNamespace(type=kind, agent=SimpleNamespace(name=agent), **kw)

    def test_handoff_items_and_agent_names(self):
        from agentverify.frameworks.openai_agents import from_openai_agents

        run = SimpleNamespace(
            new_items=[
                self._item("handoff_call_item", "Triage", raw_item={"name": "transfer_to_billing", "arguments": "{}"}),
                self._item("handoff_output_item", "Triage", output=None, raw_item={"output": '{"assistant": "Billing"}'}),
                self._item("tool_call_item", "Billing", raw_item={"name": "refund", "arguments": '{"id": 7}'}),
                self._item("tool_call_output_item", "Billing", output="ok", raw_item={"output": "ok"}),
                self._item("message_output_item", "Billing", raw_item=None),
                SimpleNamespace(type="message_output_item", agent=None, raw_item=None),
            ],
            final_output="Refunded.",
            context_wrapper=None,
        )
        result = from_openai_agents(run)
        assert [(s.agent, [t.name for t in s.tool_calls]) for s in result.steps] == [
            ("Triage", ["transfer_to_billing"]),
            ("Billing", ["refund"]),
        ]
        # The tool name carries a normalized agent name; the handoff uses the agent's own spelling.
        assert result.handoffs == [Handoff("Triage", "Billing", 0)]


def test_edge_shapes():
    from agentverify.cassette.recorder import _tool_names_in
    from agentverify.frameworks.openai_agents import from_openai_agents

    assert _tool_names_in({"role": "assistant", "tool_calls": ["bad", {"function": {"name": 3}}, {"name": "x"}]}) == ["x"]
    assert str(HandoffError("only a message")) == "only a message"
    agent = SimpleNamespace(name="A")
    run = SimpleNamespace(
        new_items=[
            SimpleNamespace(type="tool_call_item", agent=agent, raw_item={"name": "f", "arguments": "{}"}),
            SimpleNamespace(type="tool_call_item", agent=agent, raw_item={"name": "g", "arguments": "{}"}),
        ],
        final_output=None,
        context_wrapper=None,
    )
    [step] = from_openai_agents(run).steps
    assert step.agent == "A" and [t.name for t in step.tool_calls] == ["f", "g"]


def test_agent_names_match_as_handoff_tools_spell_them():
    cassette_view = ExecutionResult(
        steps=[Step(index=0, tool_calls=[ToolCall("transfer_to_billing_agent")]), Step(index=1, agent="billing_agent")]
    )
    assert_handoffs(cassette_view, [Handoff(None, "Billing Agent")])
    assert_step(cassette_view, step=1, agent="Billing Agent")
    with pytest.raises(HandoffError):
        assert_no_handoff(cassette_view, to_agent="Billing Agent")


class TestReviewEdgeCases:
    def test_only_the_first_handoff_in_a_response_counts(self):
        result = ExecutionResult(
            steps=[
                Step(index=0, agent="Triage", tool_calls=[ToolCall("transfer_to_a"), ToolCall("transfer_to_b")]),
                Step(index=1, agent="a"),
            ]
        )
        assert result.handoffs == [Handoff("Triage", "a", 0)]

    def test_non_ascii_agent_names_stay_distinct(self):
        result = ExecutionResult(steps=[Step(index=0, agent="受付"), Step(index=1, agent="請求")])
        assert result.handoffs == [Handoff("受付", "請求", None)]
        with pytest.raises(StepAgentError):
            assert_step(result, step=1, agent="返金")
        assert_step(result, step=1, agent="請求")

    def test_expected_step_is_compared(self):
        with pytest.raises(HandoffError):
            assert_handoffs(_supervisor_run(), [Handoff("supervisor", "research_expert", step=3)], order=OrderMode.ANY_ORDER)
        assert_handoffs(_supervisor_run(), [Handoff("supervisor", "research_expert", step=0)], order=OrderMode.ANY_ORDER)

    def test_probe_steps_are_not_attributed(self):
        steps = attribute_agents([Step(index=0, tool_calls=[ToolCall("transfer_to_a")]), Step(index=1, source="probe")])
        assert steps[1].agent is None

    def test_gemini_and_sdk_object_messages(self):
        from agentverify.cassette.recorder import _tool_names_in

        gemini = {"role": "model", "parts": [{"text": "x"}, {"function_call": {"name": "transfer_to_a"}}, {"functionCall": {"name": "g"}}]}
        assert _tool_names_in(gemini) == ["transfer_to_a", "g"]
        pydantic = pytest.importorskip("pydantic")

        class Message(pydantic.BaseModel):
            role: str
            tool_calls: list

        obj = Message(role="assistant", tool_calls=[{"function": {"name": "transfer_back_to_supervisor"}}])
        assert _tool_names_in(obj) == ["transfer_back_to_supervisor"]


class TestHandoffsFromTheFramework:
    """Framework adapters report handoffs from agent changes, not from tool names."""

    def test_handoffs_from_agents(self):
        from agentverify.models import handoffs_from_agents

        steps = [
            Step(index=0, agent="Bank", tool_calls=[ToolCall("transfer_to_savings")]),  # an ordinary tool
            Step(index=1, agent="Bank", tool_calls=[ToolCall("route_billing")]),  # a renamed handoff tool
            Step(index=2, source="probe"),
            Step(index=3),
            Step(index=4, agent="Billing"),
        ]
        assert handoffs_from_agents(steps) == [Handoff("Bank", "Billing", 1)]
        result = ExecutionResult(steps=steps, handoffs=handoffs_from_agents(steps))
        assert result.handoffs == [Handoff("Bank", "Billing", 1)]

    def test_explicit_handoffs_round_trip(self):
        result = ExecutionResult(steps=[Step(index=0, agent="a")], handoffs=[Handoff("a", "b", 0)])
        again = ExecutionResult.from_dict(result.to_dict())
        assert again.handoffs == [Handoff("a", "b", 0)]
        assert "handoffs" not in ExecutionResult(steps=[]).to_dict()
        assert ExecutionResult.from_dict({"steps": [], "handoffs": []}).handoffs == []

    def test_agents_sdk_ordinary_tool_and_renamed_handoff(self, monkeypatch):
        pytest.importorskip("agents")
        import asyncio
        from unittest.mock import patch

        import openai.resources.responses as responses_mod
        from agents import Agent, Runner, function_tool, handoff, set_tracing_disabled

        from agentverify.cassette.adapters.base import NormalizedResponse
        from agentverify.cassette.adapters.openai_responses import denormalize_response
        from agentverify.frameworks.openai_agents import from_openai_agents

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        set_tracing_disabled(True)

        @function_tool
        def transfer_to_savings(amount: int) -> str:
            return f"moved {amount}"

        billing = Agent(name="Billing", instructions="Bill.")
        bank = Agent(
            name="Bank",
            instructions="Bank.",
            tools=[transfer_to_savings],
            handoffs=[handoff(billing, tool_name_override="route_billing")],
        )
        turns = iter(
            [
                [{"name": "transfer_to_savings", "arguments": '{"amount": 5}', "id": "c1"}],
                [{"name": "route_billing", "arguments": "{}", "id": "c2"}],
                None,
            ]
        )

        async def live(*args, **kwargs):
            calls = next(turns)
            return denormalize_response(NormalizedResponse(tool_calls=calls, content=None if calls else "Billed."))

        with patch.object(responses_mod.AsyncResponses, "create", new=live):
            run = asyncio.run(Runner.run(bank, "Save 5, then bill me."))
        result = from_openai_agents(run)
        assert result.handoffs == [Handoff("Bank", "Billing", 1)]
        assert_no_handoff(result, to_agent="savings")
