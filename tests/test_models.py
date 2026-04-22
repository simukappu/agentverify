"""Tests for agentverify.models — ToolCall, TokenUsage, ExecutionResult."""

import json

import pytest

from agentverify.models import ExecutionResult, TokenUsage, ToolCall


class TestToolCall:
    def test_basic_creation(self):
        tc = ToolCall(name="search", arguments={"q": "test"})
        assert tc.name == "search"
        assert tc.arguments == {"q": "test"}
        assert tc.result is None

    def test_default_arguments(self):
        tc = ToolCall(name="noop")
        assert tc.arguments == {}

    def test_with_result(self):
        tc = ToolCall(name="calc", arguments={"x": 1}, result=42)
        assert tc.result == 42

    def test_frozen(self):
        tc = ToolCall(name="a")
        with pytest.raises(AttributeError):
            tc.name = "b"


class TestTokenUsage:
    def test_defaults(self):
        tu = TokenUsage()
        assert tu.input_tokens == 0
        assert tu.output_tokens == 0
        assert tu.total_tokens == 0

    def test_total_tokens(self):
        tu = TokenUsage(input_tokens=100, output_tokens=50)
        assert tu.total_tokens == 150

    def test_equality(self):
        a = TokenUsage(input_tokens=10, output_tokens=5)
        b = TokenUsage(input_tokens=10, output_tokens=5)
        assert a == b


class TestExecutionResult:
    def test_defaults(self):
        r = ExecutionResult()
        assert r.tool_calls == []
        assert r.token_usage is None
        assert r.total_cost_usd is None
        assert r.final_output is None
        assert r.duration_ms is None

    def test_from_dict_full(self):
        data = {
            "tool_calls": [
                {"name": "search", "arguments": {"q": "hi"}, "result": "found"},
            ],
            "token_usage": {"input_tokens": 10, "output_tokens": 5},
            "total_cost_usd": 0.01,
            "final_output": "done",
            "duration_ms": 1234.5,
        }
        r = ExecutionResult.from_dict(data)
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"
        assert r.tool_calls[0].arguments == {"q": "hi"}
        assert r.tool_calls[0].result == "found"
        assert r.token_usage.input_tokens == 10
        assert r.token_usage.output_tokens == 5
        assert r.total_cost_usd == 0.01
        assert r.final_output == "done"
        assert r.duration_ms == 1234.5

    def test_from_dict_minimal(self):
        r = ExecutionResult.from_dict({})
        assert r.tool_calls == []
        assert r.token_usage is None
        assert r.total_cost_usd is None
        assert r.final_output is None

    def test_from_dict_missing_name_key(self):
        with pytest.raises(ValueError, match="tool_calls\\[0\\].*'name' key"):
            ExecutionResult.from_dict({"tool_calls": [{"arguments": {"x": 1}}]})

    def test_from_dict_non_dict_tool_call(self):
        with pytest.raises(ValueError, match="tool_calls\\[1\\].*'name' key"):
            ExecutionResult.from_dict({"tool_calls": [{"name": "a"}, "not_a_dict"]})

    def test_from_dict_null_token_usage(self):
        r = ExecutionResult.from_dict({"token_usage": None})
        assert r.token_usage is None

    def test_from_json(self):
        data = {"tool_calls": [{"name": "a"}], "final_output": "ok"}
        r = ExecutionResult.from_json(json.dumps(data))
        assert r.tool_calls[0].name == "a"
        assert r.final_output == "ok"

    def test_to_dict(self):
        r = ExecutionResult(
            tool_calls=[ToolCall(name="x", arguments={"k": "v"}, result="r")],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            total_cost_usd=0.02,
            final_output="out",
            duration_ms=42.0,
        )
        d = r.to_dict()
        # v0.3.0 schema: steps is the single source; tool_calls is a derived property
        assert d["steps"] == [
            {
                "index": 0,
                "name": None,
                "source": "llm",
                "tool_calls": [{"name": "x", "arguments": {"k": "v"}, "result": "r"}],
                "tool_results": [],
                "output": None,
                "duration_ms": None,
                "token_usage": None,
                "input_context": None,
            }
        ]
        assert d["token_usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert d["total_cost_usd"] == 0.02
        assert d["final_output"] == "out"
        assert d["duration_ms"] == 42.0

    def test_to_dict_no_token_usage(self):
        r = ExecutionResult()
        d = r.to_dict()
        assert d["token_usage"] is None
        assert d["duration_ms"] is None

    def test_to_json(self):
        r = ExecutionResult(final_output="hello")
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["final_output"] == "hello"

    def test_round_trip(self):
        original = ExecutionResult(
            tool_calls=[ToolCall(name="a", arguments={"x": 1})],
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.05,
            final_output="result",
        )
        restored = ExecutionResult.from_json(original.to_json())
        assert restored.tool_calls[0].name == original.tool_calls[0].name
        assert restored.tool_calls[0].arguments == original.tool_calls[0].arguments
        assert restored.token_usage.input_tokens == original.token_usage.input_tokens
        assert restored.total_cost_usd == original.total_cost_usd
        assert restored.final_output == original.final_output



# ---------------------------------------------------------------------------
# Step and v0.3.0 ExecutionResult behaviour
# ---------------------------------------------------------------------------


from agentverify import Step


class TestStep:
    def test_to_dict_round_trip(self):
        s = Step(
            index=2,
            name="plan",
            source="probe",
            tool_calls=[ToolCall(name="x", arguments={"k": 1}, result="r")],
            tool_results=[{"result": "ok"}],
            output="text",
            duration_ms=12.5,
            token_usage=TokenUsage(input_tokens=3, output_tokens=7),
            input_context={"messages": [{"content": "hi"}]},
        )
        d = s.to_dict()
        restored = Step.from_dict(d)
        assert restored.index == 2
        assert restored.name == "plan"
        assert restored.source == "probe"
        assert restored.tool_calls[0].name == "x"
        assert restored.tool_results == [{"result": "ok"}]
        assert restored.output == "text"
        assert restored.duration_ms == 12.5
        assert restored.token_usage.input_tokens == 3
        assert restored.input_context == {"messages": [{"content": "hi"}]}

    def test_from_dict_rejects_invalid_source(self):
        with pytest.raises(ValueError, match="source"):
            Step.from_dict({"index": 0, "source": "bogus"})

    def test_from_dict_rejects_tool_call_without_name(self):
        with pytest.raises(ValueError, match="name"):
            Step.from_dict({
                "index": 0,
                "source": "llm",
                "tool_calls": [{"arguments": {}}],  # missing 'name'
            })

    def test_from_dict_with_minimal_input(self):
        s = Step.from_dict({})
        assert s.index == 0
        assert s.source == "llm"
        assert s.tool_calls == []


class TestExecutionResultStepsSingleSource:
    def test_tool_calls_property_derives_from_steps(self):
        r = ExecutionResult(steps=[
            Step(index=0, source="llm", tool_calls=[ToolCall("a")]),
            Step(index=1, source="llm", tool_calls=[ToolCall("b"), ToolCall("c")]),
        ])
        assert [tc.name for tc in r.tool_calls] == ["a", "b", "c"]

    def test_from_flat_tool_calls_wraps_into_single_step(self):
        r = ExecutionResult.from_flat_tool_calls([
            ToolCall("x"),
            ToolCall("y"),
        ])
        assert len(r.steps) == 1
        assert [tc.name for tc in r.steps[0].tool_calls] == ["x", "y"]

    def test_constructor_rejects_both_steps_and_tool_calls(self):
        with pytest.raises(TypeError):
            ExecutionResult(
                steps=[Step(index=0, source="llm")],
                tool_calls=[ToolCall("a")],
            )

    def test_from_dict_prefers_steps_over_legacy_tool_calls(self):
        r = ExecutionResult.from_dict({
            "steps": [{"index": 0, "source": "llm", "tool_calls": [{"name": "x"}]}],
            "tool_calls": [{"name": "ignored"}],
        })
        assert [tc.name for tc in r.tool_calls] == ["x"]

    def test_from_dict_legacy_tool_calls_only(self):
        r = ExecutionResult.from_dict({
            "tool_calls": [{"name": "a"}, {"name": "b"}],
        })
        assert len(r.steps) == 1
        assert [tc.name for tc in r.tool_calls] == ["a", "b"]

    def test_from_dict_legacy_tool_calls_rejects_missing_name(self):
        with pytest.raises(ValueError, match="name"):
            ExecutionResult.from_dict({
                "tool_calls": [{"arguments": {}}],
            })

    def test_from_dict_empty(self):
        r = ExecutionResult.from_dict({})
        assert r.steps == []
        assert r.tool_calls == []

    def test_to_dict_round_trip_with_steps(self):
        original = ExecutionResult(steps=[
            Step(index=0, source="probe", name="p", output="x"),
            Step(index=1, source="llm", tool_calls=[ToolCall("a")]),
        ])
        reloaded = ExecutionResult.from_dict(original.to_dict())
        assert len(reloaded.steps) == 2
        assert reloaded.steps[0].name == "p"
        assert reloaded.steps[0].source == "probe"
        assert reloaded.steps[1].tool_calls[0].name == "a"
