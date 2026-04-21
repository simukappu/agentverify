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
        assert d["tool_calls"] == [{"name": "x", "arguments": {"k": "v"}, "result": "r"}]
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
