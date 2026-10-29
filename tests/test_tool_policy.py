"""assert_tool_calls_allowed: tool calls checked against allowed tools and argument conditions (a small part of JSON Schema)."""

from __future__ import annotations

import math

import pytest

from agentverify import ExecutionResult, ToolPolicyError, assert_tool_calls_allowed
from agentverify._schema import violations


def _result(*calls):
    return ExecutionResult.from_dict({"steps": [{"index": 0, "tool_calls": [{"name": n, "arguments": a, **({"call_id": i} if i else {})} for n, a, i in calls]}]})


WEATHER = {"type": "object", "properties": {"city": {"type": "string", "maxLength": 10}, "days": {"type": "integer", "minimum": 1, "maximum": 7}}, "required": ["city"], "additionalProperties": False}


class TestAssertion:
    def test_passes_within_the_policy(self):
        assert_tool_calls_allowed(_result(("get_weather", {"city": "Tokyo", "days": 3}, None), ("say_hello", {"anything": 1}, None)), {"get_weather": WEATHER, "say_hello": None})
        assert_tool_calls_allowed(ExecutionResult(), {})

    def test_reports_every_violation(self):
        result = _result(("delete_file", {"path": "/etc"}, "c1"), ("get_weather", {"city": "Kuala Lumpur, MY", "days": 9, "units": "C"}, "c2"), ("get_weather", {}, None))
        with pytest.raises(ToolPolicyError) as error:
            assert_tool_calls_allowed(result, {"get_weather": WEATHER})
        message = str(error.value)
        assert message.startswith("3 tool calls outside the allowed tools")
        assert "#0 delete_file({'path': '/etc'})  [c1]\n    not an allowed tool" in message
        assert "city: longer than maxLength 10" in message and "days: 9 is above the maximum 7" in message and "units: not an allowed argument" in message
        assert "arguments: missing required 'city'" in message
        assert [v["position"] for v in error.value.violations] == [0, 1, 2] and error.value.violations[1]["call_id"] == "c2"
        with pytest.raises(ToolPolicyError, match="1 tool call outside"):
            assert_tool_calls_allowed(_result(("x", {}, None)), {})

    def test_unsupported_schemas_raise(self):
        with pytest.raises(ValueError, match=r"allowed\['f'\]: unsupported schema keyword\(s\) \['items'\]"):
            assert_tool_calls_allowed(ExecutionResult(), {"f": {"type": "array", "items": {"type": "string"}}})
        with pytest.raises(ValueError, match=r"allowed\['f'\]\.a: unsupported"):
            assert_tool_calls_allowed(ExecutionResult(), {"f": {"properties": {"a": {"oneOf": []}}}})
        with pytest.raises(ValueError, match=r"additionalProperties: a schema must be a dict"):
            assert_tool_calls_allowed(ExecutionResult(), {"f": {"additionalProperties": "no"}})
        with pytest.raises(ValueError, match="unknown type 'text'"):
            assert_tool_calls_allowed(ExecutionResult(), {"f": {"type": ["string", "text"]}})
        with pytest.raises(ValueError, match="unknown type 3"):
            assert_tool_calls_allowed(ExecutionResult(), {"f": {"type": 3}})
        with pytest.raises(ValueError, match="allowed must map"):
            assert_tool_calls_allowed(ExecutionResult(), ["f"])  # type: ignore[arg-type]
        assert_tool_calls_allowed(ExecutionResult(), {"f": {"title": "t", "description": "d", "default": 1, "examples": [], "$schema": "x", "additionalProperties": {"type": "string"}}})

    @pytest.mark.parametrize(
        "schema, message",
        [
            ({"required": "city"}, "required must be a list of names"),
            ({"enum": "abc"}, "enum must be a list"),
            ({"pattern": "("}, "pattern must be a regular expression"),
            ({"pattern": 3}, "pattern must be a regular expression"),
            ({"properties": ["x"]}, "properties must be a dict"),
            ({"minimum": "5"}, "minimum must be a number"),
            ({"maximum": True}, "maximum must be a number"),
            ({"maxLength": -1}, "maxLength must be a non-negative integer"),
            ({"minLength": 1.5}, "minLength must be a non-negative integer"),
        ],
    )
    def test_malformed_keyword_values_raise(self, schema, message):
        with pytest.raises(ValueError, match=message):
            assert_tool_calls_allowed(ExecutionResult(), {"f": schema})


class TestValues:
    """Agreed with agent-toolgate so the two do not disagree."""

    def test_booleans_are_not_numbers(self):
        assert violations(True, {"type": "integer"}) == ["arguments: expected integer, got True"]
        assert violations(False, {"type": "number"}) and not violations(True, {"type": "boolean"})

    def test_integers_take_no_floats(self):
        assert violations(3.0, {"type": "integer"}) and not violations(3, {"type": "integer"})
        assert not violations(3.5, {"type": "number"}) and not violations(3, {"type": "number"})

    def test_huge_integers_are_numbers(self):
        assert not violations(10**400, {"type": "number"}) and not violations(10**400, {"type": "integer"})

    def test_no_nan_or_infinity(self):
        assert violations(math.nan, {"type": "number"}) and violations(math.inf, {"type": "number"})
        assert violations(math.nan, {"maximum": 1}) == ["arguments: nan is not a finite number"]

    def test_enum_and_const_compare_types(self):
        assert violations(True, {"enum": [1]}) and violations(1.0, {"enum": [1]}) and not violations(1, {"enum": [1]})
        assert violations(1, {"const": True}) and not violations({"a": [1]}, {"const": {"a": [1]}})
        assert violations({"a": [1.0]}, {"const": {"a": [1]}}) and violations({"a": [1, 2]}, {"const": {"a": [1]}})
        assert violations({"b": 1}, {"const": {"a": 1}})

    def test_strings_objects_and_other_types(self):
        assert violations("ab", {"minLength": 3}) == ["arguments: shorter than minLength 3"]
        assert violations("abc", {"pattern": r"^\d+$"}) == ["arguments: does not match pattern '^\\\\d+$'"]
        assert violations(0, {"minimum": 1}) == ["arguments: 0 is below the minimum 1"]
        assert not violations(None, {"type": "null"}) and not violations([], {"type": "array"}) and violations({}, {"type": ["string", "null"]})
        assert violations({"o": {"x": 1}}, {"properties": {"o": {"properties": {"x": {"type": "string"}}}}}) == ["o.x: expected string, got 1"]
        assert violations({"n": 5}, {"additionalProperties": {"type": "string"}}) == ["n: expected string, got 5"]
