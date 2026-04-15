"""Tests for agentverify.matchers — ANY, OrderMode."""

from agentverify.matchers import ANY, OrderMode, _ANYType


class TestANY:
    def test_equals_any_value(self):
        assert ANY == 42
        assert ANY == "hello"
        assert ANY == [1, 2, 3]
        assert ANY == None
        assert ANY == {"key": "value"}

    def test_ne_always_false(self):
        assert not (ANY != 42)
        assert not (ANY != "hello")

    def test_singleton(self):
        a = _ANYType()
        b = _ANYType()
        assert a is b

    def test_hash(self):
        assert hash(ANY) == 0

    def test_repr(self):
        assert repr(ANY) == "ANY"

    def test_usable_in_dict_comparison(self):
        expected = {"name": "search", "q": ANY}
        actual = {"name": "search", "q": "anything"}
        assert expected == actual


class TestOrderMode:
    def test_values(self):
        assert OrderMode.EXACT.value == "exact"
        assert OrderMode.IN_ORDER.value == "in_order"
        assert OrderMode.ANY_ORDER.value == "any_order"

    def test_from_value(self):
        assert OrderMode("exact") == OrderMode.EXACT
        assert OrderMode("in_order") == OrderMode.IN_ORDER
        assert OrderMode("any_order") == OrderMode.ANY_ORDER
