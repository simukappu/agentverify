"""Tests for agentverify.matchers — ANY, MATCHES, OrderMode."""

import re

from agentverify.matchers import ANY, MATCHES, OrderMode, _ANYType


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


class TestMATCHES:
    def test_matches_substring(self):
        m = MATCHES(r"/points/")
        assert m == "https://api.weather.gov/points/47,122"

    def test_no_match_returns_false(self):
        m = MATCHES(r"/forecast$")
        assert m != "https://api.weather.gov/points/47,122"

    def test_non_string_never_matches(self):
        m = MATCHES(r".*")
        assert m != 42
        assert m != None
        assert m != ["hello"]
        assert m != {"url": "x"}

    def test_accepts_precompiled_pattern(self):
        pattern = re.compile(r"^https://")
        m = MATCHES(pattern)
        assert m == "https://example.com"
        assert m != "http://example.com"

    def test_anchors_must_be_explicit(self):
        """MATCHES uses re.search, so patterns match anywhere in the string."""
        m = MATCHES(r"weather")
        assert m == "the weather is nice"
        # Use anchors to require full match
        strict = MATCHES(r"^weather$")
        assert strict != "the weather is nice"
        assert strict == "weather"

    def test_repr(self):
        assert repr(MATCHES(r"/points/")) == "MATCHES('/points/')"

    def test_usable_in_dict_comparison(self):
        expected = {"url": MATCHES(r"/points/")}
        actual = {"url": "https://api.weather.gov/points/47,122"}
        assert expected == actual

    def test_ne_operator(self):
        m = MATCHES(r"/forecast")
        assert (m != "https://example.com/points/1,2") is True
        assert (m != "https://example.com/forecast") is False


class TestOrderMode:
    def test_values(self):
        assert OrderMode.EXACT.value == "exact"
        assert OrderMode.IN_ORDER.value == "in_order"
        assert OrderMode.ANY_ORDER.value == "any_order"

    def test_from_value(self):
        assert OrderMode("exact") == OrderMode.EXACT
        assert OrderMode("in_order") == OrderMode.IN_ORDER
        assert OrderMode("any_order") == OrderMode.ANY_ORDER
