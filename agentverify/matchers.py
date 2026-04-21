"""Matchers for flexible tool call argument verification.

Provides ANY (wildcard matcher), MATCHES (regex matcher), and OrderMode
(sequence comparison modes) for use with the assertion engine.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Pattern, Union


class _ANYType:
    """Singleton matcher that equals any value.

    Used as a wildcard in expected ToolCall arguments so that
    specific argument values can be ignored during assertion.
    """

    _instance: _ANYType | None = None

    def __new__(cls) -> _ANYType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __eq__(self, other: Any) -> bool:
        return True

    def __ne__(self, other: Any) -> bool:
        return False

    def __hash__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "ANY"


ANY = _ANYType()


class MATCHES:
    """Regex matcher for string argument values.

    Equal to any string where ``re.search(pattern, value)`` finds a match.
    Non-string values never match.

    Use as a wildcard in expected ToolCall arguments::

        from agentverify import ToolCall, MATCHES
        ToolCall("http_request", {"url": MATCHES(r"/points/")})

    Args:
        pattern: A regex pattern string or a pre-compiled ``re.Pattern``.
    """

    __slots__ = ("_pattern",)

    def __init__(self, pattern: Union[str, Pattern[str]]) -> None:
        if isinstance(pattern, re.Pattern):
            self._pattern = pattern
        else:
            self._pattern = re.compile(pattern)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return False
        return self._pattern.search(other) is not None

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:  # pragma: no cover — defensive
        # Not truly hashable semantically (MATCHES(".*") == MATCHES(".*") is
        # undefined), but dict/set usage shouldn't crash.
        return hash(self._pattern.pattern)

    def __repr__(self) -> str:
        return f"MATCHES({self._pattern.pattern!r})"


class OrderMode(Enum):
    """Comparison mode for assert_tool_calls."""

    EXACT = "exact"  # Same tools, same order, same count
    IN_ORDER = "in_order"  # Subsequence match (other calls in between are OK)
    ANY_ORDER = "any_order"  # Set membership only (order doesn't matter)
