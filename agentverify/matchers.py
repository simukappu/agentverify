"""Matchers for flexible tool call argument verification.

Provides ANY (wildcard matcher) and OrderMode (sequence comparison modes)
for use with the assertion engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


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


class OrderMode(Enum):
    """Comparison mode for assert_tool_calls."""

    EXACT = "exact"  # Same tools, same order, same count
    IN_ORDER = "in_order"  # Subsequence match (other calls in between are OK)
    ANY_ORDER = "any_order"  # Set membership only (order doesn't matter)
