"""Property-based tests for langchain_result_to_execution_result().

# Feature: examples, Property 2: LangChain conversion helper field preservation

**Validates: Requirements 2.4, 4.2, 4.3**
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Mock langchain_core.messages before importing converter
# ---------------------------------------------------------------------------
# The converter module imports AIMessage from langchain_core.messages.
# We provide a lightweight mock so the test runs without installing langchain.

_mock_messages_module = types.ModuleType("langchain_core.messages")


class _MockAIMessage:
    """Minimal stand-in for langchain_core.messages.AIMessage."""

    def __init__(self, usage_metadata: dict | None = None):
        self.usage_metadata = usage_metadata


_mock_messages_module.AIMessage = _MockAIMessage  # type: ignore[attr-defined]

# Register the mock modules so `from langchain_core.messages import AIMessage` works
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
sys.modules["langchain_core.messages"] = _mock_messages_module

# Allow importing converter from the parent example directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis import given, settings, strategies as st

from converter import langchain_result_to_execution_result

# ---------------------------------------------------------------------------
# Hypothesis strategies (from design doc)
# ---------------------------------------------------------------------------

tool_names = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
)

tool_inputs = st.dictionaries(
    keys=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(whitelist_categories=("L",)),
    ),
    values=st.one_of(
        st.text(max_size=50),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
    ),
    max_size=5,
)

agent_actions = st.builds(
    lambda tool, tool_input: (
        type("AgentAction", (), {"tool": tool, "tool_input": tool_input})(),
        "observation_result",
    ),
    tool=tool_names,
    tool_input=tool_inputs,
)


# ---------------------------------------------------------------------------
# Property 2: LangChain conversion helper field preservation
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    steps=st.lists(agent_actions, min_size=0, max_size=10),
    output_text=st.one_of(st.none(), st.text(max_size=200)),
)
def test_langchain_conversion_field_preservation(
    steps: list[tuple],
    output_text: str | None,
) -> None:
    """Property 2: LangChain conversion helper field preservation

    For any valid LangChain AgentExecutor output structure, the converted
    ExecutionResult preserves all fields faithfully.

    **Validates: Requirements 2.4, 4.2, 4.3**
    """
    result_dict: dict = {
        "intermediate_steps": steps,
        "output": output_text,
    }

    execution_result = langchain_result_to_execution_result(result_dict)

    # tool_calls length matches intermediate_steps length
    assert len(execution_result.tool_calls) == len(steps)

    # Each tool_calls[i].name and arguments match the original data
    for i, (action, _obs) in enumerate(steps):
        assert execution_result.tool_calls[i].name == action.tool
        assert execution_result.tool_calls[i].arguments == action.tool_input

    # final_output matches the original output
    assert execution_result.final_output == output_text
