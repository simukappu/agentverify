"""Property-based tests for strands_result_to_execution_result().

# Feature: examples, Property 1: Strands conversion helper field preservation

**Validates: Requirements 1.4, 4.1, 4.3**
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing converter from the parent example directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis import given, settings, strategies as st

from converter import strands_result_to_execution_result

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

tool_use_blocks = st.builds(
    lambda name, inp: {"toolUse": {"name": name, "input": inp}},
    name=tool_names,
    inp=tool_inputs,
)


# ---------------------------------------------------------------------------
# Mock AgentResult helper
# ---------------------------------------------------------------------------

class _MockState:
    """Minimal mock for result.state."""

    def __init__(self, messages: list[dict]):
        self.messages = messages


class _MockAgentResult:
    """Minimal mock for Strands AgentResult."""

    def __init__(
        self,
        tool_use_blocks: list[dict],
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        final_text: str | None = None,
    ):
        # Build state.messages — one message whose content holds all toolUse blocks
        self.state = _MockState(
            messages=[{"content": tool_use_blocks}] if tool_use_blocks else []
        )

        # metrics
        if input_tokens is not None and output_tokens is not None:
            self.metrics = {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
            }
        else:
            self.metrics = None

        # message (final output)
        if final_text is not None:
            self.message = {"content": [{"text": final_text}]}
        else:
            self.message = None


# ---------------------------------------------------------------------------
# Property 1: Strands conversion helper field preservation
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    blocks=st.lists(tool_use_blocks, min_size=0, max_size=10),
    input_tokens=st.integers(min_value=0, max_value=100_000),
    output_tokens=st.integers(min_value=0, max_value=100_000),
    final_text=st.one_of(st.none(), st.text(max_size=200)),
)
def test_strands_conversion_field_preservation(
    blocks: list[dict],
    input_tokens: int,
    output_tokens: int,
    final_text: str | None,
) -> None:
    """Property 1: Strands conversion helper field preservation

    For any valid Strands AgentResult structure, the converted
    ExecutionResult preserves all fields faithfully.

    **Validates: Requirements 1.4, 4.1, 4.3**
    """
    mock_result = _MockAgentResult(
        tool_use_blocks=blocks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        final_text=final_text,
    )

    execution_result = strands_result_to_execution_result(mock_result)

    # tool_calls length matches the number of toolUse blocks
    assert len(execution_result.tool_calls) == len(blocks)

    # Each tool_calls[i].name and arguments match the original data
    for i, block in enumerate(blocks):
        tool_use = block["toolUse"]
        assert execution_result.tool_calls[i].name == tool_use["name"]
        assert execution_result.tool_calls[i].arguments == tool_use["input"]

    # token_usage matches the original metrics
    assert execution_result.token_usage is not None
    assert execution_result.token_usage.input_tokens == input_tokens
    assert execution_result.token_usage.output_tokens == output_tokens

    # final_output matches the original text
    assert execution_result.final_output == final_text
