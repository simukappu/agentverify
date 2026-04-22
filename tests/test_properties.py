"""Property-based tests for agentverify using Hypothesis.

Tests Properties 1-12 as defined in the design document.
Each test is annotated with the property number and description.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentverify.assertions import (
    assert_all,
    assert_cost,
    assert_no_tool_call,
    assert_tool_calls,
)
from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.adapters.openai import OpenAIAdapter
from agentverify.cassette.io import load_cassette, save_cassette
from agentverify.errors import (
    AgentVerifyError,
    CostBudgetError,
    MultipleAssertionError,
    SafetyRuleViolationError,
    ToolCallSequenceError,
)
from agentverify.matchers import ANY, OrderMode
from agentverify.models import ExecutionResult, TokenUsage, ToolCall

# ---------------------------------------------------------------------------
# Hypothesis strategies (from design doc)
# ---------------------------------------------------------------------------

tool_names = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
)

tool_arguments = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.one_of(
        st.text(),
        st.integers(),
        st.floats(allow_nan=False),
        st.booleans(),
        st.none(),
    ),
    max_size=10,
)

tool_calls = st.builds(
    ToolCall,
    name=tool_names,
    arguments=tool_arguments,
    result=st.one_of(st.none(), st.text()),
)

token_usages = st.builds(
    TokenUsage,
    input_tokens=st.integers(min_value=0, max_value=100000),
    output_tokens=st.integers(min_value=0, max_value=100000),
)

execution_results = st.builds(
    ExecutionResult,
    tool_calls=st.lists(tool_calls, max_size=20),
    token_usage=st.one_of(st.none(), token_usages),
    total_cost_usd=st.one_of(
        st.none(), st.floats(min_value=0, max_value=1000, allow_nan=False)
    ),
    final_output=st.one_of(st.none(), st.text()),
)

# Strategy for arbitrary Python values (used by Property 5)
any_python_values = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(),
    st.binary(),
    st.lists(st.integers()),
    st.dictionaries(st.text(min_size=1, max_size=5), st.integers()),
)

# Strategy for dict keys/values used by Property 6
partial_match_keys = st.text(min_size=1, max_size=10)
partial_match_values = st.one_of(
    st.text(), st.integers(), st.floats(allow_nan=False), st.booleans()
)


# ---------------------------------------------------------------------------
# Property 1: ExecutionResult construction preserves all fields
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 1: For any valid ToolCall and
# ExecutionResult, all fields are preserved after construction.
# TokenUsage.total_tokens == input_tokens + output_tokens.
# Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5


@given(
    name=tool_names,
    arguments=tool_arguments,
    result=st.one_of(st.none(), st.text()),
    input_tokens=st.integers(min_value=0, max_value=100000),
    output_tokens=st.integers(min_value=0, max_value=100000),
    total_cost_usd=st.one_of(
        st.none(), st.floats(min_value=0, max_value=1000, allow_nan=False)
    ),
    final_output=st.one_of(st.none(), st.text()),
)
@settings(max_examples=100)
def test_property_1_construction_preserves_fields(
    name,
    arguments,
    result,
    input_tokens,
    output_tokens,
    total_cost_usd,
    final_output,
):
    """**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**"""
    # Tool_Call preserves fields
    tc = ToolCall(name=name, arguments=arguments, result=result)
    assert tc.name == name
    assert tc.arguments == arguments
    assert tc.result == result

    # TokenUsage preserves fields and total_tokens is correct
    tu = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    assert tu.input_tokens == input_tokens
    assert tu.output_tokens == output_tokens
    assert tu.total_tokens == input_tokens + output_tokens

    # ExecutionResult preserves fields
    er = ExecutionResult(
        tool_calls=[tc],
        token_usage=tu,
        total_cost_usd=total_cost_usd,
        final_output=final_output,
    )
    assert er.tool_calls == [tc]
    assert er.token_usage is tu
    assert er.total_cost_usd == total_cost_usd
    assert er.final_output == final_output


# ---------------------------------------------------------------------------
# Property 2: ExecutionResult serialization round-trip
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 2: For any valid ExecutionResult,
# to_dict() → from_dict() round-trip preserves equality.
# Same for to_json() → from_json().
# Validates: Requirements 1.6


@given(er=execution_results)
@settings(max_examples=100)
def test_property_2_serialization_round_trip(er):
    """**Validates: Requirements 1.6**"""
    # to_dict → from_dict round-trip
    d = er.to_dict()
    restored_from_dict = ExecutionResult.from_dict(d)
    assert restored_from_dict.tool_calls == er.tool_calls
    assert restored_from_dict.final_output == er.final_output
    assert restored_from_dict.total_cost_usd == er.total_cost_usd
    if er.token_usage is not None:
        assert restored_from_dict.token_usage is not None
        assert restored_from_dict.token_usage.input_tokens == er.token_usage.input_tokens
        assert restored_from_dict.token_usage.output_tokens == er.token_usage.output_tokens
    else:
        assert restored_from_dict.token_usage is None

    # to_json → from_json round-trip
    j = er.to_json()
    restored_from_json = ExecutionResult.from_json(j)
    assert restored_from_json.tool_calls == er.tool_calls
    assert restored_from_json.final_output == er.final_output
    assert restored_from_json.total_cost_usd == er.total_cost_usd
    if er.token_usage is not None:
        assert restored_from_json.token_usage is not None
        assert restored_from_json.token_usage.input_tokens == er.token_usage.input_tokens
        assert restored_from_json.token_usage.output_tokens == er.token_usage.output_tokens
    else:
        assert restored_from_json.token_usage is None


# ---------------------------------------------------------------------------
# Property 3: ExactOrder sequence assertion correctness
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 3: For any two Tool_Call sequences,
# assert_tool_calls(order=EXACT) succeeds iff sequences are identical,
# fails otherwise.
# Validates: Requirements 2.1


@given(tcs=st.lists(tool_calls, min_size=0, max_size=10))
@settings(max_examples=100)
def test_property_3_exact_order_identical_succeeds(tcs):
    """**Validates: Requirements 2.1** — identical sequences pass EXACT."""
    result = ExecutionResult(tool_calls=list(tcs))
    # Should not raise
    assert_tool_calls(result, expected=list(tcs), order=OrderMode.EXACT)


@given(
    tcs1=st.lists(tool_calls, min_size=1, max_size=10),
    tcs2=st.lists(tool_calls, min_size=1, max_size=10),
)
@settings(max_examples=100)
def test_property_3_exact_order_different_fails(tcs1, tcs2):
    """**Validates: Requirements 2.1** — different sequences fail EXACT."""
    # assert_tool_calls compares name and arguments only (not result),
    # so we must ensure the sequences differ in those dimensions.
    def _key(tc):
        return (tc.name, tc.arguments)

    assume([_key(t) for t in tcs1] != [_key(t) for t in tcs2])
    result = ExecutionResult(tool_calls=tcs1)
    try:
        assert_tool_calls(result, expected=tcs2, order=OrderMode.EXACT)
        # If it didn't raise, the sequences must actually be equal
        assert False, "Expected ToolCallSequenceError"
    except ToolCallSequenceError:
        pass


# ---------------------------------------------------------------------------
# Property 4: AnyOrder sequence assertion permutation invariance
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 4: For any Tool_Call sequence and any
# permutation, assert_tool_calls(order=ANY_ORDER) always succeeds.
# Validates: Requirements 2.3


@given(tcs=st.lists(tool_calls, min_size=0, max_size=10))
@settings(max_examples=100)
def test_property_4_any_order_permutation_succeeds(tcs):
    """**Validates: Requirements 2.3** — any permutation passes ANY_ORDER."""
    shuffled = list(tcs)
    random.shuffle(shuffled)
    result = ExecutionResult(tool_calls=list(tcs))
    # ANY_ORDER with a permutation of the same elements should always succeed
    assert_tool_calls(result, expected=shuffled, order=OrderMode.ANY_ORDER)


# ---------------------------------------------------------------------------
# Property 4b: InOrder sequence assertion subsequence verification
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 4b: For any Tool_Call sequence S and
# subsequence T, assert_tool_calls(result_with_S, expected=T,
# order=IN_ORDER) always succeeds. If T is NOT a subsequence, it fails.
# Validates: Requirements 2.2


@given(
    tcs=st.lists(tool_calls, min_size=1, max_size=10),
    data=st.data(),
)
@settings(max_examples=100)
def test_property_4b_in_order_subsequence_succeeds(tcs, data):
    """**Validates: Requirements 2.2** — subsequences pass IN_ORDER."""
    # Generate a random subsequence by picking random indices in order
    indices = sorted(
        data.draw(
            st.lists(
                st.integers(min_value=0, max_value=len(tcs) - 1),
                max_size=len(tcs),
                unique=True,
            )
        )
    )
    subsequence = [tcs[i] for i in indices]
    result = ExecutionResult(tool_calls=list(tcs))
    assert_tool_calls(result, expected=subsequence, order=OrderMode.IN_ORDER)


@given(
    tcs=st.lists(tool_calls, min_size=0, max_size=5),
    extra_tc=tool_calls,
)
@settings(max_examples=100)
def test_property_4b_in_order_non_subsequence_fails(tcs, extra_tc):
    """**Validates: Requirements 2.2** — non-subsequences fail IN_ORDER."""
    # extra_tc appended to expected but not in actual → not a subsequence
    assume(extra_tc not in tcs)
    result = ExecutionResult(tool_calls=list(tcs))
    expected = list(tcs) + [extra_tc]
    try:
        assert_tool_calls(result, expected=expected, order=OrderMode.IN_ORDER)
        assert False, "Expected ToolCallSequenceError"
    except ToolCallSequenceError:
        pass


# ---------------------------------------------------------------------------
# Property 5: ANY matcher universality
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 5: For any Python value,
# ANY == value is always True.
# Validates: Requirements 2.3


@given(value=any_python_values)
@settings(max_examples=100)
def test_property_5_any_matches_everything(value):
    """**Validates: Requirements 2.3**"""
    assert ANY == value
    assert not (ANY != value)


# ---------------------------------------------------------------------------
# Property 6: Partial argument matching verifies only specified keys
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 6: For any ToolCall with arguments that
# are a superset of expected arguments, assert_tool_calls with
# partial_args=True succeeds. Missing keys or different values → fails.
# Validates: Requirements 2.4


@given(
    expected=st.dictionaries(partial_match_keys, partial_match_values, min_size=0, max_size=5),
    extra=st.dictionaries(partial_match_keys, partial_match_values, min_size=0, max_size=5),
)
@settings(max_examples=100)
def test_property_6_partial_args_superset_succeeds(expected, extra):
    """**Validates: Requirements 2.4** — superset of expected keys matches with partial_args."""
    # actual contains all keys of expected (with same values) plus extras
    actual_args = {**extra, **expected}  # expected overwrites extra for shared keys
    result = ExecutionResult(tool_calls=[ToolCall(name="test_tool", arguments=actual_args)])
    assert_tool_calls(
        result,
        expected=[ToolCall(name="test_tool", arguments=expected)],
        partial_args=True,
    )


@given(
    expected=st.dictionaries(partial_match_keys, partial_match_values, min_size=1, max_size=5),
)
@settings(max_examples=100)
def test_property_6_partial_args_missing_key_fails(expected):
    """**Validates: Requirements 2.4** — missing key fails with partial_args."""
    # Remove one key from actual
    key_to_remove = random.choice(list(expected.keys()))
    actual_args = {k: v for k, v in expected.items() if k != key_to_remove}
    result = ExecutionResult(tool_calls=[ToolCall(name="test_tool", arguments=actual_args)])
    try:
        assert_tool_calls(
            result,
            expected=[ToolCall(name="test_tool", arguments=expected)],
            partial_args=True,
        )
        assert False, "Expected ToolCallSequenceError"
    except ToolCallSequenceError:
        pass


@given(
    expected=st.dictionaries(partial_match_keys, st.integers(), min_size=1, max_size=5),
)
@settings(max_examples=100)
def test_property_6_partial_args_different_value_fails(expected):
    """**Validates: Requirements 2.4** — different value fails with partial_args."""
    # Change one value in actual
    key_to_change = random.choice(list(expected.keys()))
    actual_args = dict(expected)
    # Ensure the value is actually different
    actual_args[key_to_change] = expected[key_to_change] + 1
    result = ExecutionResult(tool_calls=[ToolCall(name="test_tool", arguments=actual_args)])
    try:
        assert_tool_calls(
            result,
            expected=[ToolCall(name="test_tool", arguments=expected)],
            partial_args=True,
        )
        assert False, "Expected ToolCallSequenceError"
    except ToolCallSequenceError:
        pass


# ---------------------------------------------------------------------------
# Property 7: Budget assertion correctness
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 7: For any ExecutionResult and budget
# limit, assert_cost() succeeds iff actual <= limit, fails otherwise.
# Validates: Requirements 3.1, 3.2


@given(
    input_tokens=st.integers(min_value=0, max_value=50000),
    output_tokens=st.integers(min_value=0, max_value=50000),
    max_tokens=st.integers(min_value=0, max_value=200000),
)
@settings(max_examples=100)
def test_property_7_token_budget(input_tokens, output_tokens, max_tokens):
    """**Validates: Requirements 3.1**"""
    tu = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    er = ExecutionResult(token_usage=tu)
    actual_total = tu.total_tokens
    if actual_total <= max_tokens:
        # Should not raise
        assert_cost(er, max_tokens=max_tokens)
    else:
        try:
            assert_cost(er, max_tokens=max_tokens)
            assert False, "Expected CostBudgetError"
        except CostBudgetError as e:
            assert e.actual == actual_total
            assert e.limit == max_tokens


@given(
    cost=st.floats(min_value=0, max_value=500, allow_nan=False),
    max_cost=st.floats(min_value=0, max_value=1000, allow_nan=False),
)
@settings(max_examples=100)
def test_property_7_cost_budget(cost, max_cost):
    """**Validates: Requirements 3.2**"""
    er = ExecutionResult(total_cost_usd=cost)
    if cost <= max_cost:
        assert_cost(er, max_cost_usd=max_cost)
    else:
        try:
            assert_cost(er, max_cost_usd=max_cost)
            assert False, "Expected CostBudgetError"
        except CostBudgetError as e:
            assert e.actual == cost
            assert e.limit == max_cost


# ---------------------------------------------------------------------------
# Property 8: Forbidden tool detection reports all violations
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 8: For any Tool_Call sequence and
# forbidden tool list, assert_no_tool_call() succeeds iff intersection
# is empty. Failure reports count == actual forbidden call count.
# Validates: Requirements 4.1, 4.2, 4.3


@given(
    tcs=st.lists(tool_calls, min_size=0, max_size=10),
    forbidden=st.lists(tool_names, min_size=0, max_size=5),
)
@settings(max_examples=100)
def test_property_8_forbidden_tool_detection(tcs, forbidden):
    """**Validates: Requirements 4.1, 4.2, 4.3**"""
    forbidden_set = set(forbidden)
    er = ExecutionResult(tool_calls=tcs)
    expected_violation_count = sum(1 for tc in tcs if tc.name in forbidden_set)

    if expected_violation_count == 0:
        # Should not raise
        assert_no_tool_call(er, forbidden_tools=forbidden)
    else:
        try:
            assert_no_tool_call(er, forbidden_tools=forbidden)
            assert False, "Expected SafetyRuleViolationError"
        except SafetyRuleViolationError as e:
            assert len(e.violations) == expected_violation_count


# ---------------------------------------------------------------------------
# Property 9: Cassette record/replay round-trip
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 9: For any list of
# NormalizedRequest/NormalizedResponse pairs, save to cassette YAML then
# load back produces identical interactions.
# Validates: Requirements 5.5

# Strategy for NormalizedRequest
# Use printable text to avoid YAML encoding round-trip issues with control chars
yaml_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), min_codepoint=32, max_codepoint=126),
    max_size=50,
)

normalized_messages = st.lists(
    st.fixed_dictionaries(
        {"role": st.sampled_from(["user", "assistant", "system"]), "content": yaml_safe_text}
    ),
    min_size=1,
    max_size=3,
)

normalized_requests = st.builds(
    NormalizedRequest,
    messages=normalized_messages,
    model=yaml_safe_text.filter(lambda s: len(s) > 0),
    tools=st.just(None),
    parameters=st.just({}),
)

normalized_tool_calls_list = st.one_of(
    st.none(),
    st.lists(
        st.fixed_dictionaries(
            {
                "name": yaml_safe_text.filter(lambda s: len(s) > 0),
                "arguments": yaml_safe_text,
            }
        ),
        min_size=1,
        max_size=3,
    ),
)

normalized_responses = st.builds(
    NormalizedResponse,
    content=st.one_of(st.none(), yaml_safe_text),
    tool_calls=normalized_tool_calls_list,
    token_usage=st.one_of(st.none(), token_usages),
)

normalized_interactions = st.lists(
    st.tuples(normalized_requests, normalized_responses),
    min_size=0,
    max_size=5,
)


@given(interactions=normalized_interactions)
@settings(max_examples=100)
def test_property_9_cassette_round_trip(interactions):
    """**Validates: Requirements 5.5**"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cassette_path = Path(tmpdir) / "test_cassette.yaml"
        save_cassette(cassette_path, interactions, provider="test", model="test-model")
        _metadata, loaded = load_cassette(cassette_path)

        assert len(loaded) == len(interactions)
        for (orig_req, orig_resp), (loaded_req, loaded_resp) in zip(
            interactions, loaded
        ):
            # Request fields
            assert loaded_req.messages == orig_req.messages
            assert loaded_req.model == orig_req.model
            assert loaded_req.tools == orig_req.tools

            # Response fields
            assert loaded_resp.content == orig_resp.content
            assert loaded_resp.tool_calls == orig_resp.tool_calls
            if orig_resp.token_usage is not None:
                assert loaded_resp.token_usage is not None
                assert loaded_resp.token_usage.input_tokens == orig_resp.token_usage.input_tokens
                assert loaded_resp.token_usage.output_tokens == orig_resp.token_usage.output_tokens
            else:
                assert loaded_resp.token_usage is None


# ---------------------------------------------------------------------------
# Property 10: Provider adapter normalize round-trip (OpenAI)
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 10: For any NormalizedResponse, OpenAI
# adapter's denormalize → normalize round-trip preserves content,
# tool_calls, and token_usage.
# Validates: Requirements 5.7, 5.8

# Strategy for NormalizedResponse suitable for OpenAI adapter round-trip.
# tool_call arguments must be valid JSON strings for OpenAI format.
openai_tool_calls_list = st.one_of(
    st.none(),
    st.lists(
        st.fixed_dictionaries(
            {
                "name": tool_names,
                "arguments": st.dictionaries(
                    st.text(min_size=1, max_size=10),
                    st.one_of(st.text(max_size=20), st.integers()),
                    max_size=3,
                ).map(lambda d: json.dumps(d)),
            }
        ),
        min_size=1,
        max_size=3,
    ),
)

openai_normalized_responses = st.builds(
    NormalizedResponse,
    content=st.one_of(st.none(), st.text(max_size=50)),
    tool_calls=openai_tool_calls_list,
    token_usage=st.one_of(st.none(), token_usages),
)


@given(norm_resp=openai_normalized_responses)
@settings(max_examples=100)
def test_property_10_openai_adapter_round_trip(norm_resp):
    """**Validates: Requirements 5.7, 5.8**"""
    adapter = OpenAIAdapter()

    # denormalize → normalize round-trip
    raw = adapter.denormalize_response(norm_resp)
    restored = adapter.normalize_response(raw)

    # Content preserved
    assert restored.content == norm_resp.content

    # Tool calls preserved
    if norm_resp.tool_calls is not None:
        assert restored.tool_calls is not None
        assert len(restored.tool_calls) == len(norm_resp.tool_calls)
        for orig_tc, restored_tc in zip(norm_resp.tool_calls, restored.tool_calls):
            assert restored_tc["name"] == orig_tc["name"]
            assert restored_tc["arguments"] == orig_tc["arguments"]
    else:
        assert restored.tool_calls is None

    # Token usage preserved
    if norm_resp.token_usage is not None:
        assert restored.token_usage is not None
        assert restored.token_usage.input_tokens == norm_resp.token_usage.input_tokens
        assert restored.token_usage.output_tokens == norm_resp.token_usage.output_tokens
    else:
        assert restored.token_usage is None


# ---------------------------------------------------------------------------
# Property 11: First mismatch index correctness
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 11: For any two different Tool_Call
# sequences, ToolCallSequenceError.first_mismatch_index equals the first
# position where they differ.
# Validates: Requirements 7.2


@given(
    tcs1=st.lists(tool_calls, min_size=1, max_size=10),
    tcs2=st.lists(tool_calls, min_size=1, max_size=10),
)
@settings(max_examples=100)
def test_property_11_first_mismatch_index(tcs1, tcs2):
    """**Validates: Requirements 7.2**"""

    # assert_tool_calls compares name + arguments only (not result),
    # so we define "match" the same way the assertion engine does.
    def _matches(a: ToolCall, b: ToolCall) -> bool:
        return a.name == b.name and a.arguments == b.arguments

    # Compute whether the sequences are "same" from the assertion engine's perspective
    same_length = len(tcs1) == len(tcs2)
    all_match = same_length and all(_matches(a, b) for a, b in zip(tcs1, tcs2))
    assume(not all_match)

    # Compute expected first mismatch index
    expected_mismatch = min(len(tcs1), len(tcs2))
    for i in range(min(len(tcs1), len(tcs2))):
        if not _matches(tcs1[i], tcs2[i]):
            expected_mismatch = i
            break

    result = ExecutionResult(tool_calls=tcs1)
    try:
        assert_tool_calls(result, expected=tcs2, order=OrderMode.EXACT)
        assert False, "Expected ToolCallSequenceError"
    except ToolCallSequenceError as e:
        assert e.first_mismatch_index == expected_mismatch


# ---------------------------------------------------------------------------
# Property 12: assert_all collects all failures
# ---------------------------------------------------------------------------
# Feature: agentverify, Property 12: For N assertion callables where M
# fail, assert_all() raises MultipleAssertionError with exactly M errors.
# Validates: Requirements 7.3


@given(
    n_pass=st.integers(min_value=0, max_value=5),
    n_fail=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100)
def test_property_12_assert_all_collects_failures(n_pass, n_fail):
    """**Validates: Requirements 7.3**"""
    er = ExecutionResult()

    def passing_assertion(r: ExecutionResult) -> None:
        pass

    def failing_assertion(r: ExecutionResult) -> None:
        raise AgentVerifyError("deliberate failure")

    assertions = (
        [passing_assertion] * n_pass + [failing_assertion] * n_fail
    )
    # Shuffle to ensure order doesn't matter
    random.shuffle(assertions)

    if n_fail == 0:
        # Should not raise
        assert_all(er, *assertions)
    else:
        try:
            assert_all(er, *assertions)
            assert False, "Expected MultipleAssertionError"
        except MultipleAssertionError as e:
            assert len(e.errors) == n_fail



# ---------------------------------------------------------------------------
# Step-level properties (v0.3.0)
# ---------------------------------------------------------------------------


from agentverify import Step


@given(
    steps_data=st.lists(
        st.lists(
            st.builds(
                ToolCall,
                name=st.text(min_size=1, max_size=10),
                arguments=st.dictionaries(
                    keys=st.text(min_size=1, max_size=5),
                    values=st.integers(),
                    max_size=3,
                ),
            ),
            max_size=5,
        ),
        max_size=6,
    ),
)
@settings(max_examples=50, deadline=None)
def test_property_flatten_steps_equals_tool_calls(steps_data):
    """Flattening steps must equal the derived tool_calls property."""
    steps = [
        Step(index=i, source="llm", tool_calls=tcs)
        for i, tcs in enumerate(steps_data)
    ]
    result = ExecutionResult(steps=steps)
    expected_flat = [tc for step_tcs in steps_data for tc in step_tcs]
    assert list(result.tool_calls) == expected_flat


@given(
    steps_data=st.lists(
        st.lists(
            st.text(min_size=1, max_size=5),
            max_size=3,
        ),
        max_size=5,
    ),
)
@settings(max_examples=30, deadline=None)
def test_property_to_dict_from_dict_round_trip_with_steps(steps_data):
    """Step round-trip via to_dict/from_dict preserves structure."""
    steps = [
        Step(
            index=i,
            source="llm",
            tool_calls=[ToolCall(name=n) for n in names],
        )
        for i, names in enumerate(steps_data)
    ]
    original = ExecutionResult(steps=steps)
    reloaded = ExecutionResult.from_dict(original.to_dict())
    assert len(reloaded.steps) == len(steps)
    for a, b in zip(reloaded.steps, steps):
        assert a.index == b.index
        assert a.source == b.source
        assert [tc.name for tc in a.tool_calls] == [tc.name for tc in b.tool_calls]
