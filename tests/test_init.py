"""Tests for agentverify public API imports."""

from agentverify import (
    ANY,
    CassetteMode,
    ExecutionResult,
    LLMCassetteRecorder,
    MATCHES,
    MockLLM,
    OrderMode,
    TokenUsage,
    ToolCall,
    assert_all,
    assert_cost,
    assert_final_output,
    assert_latency,
    assert_no_tool_call,
    assert_tool_calls,
    mock_response,
)


def test_all_public_names_importable():
    import agentverify

    for name in agentverify.__all__:
        assert hasattr(agentverify, name), f"{name} missing from agentverify"


def test_public_api_types():
    assert ToolCall is not None
    assert ExecutionResult is not None
    assert TokenUsage is not None
    assert ANY is not None
    assert MATCHES is not None
    assert OrderMode is not None
    assert callable(assert_tool_calls)
    assert callable(assert_cost)
    assert callable(assert_latency)
    assert callable(assert_no_tool_call)
    assert callable(assert_final_output)
    assert callable(assert_all)
    assert LLMCassetteRecorder is not None
    assert CassetteMode is not None
    assert MockLLM is not None
    assert callable(mock_response)


def test_tool_result_assertions_public_api():
    """The tool result assertion primitives and errors are exported."""
    import agentverify

    assert callable(agentverify.assert_tool_invocation_succeeded)
    assert callable(agentverify.assert_no_tool_errors)
    assert callable(agentverify.assert_tool_result_matches)
    assert callable(agentverify.assert_retry_count)
    assert issubclass(agentverify.ToolInvocationError, AssertionError)
    assert issubclass(agentverify.ToolResultMatchError, AssertionError)
    assert issubclass(agentverify.RetryBudgetError, AssertionError)
