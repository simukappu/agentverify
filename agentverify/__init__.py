"""agentverify — deterministic testing for AI agents.

Covers tool call sequences, workflow logic, and step-to-step data flow.  Public API re-exports for convenient access::

    from agentverify import (
        ToolCall, ExecutionResult, Step, TokenUsage,
        ANY, MATCHES, OrderMode,
        assert_tool_calls, assert_cost, assert_no_tool_call,
        assert_final_output, assert_latency, assert_all,
        assert_step, assert_step_output, assert_step_uses_result_from,
        step_probe,
        LLMCassetteRecorder, CassetteMode,
        MockLLM, mock_response,
    )

Framework adapters::

    from agentverify.frameworks.strands import from_strands
    from agentverify.frameworks.langchain import from_langchain
    from agentverify.frameworks.langgraph import from_langgraph
    from agentverify.frameworks.openai_agents import from_openai_agents
"""

from agentverify.assertions import (
    assert_all,
    assert_cost,
    assert_final_output,
    assert_latency,
    assert_no_tool_call,
    assert_no_tool_errors,
    assert_retry_count,
    assert_step,
    assert_step_output,
    assert_step_uses_result_from,
    assert_tool_calls,
    assert_tool_invocation_succeeded,
    assert_tool_result_matches,
)
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder
from agentverify.cassette.sanitize import DEFAULT_PATTERNS, SanitizePattern
from agentverify.errors import (
    CassetteRequestMismatchError,
    RetryBudgetError,
    StepDependencyError,
    StepIndexError,
    StepNameAmbiguousError,
    StepNameNotFoundError,
    StepOutputError,
    ToolInvocationError,
    ToolResultMatchError,
)
from agentverify.matchers import ANY, MATCHES, OrderMode
from agentverify.mocking import MockLLM, mock_response
from agentverify.models import ExecutionResult, Step, TokenUsage, ToolCall
from agentverify.probe import ProbeHandle, step_probe

__all__ = [
    # Data models
    "ToolCall",
    "ExecutionResult",
    "Step",
    "TokenUsage",
    # Matchers
    "ANY",
    "MATCHES",
    "OrderMode",
    # Assertions
    "assert_tool_calls",
    "assert_cost",
    "assert_latency",
    "assert_no_tool_call",
    "assert_final_output",
    "assert_all",
    "assert_step",
    "assert_step_output",
    "assert_step_uses_result_from",
    # Tool result assertions (tool invocation outcome)
    "assert_tool_invocation_succeeded",
    "assert_no_tool_errors",
    "assert_tool_result_matches",
    "assert_retry_count",
    # step_probe
    "step_probe",
    "ProbeHandle",
    # Cassette
    "LLMCassetteRecorder",
    "CassetteMode",
    "CassetteRequestMismatchError",
    "SanitizePattern",
    "DEFAULT_PATTERNS",
    # Mocking
    "MockLLM",
    "mock_response",
    # Step errors
    "StepIndexError",
    "StepNameNotFoundError",
    "StepNameAmbiguousError",
    "StepOutputError",
    "StepDependencyError",
    # Tool result errors
    "ToolInvocationError",
    "ToolResultMatchError",
    "RetryBudgetError",
]
