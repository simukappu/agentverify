"""agentverify — deterministic testing for AI agent tool call sequences.

Public API re-exports for convenient access::

    from agentverify import (
        ToolCall, ExecutionResult, TokenUsage,
        ANY, OrderMode,
        assert_tool_calls, assert_cost, assert_no_tool_call,
        assert_final_output, assert_all,
        LLMCassetteRecorder, CassetteMode,
    )

Framework adapters (no converter needed)::

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
    assert_tool_calls,
)
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder
from agentverify.cassette.sanitize import DEFAULT_PATTERNS, SanitizePattern
from agentverify.errors import CassetteRequestMismatchError
from agentverify.matchers import ANY, OrderMode
from agentverify.mocking import MockLLM, mock_response
from agentverify.models import ExecutionResult, TokenUsage, ToolCall

__all__ = [
    # Data models
    "ToolCall",
    "ExecutionResult",
    "TokenUsage",
    # Matchers
    "ANY",
    "OrderMode",
    # Assertions
    "assert_tool_calls",
    "assert_cost",
    "assert_latency",
    "assert_no_tool_call",
    "assert_final_output",
    "assert_all",
    # Cassette
    "LLMCassetteRecorder",
    "CassetteMode",
    "CassetteRequestMismatchError",
    "SanitizePattern",
    "DEFAULT_PATTERNS",
    # Mocking
    "MockLLM",
    "mock_response",
]
