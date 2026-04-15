"""Mock GitHub MCP Server for agentverify examples.

A FastMCP-based mock GitHub MCP server that returns hardcoded issue data.
Used by the LangChain Issue Triage Agent example to enable testing
without a real GitHub token.

The target repository can be configured via the MOCK_REPO environment
variable (default: simukappu/agentverify).
"""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mock-github")

# Target repository (configurable via environment variable, default: simukappu/agentverify)
TARGET_REPO = os.environ.get("MOCK_REPO", "simukappu/agentverify")

# Hardcoded mock issue data
MOCK_ISSUES = [
    {
        "number": 1,
        "title": "Add async support for cassette recorder",
        "body": (
            "The LLMCassetteRecorder currently only supports synchronous usage. "
            "It would be great to have native async support for testing async agents "
            "and tools that use asyncio."
        ),
        "state": "open",
        "labels": [{"name": "enhancement"}],
        "user": {"login": "contributor1"},
        "created_at": "2025-01-15T10:00:00Z",
    },
    {
        "number": 2,
        "title": "assert_tool_calls fails with partial_args and nested dicts",
        "body": (
            "When using partial_args=True with nested dictionary arguments, "
            "the assertion incorrectly fails even though the expected keys match. "
            "Reproduction: use ToolCall with nested dict arguments and partial_args=True."
        ),
        "state": "open",
        "labels": [{"name": "bug"}],
        "user": {"login": "contributor2"},
        "created_at": "2025-01-16T14:30:00Z",
    },
    {
        "number": 3,
        "title": "Question: How to test agents with streaming responses?",
        "body": (
            "I'm trying to test an agent that uses streaming responses from the LLM. "
            "How should I structure the ExecutionResult when the response is streamed? "
            "Is there a recommended pattern for this?"
        ),
        "state": "open",
        "labels": [{"name": "question"}],
        "user": {"login": "contributor3"},
        "created_at": "2025-01-17T09:15:00Z",
    },
]

MOCK_LABELS = [
    {"name": "bug", "description": "Something isn't working"},
    {"name": "enhancement", "description": "New feature or request"},
    {"name": "question", "description": "Further information is requested"},
    {"name": "priority:high", "description": "High priority"},
    {"name": "priority:low", "description": "Low priority"},
]


@mcp.tool()
def list_issues(repo: str, state: str = "open") -> list[dict]:
    """Retrieve the list of issues for a repository."""
    return [i for i in MOCK_ISSUES if i["state"] == state]


@mcp.tool()
def get_issue(repo: str, issue_number: int) -> dict:
    """Retrieve the details of a specified issue."""
    for issue in MOCK_ISSUES:
        if issue["number"] == issue_number:
            return issue
    return {"error": f"Issue #{issue_number} not found"}


@mcp.tool()
def list_labels(repo: str) -> list[dict]:
    """Retrieve the list of labels for a repository."""
    return MOCK_LABELS


if __name__ == "__main__":
    mcp.run(transport="stdio")
