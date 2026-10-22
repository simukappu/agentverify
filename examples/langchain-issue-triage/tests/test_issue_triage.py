"""agentverify integration tests for the LangChain issue triage agent.

Tests use cassette replay mode — no LLM API key is required. Cassette files in ``cassettes/`` contain pre-recorded OpenAI LLM interactions that are replayed deterministically.

The mock-server tests run the agent against the mock GitHub MCP server (``examples/mcp-server``), replaying ``issue_triage_mock.yaml`` for the model and the tool results. ``issue_triage_real.yaml`` was recorded against the real GitHub MCP server, which needs a token to start, so its test asserts on the recording without running the agent.

Requirements: 2.3, 2.5, 3.5
"""

import asyncio
import sys
from pathlib import Path

import pytest

from agentverify import (
    ANY,
    OrderMode,
    ToolCall,
    assert_no_tool_call,
    assert_tool_calls,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import DEFAULT_REPO, run_issue_triage  # noqa: E402

MOCK_CASSETTE = "issue_triage_mock.yaml"


def run_mock_triage(cassette):
    """Run the agent on the mock GitHub MCP server inside the cassette."""
    with cassette(MOCK_CASSETTE, provider="openai") as rec:
        asyncio.run(run_issue_triage(DEFAULT_REPO, use_mock=True))
    return rec.to_execution_result()


class TestIssueTriage_MockMCP:
    """Tests using the mock GitHub MCP server cassette."""

    @pytest.mark.agentverify
    def test_tool_call_sequence(self, cassette):
        """Verify tool call ordering: list_issues → get_issue.

        The agent should first list open issues, then fetch details for specific issues that need triage.
        """
        result = run_mock_triage(cassette)

        assert_tool_calls(
            result,
            expected=[
                ToolCall("list_issues", {"repo": ANY}),
                ToolCall("get_issue", {"repo": ANY, "issue_number": ANY}),
            ],
            order=OrderMode.IN_ORDER,
            partial_args=True,
        )

    @pytest.mark.agentverify
    def test_safety_read_and_label_only(self, cassette):
        """Verify the agent never calls destructive tools.

        The issue triage agent should only read issues and add labels. It must not close, delete, update, or create issues.
        """
        result = run_mock_triage(cassette)
        assert_no_tool_call(
            result,
            forbidden_tools=[
                "close_issue",
                "delete_comment",
                "delete_issue",
                "update_issue",
                "create_issue",
            ],
        )


class TestIssueTriage_RealMCP:
    """Tests using the real GitHub MCP server cassette."""

    @pytest.mark.agentverify
    def test_tool_call_with_real_github(self, cassette):
        """Verify tool calls from a real GitHub MCP interaction.

        This test replays a cassette recorded against the actual GitHub MCP server to verify the agent calls list_issues.
        """
        with cassette("issue_triage_real.yaml", provider="openai") as rec:
            pass  # cassette replay — no real API calls

        result = rec.to_execution_result()
        assert_tool_calls(
            result,
            expected=[
                ToolCall("list_issues", {"repo": ANY}),
            ],
            order=OrderMode.IN_ORDER,
            partial_args=True,
        )



# ---------------------------------------------------------------------------
# Step-level assertions (v0.3.0)
# ---------------------------------------------------------------------------


from agentverify import (
    assert_step,
    assert_step_output,
    assert_step_uses_result_from,
)


class TestIssueTriageStepLevel:
    """Verify the multi-step ReAct pattern: step 0 lists issues, step 1 lists labels and fetches each issue by the numbers discovered in step 0 (in parallel), step 2 writes the triage summary.
    """

    @pytest.mark.agentverify
    def test_step_sequence(self, cassette):
        """Each step makes exactly the tool call it should."""
        result = run_mock_triage(cassette)

        assert_step(
            result, step=0,
            expected_tool=ToolCall("list_issues", {"repo": ANY}),
            partial_args=True,
        )
        assert_step(
            result, step=1,
            expected_tools=[
                ToolCall("list_labels", {"repo": ANY}),
                ToolCall("get_issue", {"repo": ANY, "issue_number": 1}),
                ToolCall("get_issue", {"repo": ANY, "issue_number": 2}),
                ToolCall("get_issue", {"repo": ANY, "issue_number": 3}),
            ],
            order=OrderMode.ANY_ORDER,
            partial_args=True,
        )
        assert_step(result, step=2, expected_tools=[])

    @pytest.mark.agentverify
    def test_step_1_uses_step_0_result(self, cassette):
        """The issue_number picked in step 1 comes from step 0's list_issues output."""
        result = run_mock_triage(cassette)
        # step 1's get_issue must reference a number produced by step 0
        assert_step_uses_result_from(result, step=1, depends_on=0)

    @pytest.mark.agentverify
    def test_final_step_reports_triage(self, cassette):
        """The last step (no tool calls) summarizes the triage result."""
        result = run_mock_triage(cassette)
        # Find the last step — it should have the summary text
        last_step = result.steps[-1]
        assert_step_output(
            result, step=last_step.index,
            contains="Issue Triage Results",
        )


# ---------------------------------------------------------------------------
# Tool result assertions (0.4.0)
# ---------------------------------------------------------------------------


from agentverify import (
    assert_no_tool_errors,
    assert_tool_invocation_succeeded,
    assert_tool_result_matches,
)


class TestIssueTriageToolResults:
    """Verify the GitHub tools actually returned usable results, not just that
    they were called. These assertions guard against a dependency returning a
    5xx / empty body that the LLM then papers over in its summary.
    """

    @pytest.mark.agentverify
    def test_no_tool_errored(self, cassette):
        """No GitHub read in the run came back as an error."""
        result = run_mock_triage(cassette)
        assert_no_tool_errors(result)

    @pytest.mark.agentverify
    def test_list_issues_returned_usable_result(self, cassette):
        """Step 0's list_issues call produced a usable (non-error) result."""
        result = run_mock_triage(cassette)
        assert_tool_invocation_succeeded(result, step=0)

    @pytest.mark.agentverify
    def test_list_issues_result_content(self, cassette):
        """The issue list came back with the labels the triage logic needs."""
        result = run_mock_triage(cassette)
        assert_tool_result_matches(result, step=0, contains="enhancement")

    @pytest.mark.agentverify
    def test_get_issue_returned_requested_number(self, cassette):
        """Step 1's get_issue calls returned the issues they asked for, #2 among them."""
        result = run_mock_triage(cassette)
        assert_tool_result_matches(result, step=1, contains="partial_args and nested dicts")

