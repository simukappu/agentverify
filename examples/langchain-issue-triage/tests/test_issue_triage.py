"""agentverify integration tests for the LangChain issue triage agent.

Tests use cassette replay mode — no LLM API key is required.
Cassette files in ``cassettes/`` contain pre-recorded OpenAI LLM
interactions that are replayed deterministically.

Requirements: 2.3, 2.5, 3.5
"""

import pytest

from agentverify import (
    ANY,
    OrderMode,
    ToolCall,
    assert_no_tool_call,
    assert_tool_calls,
)


class TestIssueTriage_MockMCP:
    """Tests using the mock GitHub MCP server cassette."""

    @pytest.mark.agentverify
    def test_tool_call_sequence(self, cassette):
        """Verify tool call ordering: list_issues → get_issue.

        The agent should first list open issues, then fetch details
        for specific issues that need triage.
        """
        with cassette("issue_triage_mock.yaml", provider="openai") as rec:
            pass  # cassette replay — no real API calls

        result = rec.to_execution_result()

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

        The issue triage agent should only read issues and add labels.
        It must not close, delete, update, or create issues.
        """
        with cassette("issue_triage_mock.yaml", provider="openai") as rec:
            pass  # cassette replay

        result = rec.to_execution_result()
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

        This test replays a cassette recorded against the actual
        GitHub MCP server to verify the agent calls list_issues.
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
    """Verify the multi-step ReAct pattern:
    step 0 lists issues, step 1 fetches issue details using the number
    discovered in step 0, step 2 lists labels, step 3 produces the
    final triage summary.
    """

    @pytest.mark.agentverify
    def test_step_sequence(self, cassette):
        """Each step makes exactly the tool call it should."""
        with cassette("issue_triage_mock.yaml", provider="openai") as rec:
            pass

        result = rec.to_execution_result()

        assert_step(
            result, step=0,
            expected_tool=ToolCall("list_issues", {"repo": ANY}),
            partial_args=True,
        )
        assert_step(
            result, step=1,
            expected_tool=ToolCall(
                "get_issue", {"repo": ANY, "issue_number": ANY},
            ),
            partial_args=True,
        )
        assert_step(
            result, step=2,
            expected_tool=ToolCall("list_labels", {"repo": ANY}),
            partial_args=True,
        )

    @pytest.mark.agentverify
    def test_step_1_uses_step_0_result(self, cassette):
        """The issue_number picked in step 1 comes from step 0's list_issues output."""
        with cassette("issue_triage_mock.yaml", provider="openai") as rec:
            pass

        result = rec.to_execution_result()
        # step 1's get_issue must reference a number produced by step 0
        assert_step_uses_result_from(result, step=1, depends_on=0)

    @pytest.mark.agentverify
    def test_final_step_reports_triage(self, cassette):
        """The last step (no tool calls) summarizes the triage result."""
        with cassette("issue_triage_mock.yaml", provider="openai") as rec:
            pass

        result = rec.to_execution_result()
        # Find the last step — it should have the summary text
        last_step = result.steps[-1]
        assert_step_output(
            result, step=last_step.index,
            contains="Issue Triage Results",
        )
