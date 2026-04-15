"""Example-based tests for the mock GitHub MCP server.

Validates: Requirements 3.2, 3.3
"""

import sys
from pathlib import Path

# Ensure the mcp-server directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from github_server import get_issue, list_issues, list_labels


class TestListIssues:
    """Tests for the list_issues tool."""

    def test_returns_list_of_issues(self):
        """list_issues returns a non-empty list of open issues."""
        result = list_issues(repo="simukappu/agentverify", state="open")
        assert isinstance(result, list)
        assert len(result) >= 1
        # Each issue should have expected fields
        for issue in result:
            assert "number" in issue
            assert "title" in issue
            assert "state" in issue
            assert issue["state"] == "open"


class TestGetIssue:
    """Tests for the get_issue tool."""

    def test_returns_details_for_known_issue(self):
        """get_issue returns full details for issue #2."""
        result = get_issue(repo="simukappu/agentverify", issue_number=2)
        assert isinstance(result, dict)
        assert result["number"] == 2
        assert "title" in result
        assert "body" in result
        assert "labels" in result
        assert "error" not in result

    def test_returns_error_for_unknown_issue(self):
        """get_issue returns an error response for a non-existent issue."""
        result = get_issue(repo="simukappu/agentverify", issue_number=999)
        assert isinstance(result, dict)
        assert "error" in result
        assert "999" in result["error"]


class TestListLabels:
    """Tests for the list_labels tool."""

    def test_returns_list_of_labels(self):
        """list_labels returns a non-empty list of labels."""
        result = list_labels(repo="simukappu/agentverify")
        assert isinstance(result, list)
        assert len(result) >= 1
        # Each label should have a name and description
        for label in result:
            assert "name" in label
            assert "description" in label
