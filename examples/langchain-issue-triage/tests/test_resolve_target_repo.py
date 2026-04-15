# Feature: examples, Property 4: Issue triage agent target repository resolution priority
"""Property-based tests for resolve_target_repo().

**Validates: Requirements 2.2**

Tests the priority resolution logic:
1. CLI arg takes priority when specified
2. Env var ISSUE_TRIAGE_REPO is used when CLI is None
3. Default simukappu/agentverify is used when both are unset
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock unavailable framework modules before importing agent
for mod_name in (
    "langchain", "langchain.agents", "langchain_core",
    "langchain_core.prompts", "langchain_openai",
    "langchain_mcp_adapters", "langchain_mcp_adapters.tools",
    "langchain_mcp_adapters.client",
    "langgraph", "langgraph.prebuilt",
    "mcp", "mcp.client", "mcp.client.stdio",
):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Allow importing from the parent package
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis import given, settings, HealthCheck, strategies as st

from agent import resolve_target_repo, DEFAULT_REPO


# Strategy: generate repo names in owner/repo format
repo_names = st.from_regex(r"[a-zA-Z0-9_-]{1,15}/[a-zA-Z0-9_-]{1,15}", fullmatch=True)

# Common settings for all property tests
pbt_settings = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class TestResolveTargetRepoProperty:
    """Property 4: resolve_target_repo() resolution priority."""

    @pbt_settings
    @given(cli_repo=repo_names)
    def test_cli_arg_takes_priority(self, cli_repo: str, monkeypatch):
        """CLI arg is used when specified, regardless of env var."""
        monkeypatch.setenv("ISSUE_TRIAGE_REPO", "env-owner/env-repo")
        result = resolve_target_repo(cli_repo)
        assert result == cli_repo

    @pbt_settings
    @given(env_repo=repo_names)
    def test_env_var_used_when_cli_is_none(self, env_repo: str, monkeypatch):
        """Env var ISSUE_TRIAGE_REPO is used when CLI is None."""
        monkeypatch.setenv("ISSUE_TRIAGE_REPO", env_repo)
        result = resolve_target_repo(None)
        assert result == env_repo

    def test_default_used_when_both_unset(self, monkeypatch):
        """Default simukappu/agentverify is used when both are unset."""
        monkeypatch.delenv("ISSUE_TRIAGE_REPO", raising=False)
        result = resolve_target_repo(None)
        assert result == DEFAULT_REPO
        assert result == "simukappu/agentverify"

    @pbt_settings
    @given(cli_repo=repo_names)
    def test_return_is_string_with_cli(self, cli_repo: str, monkeypatch):
        """Return value is always a string when CLI arg is given."""
        monkeypatch.delenv("ISSUE_TRIAGE_REPO", raising=False)
        result = resolve_target_repo(cli_repo)
        assert isinstance(result, str)

    @pbt_settings
    @given(env_repo=repo_names)
    def test_return_is_string_with_env(self, env_repo: str, monkeypatch):
        """Return value is always a string when env var is set."""
        monkeypatch.setenv("ISSUE_TRIAGE_REPO", env_repo)
        result = resolve_target_repo(None)
        assert isinstance(result, str)

    def test_return_is_string_default(self, monkeypatch):
        """Return value is always a string for default case."""
        monkeypatch.delenv("ISSUE_TRIAGE_REPO", raising=False)
        result = resolve_target_repo(None)
        assert isinstance(result, str)
