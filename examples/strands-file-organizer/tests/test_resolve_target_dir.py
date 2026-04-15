# Feature: examples, Property 3: File organizer agent target directory resolution priority
"""Property-based tests for resolve_target_dir().

**Validates: Requirements 1.2**

Tests the priority resolution logic:
1. CLI arg takes priority when specified
2. Env var FILE_ORGANIZER_TARGET_DIR is used when CLI is None
3. Default (repo root) is used when both are unset
4. Return value is always an absolute path
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock unavailable framework modules before importing agent
for mod_name in (
    "strands", "strands.tools", "strands.tools.mcp",
    "mcp", "mcp.client", "mcp.client.stdio",
):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Allow importing from the parent package
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis import given, settings, HealthCheck, strategies as st

from agent import resolve_target_dir


# Strategy: generate directory paths like /tmp/abcXYZ_123
dir_paths = st.from_regex(r"/tmp/[a-zA-Z0-9_]{1,20}", fullmatch=True)

# Common settings for all property tests
pbt_settings = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class TestResolveTargetDirProperty:
    """Property 3: resolve_target_dir() resolution priority."""

    @pbt_settings
    @given(cli_dir=dir_paths)
    def test_cli_arg_takes_priority(self, cli_dir: str, monkeypatch):
        """CLI arg is used when specified, regardless of env var."""
        monkeypatch.setenv("FILE_ORGANIZER_TARGET_DIR", "/tmp/env_value")
        result = resolve_target_dir(cli_dir)
        assert result == Path(cli_dir).resolve()
        assert result.is_absolute()

    @pbt_settings
    @given(env_dir=dir_paths)
    def test_env_var_used_when_cli_is_none(self, env_dir: str, monkeypatch):
        """Env var FILE_ORGANIZER_TARGET_DIR is used when CLI is None."""
        monkeypatch.setenv("FILE_ORGANIZER_TARGET_DIR", env_dir)
        result = resolve_target_dir(None)
        assert result == Path(env_dir).resolve()
        assert result.is_absolute()

    def test_default_used_when_both_unset(self, monkeypatch):
        """Default (repo root) is used when both CLI and env var are unset."""
        monkeypatch.delenv("FILE_ORGANIZER_TARGET_DIR", raising=False)
        result = resolve_target_dir(None)
        expected = (Path(__file__).parent.parent / ".." / "..").resolve()
        assert result == expected
        assert result.is_absolute()

    @pbt_settings
    @given(cli_dir=dir_paths)
    def test_return_is_always_absolute_with_cli(self, cli_dir: str, monkeypatch):
        """Return value is always an absolute Path when CLI arg is given."""
        monkeypatch.delenv("FILE_ORGANIZER_TARGET_DIR", raising=False)
        result = resolve_target_dir(cli_dir)
        assert isinstance(result, Path)
        assert result.is_absolute()

    @pbt_settings
    @given(env_dir=dir_paths)
    def test_return_is_always_absolute_with_env(self, env_dir: str, monkeypatch):
        """Return value is always an absolute Path when env var is set."""
        monkeypatch.setenv("FILE_ORGANIZER_TARGET_DIR", env_dir)
        result = resolve_target_dir(None)
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_return_is_always_absolute_default(self, monkeypatch):
        """Return value is always an absolute Path for default case."""
        monkeypatch.delenv("FILE_ORGANIZER_TARGET_DIR", raising=False)
        result = resolve_target_dir(None)
        assert isinstance(result, Path)
        assert result.is_absolute()
