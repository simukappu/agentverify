"""pytest fixtures for agentverify.

Provides the ``cassette`` fixture for LLM cassette recording and replay.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Union

import pytest

from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder, OnMissingRequest
from agentverify.cassette.sanitize import SanitizePattern

STALE_PROPERTY = "agentverify_stale_cassettes"

#: Sentinel value to distinguish "caller didn't pass mode" from "caller
#: explicitly passed CassetteMode.AUTO".
_MODE_NOT_SET = object()


def _ini(config, name: str):  # type: ignore[no-untyped-def]
    """An ini setting the plugin registers, or None when the plugin is not loaded (``-p no:agentverify`` with the fixture imported)."""
    try:
        return config.getini(name) or None
    except ValueError:
        return None


@pytest.fixture
def cassette(request):
    """Factory fixture that creates :class:`LLMCassetteRecorder` instances.

    Usage::

        def test_agent(cassette):
            with cassette("my_test.yaml", provider="openai") as rec:
                # agent code that calls the LLM
                ...
            result = rec.to_execution_result()

    The cassette mode is resolved in the following priority order:

    1. ``mode`` argument passed to the factory call (highest priority)
    2. ``--cassette-mode`` pytest CLI option
    3. ``CassetteMode.AUTO`` (default)

    By default the cassette file is stored in a ``cassettes/`` directory next to the test file.  Override with the *cassette_dir* parameter.
    """
    from agentverify.cassette.recorder import _stale_cassettes

    def _report_stale() -> None:
        # The test's stale cassettes travel on its report, so the summary lists them under pytest-xdist too.
        mine = [entry for entry in _stale_cassettes if entry[1] == request.node.nodeid]
        if mine:
            _stale_cassettes[:] = [entry for entry in _stale_cassettes if entry[1] != request.node.nodeid]
            request.node.user_properties.append((STALE_PROPERTY, mine))

    request.addfinalizer(_report_stale)

    def _cassette(
        name: str,
        mode: Union[str, CassetteMode, object] = _MODE_NOT_SET,
        provider: str = "openai",
        cassette_dir: Union[str, Path, None] = None,
        on_missing: Union[str, OnMissingRequest] = OnMissingRequest.ERROR,
        match_requests: Union[bool, None] = None,
        sanitize: Union[bool, list[SanitizePattern]] = True,
        replay_tools: bool = True,
        match_ignore: Union[list[str], None] = None,
        max_age: Union[str, timedelta, None, object] = _MODE_NOT_SET,
        on_expired: Union[str, None] = None,
    ) -> LLMCassetteRecorder:
        # Resolve mode: explicit arg > CLI option > default AUTO
        if mode is _MODE_NOT_SET:
            cli_mode = request.config.getoption("--cassette-mode", default=None)
            if cli_mode is not None:
                mode = CassetteMode(cli_mode)
            else:
                mode = CassetteMode.AUTO
        elif isinstance(mode, str):
            mode = CassetteMode(mode)

        if isinstance(on_missing, str):
            on_missing = OnMissingRequest(on_missing)

        # Resolve match_requests: explicit arg > CLI option > default True
        if match_requests is None:
            no_match = request.config.getoption(
                "--no-cassette-match-requests", default=False
            )
            match_requests = not no_match

        # Resolve the expiry: explicit arg > --cassette-max-age > the agentverify_cassette_max_age ini setting
        if max_age is _MODE_NOT_SET:
            max_age = request.config.getoption("--cassette-max-age", default=None) or _ini(request.config, "agentverify_cassette_max_age")
        if on_expired is None:
            on_expired = request.config.getoption("--cassette-expired", default=None) or "error"

        from agentverify.live import live_settings

        if live_settings(request.node) is not None:
            # A live run: call the real model and tools, and save nothing (a path that does not exist).
            return LLMCassetteRecorder(
                cassette_path=Path(tempfile.gettempdir()) / "agentverify-live" / f"{uuid.uuid4().hex}-{name}",
                mode=CassetteMode.AUTO,
                provider=provider,
                on_missing=on_missing,
                sanitize=sanitize,
                replay_tools=replay_tools,
            )
        if cassette_dir is None:
            # Resolve relative to the test file location
            test_file = Path(str(request.fspath))
            cassette_dir = test_file.parent / "cassettes"
        else:
            cassette_dir = Path(cassette_dir)

        cassette_dir.mkdir(parents=True, exist_ok=True)
        cassette_path = cassette_dir / name

        recorder = LLMCassetteRecorder(
            cassette_path=cassette_path,
            mode=mode,
            provider=provider,
            on_missing=on_missing,
            match_requests=match_requests,
            sanitize=sanitize,
            replay_tools=replay_tools,
            match_ignore=match_ignore,
            max_age=max_age,  # type: ignore[arg-type]
            on_expired=on_expired,
        )
        recorder.test_id = request.node.nodeid
        return recorder

    return _cassette
