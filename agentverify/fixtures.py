"""pytest fixtures for agentverify.

Provides the ``cassette`` fixture for LLM cassette recording and replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pytest

from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder, OnMissingRequest

#: Sentinel value to distinguish "caller didn't pass mode" from "caller
#: explicitly passed CassetteMode.AUTO".
_MODE_NOT_SET = object()


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

    By default the cassette file is stored in a ``cassettes/`` directory
    next to the test file.  Override with the *cassette_dir* parameter.
    """

    def _cassette(
        name: str,
        mode: Union[str, CassetteMode, object] = _MODE_NOT_SET,
        provider: str = "openai",
        cassette_dir: Union[str, Path, None] = None,
        on_missing: Union[str, OnMissingRequest] = OnMissingRequest.ERROR,
        match_requests: Union[bool, None] = None,
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

        if cassette_dir is None:
            # Resolve relative to the test file location
            test_file = Path(str(request.fspath))
            cassette_dir = test_file.parent / "cassettes"
        else:
            cassette_dir = Path(cassette_dir)

        cassette_dir.mkdir(parents=True, exist_ok=True)
        cassette_path = cassette_dir / name

        return LLMCassetteRecorder(
            cassette_path=cassette_path,
            mode=mode,
            provider=provider,
            on_missing=on_missing,
            match_requests=match_requests,
        )

    return _cassette
