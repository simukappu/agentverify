"""pytest configuration for the Strands subject benchmark cells.

Re-exports the ``cassette`` fixture and makes the bench-local helpers (currently ``_strands_agent_factory``) importable from the three test files in this directory. The benchmark's ``pyproject.toml`` sets ``importmode=importlib``, which does not auto-prepend the test file's directory to ``sys.path``; we add it here so ``from _strands_agent_factory import build_weather_agent`` works from any of the cell tests.

The factory module name is intentionally unique per subject (``_strands_agent_factory`` here, ``_langgraph_agent_factory`` next door) so that running both subjects in a single pytest invocation does not collide on a shared module name.

We do not re-import the agentverify plugin's ``pytest_addoption`` / ``pytest_configure`` here, because they are already registered through the ``project.entry-points.pytest11`` mapping in ``pyproject.toml``. Re-importing would register the same options twice on Python 3.14+ pytest plugin discovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agentverify.fixtures import cassette  # noqa: E402, F401
