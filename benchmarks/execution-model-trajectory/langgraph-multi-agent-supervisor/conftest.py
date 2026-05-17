"""pytest configuration for the LangGraph subject benchmark cells.

Re-exports the ``cassette`` fixture and makes the bench-local helpers (currently ``_langgraph_agent_factory``) importable from the three test files in this directory. The benchmark's ``pyproject.toml`` sets ``importmode=importlib``, which does not auto-prepend the test file's directory to ``sys.path``; we add it here so ``from _langgraph_agent_factory import build_supervisor`` works from any of the cell tests.

The factory module name is intentionally unique per subject (``_langgraph_agent_factory`` here, ``_strands_agent_factory`` next door) so that running both subjects in a single pytest invocation does not collide on a shared module name.

A real ``OPENAI_API_KEY`` is required at LangGraph agent construction time; we fall back to a placeholder so cassette-replay tests under Scenario 2 still construct ``ChatOpenAI`` cleanly. Scenario 1 requires a real key and will fail at the live LLM call if the placeholder is left in place.

We do not re-import the agentverify plugin's ``pytest_addoption`` / ``pytest_configure`` here, because they are already registered through the ``project.entry-points.pytest11`` mapping in ``pyproject.toml``. Re-importing would register the same options twice on Python 3.14+ pytest plugin discovery.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder-for-cassette-replay")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agentverify.fixtures import cassette  # noqa: E402, F401
