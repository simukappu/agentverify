"""pytest configuration for langchain-issue-triage example tests.

Registers the agentverify ``cassette`` fixture for cassette replay testing. The agentverify pytest plugin auto-registers via the ``project.entry-points.pytest11`` mapping in ``pyproject.toml``, so we only need to import the fixture here.
"""

from agentverify.fixtures import cassette  # noqa: F401

import os

import pytest


@pytest.fixture(autouse=True)
def _openai_key(monkeypatch):
    """ChatOpenAI needs a key to start; replay never sends it."""
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
