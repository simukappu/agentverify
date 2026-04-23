"""pytest configuration for langgraph-multi-agent-supervisor example tests.

Registers the agentverify ``cassette`` fixture and ``--cassette-mode``
CLI option for cassette replay testing.
"""

import os

# Set a dummy API key if none is configured. langchain-openai requires an
# API key at client construction time even when the underlying calls are
# being intercepted for cassette replay, so we provide a placeholder to
# keep replay-mode tests self-contained.
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder-for-cassette-replay")

from agentverify.fixtures import cassette  # noqa: F401, E402
from agentverify.plugin import pytest_addoption, pytest_configure  # noqa: F401, E402
