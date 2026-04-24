"""pytest configuration for openai-agents-llm-as-a-judge example tests.

Registers the agentverify ``cassette`` fixture and ``--cassette-mode``
CLI option for cassette replay testing.
"""

import os

# Set a dummy API key if none is configured. The OpenAI Agents SDK
# requires an API key at client construction time even when the calls
# are being intercepted for cassette replay.
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder-for-cassette-replay")

from agentverify.fixtures import cassette  # noqa: F401, E402
from agentverify.plugin import pytest_addoption, pytest_configure  # noqa: F401, E402
