"""pytest configuration for custom-converter-python-agent example tests.

Registers the agentverify ``cassette`` fixture and ``--cassette-mode``
CLI option for cassette replay testing.
"""

import os

# Set a dummy API key if none is configured. The Anthropic SDK requires
# an API key at client construction time even when the underlying calls
# are intercepted for cassette replay.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-placeholder-for-cassette-replay")

from agentverify.fixtures import cassette  # noqa: F401, E402
from agentverify.plugin import pytest_addoption, pytest_configure  # noqa: F401, E402
