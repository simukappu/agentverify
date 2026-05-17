"""pytest configuration for custom-converter-python-agent example tests.

Registers the agentverify ``cassette`` fixture for cassette replay testing. The agentverify pytest plugin auto-registers via the ``project.entry-points.pytest11`` mapping in ``pyproject.toml``, so we only need to import the fixture here.
"""

import os

# Set a dummy API key if none is configured. The Anthropic SDK requires an API key at client construction time even when the underlying calls are intercepted for cassette replay.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-placeholder-for-cassette-replay")

from agentverify.fixtures import cassette  # noqa: F401, E402
