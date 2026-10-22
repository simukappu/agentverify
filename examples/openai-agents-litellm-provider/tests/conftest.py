"""pytest configuration for the openai-agents-litellm-provider example tests.

The agentverify pytest plugin auto-registers via its ``pytest11`` entry point, so only the fixture is imported here.
"""

import os

# The OpenAI Agents SDK exports traces to OpenAI over its own HTTP client, outside the cassette; keep tests offline.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

from agentverify.fixtures import cassette  # noqa: F401, E402
