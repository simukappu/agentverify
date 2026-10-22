"""pytest configuration for the google-adk-agent-team example tests.

The agentverify pytest plugin auto-registers via its ``pytest11`` entry point, so only the fixture is imported here.
"""

from agentverify.fixtures import cassette  # noqa: F401
