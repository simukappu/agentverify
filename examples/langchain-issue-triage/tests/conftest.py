"""pytest configuration for langchain-issue-triage example tests.

Registers the agentverify ``cassette`` fixture for cassette replay testing. The agentverify pytest plugin auto-registers via the ``project.entry-points.pytest11`` mapping in ``pyproject.toml``, so we only need to import the fixture here.
"""

from agentverify.fixtures import cassette  # noqa: F401
