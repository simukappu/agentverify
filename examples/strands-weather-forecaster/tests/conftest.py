"""pytest configuration for strands-weather-forecaster example tests.

The agentverify pytest plugin auto-registers via the ``project.entry-points.pytest11`` mapping in ``pyproject.toml``, so we only need to import the cassette fixture here.
"""

from agentverify.fixtures import cassette  # noqa: F401
