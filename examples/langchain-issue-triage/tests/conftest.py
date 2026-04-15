"""pytest configuration for langchain-issue-triage example tests.

Registers the agentverify ``cassette`` fixture and ``--cassette-mode``
CLI option for cassette replay testing.
"""

from agentverify.fixtures import cassette  # noqa: F401
from agentverify.plugin import pytest_addoption, pytest_configure  # noqa: F401
