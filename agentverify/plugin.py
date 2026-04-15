"""pytest plugin entry point for agentverify.

Registers the ``agentverify`` marker, the ``--cassette-mode`` CLI option,
and auto-loads fixtures when the plugin is discovered via the ``pytest11``
entry point.
"""

from agentverify.fixtures import cassette  # noqa: F401 — auto-register fixture


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    """Add the ``--cassette-mode`` command-line option."""
    parser.addoption(
        "--cassette-mode",
        default=None,
        choices=["auto", "record", "replay"],
        help="Override cassette mode for all tests (auto, record, replay).",
    )


def pytest_configure(config):  # type: ignore[no-untyped-def]
    """Register the agentverify marker."""
    config.addinivalue_line(
        "markers",
        "agentverify: mark test as an agentverify agent test",
    )
