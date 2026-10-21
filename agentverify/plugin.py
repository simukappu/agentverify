"""pytest plugin entry point for agentverify.

Registers the ``agentverify`` marker, the ``--cassette-mode`` CLI option, and auto-loads fixtures when the plugin is discovered via the ``pytest11`` entry point.
"""

from agentverify.fixtures import cassette  # noqa: F401 — auto-register fixture


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    """Add the ``--cassette-mode`` and ``--cassette-match-requests`` command-line options."""
    parser.addoption(
        "--cassette-mode",
        default=None,
        choices=["auto", "record", "replay"],
        help="Override cassette mode for all tests (auto, record, replay).",
    )
    parser.addoption(
        "--no-cassette-match-requests",
        action="store_true",
        default=False,
        help="Disable request matching during cassette replay.",
    )


def pytest_configure(config):  # type: ignore[no-untyped-def]
    """Register the agentverify marker."""
    config.addinivalue_line(
        "markers",
        "agentverify: mark test as an agentverify agent test",
    )


def pytest_terminal_summary(terminalreporter):  # type: ignore[no-untyped-def]
    """List the cassettes that no longer match the requests the agent makes, with the command to re-record them."""
    from agentverify.cassette.recorder import _stale_cassettes

    if not _stale_cassettes:
        return
    seen: dict[str, tuple[str | None, str]] = {}
    for path, test_id, field in _stale_cassettes:
        seen.setdefault(path, (test_id, field))
    terminalreporter.section("agentverify: stale cassettes")
    for path, (test_id, field) in seen.items():
        terminalreporter.write_line(f"{path}  ({field} changed)")
    tests = sorted({test_id for test_id, _ in seen.values() if test_id})
    if tests:
        import shlex

        terminalreporter.write_line("")
        terminalreporter.write_line(f"Re-record them with (from {terminalreporter.config.rootpath}):")
        terminalreporter.write_line("  pytest --cassette-mode=record " + " ".join(shlex.quote(t) for t in tests))
    _stale_cassettes.clear()
