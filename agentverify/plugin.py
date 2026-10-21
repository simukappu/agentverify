"""pytest plugin entry point for agentverify.

Registers the ``agentverify`` marker, the ``--cassette-mode`` CLI option, and auto-loads fixtures when the plugin is discovered via the ``pytest11`` entry point.
"""

import pytest

from agentverify.fixtures import cassette  # noqa: F401 — auto-register fixture


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    """Add the ``--cassette-mode``, ``--no-cassette-match-requests``, and ``--agentverify-live`` command-line options."""
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
    parser.addoption(
        "--agentverify-live",
        action="store_true",
        default=False,
        help="Run tests marked agentverify(runs=N, min_pass_rate=R) N times against the live model, saving no cassettes.",
    )


def pytest_configure(config):  # type: ignore[no-untyped-def]
    """Register the agentverify marker."""
    config.addinivalue_line(
        "markers",
        "agentverify(runs=None, min_pass_rate=1.0): an agentverify agent test; with --agentverify-live, run it runs times live and require min_pass_rate",
    )


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    """Check the live-run settings of every marked test before any runs."""
    from agentverify.live import live_settings

    for item in items:
        live_settings(item)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):  # type: ignore[no-untyped-def]
    """Under --agentverify-live, run a test marked with runs= repeatedly and judge it by its pass rate."""
    from agentverify.live import run_protocol

    return run_protocol(item, nextitem)


def pytest_runtest_logreport(report):  # type: ignore[no-untyped-def]
    from agentverify.live import collect

    collect(report)


def pytest_terminal_summary(terminalreporter):  # type: ignore[no-untyped-def]
    """List the cassettes that no longer match the requests the agent makes, with the command to re-record them, and the pass rates of live runs."""
    from agentverify.cassette.recorder import _stale_cassettes
    from agentverify.live import summary

    summary(terminalreporter)

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
