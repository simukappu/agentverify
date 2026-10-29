"""pytest plugin entry point for agentverify.

Registers the ``agentverify`` marker, the ``--cassette-mode`` CLI option, and auto-loads fixtures when the plugin is discovered via the ``pytest11`` entry point.
"""

from pathlib import Path

import pytest

from agentverify.fixtures import cassette  # noqa: F401 — auto-register fixture


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    """Add the cassette and live-run command-line options and the ``agentverify_cassette_max_age`` ini setting."""
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
        "--cassette-max-age",
        default=None,
        help="Fail replaying cassettes recorded longer ago than this (e.g. 30d, 12h). Overrides the agentverify_cassette_max_age ini setting.",
    )
    parser.addoption(
        "--cassette-expired",
        default=None,
        choices=["error", "warn"],
        help="What replaying an expired cassette does: error (default) or warn.",
    )
    parser.addoption(
        "--cassette-report",
        action="store_true",
        default=False,
        help="List the cassettes next to the collected tests (recorded when, with which model, expired or not) instead of running the tests.",
    )
    parser.addini(
        "agentverify_cassette_max_age",
        default="",
        help="Fail replaying cassettes recorded longer ago than this (e.g. 30d). Empty: no limit.",
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
def pytest_runtestloop(session):  # type: ignore[no-untyped-def]
    """Under --cassette-report, list the cassettes instead of running the tests."""
    if not session.config.getoption("--cassette-report", default=False):
        return None
    from agentverify.cassette.report import report

    report(session)
    return True


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):  # type: ignore[no-untyped-def]
    """Under --agentverify-live, run a test marked with runs= repeatedly and judge it by its pass rate."""
    from agentverify.live import run_protocol

    return run_protocol(item, nextitem)


def pytest_runtest_logreport(report):  # type: ignore[no-untyped-def]
    from agentverify.cassette.recorder import _stale_cassettes
    from agentverify.fixtures import STALE_PROPERTY
    from agentverify.live import collect

    collect(report)
    if report.when == "teardown":
        for name, value in report.user_properties:
            if name == STALE_PROPERTY:
                _stale_cassettes.extend(tuple(entry) for entry in value)


def pytest_terminal_summary(terminalreporter):  # type: ignore[no-untyped-def]
    """List the cassettes that no longer match the requests the agent makes, with the command to re-record them, and the pass rates of live runs."""
    from agentverify.cassette.recorder import _stale_cassettes
    from agentverify.live import summary

    summary(terminalreporter)

    if not _stale_cassettes:
        return
    root = terminalreporter.config.rootpath
    stale: dict[str, tuple[set[str], set[str]]] = {}
    for path, test_id, field in _stale_cassettes:
        fields, tests = stale.setdefault(path, (set(), set()))
        fields.add(field)
        if test_id:
            tests.add(test_id)
    terminalreporter.section("agentverify: stale cassettes")
    for path, (fields, tests) in stale.items():
        try:
            shown = str(Path(path).relative_to(root))
        except ValueError:
            shown = path
        changed = [f for f in sorted(fields) if not f.startswith("expired")]
        reasons = ([f"{', '.join(changed)} changed"] if changed else []) + [f for f in sorted(fields) if f.startswith("expired")]
        terminalreporter.write_line(f"{shown}  ({'; '.join(reasons)})")
    every_test = sorted(set().union(*(tests for _, tests in stale.values())))
    if every_test:
        import shlex

        terminalreporter.write_line("")
        terminalreporter.write_line(f"Re-record them with (from {root}):")
        terminalreporter.write_line("  pytest --cassette-mode=record " + " ".join(shlex.quote(t) for t in every_test))
    _stale_cassettes.clear()
