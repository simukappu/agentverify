"""Run a test several times against the live model and judge it by its pass rate.

A test marked ``@pytest.mark.agentverify(runs=5, min_pass_rate=0.8)`` runs once as usual (replaying its cassette). With ``--agentverify-live`` it runs *runs* times instead, with every ``cassette`` calling the real model and tools and saving nothing, and passes when at least ``min_pass_rate`` of the runs pass. The assertions are the test's own; no judge model is involved.

Each run is a full run of the test: its function-scoped fixtures are set up and torn down again, and async test plugins (pytest-asyncio, anyio) run it as they would once. A run that skips does not count toward the rate.
"""

from __future__ import annotations

from typing import Any

import pytest

# (test id, passed runs, counted runs, required rate, failed runs) for the summary, gathered from the
# reports so it works under pytest-xdist too.
_live_results: list[tuple[str, int, int, float, list[str]]] = []
_PROPERTY = "agentverify_live"


def live_settings(item: Any) -> tuple[int, float] | None:
    """The (runs, min_pass_rate) a live session applies to *item*, or None to run it as usual."""
    if not item.config.getoption("--agentverify-live", default=False):
        return None
    # The nearest agentverify marker that sets runs; a bare marker on the function does not hide one on its class or module.
    marker = next((m for m in item.iter_markers("agentverify") if "runs" in m.kwargs), None)
    if marker is None:
        return None
    runs = marker.kwargs["runs"]
    min_pass_rate = marker.kwargs.get("min_pass_rate", 1.0)
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise pytest.UsageError(f"{item.nodeid}: agentverify(runs=...) must be a positive integer, got {runs!r}")
    if isinstance(min_pass_rate, bool) or not isinstance(min_pass_rate, (int, float)) or not 0 <= min_pass_rate <= 1:
        raise pytest.UsageError(f"{item.nodeid}: agentverify(min_pass_rate=...) must be between 0 and 1, got {min_pass_rate!r}")
    return runs, float(min_pass_rate)


def _first_line(report: Any) -> str:
    text = str(getattr(report, "longreprtext", "") or "").strip().splitlines()
    error = next((line for line in text if line.startswith("E ")), text[-1] if text else "")
    return error.removeprefix("E").strip()


def run_protocol(item: Any, nextitem: Any) -> bool | None:
    """Run *item* repeatedly when the live settings apply, and log one combined outcome; True when handled here."""
    settings = live_settings(item)
    if settings is None:
        return None
    from _pytest.runner import runtestprotocol

    runs, min_pass_rate = settings
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    passed, counted, failures = 0, 0, []
    reports: list[Any] = []
    for run in range(1, runs + 1):
        if run > 1:
            item._initrequest()  # fresh fixtures for the next run, as pytest-rerunfailures does
        reports = runtestprotocol(item, nextitem=nextitem, log=False)
        if any(r.skipped for r in reports) and not any(r.failed for r in reports):
            continue
        counted += 1
        failed = next((r for r in reports if r.failed), None)
        if failed is None:
            passed += 1
        else:
            failures.append(f"run {run}: {_first_line(failed)}")
    if counted:
        reports = _verdict(reports, passed, counted, min_pass_rate, failures)
    else:  # every run skipped: the last run's skip stands
        skipped = next(r for r in reports if r.skipped)
        skipped.user_properties.append((_PROPERTY, (0, 0, min_pass_rate, [])))
    for report in reports:
        item.ihook.pytest_runtest_logreport(report=report)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def _verdict(reports: list[Any], passed: int, counted: int, min_pass_rate: float, failures: list[str]) -> list[Any]:
    """The last run's reports, with its setup and teardown passed (a failure or skip there is one of the runs) and its call carrying the combined outcome.

    Every run's failures are in the verdict already, so the last run's own setup or teardown failure is not reported again.
    """
    import copy

    setup = next(r for r in reports if r.when == "setup")
    call = next((r for r in reports if r.when == "call"), None)
    if call is None:  # the last run failed in setup: its verdict goes on a call report of its own
        call = copy.copy(setup)
        call.when, call.sections, call.user_properties = "call", [], list(setup.user_properties)
    call.user_properties.append((_PROPERTY, (passed, counted, min_pass_rate, failures)))
    if passed / counted >= min_pass_rate:
        call.outcome, call.longrepr = "passed", None
    else:
        call.outcome = "failed"
        call.longrepr = f"passed {passed} of {counted} live runs ({passed / counted:.0%}), below the required {min_pass_rate:.0%}\n" + "\n".join(f"  {f}" for f in failures)
    for report in reports:
        if report.when != "call" and not report.passed:
            report.outcome, report.longrepr = "passed", None
    return [setup, call, *(r for r in reports if r.when == "teardown")]


def collect(report: Any) -> None:
    for name, value in getattr(report, "user_properties", []):
        if name == _PROPERTY:
            passed, counted, required, failures = value
            _live_results.append((report.nodeid, passed, counted, required, list(failures)))


def summary(terminalreporter: Any) -> None:
    if not _live_results:
        return
    terminalreporter.section("agentverify: live runs")
    for test_id, passed, counted, required, failures in _live_results:
        rate = f"{passed / counted:.0%}" if counted else "no runs counted"
        terminalreporter.write_line(f"{test_id}: {passed}/{counted} passed ({rate}), required {required:.0%}")
        for failure in failures:
            terminalreporter.write_line(f"  {failure}")
    _live_results.clear()
