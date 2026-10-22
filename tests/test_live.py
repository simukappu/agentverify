"""Tests for live runs judged by pass rate (--agentverify-live)."""

from __future__ import annotations

import pytest

# Imported here so pytester's in-process runs, which drop modules first imported during a run, keep them.
import agentverify.cassette.tool_hooks  # noqa: F401
import agentverify.live  # noqa: F401
import agentverify.plugin  # noqa: F401
from agentverify.cassette.recorder import _resolve_provider

_resolve_provider("openai")  # imports every installed provider SDK now, not inside a run

pytest_plugins = ["pytester"]

FLAKY = """
import pytest

RUNS = []

@pytest.mark.agentverify(runs=5, min_pass_rate={rate})
def test_flaky():
    RUNS.append(1)
    assert len(RUNS) != 3, "the third run routes to the wrong agent"
"""


def _run(pytester, source, *args):
    pytester.makepyfile(source)
    return pytester.runpytest_inprocess("-p", "agentverify", *args)


def test_passes_at_the_required_rate_and_reports_it(pytester):
    result = _run(pytester, FLAKY.format(rate=0.8), "--agentverify-live")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*agentverify: live runs*", "*test_flaky: 4/5 passed (80%), required 80%"])


def test_fails_below_the_required_rate_with_each_failed_run(pytester):
    result = _run(pytester, FLAKY.format(rate=1.0), "--agentverify-live")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*passed 4 of 5 live runs (80%), below the required 100%*", "*run 3: AssertionError: the third run routes to the wrong agent*"])


def test_without_the_flag_the_test_runs_once(pytester):
    result = _run(pytester, FLAKY.format(rate=0.8))
    result.assert_outcomes(passed=1)
    assert "live runs" not in result.stdout.str()


def test_live_cassettes_call_the_real_api_and_save_nothing(pytester):
    pytester.mkdir("cassettes")
    (pytester.path / "cassettes" / "existing.yaml").write_text("interactions: []\n")
    source = """
import pytest
from agentverify.cassette.recorder import CassetteMode

MODES = []

@pytest.mark.agentverify(runs=2)
def test_agent(cassette):
    with cassette("existing.yaml", mode="replay") as rec:
        MODES.append(rec.mode)
    assert rec.mode == CassetteMode.AUTO and "agentverify-live" in str(rec.cassette_path) and not rec.cassette_path.exists()

def test_unmarked(cassette):
    with cassette("existing.yaml") as rec:
        assert rec.mode == CassetteMode.REPLAY
"""
    result = _run(pytester, source, "--agentverify-live")
    result.assert_outcomes(passed=2)
    assert (pytester.path / "cassettes" / "existing.yaml").read_text() == "interactions: []\n"


@pytest.mark.parametrize(
    "marker, message",
    [
        ("runs=0", "must be a positive integer"),
        ("runs='5'", "must be a positive integer"),
        ("runs=3, min_pass_rate=1.5", "must be between 0 and 1"),
        ("runs=3, min_pass_rate='high'", "must be between 0 and 1"),
    ],
)
def test_bad_settings_are_usage_errors(pytester, marker, message):
    source = f"""
import pytest

@pytest.mark.agentverify({marker})
def test_x():
    pass
"""
    result = _run(pytester, source, "--agentverify-live")
    assert message in result.stdout.str() + result.stderr.str()


def test_each_run_gets_fresh_fixtures_and_skips_and_fails_count_as_runs(pytester):
    source = """
import pytest

RUNS = []

@pytest.fixture
def state():
    yield []

@pytest.mark.agentverify(runs=4, min_pass_rate=0.5)
def test_runs(state):
    RUNS.append(1)
    state.append(1)
    assert state == [1], "leaked"
    if len(RUNS) == 2:
        pytest.skip("rate limited")
    if len(RUNS) == 3:
        pytest.fail("wrong route")

@pytest.mark.agentverify(runs=2)
def test_always_skips():
    pytest.skip("no key")

@pytest.fixture
def broken():
    raise RuntimeError("setup failed")

@pytest.mark.agentverify(runs=2)
def test_setup_fails(broken):
    pass
"""
    result = _run(pytester, source, "--agentverify-live", "-p", "no:randomly")
    result.assert_outcomes(passed=1, skipped=1, failed=1)  # the setup errors are in the verdict
    out = result.stdout.str()
    assert "test_runs: 2/3 passed (67%), required 50%" in out and "run 3: Failed: wrong route" in out
    assert "test_always_skips: 0/0 passed (no runs counted)" in out


def test_a_bare_marker_does_not_hide_runs_on_the_class(pytester):
    source = """
import pytest

RUNS = []

@pytest.mark.agentverify(runs=3)
class TestAgent:
    @pytest.mark.agentverify
    def test_x(self):
        RUNS.append(1)
"""
    result = _run(pytester, source, "--agentverify-live")
    result.stdout.fnmatch_lines(["*test_x: 3/3 passed (100%), required 100%"])


def test_booleans_are_not_settings(pytester):
    source = """
import pytest

@pytest.mark.agentverify(runs=True)
def test_x():
    pass
"""
    result = _run(pytester, source, "--agentverify-live")
    assert "must be a positive integer" in result.stdout.str() + result.stderr.str()


def test_other_user_properties_are_left_alone(pytester):
    source = """
import pytest

@pytest.mark.agentverify(runs=1)
def test_x(record_property):
    record_property("model", "m")
"""
    result = _run(pytester, source, "--agentverify-live")
    result.stdout.fnmatch_lines(["*test_x: 1/1 passed (100%), required 100%"])


def test_the_last_run_failing_in_setup_or_teardown_is_reported_once(pytester):
    source = """
import pytest

SETUPS, TEARDOWNS = [], []

@pytest.fixture
def flaky_setup():
    SETUPS.append(1)
    if len(SETUPS) == 3:
        raise RuntimeError("setup failed")

@pytest.fixture
def flaky_teardown():
    yield
    TEARDOWNS.append(1)
    if len(TEARDOWNS) == 3:
        raise RuntimeError("teardown failed")

@pytest.mark.agentverify(runs=3, min_pass_rate=0.6)
def test_setup(flaky_setup):
    pass

@pytest.mark.agentverify(runs=3, min_pass_rate=0.6)
def test_teardown(flaky_teardown):
    pass

SKIPS = []

@pytest.fixture
def skips_last():
    SKIPS.append(1)
    if len(SKIPS) == 2:
        pytest.skip("rate limited")

@pytest.mark.agentverify(runs=2)
def test_last_run_skips(skips_last):
    pass
"""
    result = _run(pytester, source, "--agentverify-live", "-p", "no:randomly")
    result.assert_outcomes(passed=3)
    out = result.stdout.str()
    assert "test_setup: 2/3 passed (67%), required 60%" in out and "run 3: RuntimeError: setup failed" in out
    assert "test_teardown: 2/3 passed (67%), required 60%" in out and "run 3: RuntimeError: teardown failed" in out


def test_stale_cassettes_travel_on_the_reports(pytester):
    """The summary is built from the test reports, as pytest-xdist workers send them."""
    source = """
import pathlib
import pytest
from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.io import save_cassette
from agentverify.errors import CassetteRequestMismatchError

def _request(text):
    return NormalizedRequest(messages=[{"role": "user", "content": text}], model="m")

@pytest.mark.parametrize("n", [1, 2])
def test_stale(cassette, tmp_path, n):
    path = pathlib.Path(__file__).parent / "cassettes" / "shared.yaml"
    path.parent.mkdir(exist_ok=True)
    save_cassette(path, [(_request("old"), NormalizedResponse(content="ok"))], provider="openai")
    rec = cassette("shared.yaml", mode="replay")
    with pytest.raises(CassetteRequestMismatchError):
        with rec:
            rec.lookup(_request("new"))
"""
    result = _run(pytester, source)
    result.assert_outcomes(passed=2)
    out = result.stdout.str()
    assert "cassettes/shared.yaml  (user messages changed)" in out and "/cassettes/shared.yaml" not in out
    assert "::test_stale[1]' " in out and "::test_stale[2]'" in out
