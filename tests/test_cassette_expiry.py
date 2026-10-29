"""Replaying a cassette older than its max_age fails, or warns; --cassette-report lists the cassettes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import yaml

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.expiry import check_on_expired, expiry, parse_max_age, recorded_at
from agentverify.cassette.io import save_cassette
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder, _stale_cassettes
from agentverify.errors import CassetteExpiredError

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _clear():
    _stale_cassettes.clear()
    yield
    _stale_cassettes.clear()


def _cassette(path, days_ago: float | None = 0, raw: str | None = None):
    save_cassette(path, [(NormalizedRequest(messages=[], model="m"), NormalizedResponse(content="ok"))], provider="openai")
    document = yaml.safe_load(path.read_text())
    if raw is not None:
        document["metadata"]["recorded_at"] = raw
    elif days_ago is None:
        del document["metadata"]["recorded_at"]
    else:
        document["metadata"]["recorded_at"] = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    path.write_text(yaml.safe_dump(document))
    return path


class TestDurations:
    def test_parse(self):
        assert parse_max_age("30d") == timedelta(days=30)
        assert parse_max_age(" 12h ") == timedelta(hours=12)
        assert parse_max_age("1.5d") == timedelta(days=1.5)
        assert parse_max_age(timedelta(days=2)) == timedelta(days=2) and parse_max_age(None) is None
        for bad in ("30", "30m", 30, "d"):
            with pytest.raises(ValueError, match="max_age"):
                parse_max_age(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="on_expired"):
            check_on_expired("ignore")

    def test_recorded_at(self):
        naive = recorded_at({"recorded_at": "2026-10-01T00:00:00"})
        assert naive == datetime(2026, 10, 1, tzinfo=timezone.utc)  # no time zone: UTC
        assert recorded_at({"recorded_at": "2026-10-01T00:00:00Z"}) == naive
        assert recorded_at({"recorded_at": datetime(2026, 10, 1)}) == naive  # YAML loads unquoted timestamps as datetimes
        assert recorded_at({"recorded_at": "yesterday"}) is None and recorded_at({}) is None

    def test_expiry(self):
        now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        assert expiry({"recorded_at": "2026-10-20T00:00:00+00:00"}, timedelta(days=30), now) is None
        reason = expiry({"recorded_at": "2026-09-01T00:00:00+00:00"}, timedelta(days=30), now)
        assert reason == "expired: recorded 2026-09-01 (58 days ago), due for re-recording by 2026-10-01"
        assert expiry({}, timedelta(days=30), now) == "expired: no recorded_at in its metadata"


class TestRecorder:
    def test_an_expired_cassette_fails_before_the_agent_runs(self, tmp_path):
        path = _cassette(tmp_path / "old.yaml", days_ago=40)
        with pytest.raises(CassetteExpiredError, match=r"old.yaml: expired: recorded .* \(40 days ago\).*--cassette-mode=record"):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, max_age="30d"):
                pytest.fail("the block must not run")  # pragma: no cover
        assert [entry[2][:8] for entry in _stale_cassettes] == ["expired:"]

    def test_warn_replays_anyway(self, tmp_path):
        path = _cassette(tmp_path / "old.yaml", days_ago=40)
        with pytest.warns(UserWarning, match="expired"):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, max_age="30d", on_expired="warn") as rec:
                assert rec.lookup(NormalizedRequest(messages=[], model="m")).content == "ok"

    def test_fresh_unlimited_recorded_and_undated(self, tmp_path):
        fresh = _cassette(tmp_path / "fresh.yaml", days_ago=1)
        with LLMCassetteRecorder(fresh, mode=CassetteMode.REPLAY, max_age="30d"):
            pass
        old = _cassette(tmp_path / "old.yaml", days_ago=400)
        with LLMCassetteRecorder(old, mode=CassetteMode.REPLAY):  # no limit
            pass
        with LLMCassetteRecorder(old, mode=CassetteMode.RECORD, max_age="30d"):  # recording replaces it
            pass
        undated = _cassette(tmp_path / "undated.yaml", days_ago=None)
        with pytest.raises(CassetteExpiredError, match="no recorded_at"):
            with LLMCassetteRecorder(undated, mode=CassetteMode.REPLAY, max_age="30d"):
                pass  # pragma: no cover
        with pytest.raises(ValueError, match="on_expired"):
            LLMCassetteRecorder(fresh, max_age="30d", on_expired="skip")


def _run(pytester, *args, ini=""):
    pytester.makepyfile(
        test_agent="""
import pytest
from agentverify.cassette.adapters.base import NormalizedRequest

@pytest.mark.agentverify
def test_old(cassette):
    with cassette("old.yaml") as rec:
        rec.lookup(NormalizedRequest(messages=[], model="m"))

def test_fresh(cassette):
    with cassette("fresh.yaml", max_age="1d") as rec:
        rec.lookup(NormalizedRequest(messages=[], model="m"))
"""
    )
    if ini:
        pytester.makeini(ini)
    cassettes = pytester.path / "cassettes"
    cassettes.mkdir(exist_ok=True)
    _cassette(cassettes / "old.yaml", days_ago=40)
    _cassette(cassettes / "fresh.yaml", days_ago=0.5)
    (cassettes / "notes.json").write_text("not json")
    return pytester.runpytest_inprocess("-p", "agentverify", *args)


class TestPytest:
    def test_ini_limit_fails_and_lists_the_cassette(self, pytester):
        result = _run(pytester, ini="[pytest]\nagentverify_cassette_max_age = 30d\n")
        result.assert_outcomes(passed=1, failed=1)
        out = result.stdout.str()
        assert "CassetteExpiredError" in out
        assert "cassettes/old.yaml  (expired: recorded" in out and "test_agent.py::test_old" in out.split("Re-record them with")[1]

    def test_the_command_line_overrides_the_ini_and_can_warn(self, pytester):
        result = _run(pytester, "--cassette-max-age=60d", ini="[pytest]\nagentverify_cassette_max_age = 30d\n")
        result.assert_outcomes(passed=2)
        result = _run(pytester, "--cassette-max-age=30d", "--cassette-expired=warn")
        result.assert_outcomes(passed=2)
        assert "UserWarning:" in result.stdout.str() and "old.yaml: expired" in result.stdout.str()

    def test_report_lists_cassettes_without_running_tests(self, pytester):
        result = _run(pytester, "--cassette-report", "--cassette-max-age=30d")
        out = result.stdout.str()
        assert "agentverify: cassettes" in out and " passed" not in out
        assert "status (max age 30d)" in out and "openai" in out
        lines = {line.split()[0]: line for line in out.splitlines() if line.startswith("cassettes/")}
        assert "expired: recorded" in lines["cassettes/old.yaml"] and lines["cassettes/fresh.yaml"].rstrip().endswith("ok")
        assert "unreadable: " in lines["cassettes/notes.json"]

    def test_report_without_a_limit_or_cassettes(self, pytester):
        result = _run(pytester, "--cassette-report", "--cassette-max-age=12h")
        assert "status (max age 12h)" in result.stdout.str()
        pytester.makepyfile(test_other="def test_x():\n    pass\n")
        empty = pytester.mkdir("empty")
        (empty / "test_y.py").write_text("def test_y():\n    pass\n")
        result = pytester.runpytest_inprocess("-p", "agentverify", "--cassette-report", str(empty))
        assert "No cassettes next to the collected tests." in result.stdout.str()
        result = _run(pytester, "--cassette-report")
        assert "  status" in result.stdout.str().splitlines()[[i for i, l in enumerate(result.stdout.str().splitlines()) if l.startswith("cassette ")][0]]


def test_paths_outside_the_rootdir_and_an_explicit_on_expired(tmp_path, cassette):
    from pathlib import Path

    from agentverify.cassette.report import _relative

    assert _relative(Path("/elsewhere/c.yaml"), Path("/root")) == "/elsewhere/c.yaml"
    _cassette(tmp_path / "old.yaml", days_ago=40)
    with pytest.warns(UserWarning, match="expired"):
        with cassette("old.yaml", cassette_dir=tmp_path, mode="replay", max_age="30d", on_expired="warn"):
            pass
