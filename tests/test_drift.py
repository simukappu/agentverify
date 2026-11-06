"""Drift: replayed tool results, messages, and parameters compared with the recording (off by default)."""

from __future__ import annotations

import pytest

from agentverify import CassetteDriftError
from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.drift import compile_ignore, describe, drift
from agentverify.cassette.io import save_cassette
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder, _stale_cassettes

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _clear():
    _stale_cassettes.clear()
    yield
    _stale_cassettes.clear()


def _request(result: str, extra=(), **parameters):
    messages = [
        {"role": "user", "content": "events?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "camera_events", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": result},
        *extra,
    ]
    return NormalizedRequest(model="m", messages=messages, parameters=parameters)


RECORDED = _request('{"events": [{"kind": "motion"}], "now": "10:00"}', temperature=0)


class TestCompare:
    def test_what_differs(self):
        assert drift(RECORDED, RECORDED, 3, []) == []
        changed = _request('{"events": [{"kind": "motion", "description": null}], "now": "10:05"}', temperature=1)
        assert drift(RECORDED, changed, 3, []) == [
            "request[3].message[2](tool).tool_result[0](camera_events).events[0].description",  # a new null counts
            "request[3].message[2](tool).tool_result[0](camera_events).now",
            "request[3].parameters.temperature",
        ]
        assert drift(RECORDED, changed, 3, compile_ignore(["request[*].message[*].tool_result[*].now", "*.temperature", "*.description"])) == []
        longer = _request('{"events": [{"kind": "motion"}], "now": "10:00"}', extra=[{"role": "assistant", "content": "done"}], temperature=0)
        assert drift(RECORDED, longer, 0, []) == ["request[0].messages (roles user,assistant,tool → user,assistant,tool,assistant)"]

    def test_brackets_in_globs_are_literal_and_describe_truncates(self):
        [pattern] = compile_ignore(["request[1].x"])
        assert pattern.match("request[1].x") and not pattern.match("request1.x")
        assert describe([str(i) for i in range(7)]) == "0, 1, 2, 3, 4, and 2 more"


class TestRecorder:
    def _cassette(self, tmp_path):
        path = tmp_path / "c.yaml"
        save_cassette(path, [(RECORDED, NormalizedResponse(content="ok"))], provider="openai")
        return path

    def test_off_by_default(self, tmp_path):
        with LLMCassetteRecorder(self._cassette(tmp_path), mode=CassetteMode.REPLAY) as rec:
            rec.lookup(_request('{"events": [], "now": "11:00"}', temperature=0))
        assert _stale_cassettes == []

    def test_warn_lists_and_fail_raises(self, tmp_path):
        path = self._cassette(tmp_path)
        now = _request('{"events": [{"kind": "motion"}], "now": "11:00"}', temperature=0)
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, drift="warn") as rec:
            assert rec.lookup(now).content == "ok"
        assert _stale_cassettes[0][2] == "drift: request[0].message[2](tool).tool_result[0](camera_events).now"
        with pytest.raises(CassetteDriftError, match=r"request 0 differs from the recording\n\n  request\[0\]\.message\[2\]"):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, drift="fail") as rec:
                try:
                    rec.lookup(now)
                except CassetteDriftError:
                    pass  # a framework catching it; the cassette raises it again on exit
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, drift="fail", drift_ignore=["*.now"]) as rec:
            rec.lookup(now)
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, drift="fail", match_ignore=[r"\d\d:\d\d"]) as rec:
            rec.lookup(now)  # match_ignore applies too
        with pytest.raises(ValueError, match="drift must be one of"):
            LLMCassetteRecorder(path, drift="loud")


def test_pytest_settings_and_summary(pytester):
    pytester.makepyfile(
        """
from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.io import save_cassette
import pathlib

def _request(now):
    return NormalizedRequest(model="m", messages=[{"role": "tool", "tool_call_id": "c", "content": now}])

def _setup():
    path = pathlib.Path(__file__).parent / "cassettes" / "d.yaml"
    path.parent.mkdir(exist_ok=True)
    save_cassette(path, [(_request("10:00"), NormalizedResponse(content="ok"))], provider="openai")

def test_default(cassette):
    _setup()
    with cassette("d.yaml") as rec:
        rec.lookup(_request("11:00"))

def test_strict(cassette):
    _setup()
    with cassette("d.yaml", drift="fail") as rec:
        rec.lookup(_request("11:00"))
"""
    )
    pytester.makeini("[pytest]\nagentverify_cassette_drift = warn\n")
    result = pytester.runpytest_inprocess("-p", "agentverify")
    result.assert_outcomes(passed=1, failed=1)
    out = result.stdout.str()
    assert "CassetteDriftError" in out and "cassettes/d.yaml  (drift: request[0].message[0](tool).tool_result[0](?))" in out
    result = pytester.runpytest_inprocess("-p", "agentverify", "--cassette-drift=off", "-k", "default")
    result.assert_outcomes(passed=1)
    assert "stale cassettes" not in result.stdout.str()


def test_drift_works_without_request_matching(tmp_path):
    path = tmp_path / "c.yaml"
    save_cassette(path, [(RECORDED, NormalizedResponse(content="ok"))], provider="openai")
    with pytest.raises(CassetteDriftError):
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, match_requests=False, drift="fail") as rec:
            rec.lookup(_request('{"events": [], "now": "10:00"}', model="other", temperature=0))
