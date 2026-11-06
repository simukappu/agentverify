"""--cassette-mode=record-missing: replay the cassettes that exist, record the ones that do not."""

from __future__ import annotations

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.io import load_document, save_cassette
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder, _new_cassettes

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _clear():
    _new_cassettes.clear()
    yield
    _new_cassettes.clear()


def test_the_recorder_replays_or_records(tmp_path):
    existing = tmp_path / "existing.yaml"
    save_cassette(existing, [(NormalizedRequest(messages=[], model="m"), NormalizedResponse(content="recorded"))], provider="openai")
    with LLMCassetteRecorder(existing, mode=CassetteMode.RECORD_MISSING) as rec:
        assert rec.mode == CassetteMode.REPLAY
        assert rec.lookup(NormalizedRequest(messages=[], model="m")).content == "recorded"
    missing = tmp_path / "missing.yaml"
    with LLMCassetteRecorder(missing, mode=CassetteMode.RECORD_MISSING, sanitize=True) as rec:
        assert rec.mode == CassetteMode.RECORD
        rec.record(NormalizedRequest(messages=[{"role": "user", "content": "key sk-abcdefghijklmnopqrstuvwxyz0123"}], model="m"), NormalizedResponse(content="live"))
    document = load_document(missing)
    assert "sk-abcdefghijklmnopqrstuvwxyz0123" not in str(document)  # sanitized like any recording
    assert [path for path, _ in _new_cassettes] == [str(missing)]


def test_pytest_lists_the_new_cassettes(pytester):
    pytester.makepyfile(
        """
from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse

def test_one(cassette):
    with cassette("one.yaml") as rec:
        if rec.mode.value == "record":
            rec.record(NormalizedRequest(messages=[], model="m"), NormalizedResponse(content="hi"))
        else:
            assert rec.lookup(NormalizedRequest(messages=[], model="m")).content == "hi"
"""
    )
    first = pytester.runpytest_inprocess("-p", "agentverify", "--cassette-mode=record-missing")
    first.assert_outcomes(passed=1)
    assert "agentverify: new cassettes" in first.stdout.str() and "cassettes/one.yaml" in first.stdout.str()
    second = pytester.runpytest_inprocess("-p", "agentverify", "--cassette-mode=record-missing")
    second.assert_outcomes(passed=1)
    assert "new cassettes" not in second.stdout.str()


def test_a_failed_recording_is_not_kept(tmp_path):
    path = tmp_path / "cut.yaml"
    with pytest.raises(RuntimeError):
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD_MISSING) as rec:
            rec.record(NormalizedRequest(messages=[], model="m"), NormalizedResponse(content="half"))
            raise RuntimeError("rate limited")
    assert not path.exists() and _new_cassettes == []
