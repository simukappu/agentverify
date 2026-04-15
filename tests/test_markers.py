"""Tests for agentverify.markers — MARKER_NAME, MARKER_DESCRIPTION."""

from agentverify.markers import MARKER_DESCRIPTION, MARKER_NAME


def test_marker_name():
    assert MARKER_NAME == "agentverify"


def test_marker_description():
    assert isinstance(MARKER_DESCRIPTION, str)
    assert len(MARKER_DESCRIPTION) > 0
