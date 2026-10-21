"""Tests for agentverify.plugin — pytest_configure, pytest_addoption, and fixture import."""

from unittest.mock import MagicMock

from agentverify.plugin import cassette, pytest_addoption, pytest_configure


def test_pytest_configure_registers_marker():
    config = MagicMock()
    pytest_configure(config)
    [(kind, line)] = [c[0] for c in config.addinivalue_line.call_args_list]
    assert kind == "markers" and line.startswith("agentverify(runs=None, min_pass_rate=1.0):")


def test_cassette_fixture_is_imported():
    """The plugin module re-exports the cassette fixture."""
    assert cassette is not None
    assert callable(cassette)


def test_pytest_addoption_registers_cassette_mode():
    """pytest_addoption registers the --cassette-mode and --cassette-match-requests options."""
    parser = MagicMock()
    pytest_addoption(parser)
    assert parser.addoption.call_count == 3

    # First call: --cassette-mode
    first_call = parser.addoption.call_args_list[0]
    assert first_call[0][0] == "--cassette-mode"
    assert "record" in first_call[1]["choices"]
    assert "replay" in first_call[1]["choices"]
    assert "auto" in first_call[1]["choices"]

    # Second call: --no-cassette-match-requests
    second_call = parser.addoption.call_args_list[1]
    assert second_call[0][0] == "--no-cassette-match-requests"
    assert second_call[1]["action"] == "store_true"
    assert second_call[1]["default"] is False

    third_call = parser.addoption.call_args_list[2]
    assert third_call[0][0] == "--agentverify-live" and third_call[1]["default"] is False
