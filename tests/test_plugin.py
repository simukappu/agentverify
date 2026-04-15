"""Tests for agentverify.plugin — pytest_configure, pytest_addoption, and fixture import."""

from unittest.mock import MagicMock

from agentverify.plugin import cassette, pytest_addoption, pytest_configure


def test_pytest_configure_registers_marker():
    config = MagicMock()
    pytest_configure(config)
    config.addinivalue_line.assert_called_once_with(
        "markers",
        "agentverify: mark test as an agentverify agent test",
    )


def test_cassette_fixture_is_imported():
    """The plugin module re-exports the cassette fixture."""
    assert cassette is not None
    assert callable(cassette)


def test_pytest_addoption_registers_cassette_mode():
    """pytest_addoption registers the --cassette-mode option."""
    parser = MagicMock()
    pytest_addoption(parser)
    parser.addoption.assert_called_once()
    call_kwargs = parser.addoption.call_args
    assert call_kwargs[0][0] == "--cassette-mode"
    assert "record" in call_kwargs[1]["choices"]
    assert "replay" in call_kwargs[1]["choices"]
    assert "auto" in call_kwargs[1]["choices"]
