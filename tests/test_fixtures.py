"""Tests for agentverify.fixtures — the cassette fixture factory."""

from pathlib import Path

from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder


class TestCassetteFixture:
    """Test the cassette fixture via pytest's fixture injection."""

    def test_returns_factory(self, cassette):
        assert callable(cassette)

    def test_factory_creates_recorder(self, cassette, tmp_path):
        rec = cassette("test.yaml", mode=CassetteMode.RECORD, cassette_dir=tmp_path)
        assert isinstance(rec, LLMCassetteRecorder)
        assert rec.mode == CassetteMode.RECORD

    def test_default_cassette_dir(self, cassette):
        """Default cassette dir is 'cassettes/' next to the test file."""
        rec = cassette("my.yaml", mode=CassetteMode.RECORD)
        assert "cassettes" in str(rec.cassette_path)
        assert rec.cassette_path.name == "my.yaml"

    def test_custom_cassette_dir(self, cassette, tmp_path):
        custom_dir = tmp_path / "custom"
        rec = cassette("test.yaml", mode=CassetteMode.RECORD, cassette_dir=custom_dir)
        assert rec.cassette_path == custom_dir / "test.yaml"

    def test_string_mode_conversion(self, cassette, tmp_path):
        rec = cassette("test.yaml", mode="record", cassette_dir=tmp_path)
        assert rec.mode == CassetteMode.RECORD

    def test_provider_passed_through(self, cassette, tmp_path):
        rec = cassette("test.yaml", mode=CassetteMode.RECORD, provider="openai", cassette_dir=tmp_path)
        assert rec._adapter.name == "openai"

    def test_cassette_dir_as_string(self, cassette, tmp_path):
        rec = cassette("test.yaml", mode=CassetteMode.RECORD, cassette_dir=str(tmp_path / "str_dir"))
        assert rec.cassette_path == tmp_path / "str_dir" / "test.yaml"

    def test_default_mode_is_auto(self, cassette, tmp_path):
        """When no mode is specified and no CLI option, default is AUTO."""
        rec = cassette("test.yaml", cassette_dir=tmp_path)
        assert rec.mode == CassetteMode.AUTO

    def test_string_on_missing_conversion(self, cassette, tmp_path):
        """on_missing can be passed as a string and is converted to enum."""
        from agentverify.cassette.recorder import OnMissingRequest

        rec = cassette("test.yaml", mode=CassetteMode.RECORD, cassette_dir=tmp_path, on_missing="error")
        assert rec.on_missing == OnMissingRequest.ERROR


class TestCassetteFixtureCLIOption:
    """Test the cassette fixture with --cassette-mode CLI option."""

    def test_cli_mode_override(self, cassette, tmp_path, request):
        """When --cassette-mode is set, it overrides the default AUTO mode."""
        # Simulate the CLI option being set
        original = request.config.getoption
        request.config.getoption = lambda key, default=None: "record" if key == "--cassette-mode" else original(key, default=default)
        try:
            rec = cassette("test.yaml", cassette_dir=tmp_path)
            assert rec.mode == CassetteMode.RECORD
        finally:
            request.config.getoption = original
