"""Tests for agentverify.cassette.io — load_cassette, save_cassette."""

import json

import pytest
import yaml

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.io import load_cassette, save_cassette
from agentverify.models import TokenUsage


def _make_interactions():
    req = NormalizedRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4.1",
        tools=[{"name": "search", "parameters": {}}],
        parameters={"temperature": 0.0},
    )
    resp = NormalizedResponse(
        content="Hi there",
        tool_calls=[{"name": "search", "arguments": "{}"}],
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    return [(req, resp)]


# ---------------------------------------------------------------------------
# YAML format
# ---------------------------------------------------------------------------


class TestYAML:
    def test_save_and_load_yaml(self, tmp_path):
        path = tmp_path / "test.yaml"
        interactions = _make_interactions()
        save_cassette(path, interactions, provider="openai", model="gpt-4.1")

        assert path.exists()
        metadata, loaded = load_cassette(path)
        assert metadata["provider"] == "openai"
        assert metadata["model"] == "gpt-4.1"
        assert len(loaded) == 1
        req, resp = loaded[0]
        assert req.model == "gpt-4.1"
        assert req.messages == [{"role": "user", "content": "Hello"}]
        assert req.tools == [{"name": "search", "parameters": {}}]
        assert req.parameters == {"temperature": 0.0}
        assert resp.content == "Hi there"
        assert resp.tool_calls == [{"name": "search", "arguments": "{}"}]
        assert resp.token_usage.input_tokens == 10
        assert resp.token_usage.output_tokens == 5

    def test_yml_extension(self, tmp_path):
        path = tmp_path / "test.yml"
        save_cassette(path, _make_interactions(), provider="openai")
        metadata, loaded = load_cassette(path)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# JSON format
# ---------------------------------------------------------------------------


class TestJSON:
    def test_save_and_load_json(self, tmp_path):
        path = tmp_path / "test.json"
        interactions = _make_interactions()
        save_cassette(path, interactions, provider="openai", model="gpt-4.1")

        assert path.exists()
        metadata, loaded = load_cassette(path)
        assert metadata["provider"] == "openai"
        assert len(loaded) == 1

    def test_json_content_valid(self, tmp_path):
        path = tmp_path / "test.json"
        save_cassette(path, _make_interactions())
        raw = json.loads(path.read_text())
        assert "metadata" in raw
        assert "interactions" in raw


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unsupported_extension_load(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported cassette file extension"):
            load_cassette(path)

    def test_unsupported_extension_save(self, tmp_path):
        path = tmp_path / "test.txt"
        with pytest.raises(ValueError, match="Unsupported cassette file extension"):
            save_cassette(path, [])

    def test_empty_interactions(self, tmp_path):
        path = tmp_path / "empty.yaml"
        save_cassette(path, [], provider="test")
        metadata, loaded = load_cassette(path)
        assert loaded == []

    def test_no_token_usage(self, tmp_path):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(content="hi")
        path = tmp_path / "no_tokens.yaml"
        save_cassette(path, [(req, resp)])
        _, loaded = load_cassette(path)
        assert loaded[0][1].token_usage is None

    def test_auto_model_from_interactions(self, tmp_path):
        """When model="" is passed, it should be inferred from the first interaction."""
        req = NormalizedRequest(messages=[], model="gpt-4.1-mini")
        resp = NormalizedResponse(content="ok")
        path = tmp_path / "auto_model.yaml"
        save_cassette(path, [(req, resp)], provider="openai")
        metadata, _ = load_cassette(path)
        assert metadata["model"] == "gpt-4.1-mini"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.yaml"
        save_cassette(path, _make_interactions())
        assert path.exists()

    def test_load_minimal_yaml(self, tmp_path):
        """Load a YAML file with no metadata or interactions keys."""
        path = tmp_path / "minimal.yaml"
        path.write_text("{}")
        metadata, loaded = load_cassette(path)
        assert metadata == {}
        assert loaded == []

    def test_load_empty_yaml(self, tmp_path):
        """Load a YAML file that is completely empty (safe_load returns None)."""
        path = tmp_path / "empty_file.yaml"
        path.write_text("")
        metadata, loaded = load_cassette(path)
        assert metadata == {}
        assert loaded == []

    def test_response_no_tools_no_content(self, tmp_path):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse()
        path = tmp_path / "bare.json"
        save_cassette(path, [(req, resp)])
        _, loaded = load_cassette(path)
        assert loaded[0][1].content is None
        assert loaded[0][1].tool_calls is None

    def test_request_no_tools_no_params(self, tmp_path):
        req = NormalizedRequest(messages=[{"role": "user", "content": "hi"}], model="m")
        resp = NormalizedResponse(content="ok")
        path = tmp_path / "no_tools.yaml"
        save_cassette(path, [(req, resp)])
        raw = yaml.safe_load(path.read_text())
        # tools and parameters should not be in serialized output when empty/None
        req_data = raw["interactions"][0]["request"]
        assert "tools" not in req_data
        assert "parameters" not in req_data
