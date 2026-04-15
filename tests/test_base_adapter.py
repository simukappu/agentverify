"""Tests for agentverify.cassette.adapters.base — NormalizedRequest, NormalizedResponse dataclasses."""

from agentverify.cassette.adapters.base import (
    NormalizedRequest,
    NormalizedResponse,
)
from agentverify.models import TokenUsage


class TestNormalizedRequest:
    def test_defaults(self):
        req = NormalizedRequest(messages=[], model="gpt-4.1")
        assert req.messages == []
        assert req.model == "gpt-4.1"
        assert req.tools is None
        assert req.parameters == {}

    def test_with_tools(self):
        req = NormalizedRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4.1",
            tools=[{"name": "search", "parameters": {}}],
        )
        assert req.tools is not None
        assert len(req.tools) == 1

    def test_with_parameters(self):
        req = NormalizedRequest(
            messages=[],
            model="m",
            parameters={"temperature": 0.5},
        )
        assert req.parameters == {"temperature": 0.5}


class TestNormalizedResponse:
    def test_defaults(self):
        resp = NormalizedResponse()
        assert resp.content is None
        assert resp.tool_calls is None
        assert resp.token_usage is None
        assert resp.raw_metadata == {}

    def test_with_content(self):
        resp = NormalizedResponse(content="hello")
        assert resp.content == "hello"

    def test_with_tool_calls(self):
        resp = NormalizedResponse(
            tool_calls=[{"name": "search", "arguments": "{}"}]
        )
        assert len(resp.tool_calls) == 1

    def test_with_token_usage(self):
        tu = TokenUsage(input_tokens=10, output_tokens=5)
        resp = NormalizedResponse(token_usage=tu)
        assert resp.token_usage.total_tokens == 15

    def test_with_raw_metadata(self):
        resp = NormalizedResponse(raw_metadata={"model": "gpt-4.1", "id": "abc"})
        assert resp.raw_metadata["model"] == "gpt-4.1"
