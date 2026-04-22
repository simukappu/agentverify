"""Tests for agentverify.mocking — MockLLM and mock_response."""

import pytest

from agentverify import (
    MockLLM,
    ToolCall,
    assert_cost,
    assert_tool_calls,
    mock_response,
)
from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.recorder import CassetteMode, OnMissingRequest
from agentverify.errors import CassetteMissingRequestError
from agentverify.models import TokenUsage


# ---------------------------------------------------------------------------
# mock_response builder
# ---------------------------------------------------------------------------


class TestMockResponse:
    def test_content_only(self):
        resp = mock_response(content="hello")
        assert isinstance(resp, NormalizedResponse)
        assert resp.content == "hello"
        assert resp.tool_calls is None
        assert resp.token_usage is None

    def test_tool_calls_tuple_form(self):
        resp = mock_response(tool_calls=[("search", {"q": "x"}), ("fetch", {"url": "u"})])
        assert resp.tool_calls == [
            {"name": "search", "arguments": {"q": "x"}},
            {"name": "fetch", "arguments": {"url": "u"}},
        ]

    def test_tool_calls_dict_form(self):
        resp = mock_response(
            tool_calls=[{"name": "search", "arguments": {"q": "x"}}]
        )
        assert resp.tool_calls == [{"name": "search", "arguments": {"q": "x"}}]

    def test_tool_calls_dict_without_arguments(self):
        resp = mock_response(tool_calls=[{"name": "noop"}])
        assert resp.tool_calls == [{"name": "noop", "arguments": {}}]

    def test_tool_calls_invalid_type(self):
        with pytest.raises(TypeError, match="must be tuple or dict"):
            mock_response(tool_calls=["not_a_tuple_or_dict"])

    def test_token_usage_populated(self):
        resp = mock_response(content="x", input_tokens=10, output_tokens=5)
        assert resp.token_usage == TokenUsage(input_tokens=10, output_tokens=5)

    def test_token_usage_omitted_when_zero(self):
        resp = mock_response(content="x")
        assert resp.token_usage is None

    def test_token_usage_populated_when_only_output(self):
        resp = mock_response(content="x", output_tokens=5)
        assert resp.token_usage == TokenUsage(input_tokens=0, output_tokens=5)


# ---------------------------------------------------------------------------
# MockLLM — core behaviour
# ---------------------------------------------------------------------------


class TestMockLLMInit:
    def test_init_defaults(self):
        mock = MockLLM([], provider="openai")
        assert mock._adapter.name == "openai"
        assert mock.mode == CassetteMode.REPLAY
        assert mock.on_missing == OnMissingRequest.ERROR
        assert mock._interactions == []
        assert mock._replay_index == 0
        assert mock._duration_ms is None

    def test_init_copies_responses(self):
        """MockLLM should not mutate the caller's list."""
        resps = [mock_response(content="a")]
        mock = MockLLM(resps, provider="openai")
        resps.append(mock_response(content="b"))
        assert len(mock._responses) == 1


class TestMockLLMLookup:
    def test_sequential_lookup(self):
        mock = MockLLM(
            [mock_response(content="first"), mock_response(content="second")],
            provider="openai",
        )
        req = NormalizedRequest(messages=[], model="gpt-4")
        assert mock.lookup(req).content == "first"
        assert mock.lookup(req).content == "second"

    def test_lookup_records_interaction(self):
        mock = MockLLM([mock_response(content="x")], provider="openai")
        req = NormalizedRequest(messages=[{"role": "user", "content": "hi"}], model="gpt-4")
        mock.lookup(req)
        assert len(mock._interactions) == 1
        assert mock._interactions[0][0] is req

    def test_lookup_exhausted_raises(self):
        mock = MockLLM([], provider="openai")
        req = NormalizedRequest(messages=[], model="gpt-4")
        with pytest.raises(CassetteMissingRequestError, match="ran out of predefined"):
            mock.lookup(req)

    def test_lookup_exhausted_after_consuming_all(self):
        mock = MockLLM([mock_response(content="only")], provider="openai")
        req = NormalizedRequest(messages=[], model="gpt-4")
        mock.lookup(req)
        with pytest.raises(CassetteMissingRequestError):
            mock.lookup(req)


class TestMockLLMContextManager:
    def test_enter_exit_captures_duration(self):
        mock = MockLLM([], provider="openai")
        with mock:
            pass
        assert mock._duration_ms is not None
        assert mock._duration_ms >= 0

    def test_context_manager_returns_self(self):
        mock = MockLLM([], provider="openai")
        with mock as rec:
            assert rec is mock

    def test_exit_without_enter_is_safe(self):
        """Calling __exit__ without __enter__ should be a no-op (defensive)."""
        mock = MockLLM([], provider="openai")
        mock.__exit__(None, None, None)
        assert mock._duration_ms is None
        assert mock._patch_ctx is None


# ---------------------------------------------------------------------------
# MockLLM — to_execution_result
# ---------------------------------------------------------------------------


class TestMockLLMToExecutionResult:
    def test_empty_session(self):
        mock = MockLLM([], provider="openai")
        with mock:
            pass
        result = mock.to_execution_result()
        assert result.tool_calls == []
        assert result.token_usage is None
        assert result.final_output is None
        assert result.duration_ms is not None

    def test_collects_tool_calls_and_content(self):
        mock = MockLLM(
            [
                mock_response(
                    tool_calls=[("search", {"q": "Tokyo"})],
                    input_tokens=5,
                    output_tokens=3,
                ),
                mock_response(content="Tokyo is sunny", input_tokens=10, output_tokens=5),
            ],
            provider="openai",
        )
        req = NormalizedRequest(messages=[], model="gpt-4")
        mock.lookup(req)
        mock.lookup(req)

        result = mock.to_execution_result()
        assert result.tool_calls == [ToolCall("search", {"q": "Tokyo"})]
        assert result.token_usage.input_tokens == 15
        assert result.token_usage.output_tokens == 8
        assert result.final_output == "Tokyo is sunny"

    def test_tool_call_arguments_json_string(self):
        """Some adapters may hand tool call arguments as JSON strings."""
        mock = MockLLM(
            [
                NormalizedResponse(
                    tool_calls=[{"name": "calc", "arguments": '{"x": 1}'}],
                )
            ],
            provider="openai",
        )
        mock.lookup(NormalizedRequest(messages=[], model="m"))
        result = mock.to_execution_result()
        assert result.tool_calls[0].arguments == {"x": 1}

    def test_tool_call_arguments_invalid_json_string(self):
        mock = MockLLM(
            [
                NormalizedResponse(
                    tool_calls=[{"name": "broken", "arguments": "not-json"}],
                )
            ],
            provider="openai",
        )
        mock.lookup(NormalizedRequest(messages=[], model="m"))
        result = mock.to_execution_result()
        assert result.tool_calls[0].arguments == {}


# ---------------------------------------------------------------------------
# MockLLM — end-to-end with a real SDK
# ---------------------------------------------------------------------------


class TestMockLLMEndToEnd:
    """Patch the OpenAI SDK via the provider adapter and verify that a
    caller using the SDK ends up with mocked responses."""

    def test_openai_routing_without_real_llm(self):
        openai = pytest.importorskip("openai")
        client = openai.OpenAI(api_key="dummy")

        with MockLLM(
            [
                mock_response(
                    tool_calls=[("get_weather", {"city": "Tokyo"})],
                    input_tokens=5,
                    output_tokens=3,
                ),
                mock_response(
                    content="Tokyo is sunny, 22C", input_tokens=10, output_tokens=5
                ),
            ],
            provider="openai",
        ) as rec:
            r1 = client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": "hi"}]
            )
            assert r1.choices[0].message.tool_calls[0].function.name == "get_weather"
            r2 = client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": "follow up"}]
            )
            assert r2.choices[0].message.content == "Tokyo is sunny, 22C"

        result = rec.to_execution_result()
        assert_tool_calls(
            result, expected=[ToolCall("get_weather", {"city": "Tokyo"})]
        )
        assert_cost(result, max_tokens=30)


