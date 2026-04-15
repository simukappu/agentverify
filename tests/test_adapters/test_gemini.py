"""Tests for the Google Gemini SDK provider adapter.

Covers normalize_request, normalize_response, denormalize_response,
and the patch() context manager.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as unittest_mock_patch

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.adapters.gemini import GeminiAdapter
from agentverify.models import TokenUsage


@pytest.fixture
def adapter() -> GeminiAdapter:
    return GeminiAdapter()


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------


class TestName:
    def test_name(self, adapter: GeminiAdapter) -> None:
        assert adapter.name == "gemini"


# ---------------------------------------------------------------------------
# normalize_request
# ---------------------------------------------------------------------------


class TestNormalizeRequest:
    def test_basic_string_contents(self, adapter: GeminiAdapter) -> None:
        raw = {
            "model": "gemini-2.5-pro",
            "contents": "What's the weather in Tokyo?",
        }
        result = adapter.normalize_request(raw)
        assert result.messages == [{"role": "user", "content": "What's the weather in Tokyo?"}]
        assert result.model == "gemini-2.5-pro"
        assert result.tools is None
        assert result.parameters == {}

    def test_list_contents_with_strings(self, adapter: GeminiAdapter) -> None:
        raw = {
            "model": "gemini-2.5-pro",
            "contents": ["Hello", "World"],
        }
        result = adapter.normalize_request(raw)
        assert len(result.messages) == 2
        assert result.messages[0] == {"role": "user", "content": "Hello"}
        assert result.messages[1] == {"role": "user", "content": "World"}

    def test_list_contents_with_dicts(self, adapter: GeminiAdapter) -> None:
        raw = {
            "model": "gemini-2.5-pro",
            "contents": [{"role": "user", "content": "Hi there"}],
        }
        result = adapter.normalize_request(raw)
        assert result.messages == [{"role": "user", "content": "Hi there"}]

    def test_request_with_tools_dict(self, adapter: GeminiAdapter) -> None:
        raw = {
            "model": "gemini-2.5-pro",
            "contents": "Weather?",
            "config": {
                "tools": [
                    {
                        "function_declarations": [
                            {
                                "name": "get_weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"location": {"type": "string"}},
                                },
                            }
                        ]
                    }
                ],
                "temperature": 0.0,
            },
        }
        result = adapter.normalize_request(raw)
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "get_weather"
        assert "properties" in result.tools[0]["parameters"]
        assert result.parameters == {"temperature": 0.0}

    def test_request_with_tools_sdk_objects(self, adapter: GeminiAdapter) -> None:
        """Tools provided as SDK-like objects with function_declarations."""
        func_decl = MagicMock()
        func_decl.name = "search"
        func_decl.parameters = {"type": "object"}

        tool = MagicMock()
        tool.function_declarations = [func_decl]

        config = MagicMock()
        config.tools = [tool]
        config.temperature = 0.5
        config.top_p = None
        config.top_k = None
        config.max_output_tokens = None
        config.stop_sequences = None
        config.candidate_count = None
        config.response_mime_type = None

        raw = {
            "model": "gemini-2.5-pro",
            "contents": "Search for something",
            "config": config,
        }
        result = adapter.normalize_request(raw)
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "search"

    def test_extra_parameters(self, adapter: GeminiAdapter) -> None:
        raw = {
            "model": "gemini-2.5-pro",
            "contents": "Hi",
            "config": {"temperature": 0.7, "top_k": 40},
        }
        result = adapter.normalize_request(raw)
        assert result.parameters == {"temperature": 0.7, "top_k": 40}

    def test_empty_request(self, adapter: GeminiAdapter) -> None:
        result = adapter.normalize_request({})
        assert result.messages == []
        assert result.model == ""
        assert result.tools is None
        assert result.parameters == {}

    def test_empty_tools_list(self, adapter: GeminiAdapter) -> None:
        raw = {
            "model": "gemini-2.5-pro",
            "contents": "Hi",
            "config": {"tools": []},
        }
        result = adapter.normalize_request(raw)
        assert result.tools is None

    def test_non_list_non_string_contents(self, adapter: GeminiAdapter) -> None:
        """Non-string, non-list contents results in empty messages."""
        raw = {"model": "gemini-2.5-pro", "contents": 42}
        result = adapter.normalize_request(raw)
        assert result.messages == []

    def test_sdk_content_objects_in_list(self, adapter: GeminiAdapter) -> None:
        """SDK Content objects in the contents list are handled gracefully."""
        content_obj = MagicMock()
        content_obj.role = "user"
        content_obj.__str__ = lambda self: "Hello from SDK"

        raw = {"model": "gemini-2.5-pro", "contents": [content_obj]}
        result = adapter.normalize_request(raw)
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"

    def test_tools_with_dict_function_declarations(self, adapter: GeminiAdapter) -> None:
        """SDK Tool objects containing dict-based function declarations."""
        tool = MagicMock()
        tool.function_declarations = [
            {"name": "calc", "parameters": {"type": "object"}},
        ]

        config = MagicMock()
        config.tools = [tool]
        config.temperature = None
        config.top_p = None
        config.top_k = None
        config.max_output_tokens = None
        config.stop_sequences = None
        config.candidate_count = None
        config.response_mime_type = None

        raw = {"model": "gemini-2.5-pro", "contents": "calc", "config": config}
        result = adapter.normalize_request(raw)
        assert result.tools is not None
        assert result.tools[0]["name"] == "calc"

    def test_sdk_config_object_without_tools(self, adapter: GeminiAdapter) -> None:
        """SDK config object with no tools attribute set."""
        config = MagicMock()
        config.tools = None
        config.temperature = 0.9
        config.top_p = None
        config.top_k = None
        config.max_output_tokens = None
        config.stop_sequences = None
        config.candidate_count = None
        config.response_mime_type = None

        raw = {
            "model": "gemini-2.5-pro",
            "contents": "Hi",
            "config": config,
        }
        result = adapter.normalize_request(raw)
        assert result.tools is None
        assert result.parameters == {"temperature": 0.9}

    def test_extra_top_level_keys(self, adapter: GeminiAdapter) -> None:
        raw = {
            "model": "gemini-2.5-pro",
            "contents": "Hi",
            "safety_settings": [{"category": "HARM_CATEGORY_DANGEROUS"}],
        }
        result = adapter.normalize_request(raw)
        assert "safety_settings" in result.parameters


# ---------------------------------------------------------------------------
# normalize_response
# ---------------------------------------------------------------------------


def _make_gemini_response(
    content: str | None = None,
    function_calls: list[dict] | None = None,
    prompt_token_count: int = 50,
    candidates_token_count: int = 20,
) -> MagicMock:
    """Build a mock that mimics a Gemini GenerateContentResponse object."""
    mock = MagicMock()

    parts = []
    if content is not None:
        text_part = MagicMock()
        text_part.text = content
        text_part.function_call = None
        parts.append(text_part)

    if function_calls:
        for fc in function_calls:
            fc_part = MagicMock()
            fc_part.text = None
            fc_obj = MagicMock()
            fc_obj.name = fc["name"]
            fc_obj.args = fc["args"]
            fc_part.function_call = fc_obj
            parts.append(fc_part)

    candidate = MagicMock()
    candidate.content.parts = parts
    mock.candidates = [candidate]

    usage = MagicMock()
    usage.prompt_token_count = prompt_token_count
    usage.candidates_token_count = candidates_token_count
    mock.usage_metadata = usage

    return mock


class TestNormalizeResponse:
    def test_text_response(self, adapter: GeminiAdapter) -> None:
        raw = _make_gemini_response(content="Hello world")
        result = adapter.normalize_response(raw)
        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 50
        assert result.token_usage.output_tokens == 20

    def test_function_call_response(self, adapter: GeminiAdapter) -> None:
        raw = _make_gemini_response(
            function_calls=[
                {"name": "get_weather", "args": {"location": "Tokyo"}},
            ]
        )
        result = adapter.normalize_response(raw)
        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["arguments"] == '{"location": "Tokyo"}'

    def test_no_usage(self, adapter: GeminiAdapter) -> None:
        raw = _make_gemini_response(content="Hi")
        raw.usage_metadata = None
        result = adapter.normalize_response(raw)
        assert result.token_usage is None

    def test_mixed_text_and_function_call(self, adapter: GeminiAdapter) -> None:
        """Response with both text and function_call parts."""
        raw = _make_gemini_response(
            content="Let me check.",
            function_calls=[{"name": "get_weather", "args": {"location": "Tokyo"}}],
        )
        result = adapter.normalize_response(raw)
        assert result.content == "Let me check."
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_multiple_function_calls(self, adapter: GeminiAdapter) -> None:
        """Response with multiple function_call parts."""
        raw = _make_gemini_response(
            function_calls=[
                {"name": "get_weather", "args": {"location": "Tokyo"}},
                {"name": "get_news", "args": {"topic": "tech"}},
            ]
        )
        result = adapter.normalize_response(raw)
        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_news"

    def test_empty_candidates(self, adapter: GeminiAdapter) -> None:
        """Response with no candidates."""
        mock = MagicMock()
        mock.candidates = []
        mock.usage_metadata = None
        result = adapter.normalize_response(mock)
        assert result.content is None
        assert result.tool_calls is None
        assert result.token_usage is None

    def test_empty_args(self, adapter: GeminiAdapter) -> None:
        """Function call with empty args."""
        raw = _make_gemini_response(
            function_calls=[{"name": "no_args_tool", "args": {}}]
        )
        result = adapter.normalize_response(raw)
        assert result.tool_calls is not None
        assert result.tool_calls[0]["arguments"] == "{}"

    def test_none_args(self, adapter: GeminiAdapter) -> None:
        """Function call with None args."""
        raw = _make_gemini_response(
            function_calls=[{"name": "no_args_tool", "args": None}]
        )
        # Override the mock to set args to None
        fc_part = raw.candidates[0].content.parts[-1]
        fc_part.function_call.args = None
        result = adapter.normalize_response(raw)
        assert result.tool_calls is not None
        assert result.tool_calls[0]["arguments"] == "{}"

    def test_non_dict_args(self, adapter: GeminiAdapter) -> None:
        """Function call with non-dict args (e.g. MapComposite from protobuf)."""
        # Simulate a protobuf MapComposite that is not a dict but is iterable
        mock_args = MagicMock()
        mock_args.__iter__ = MagicMock(return_value=iter(["location"]))
        mock_args.__getitem__ = MagicMock(return_value="Tokyo")
        # Make isinstance(args, dict) return False
        type(mock_args).__instancecheck__ = lambda cls, inst: False

        raw = _make_gemini_response(
            function_calls=[{"name": "get_weather", "args": {"location": "Tokyo"}}]
        )
        # Replace args with our non-dict mock
        fc_part = raw.candidates[0].content.parts[-1]
        fc_part.function_call.args = mock_args
        result = adapter.normalize_response(raw)
        assert result.tool_calls is not None
        # dict(mock_args) should produce something from the mock

    def test_multiple_text_parts_concatenation(self, adapter: GeminiAdapter) -> None:
        """Multiple text parts in a response are concatenated."""
        mock = MagicMock()
        part1 = MagicMock()
        part1.text = "Hello "
        part1.function_call = None
        part2 = MagicMock()
        part2.text = "World"
        part2.function_call = None
        candidate = MagicMock()
        candidate.content.parts = [part1, part2]
        mock.candidates = [candidate]
        mock.usage_metadata = None
        result = adapter.normalize_response(mock)
        assert result.content == "Hello World"


# ---------------------------------------------------------------------------
# denormalize_response
# ---------------------------------------------------------------------------


class TestDenormalizeResponse:
    def test_text_response(self, adapter: GeminiAdapter) -> None:
        normalized = NormalizedResponse(
            content="Hello",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        result = adapter.denormalize_response(normalized)
        assert result.candidates[0].content.parts[0].text == "Hello"
        assert result.candidates[0].content.parts[0].function_call is None
        assert result.candidates[0].finish_reason == "STOP"
        assert result.usage_metadata.prompt_token_count == 10
        assert result.usage_metadata.candidates_token_count == 5
        assert result.text == "Hello"

    def test_function_call_response(self, adapter: GeminiAdapter) -> None:
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
            ],
            token_usage=TokenUsage(input_tokens=20, output_tokens=10),
        )
        result = adapter.denormalize_response(normalized)
        assert result.candidates[0].finish_reason == "FUNCTION_CALL"
        parts = result.candidates[0].content.parts
        assert len(parts) == 1
        assert parts[0].function_call.name == "get_weather"
        assert parts[0].function_call.args == {"location": "Tokyo"}
        assert parts[0].text is None

    def test_no_usage(self, adapter: GeminiAdapter) -> None:
        normalized = NormalizedResponse(content="Hi")
        result = adapter.denormalize_response(normalized)
        assert result.usage_metadata is None

    def test_invalid_json_arguments(self, adapter: GeminiAdapter) -> None:
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "some_tool", "arguments": "not-valid-json"},
            ],
        )
        result = adapter.denormalize_response(normalized)
        parts = result.candidates[0].content.parts
        assert parts[0].function_call.args == {}

    def test_dict_arguments(self, adapter: GeminiAdapter) -> None:
        """When arguments is already a dict, it's used directly."""
        normalized = NormalizedResponse(
            tool_calls=[
                {"name": "some_tool", "arguments": {"key": "value"}},
            ],
        )
        result = adapter.denormalize_response(normalized)
        parts = result.candidates[0].content.parts
        assert parts[0].function_call.args == {"key": "value"}

    def test_text_property(self, adapter: GeminiAdapter) -> None:
        """The .text property returns concatenated text from parts."""
        normalized = NormalizedResponse(content="Hello world")
        result = adapter.denormalize_response(normalized)
        assert result.text == "Hello world"

    def test_text_property_none_for_function_call(self, adapter: GeminiAdapter) -> None:
        """The .text property returns None when there are no text parts."""
        normalized = NormalizedResponse(
            tool_calls=[{"name": "tool", "arguments": "{}"}],
        )
        result = adapter.denormalize_response(normalized)
        assert result.text is None

    def test_text_property_empty_candidates(self, adapter: GeminiAdapter) -> None:
        """The .text property returns None when candidates list is empty."""
        from agentverify.cassette.adapters.gemini import _GenerateContentResponse

        resp = _GenerateContentResponse(candidates=[])
        assert resp.text is None


# ---------------------------------------------------------------------------
# Round-trip: normalize_response → denormalize_response preserves semantics
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_text_round_trip(self, adapter: GeminiAdapter) -> None:
        raw = _make_gemini_response(
            content="Hello world", prompt_token_count=50, candidates_token_count=20
        )
        normalized = adapter.normalize_response(raw)
        restored = adapter.denormalize_response(normalized)
        re_normalized = adapter.normalize_response(restored)

        assert re_normalized.content == normalized.content
        assert re_normalized.tool_calls == normalized.tool_calls
        assert re_normalized.token_usage == normalized.token_usage

    def test_function_call_round_trip(self, adapter: GeminiAdapter) -> None:
        raw = _make_gemini_response(
            function_calls=[
                {"name": "search", "args": {"q": "test"}},
                {"name": "fetch", "args": {"url": "https://example.com"}},
            ],
            prompt_token_count=100,
            candidates_token_count=30,
        )
        normalized = adapter.normalize_response(raw)
        restored = adapter.denormalize_response(normalized)
        re_normalized = adapter.normalize_response(restored)

        assert re_normalized.content == normalized.content
        assert re_normalized.tool_calls == normalized.tool_calls
        assert re_normalized.token_usage == normalized.token_usage


# ---------------------------------------------------------------------------
# patch() context manager
# ---------------------------------------------------------------------------


class TestPatch:
    """Test the patch() context manager for record and replay modes."""

    def test_patch_record_mode(self, adapter: GeminiAdapter) -> None:
        """In RECORD mode, the real API is called and the response is recorded."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.RECORD

        fake_response = _make_gemini_response(content="recorded answer")

        with unittest_mock_patch(
            "google.genai.models.Models.generate_content",
            return_value=fake_response,
        ) as mock_api:
            with adapter.patch(recorder):
                import google.genai.models

                client = MagicMock(spec=google.genai.models.Models)
                result = google.genai.models.Models.generate_content(
                    client,
                    model="gemini-2.5-pro",
                    contents="hi",
                )

        recorder.record.assert_called_once()

    def test_patch_replay_mode(self, adapter: GeminiAdapter) -> None:
        """In REPLAY mode, the cassette is looked up and denormalized."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = NormalizedResponse(
            content="replayed answer",
            token_usage=TokenUsage(input_tokens=5, output_tokens=3),
        )

        with unittest_mock_patch(
            "google.genai.models.Models.generate_content",
            return_value=MagicMock(),
        ):
            with adapter.patch(recorder):
                import google.genai.models

                client = MagicMock(spec=google.genai.models.Models)
                result = google.genai.models.Models.generate_content(
                    client,
                    model="gemini-2.5-pro",
                    contents="hi",
                )

        assert result.candidates[0].content.parts[0].text == "replayed answer"
        recorder.lookup.assert_called_once()

    def test_patch_replay_missing_raises(self, adapter: GeminiAdapter) -> None:
        """In REPLAY mode, if lookup returns None, CassetteMissingRequestError is raised."""
        from agentverify.cassette.recorder import CassetteMode
        from agentverify.errors import CassetteMissingRequestError

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None

        with unittest_mock_patch(
            "google.genai.models.Models.generate_content",
            return_value=MagicMock(),
        ):
            with adapter.patch(recorder):
                import google.genai.models

                client = MagicMock(spec=google.genai.models.Models)
                with pytest.raises(CassetteMissingRequestError):
                    google.genai.models.Models.generate_content(
                        client,
                        model="gemini-2.5-pro",
                        contents="hi",
                    )

    def test_patch_auto_mode(self, adapter: GeminiAdapter) -> None:
        """AUTO mode behaves like RECORD — calls real API and records."""
        from agentverify.cassette.recorder import CassetteMode

        recorder = MagicMock()
        recorder.mode = CassetteMode.AUTO

        fake_response = _make_gemini_response(content="auto answer")

        with unittest_mock_patch(
            "google.genai.models.Models.generate_content",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import google.genai.models

                client = MagicMock(spec=google.genai.models.Models)
                result = google.genai.models.Models.generate_content(
                    client,
                    model="gemini-2.5-pro",
                    contents="hi",
                )

        recorder.record.assert_called_once()

    def test_patch_replay_fallback(self, adapter: GeminiAdapter) -> None:
        """In REPLAY mode with FALLBACK, if lookup returns None, the real API is called."""
        from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None
        recorder.on_missing = OnMissingRequest.FALLBACK

        fake_response = _make_gemini_response(content="fallback answer")

        with unittest_mock_patch(
            "google.genai.models.Models.generate_content",
            return_value=fake_response,
        ):
            with adapter.patch(recorder):
                import google.genai.models

                client = MagicMock(spec=google.genai.models.Models)
                result = google.genai.models.Models.generate_content(
                    client,
                    model="gemini-2.5-pro",
                    contents="hi",
                )

        recorder.record.assert_called_once()
