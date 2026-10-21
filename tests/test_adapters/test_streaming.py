"""Tests for recording and replaying streamed LLM responses."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch as unittest_mock_patch

import pytest

from agentverify.cassette.adapters.base import (
    NormalizedRequest,
    NormalizedResponse,
    StreamAccumulator,
    StreamingWire,
    _ReplayedHTTPResponse,
    call_async,
    call_sync,
    patch_scope,
    record_stream,
    replayed_sdk_stream,
    tap_stream,
)
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder
from agentverify.errors import CassetteUnsupportedCallError
from agentverify.models import TokenUsage

RESPONSE = NormalizedResponse(
    content="Sunny",
    tool_calls=[{"name": "get_weather", "arguments": '{"city": "Tokyo"}', "id": "call_1"}],
    token_usage=TokenUsage(input_tokens=7, output_tokens=3),
    raw_metadata={"model": "m"},
)


class _Collect(StreamAccumulator):
    def __init__(self) -> None:
        self.items: list = []

    def add(self, item):
        self.items.append(item)

    def result(self):
        return NormalizedResponse(content="".join(self.items) or None)


def _finished():
    results: list[NormalizedResponse] = []
    return results, results.append


# ---------------------------------------------------------------------------
# The stream tap
# ---------------------------------------------------------------------------


class _NextStream:
    """Shaped like the OpenAI / Anthropic SDK stream: __next__ reads _iterator."""

    def __init__(self, items):
        self._iterator = iter(items)
        self.closed = False

    def __next__(self):
        return next(self._iterator)

    def __iter__(self):
        yield from self._iterator

    def close(self):
        self.closed = True


class _IterStream:
    """Shaped like botocore's EventStream: a generator __iter__ and no __next__."""

    def __init__(self, items):
        self._items = items

    def __iter__(self):
        yield from self._items


class _AsyncStream:
    def __init__(self, items):
        self._items = iter(items)
        self.closed = False

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None

    def __aiter__(self):
        return self

    async def close(self):
        self.closed = True


class TestTapStream:
    def test_next_stream_records_when_exhausted_and_keeps_its_class(self):
        results, finish = _finished()
        stream = tap_stream(_NextStream(["a", "b"]), _Collect(), finish)
        assert isinstance(stream, _NextStream) and type(stream).__name__ == "_NextStream"
        assert next(stream) == "a"
        assert list(stream) == ["b"]
        assert [r.content for r in results] == ["ab"]
        stream.close()  # closing after the end records nothing more
        assert stream.closed and len(results) == 1

    def test_close_before_the_end_records_what_was_read(self):
        results, finish = _finished()
        stream = tap_stream(_NextStream(["a", "b"]), _Collect(), finish)
        next(stream)
        stream.close()
        assert [r.content for r in results] == ["a"]

    def test_iter_only_stream(self):
        results, finish = _finished()
        assert list(tap_stream(_IterStream(["x", "y"]), _Collect(), finish)) == ["x", "y"]
        assert results[0].content == "xy"

    def test_async_stream_with_async_close(self):
        results, finish = _finished()

        async def read():
            stream = tap_stream(_AsyncStream(["p", "q"]), _Collect(), finish)
            first = await stream.__anext__()
            await stream.close()
            return first, stream.closed

        assert asyncio.run(read()) == ("p", True)
        assert results[0].content == "p"

        results2, finish2 = _finished()

        async def read_all():
            return [item async for item in tap_stream(_AsyncStream(["r"]), _Collect(), finish2)]

        assert asyncio.run(read_all()) == ["r"]
        assert results2[0].content == "r"

    def test_generators(self):
        results, finish = _finished()
        assert list(tap_stream((c for c in "hi"), _Collect(), finish)) == ["h", "i"]

        async def agen():
            yield "o"

        async def read():
            return [item async for item in tap_stream(agen(), _Collect(), finish)]

        assert asyncio.run(read()) == ["o"]
        assert [r.content for r in results] == ["hi", "o"]

    def test_objects_that_are_not_streams_are_returned_as_they_are(self):
        results, finish = _finished()
        assert tap_stream("not a stream", _Collect(), finish) == "not a stream"
        assert results == []

    def test_tapped_classes_are_reused(self):
        first = tap_stream(_NextStream([]), _Collect(), lambda r: None)
        second = tap_stream(_NextStream([]), _Collect(), lambda r: None)
        assert type(first) is type(second)


class TestRecordStream:
    def test_slot_is_reserved_in_request_order_and_filled_later(self):
        rec = LLMCassetteRecorder("/nonexistent/never-saved.yaml", mode=CassetteMode.AUTO)
        wire = MagicMock(spec=StreamingWire)
        wire.stream_accumulator.return_value = _Collect()
        wire.tap_response.side_effect = lambda response, tap: tap(response)
        stream = record_stream(rec, wire, NormalizedRequest(messages=[], model="m"), _NextStream(["a"]))
        rec.record(NormalizedRequest(messages=[], model="m"), NormalizedResponse(content="second"))
        assert rec._interactions[0][1].content is None
        list(stream)
        assert [r.content for _, r in rec._interactions] == ["a", "second"]

    def test_an_unread_stream_is_filled_in_when_the_cassette_closes(self, tmp_path):
        path = tmp_path / "unread.yaml"
        from agentverify.cassette.adapters.openai import OpenAIAdapter

        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai") as rec:
            stream = record_stream(rec, OpenAIAdapter(), NormalizedRequest(messages=[], model="m"), _NextStream([]))
            assert rec._pending_streams
        assert rec._pending_streams == []
        del stream
        assert "interactions:" in path.read_text()

    def test_recorder_without_pending_list(self):
        recorder = SimpleNamespace(record=MagicMock())
        wire = MagicMock(spec=StreamingWire)
        wire.stream_accumulator.return_value = _Collect()
        wire.tap_response.side_effect = lambda response, tap: tap(response)
        list(record_stream(recorder, wire, NormalizedRequest(messages=[], model="m"), _NextStream(["z"])))
        assert recorder.record.call_args[0][1].content == "z"

    def test_default_tap_response_taps_the_response_itself(self):
        class Wire(StreamingWire):
            def stream_accumulator(self):
                return _Collect()

            def replay_stream(self, normalized, kwargs, is_async):
                return None

        assert Wire().tap_response("r", lambda s: s + "!") == "r!"


class TestReplayedSdkStream:
    def test_sync_and_async(self):
        stream = replayed_sdk_stream(_NextStream, _AsyncStream, [1, 2], is_async=False)
        assert isinstance(stream, _NextStream) and list(stream) == [1, 2]
        astream = replayed_sdk_stream(_NextStream, _AsyncStream, [3], is_async=True)
        assert isinstance(astream, _AsyncStream)

        async def first():
            return await astream._iterator.__anext__()

        assert asyncio.run(first()) == 3

    def test_http_response_stand_in(self):
        response = _ReplayedHTTPResponse()
        response.close()
        asyncio.run(response.aclose())
        assert response.status_code == 200 and response.headers == {}


def test_reject_streaming_async_raises_while_recording():
    from agentverify.cassette.adapters.base import PatchScope, reject_streaming

    async def original():  # pragma: no cover — never called
        return "real"

    record = MagicMock()
    record.mode = CassetteMode.RECORD
    wrapper = reject_streaming(PatchScope(), record, "p", "x.astream", original, lambda: original)
    with pytest.raises(CassetteUnsupportedCallError, match="x.astream"):
        asyncio.run(wrapper())


class TestNonStreamingWire:
    """A custom adapter without streaming support keeps the 0.6.0 behaviour."""

    def _adapter(self):
        adapter = MagicMock()
        adapter.name = "custom"
        return adapter

    def test_rejects_outside_auto_and_passes_through_in_auto(self):
        scope = SimpleNamespace(active=True)
        record = MagicMock()
        record.mode = CassetteMode.RECORD
        with pytest.raises(CassetteUnsupportedCallError, match=r"create\(stream=True\)"):
            call_sync(scope, record, self._adapter(), "create", {"stream": True}, lambda: "live")
        auto = MagicMock()
        auto.mode = CassetteMode.AUTO
        assert call_sync(scope, auto, self._adapter(), "create", {"stream": True}, lambda: "live") == "live"

        async def live():
            return "alive"

        assert asyncio.run(call_async(scope, auto, self._adapter(), "create", {"stream": True}, live)) == "alive"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _replay_recorder(response: NormalizedResponse = RESPONSE) -> MagicMock:
    recorder = MagicMock()
    recorder.mode = CassetteMode.REPLAY
    recorder.lookup.return_value = response
    return recorder


class TestOpenAIChat:
    @pytest.fixture(autouse=True)
    def _openai(self):
        pytest.importorskip("openai")

    def _client(self, is_async=False):
        import openai

        return (openai.AsyncOpenAI if is_async else openai.OpenAI)(api_key="k", max_retries=0)

    def test_record_then_replay_through_the_stream_helper(self, tmp_path):
        import openai.resources.chat.completions as completions

        from agentverify.cassette.adapters.openai import OpenAIAdapter

        path = tmp_path / "chat.yaml"

        def live(self, **kwargs):
            return OpenAIAdapter().replay_stream(RESPONSE, kwargs, False)

        messages = [{"role": "user", "content": "hi"}]
        with unittest_mock_patch.object(completions.Completions, "create", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai"):
                chunks = list(
                    self._client().chat.completions.create(
                        model="m", messages=messages, stream=True, stream_options={"include_usage": True}
                    )
                )
        assert chunks[-1].usage.prompt_tokens == 7
        with unittest_mock_patch.object(completions.Completions, "create", side_effect=AssertionError("live")):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="openai") as rec:
                with self._client().chat.completions.stream(model="m", messages=messages) as stream:
                    final = stream.get_final_completion()
        assert final.choices[0].message.content == "Sunny"
        assert final.choices[0].message.tool_calls[0].id == "call_1"
        assert rec.to_execution_result().token_usage == TokenUsage(7, 3)

    def test_a_non_streamed_recording_replays_as_a_stream_and_back(self):
        from agentverify.cassette.adapters.openai import OpenAIAdapter

        with OpenAIAdapter().patch(_replay_recorder()):
            chunks = list(self._client().chat.completions.create(model="m", messages=[], stream=True))
            plain = self._client().chat.completions.create(model="m", messages=[])
        assert chunks[0].choices[0].delta.content == "Sunny"
        assert all(c.usage is None for c in chunks)  # usage only when stream_options asks for it
        assert plain.choices[0].message.content == "Sunny"

    def test_async(self):
        from agentverify.cassette.adapters.openai import OpenAIAdapter

        async def read():
            stream = await self._client(True).chat.completions.create(model="m", messages=[], stream=True)
            async with stream:
                return [c async for c in stream]

        with OpenAIAdapter().patch(_replay_recorder()):
            chunks = asyncio.run(read())
        assert chunks[-1].choices[0].finish_reason == "tool_calls"

    def test_accumulator_edge_cases(self):
        from agentverify.cassette.adapters.openai import ChatChunkAccumulator, chat_chunks, wants_usage

        acc = ChatChunkAccumulator()
        delta = lambda **kw: SimpleNamespace(content=None, tool_calls=None, **kw)  # noqa: E731
        acc.add(SimpleNamespace(model="m", usage=None, choices=[SimpleNamespace(index=0, delta=None)]))
        acc.add(SimpleNamespace(usage=None, choices=[SimpleNamespace(index=1, delta=delta())]))
        call = SimpleNamespace(index=None, id=None, function=None)
        acc.add(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=None), choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="x", tool_calls=[call]))]))
        result = acc.result()
        assert result.content == "x" and result.tool_calls == [{"name": "", "arguments": ""}] and result.token_usage is None
        assert result.raw_metadata == {"model": "m"}
        assert ChatChunkAccumulator().result() == NormalizedResponse()
        chunks = chat_chunks(NormalizedResponse(tool_calls=[{"name": "f", "arguments": {"a": 1}}]), include_usage=True)
        assert json.loads(chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"  # no usage recorded, no usage chunk
        assert not wants_usage({"stream_options": "bad"}) and wants_usage({"stream_options": {"include_usage": True}})

    def test_mock_llm_streams(self):
        from agentverify import MockLLM, mock_response

        with MockLLM([mock_response(content="mocked")], provider="openai"):
            chunks = list(self._client().chat.completions.create(model="m", messages=[], stream=True))
        assert chunks[0].choices[0].delta.content == "mocked"


class TestOpenAIResponses:
    @pytest.fixture(autouse=True)
    def _openai(self):
        pytest.importorskip("openai")

    def _client(self, is_async=False):
        import openai

        return (openai.AsyncOpenAI if is_async else openai.OpenAI)(api_key="k", max_retries=0)

    def test_record_then_replay_through_the_stream_helper(self, tmp_path):
        import openai.resources.responses as responses

        from agentverify.cassette.adapters.openai_responses import ResponsesWire

        path = tmp_path / "responses.yaml"

        def live(self, **kwargs):
            return ResponsesWire().replay_stream(RESPONSE, kwargs, False)

        with unittest_mock_patch.object(responses.Responses, "create", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai"):
                events = [e.type for e in self._client().responses.create(model="m", input="hi", stream=True)]
        assert events[0] == "response.created" and events[-1] == "response.completed"
        with unittest_mock_patch.object(responses.Responses, "create", side_effect=AssertionError("live")):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="openai") as rec:
                with self._client().responses.stream(model="m", input="hi") as stream:
                    final = stream.get_final_response()
        assert final.output_text == "Sunny" and final.output[1].call_id == "call_1"
        assert rec.to_execution_result().tool_calls[0].name == "get_weather"

    def test_async(self):
        from agentverify.cassette.adapters.openai import OpenAIAdapter

        async def read():
            stream = await self._client(True).responses.create(model="m", input="hi", stream=True)
            return [e.type async for e in stream]

        with OpenAIAdapter().patch(_replay_recorder(NormalizedResponse(content="only text"))):
            events = asyncio.run(read())
        assert "response.function_call_arguments.delta" not in events

    def test_accumulator_without_a_final_event(self):
        from agentverify.cassette.adapters.openai_responses import ResponseEventAccumulator, response_events

        from openai._models import construct_type
        from openai.types.responses import ResponseStreamEvent

        acc = ResponseEventAccumulator()
        for event in response_events(RESPONSE)[:-1]:  # cut before response.completed
            acc.add(construct_type(type_=ResponseStreamEvent, value=event))
        result = acc.result()
        assert result.content == "Sunny" and result.tool_calls[0]["id"] == "call_1"
        assert result.token_usage is None


class TestAnthropic:
    @pytest.fixture(autouse=True)
    def _anthropic(self):
        pytest.importorskip("anthropic")

    def _client(self, is_async=False):
        import anthropic

        return (anthropic.AsyncAnthropic if is_async else anthropic.Anthropic)(api_key="k", max_retries=0)

    def test_record_through_the_stream_helper_then_replay_create(self, tmp_path):
        import anthropic.resources.messages as messages

        from agentverify.cassette.adapters.anthropic import AnthropicAdapter

        path = tmp_path / "anthropic.yaml"

        def live(self, **kwargs):
            return AnthropicAdapter().replay_stream(RESPONSE, kwargs, False)

        args = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
        with unittest_mock_patch.object(messages.Messages, "create", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="anthropic"):
                with self._client().messages.stream(**args) as stream:
                    assert "".join(stream.text_stream) == "Sunny"
                    message = stream.get_final_message()
        assert message.content[1].input == {"city": "Tokyo"} and message.usage.output_tokens == 3
        with unittest_mock_patch.object(messages.Messages, "create", side_effect=AssertionError("live")):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="anthropic") as rec:
                events = [e.type for e in self._client().messages.create(**args, stream=True)]
        assert events[0] == "message_start" and events[-1] == "message_stop"
        assert rec.to_execution_result().steps[0].tool_calls[0].arguments == {"city": "Tokyo"}

    def test_async_stream_helper(self):
        from agentverify.cassette.adapters.anthropic import AnthropicAdapter

        async def read():
            async with self._client(True).messages.stream(model="m", max_tokens=8, messages=[]) as stream:
                return await stream.get_final_message()

        with AnthropicAdapter().patch(_replay_recorder()):
            message = asyncio.run(read())
        assert message.content[0].text == "Sunny" and message.content[1].id == "call_1"

    def test_structured_output_stream_passes_through_in_auto(self):
        import anthropic.resources.messages as messages

        from agentverify.cassette.adapters.anthropic import AnthropicAdapter

        auto = MagicMock()
        auto.mode = CassetteMode.AUTO
        with unittest_mock_patch.object(messages.Messages, "stream", return_value="original"):
            with AnthropicAdapter().patch(auto):
                assert self._client().messages.stream(model="m", max_tokens=1, messages=[], output_format=dict) == "original"

    def test_tool_use_without_a_string_id(self):
        from agentverify.cassette.adapters.anthropic import AnthropicAdapter

        block = SimpleNamespace(type="tool_use", name="f", input={}, id=None)
        normalized = AnthropicAdapter().normalize_response(SimpleNamespace(content=[block], usage=None))
        assert normalized.tool_calls == [{"name": "f", "arguments": "{}"}]

    def test_stale_stream_helper_calls_the_original(self):
        import anthropic.resources.messages as messages

        from agentverify.cassette.adapters.anthropic import AnthropicAdapter

        with unittest_mock_patch.object(messages.Messages, "stream", return_value="original"):
            with AnthropicAdapter().patch(_replay_recorder()):
                captured = messages.Messages.stream
            assert captured(MagicMock(), model="m", max_tokens=1, messages=[]) == "original"

    def test_accumulator_edge_cases(self):
        from agentverify.cassette.adapters.anthropic import MessageEventAccumulator, _is_given, message_events

        acc = MessageEventAccumulator()
        acc.add(SimpleNamespace(type="message_start", message=SimpleNamespace(model=None, usage=None)))
        acc.add(SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="t")))
        acc.add(SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="signature_delta")))
        acc.add(SimpleNamespace(type="message_delta", usage=None))
        acc.add(SimpleNamespace(type="ping"))
        acc.add(SimpleNamespace(type="content_block_start", index=1, content_block=SimpleNamespace(type="thinking")))
        result = acc.result()
        assert result.content == "t" and result.token_usage is None and result.raw_metadata == {}
        events = message_events(NormalizedResponse(tool_calls=[{"name": "f", "arguments": {"a": 1}}]))
        assert json.loads(events[2]["delta"]["partial_json"]) == {"a": 1}
        assert events[-2]["usage"]["output_tokens"] == 0
        assert not _is_given(None)


class TestGemini:
    @pytest.fixture(autouse=True)
    def _genai(self):
        pytest.importorskip("google.genai")

    def test_record_then_replay_sync_and_async(self, tmp_path):
        import google.genai.models as models
        from google import genai
        from google.genai import types

        path = tmp_path / "gemini.yaml"

        def live(self, **kwargs):
            part = lambda **kw: types.Content(role="model", parts=[types.Part(**kw)])  # noqa: E731
            yield types.GenerateContentResponse(candidates=[types.Candidate(content=part(text="Sun"))])
            yield types.GenerateContentResponse(
                candidates=[types.Candidate(content=part(function_call=types.FunctionCall(name="f", args={"a": 1})))],
                usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=4, candidates_token_count=2),
            )

        client = genai.Client(api_key="k")
        with unittest_mock_patch.object(models.Models, "generate_content_stream", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="gemini"):
                assert len(list(client.models.generate_content_stream(model="m", contents="hi"))) == 2
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="gemini") as rec:
            [chunk] = list(client.models.generate_content_stream(model="m", contents="hi"))
        assert chunk.candidates[0].content.parts[0].text == "Sun"
        assert rec.to_execution_result().token_usage == TokenUsage(4, 2)

        async def read():
            return [c async for c in await client.aio.models.generate_content_stream(model="m", contents="hi")]

        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="gemini"):
            assert len(asyncio.run(read())) == 1

    def test_stale_stream_wrappers_redispatch(self):
        import google.genai.models as models

        from agentverify.cassette.adapters.gemini import GeminiAdapter

        async def alive(*args, **kwargs):
            return "now"

        with unittest_mock_patch.object(models.Models, "generate_content_stream", return_value="now"), unittest_mock_patch.object(
            models.AsyncModels, "generate_content_stream", new=alive
        ):
            with GeminiAdapter().patch(_replay_recorder()):
                sync_ref = models.Models.generate_content_stream
                async_ref = models.AsyncModels.generate_content_stream
            assert sync_ref(MagicMock(), model="m", contents="x") == "now"
            assert asyncio.run(async_ref(MagicMock(), model="m", contents="x")) == "now"


class TestBedrock:
    @pytest.fixture(autouse=True)
    def _boto(self):
        pytest.importorskip("botocore")

    def _call(self, recorder, live=None):
        import botocore.client

        from agentverify.cassette.adapters.bedrock import BedrockAdapter

        with unittest_mock_patch("botocore.client.BaseClient._make_api_call", side_effect=live or AssertionError("live")):
            with BedrockAdapter().patch(recorder):
                params = {"modelId": "m", "messages": [{"role": "user", "content": [{"text": "hi"}]}]}
                return botocore.client.BaseClient._make_api_call(MagicMock(), "ConverseStream", params)

    def test_record_then_replay(self, tmp_path):
        from agentverify.cassette.adapters.bedrock import BedrockAdapter

        path = tmp_path / "bedrock.yaml"

        def live(client, operation, params):
            response = BedrockAdapter().replay_stream(RESPONSE, params, False)
            response["stream"] = _IterStream(list(response["stream"]))
            return response

        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="bedrock") as rec:
            response = self._call(rec, live)
            kinds = [next(iter(event)) for event in response["stream"]]
        assert kinds[-1] == "metadata" and "contentBlockStart" in kinds
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="bedrock") as rec:
            events = list(self._call(rec)["stream"])
        assert events[1] == {"contentBlockDelta": {"delta": {"text": "Sunny"}, "contentBlockIndex": 0}}
        result = rec.to_execution_result()
        assert result.tool_calls[0].arguments == {"city": "Tokyo"} and result.token_usage == TokenUsage(7, 3)

    def test_replay_without_usage(self):
        events = list(self._call(_replay_recorder(NormalizedResponse(content="hi")))["stream"])
        assert "metadata" not in events[-1]

    def test_accumulator_edge_cases(self):
        from agentverify.cassette.adapters.bedrock import ConverseStreamAccumulator

        acc = ConverseStreamAccumulator()
        for event in [
            "not-a-dict",
            {"contentBlockStart": {"start": {}, "contentBlockIndex": 0}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "f"}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "{bad"}}, "contentBlockIndex": 1}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t2", "name": "g"}}, "contentBlockIndex": 2}},
            {"contentBlockDelta": {"delta": {"reasoningContent": {}}, "contentBlockIndex": 3}},
            {"metadata": {"metrics": {}}},
        ]:
            acc.add(event)
        result = acc.result()
        assert [tc["arguments"] for tc in result.tool_calls] == ["{}", "{}"]
        assert result.content == "" and result.token_usage is None


class TestLiteLLM:
    @pytest.fixture(autouse=True)
    def _litellm(self):
        pytest.importorskip("litellm")

    def test_record_then_replay_sync_and_async(self, tmp_path):
        import litellm

        from agentverify.cassette.adapters.litellm import LiteLLMAdapter

        path = tmp_path / "litellm.yaml"

        def live(*args, **kwargs):
            return LiteLLMAdapter().replay_stream(RESPONSE, kwargs, False)

        args = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        with unittest_mock_patch("litellm.completion", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="litellm"):
                chunks = list(litellm.completion(**args, stream_options={"include_usage": True}))
        assert litellm.stream_chunk_builder(chunks).choices[0].message.tool_calls[0].function.name == "get_weather"

        async def read():
            return [c async for c in await litellm.acompletion(**args)]

        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="litellm") as rec:
            chunks = asyncio.run(read())
        assert chunks[0].choices[0].delta.content == "Sunny"
        assert rec.to_execution_result().token_usage == TokenUsage(7, 3)

    def test_replayed_stream_iterates_directly(self):
        from agentverify.cassette.adapters.litellm import _ReplayedStream

        assert list(iter(_ReplayedStream([1, 2]))) == [1, 2]


class TestOpenAIRawResponses:
    """``with_raw_response`` and ``with_streaming_response`` callers get the raw shape back, streamed or not."""

    @pytest.fixture(autouse=True)
    def _openai(self):
        pytest.importorskip("openai")

    def test_wrap_raw(self):
        from agentverify.cassette.adapters._openai_raw import _AsyncRawResponseWrapper, _RawResponseWrapper, raw_mode, wrap_raw

        assert wrap_raw("parsed", {}, False) == "parsed"
        assert raw_mode({"extra_headers": "bad"}) is None
        raw = wrap_raw("parsed", {"extra_headers": {"X-Stainless-Raw-Response": "raw"}}, True)
        assert type(raw) is _RawResponseWrapper and raw.parse() == "parsed" and raw.close() is None
        streamed = wrap_raw("parsed", {"extra_headers": {"X-Stainless-Raw-Response": "stream"}}, True)
        assert type(streamed) is _AsyncRawResponseWrapper

        async def use():
            value = await streamed.parse()
            await streamed.close()
            return value

        assert asyncio.run(use()) == "parsed"

    def test_tapped_raw_response_taps_once_and_delegates(self):
        from agentverify.cassette.adapters._openai_raw import _AsyncRawResponseWrapper, _RawResponseWrapper, tap_raw_or_stream

        taps: list = []
        tap = lambda s: taps.append(s) or f"tapped:{s}"  # noqa: E731
        sync = tap_raw_or_stream(_RawResponseWrapper("s"), tap)
        assert sync.parse() == "tapped:s" and sync.parse() == "tapped:s" and sync.headers == {}
        async_raw = tap_raw_or_stream(_AsyncRawResponseWrapper("a"), tap)

        async def twice():
            return await async_raw.parse(), await async_raw.parse()

        assert asyncio.run(twice()) == ("tapped:a", "tapped:a")
        assert taps == ["s", "a"]
        assert tap_raw_or_stream([1], lambda s: "stream") == "stream"

    def test_agents_sdk_streaming_path(self, tmp_path):
        """The Agents SDK streams through responses.with_streaming_response.create(stream=True)."""
        import openai
        import openai.resources.responses as responses

        from agentverify.cassette.adapters.openai_responses import ResponsesWire

        path = tmp_path / "raw-stream.yaml"

        async def live(self, **kwargs):
            return ResponsesWire().replay_stream(RESPONSE, kwargs, True)

        async def read():
            client = openai.AsyncOpenAI(api_key="k", max_retries=0)
            async with client.responses.with_streaming_response.create(model="m", input="hi", stream=True) as raw:
                stream = await raw.parse()
                return [e.type async for e in stream]

        with unittest_mock_patch.object(responses.AsyncResponses, "create", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai"):
                assert asyncio.run(read())[-1] == "response.completed"

        async def no_network(self, **kwargs):  # pragma: no cover — replay must not call it
            raise AssertionError("live")

        with unittest_mock_patch.object(responses.AsyncResponses, "create", new=no_network):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="openai") as rec:
                assert asyncio.run(read())[-1] == "response.completed"
        assert rec.to_execution_result().steps[0].tool_calls[0].name == "get_weather"


class TestReviewFindings:
    def test_a_stream_that_raises_partway_records_nothing(self, tmp_path):
        """A failed stream is dropped like a failed call, so the framework's retry takes its slot."""
        from agentverify.cassette.adapters.openai import OpenAIAdapter

        def broken():
            yield SimpleNamespace(model="m", usage=None, choices=[])
            raise RuntimeError("throttled")

        class Broken(_NextStream):
            def __init__(self):
                self._iterator = broken()

        rec = LLMCassetteRecorder(tmp_path / "x.yaml", mode=CassetteMode.RECORD)
        stream = record_stream(rec, OpenAIAdapter(), NormalizedRequest(messages=[], model="m"), Broken())
        with pytest.raises(RuntimeError):
            list(stream)
        assert rec._interactions == [] and rec._interaction_probe_stack == []
        rec.discard(NormalizedResponse())  # unknown responses are ignored

        async def abroken():
            yield 1
            raise RuntimeError("late")

        results, finish = _finished()
        failures: list = []

        async def read():
            return [item async for item in tap_stream(abroken(), _Collect(), finish, lambda: failures.append(1))]

        with pytest.raises(RuntimeError):
            asyncio.run(read())

        class BrokenAsync(_AsyncStream):
            async def __anext__(self):
                raise RuntimeError("at once")

        with pytest.raises(RuntimeError):
            asyncio.run(tap_stream(BrokenAsync([]), _Collect(), finish, lambda: failures.append(2)).__anext__())
        assert failures == [1, 2] and results == []

    def test_reading_a_lazy_stream_counts_as_the_live_call(self):
        from agentverify.cassette.adapters.base import _live_call_allowed

        seen: list = []

        def lazy():
            seen.append(_live_call_allowed.get())
            yield "a"

        stream = tap_stream(lazy(), _Collect(), lambda r: None)
        assert next(stream) == "a"
        assert seen == [True] and _live_call_allowed.get() is False  # not left set between reads

    def test_gemini_replay_miss_with_fallback_streams_live(self):
        pytest.importorskip("google.genai")
        import google.genai.models as models
        from google import genai
        from google.genai import types

        from agentverify.cassette.adapters.gemini import GeminiAdapter
        from agentverify.cassette.recorder import OnMissingRequest

        recorder = MagicMock()
        recorder.mode = CassetteMode.REPLAY
        recorder.lookup.return_value = None
        recorder.on_missing = OnMissingRequest.FALLBACK

        def live(self, **kwargs):
            # A real SDK stream sends its request here, through the guarded transport.
            from google.genai._api_client import BaseApiClient

            BaseApiClient.request_streamed(MagicMock(), "post", "path", {})
            yield types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="live")]))])

        with unittest_mock_patch.object(models.Models, "generate_content_stream", new=live):
            from google.genai import _api_client

            with unittest_mock_patch.object(_api_client.BaseApiClient, "request_streamed", return_value=None):
                with GeminiAdapter().patch(recorder):
                    chunks = list(genai.Client(api_key="k").models.generate_content_stream(model="m", contents="hi"))
        assert chunks[0].text == "live"

    def test_partly_read_responses_stream_keeps_what_arrived(self):
        pytest.importorskip("openai")
        from openai._models import construct_type
        from openai.types.responses import ResponseStreamEvent

        from agentverify.cassette.adapters.openai_responses import ResponseEventAccumulator, response_events

        events = response_events(RESPONSE)
        acc = ResponseEventAccumulator()
        for event in events:
            if event["type"] in ("response.output_item.done", "response.completed"):
                continue  # stop before any item finishes
            acc.add(construct_type(type_=ResponseStreamEvent, value=event))
        acc.add(SimpleNamespace(type="response.output_item.added", output_index=9, item=SimpleNamespace(type="reasoning")))
        acc.add(SimpleNamespace(type="response.output_text.delta", output_index=9, delta="ignored"))
        result = acc.result()
        assert result.content == "Sunny"
        assert result.tool_calls == [{"name": "get_weather", "arguments": '{"city": "Tokyo"}', "id": "call_1"}]

    def test_sdk_marker_headers_and_empty_extras_stay_out_of_the_cassette(self):
        pytest.importorskip("openai")
        from agentverify.cassette.adapters.openai import OpenAIAdapter
        from agentverify.cassette.adapters.openai_responses import normalize_request

        markers = {"X-Stainless-Helper-Method": "chat.completions.stream"}
        chat = OpenAIAdapter().normalize_request({"model": "m", "messages": [], "extra_headers": markers, "extra_body": None})
        assert chat.parameters == {}
        kept = OpenAIAdapter().normalize_request({"model": "m", "messages": [], "extra_headers": {**markers, "X-Team": "a"}})
        assert kept.parameters == {"extra_headers": {"X-Team": "a"}}
        responses = normalize_request({"model": "m", "input": "x", "extra_headers": markers, "extra_query": None})
        assert responses.parameters == {}
        assert normalize_request({"model": "m", "input": "x", "extra_headers": {"X-Team": "a"}}).parameters == {
            "extra_headers": {"X-Team": "a"}
        }

    def test_gemini_keeps_function_call_ids_and_replays_bad_json(self):
        pytest.importorskip("google.genai")
        from google.genai import types

        from agentverify.cassette.adapters.gemini import GeminiAdapter

        response = types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="f", args={}, id="fc1"))]))]
        )
        assert GeminiAdapter().normalize_response(response).tool_calls == [{"name": "f", "arguments": "{}", "id": "fc1"}]
        [chunk] = GeminiAdapter().replay_stream(NormalizedResponse(tool_calls=[{"name": "f", "arguments": "{bad"}]), {}, False)
        assert chunk.candidates[0].content.parts[0].function_call.args == {}

    def test_failure_edge_cases(self, tmp_path):
        # A read that raises after the stream was closed keeps what was recorded.
        failures: list = []
        results, finish = _finished()

        class AfterClose(_NextStream):
            def __next__(self):
                if self.closed:
                    raise RuntimeError("closed")
                return super().__next__()

        stream = tap_stream(AfterClose(["a"]), _Collect(), finish, lambda: failures.append(1))
        next(stream)
        stream.close()
        with pytest.raises(RuntimeError):
            next(stream)
        assert failures == [] and [r.content for r in results] == ["a"]

        # A recorder without discard(); and discard() skipping other interactions.
        recorder = SimpleNamespace(record=MagicMock())
        wire = MagicMock(spec=StreamingWire)
        wire.stream_accumulator.return_value = _Collect()
        wire.tap_response.side_effect = lambda response, tap: tap(response)

        def broken():
            raise RuntimeError("x")
            yield  # pragma: no cover

        with pytest.raises(RuntimeError):
            list(record_stream(recorder, wire, NormalizedRequest(messages=[], model="m"), broken()))
        rec = LLMCassetteRecorder(tmp_path / "y.yaml", mode=CassetteMode.RECORD)
        keep, drop = NormalizedResponse(content="keep"), NormalizedResponse()
        rec.record(NormalizedRequest(messages=[], model="m"), keep)
        rec.record(NormalizedRequest(messages=[], model="m"), drop)
        rec.discard(drop)
        assert [r for _, r in rec._interactions] == [keep]

    def test_gemini_replays_dict_arguments(self):
        pytest.importorskip("google.genai")
        from agentverify.cassette.adapters.gemini import GeminiAdapter

        [chunk] = GeminiAdapter().replay_stream(NormalizedResponse(tool_calls=[{"name": "f", "arguments": {"a": 1}}]), {}, False)
        assert chunk.candidates[0].content.parts[0].function_call.args == {"a": 1}


class TestAsyncBedrock:
    """aiobotocore / aioboto3 clients, through a stand-in AioBaseClient."""

    @pytest.fixture
    def aio(self, monkeypatch):
        pytest.importorskip("botocore")
        import sys
        import types

        calls: list = []

        class AioBaseClient:
            async def _make_api_call(self, operation_name, api_params):
                from agentverify.cassette.adapters.bedrock import BedrockAdapter

                calls.append(operation_name)
                if operation_name == "ConverseStream":
                    return BedrockAdapter().replay_stream(RESPONSE, api_params, True)
                if operation_name == "Converse":
                    return BedrockAdapter().denormalize_response(RESPONSE)
                return {"ok": operation_name}

        package = types.ModuleType("aiobotocore")
        module = types.ModuleType("aiobotocore.client")
        module.AioBaseClient = AioBaseClient
        monkeypatch.setitem(sys.modules, "aiobotocore", package)
        monkeypatch.setitem(sys.modules, "aiobotocore.client", module)
        return AioBaseClient, calls

    def test_record_then_replay(self, tmp_path, aio):
        AioBaseClient, calls = aio
        params = {"modelId": "m", "messages": [{"role": "user", "content": [{"text": "hi"}]}]}
        client = MagicMock()
        client.meta.service_model.service_name = "bedrock-runtime"

        async def run():
            plain = await AioBaseClient._make_api_call(client, "Converse", params)
            stream = await AioBaseClient._make_api_call(client, "ConverseStream", params)
            events = [e async for e in stream["stream"]]
            other = await AioBaseClient._make_api_call(client, "ListFoundationModels", {})
            return plain["output"]["message"]["content"][0]["text"], len(events), other

        path = tmp_path / "aio.yaml"
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="bedrock"):
            recorded = asyncio.run(run())
        assert calls == ["Converse", "ConverseStream", "ListFoundationModels"]
        with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="bedrock") as rec:
            assert asyncio.run(run()) == recorded
            with pytest.raises(CassetteUnsupportedCallError, match="InvokeModel"):
                asyncio.run(AioBaseClient._make_api_call(client, "InvokeModel", {}))
        assert calls == ["Converse", "ConverseStream", "ListFoundationModels", "ListFoundationModels"]
        assert rec.to_execution_result().token_usage == TokenUsage(14, 6)
