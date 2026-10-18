"""Tests for the guards on LLM calls the cassette adapters cannot record or replay."""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock
from unittest.mock import patch as unittest_mock_patch

import pytest

from agentverify.cassette.adapters.base import (
    PatchScope,
    allow_live_call,
    block_transport_in_replay,
    guard_unintercepted,
    patch_scope,
    reject_stream_kwarg,
    reject_streaming,
)
from agentverify.cassette.recorder import CassetteMode
from agentverify.errors import CassetteUnsupportedCallError


def _recorder(mode: CassetteMode) -> MagicMock:
    recorder = MagicMock()
    recorder.mode = mode
    return recorder


class TestBaseGuards:
    def _current(self, fn):
        return lambda: fn

    def test_patch_scope_turns_inactive_on_exit(self):
        with patch_scope() as scope:
            assert scope.active
        assert not scope.active

    def test_reject_stream_kwarg(self):
        reject_stream_kwarg("p", "x.create", {"stream": False})
        with pytest.raises(CassetteUnsupportedCallError, match=r"p: x.create\(stream=True\) is not supported"):
            reject_stream_kwarg("p", "x.create", {"stream": True})

    def test_reject_streaming(self):
        scope = PatchScope()
        original = MagicMock(return_value="real")
        installed = MagicMock(return_value="installed")
        wrapped = reject_streaming(scope, _recorder(CassetteMode.RECORD), "p", "x.stream", original, lambda: installed)
        with pytest.raises(CassetteUnsupportedCallError, match="streaming is not recorded"):
            wrapped(1)
        passthrough = reject_streaming(scope, _recorder(CassetteMode.AUTO), "p", "x.stream", original, lambda: installed)
        assert passthrough(1) == "real"
        scope.active = False
        assert wrapped(1) == "installed"

    def test_unintercepted_raises_in_replay_without_calling(self):
        original = MagicMock()
        guarded = guard_unintercepted(PatchScope(), _recorder(CassetteMode.REPLAY), "p", "x.create", original, self._current(original))
        with pytest.raises(CassetteUnsupportedCallError, match="replay would call the real API"):
            guarded(1)
        original.assert_not_called()

    def test_unintercepted_warns_and_calls_in_record(self):
        original = MagicMock(return_value="real")
        guarded = guard_unintercepted(PatchScope(), _recorder(CassetteMode.RECORD), "p", "x.create", original, self._current(original))
        with pytest.warns(UserWarning, match="x.create is not recorded"):
            assert guarded(1, k=2) == "real"
        original.assert_called_once_with(1, k=2)

    def test_unintercepted_rejects_streaming_in_record(self):
        original = MagicMock()
        guarded = guard_unintercepted(PatchScope(), _recorder(CassetteMode.RECORD), "p", "x.create", original, self._current(original))
        with pytest.raises(CassetteUnsupportedCallError, match="stream=True"):
            guarded(stream=True)

    def test_unintercepted_is_quiet_in_auto_passthrough_and_live_calls(self, recwarn):
        original = MagicMock(return_value="real")
        auto = guard_unintercepted(PatchScope(), _recorder(CassetteMode.AUTO), "p", "x.create", original, self._current(original))
        assert auto(stream=True) == "real"
        replay = guard_unintercepted(PatchScope(), _recorder(CassetteMode.REPLAY), "p", "x.create", original, self._current(original))
        with allow_live_call():
            assert replay() == "real"
        assert not recwarn.list

    def test_unintercepted_redispatches_once_inactive(self):
        scope = PatchScope()
        installed = MagicMock(return_value="installed")
        guarded = guard_unintercepted(scope, _recorder(CassetteMode.REPLAY), "p", "x.create", MagicMock(), lambda: installed)
        scope.active = False
        assert guarded(stream=True) == "installed"

    def test_unintercepted_async_entry_point(self):
        async def original(x):
            return x * 2

        replay = guard_unintercepted(PatchScope(), _recorder(CassetteMode.REPLAY), "p", "x.acreate", original, self._current(original))
        assert inspect.iscoroutinefunction(replay)
        coro = replay(1)  # raises when awaited, like the SDK call it replaces
        with pytest.raises(CassetteUnsupportedCallError):
            asyncio.run(coro)
        record = guard_unintercepted(PatchScope(), _recorder(CassetteMode.RECORD), "p", "x.acreate", original, self._current(original))
        with pytest.warns(UserWarning):
            assert asyncio.run(record(3)) == 6
        scope = PatchScope()
        stale = guard_unintercepted(scope, _recorder(CassetteMode.REPLAY), "p", "x.acreate", original, self._current(original))
        scope.active = False
        assert asyncio.run(stale(4)) == 8

    def test_transport_blocks_in_replay_unless_allowed(self):
        original = MagicMock(return_value="sent")
        blocked = block_transport_in_replay(_recorder(CassetteMode.REPLAY), "p", original, is_async=False)
        with pytest.raises(CassetteUnsupportedCallError, match="replay mode does not reach the network"):
            blocked()
        with allow_live_call():
            assert blocked() == "sent"
        assert block_transport_in_replay(_recorder(CassetteMode.RECORD), "p", original, is_async=False)() == "sent"

    def test_transport_async(self):
        async def original():
            return "sent"

        blocked = block_transport_in_replay(_recorder(CassetteMode.REPLAY), "p", original, is_async=True)
        with pytest.raises(CassetteUnsupportedCallError):
            asyncio.run(blocked())
        passed = block_transport_in_replay(_recorder(CassetteMode.RECORD), "p", original, is_async=True)
        assert asyncio.run(passed()) == "sent"

    def test_error_mentions_mock_llm(self):
        with pytest.raises(CassetteUnsupportedCallError, match="cassette or MockLLM"):
            reject_stream_kwarg("p", "x", {"stream": True})


class TestOpenAIGuards:
    @pytest.fixture(autouse=True)
    def _openai(self):
        pytest.importorskip("openai")

    def _adapter(self):
        from agentverify.cassette.adapters.openai import OpenAIAdapter

        return OpenAIAdapter()

    def _client(self):
        import openai

        return openai.OpenAI(api_key="test-key", max_retries=0)

    def test_chat_stream_true_raises_sync_and_async(self):
        import openai

        with self._adapter().patch(_recorder(CassetteMode.RECORD)):
            with pytest.raises(CassetteUnsupportedCallError, match=r"create\(stream=True\)"):
                self._client().chat.completions.create(model="m", messages=[], stream=True)
            client = openai.AsyncOpenAI(api_key="test-key", max_retries=0)
            with pytest.raises(CassetteUnsupportedCallError):
                asyncio.run(client.chat.completions.create(model="m", messages=[], stream=True))

    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.responses.create(model="m", input="hi"),
            lambda c: c.chat.completions.parse(model="m", messages=[]),
            lambda c: c.embeddings.create(model="m", input="hi"),
            lambda c: c.completions.create(model="m", prompt="hi"),
        ],
        ids=["responses", "chat-parse", "embeddings", "legacy-completions"],
    )
    def test_unrecorded_calls_raise_in_replay(self, call):
        with self._adapter().patch(_recorder(CassetteMode.REPLAY)):
            with pytest.raises(CassetteUnsupportedCallError):
                call(self._client())

    def test_responses_create_passes_through_in_record(self):
        import openai.resources.responses as responses

        with unittest_mock_patch.object(responses.Responses, "create", return_value="real"):
            with self._adapter().patch(_recorder(CassetteMode.RECORD)):
                with pytest.warns(UserWarning, match="responses.create is not recorded"):
                    assert self._client().responses.create(model="m", input="hi") == "real"

    def test_responses_streaming_entry_points_raise(self):
        with self._adapter().patch(_recorder(CassetteMode.RECORD)):
            with pytest.raises(CassetteUnsupportedCallError, match="responses.stream"):
                self._client().responses.stream(model="m", input="hi")
            with pytest.raises(CassetteUnsupportedCallError, match=r"responses.create\(stream=True\)"):
                self._client().responses.create(model="m", input="hi", stream=True)

    def test_raw_response_wrapper_cached_inside_cassette_works_after_exit(self):
        import openai.resources.chat.completions as completions

        client = self._client()
        with unittest_mock_patch.object(completions.Completions, "create", return_value="real"):
            with self._adapter().patch(_recorder(CassetteMode.REPLAY)):
                raw = client.chat.completions.with_raw_response  # cached_property binds the patched create
            # Outside the cassette the stale wrapper defers to the real create instead of replaying.
            assert raw.create(model="m", messages=[]) == "real"

    def test_patches_are_restored_on_exit(self):
        import openai._base_client as base_client
        import openai.resources.chat.completions as completions
        import openai.resources.responses as responses

        before = (completions.Completions.create, responses.Responses.create, base_client.SyncAPIClient.request)
        with self._adapter().patch(_recorder(CassetteMode.REPLAY)):
            assert responses.Responses.create is not before[1]
        assert (completions.Completions.create, responses.Responses.create, base_client.SyncAPIClient.request) == before

    def test_responses_guard_works_without_prior_import(self):
        # The SDK loads resource modules lazily; the adapter must import them itself.
        code = textwrap.dedent(
            """
            import sys
            from unittest.mock import MagicMock
            import openai
            assert "openai.resources.responses" not in sys.modules
            from agentverify.cassette.adapters.openai import OpenAIAdapter
            from agentverify.cassette.recorder import CassetteMode
            from agentverify.errors import CassetteUnsupportedCallError
            rec = MagicMock(); rec.mode = CassetteMode.REPLAY
            with OpenAIAdapter().patch(rec):
                try:
                    openai.OpenAI(api_key="k").responses.create(model="m", input="hi")
                except CassetteUnsupportedCallError:
                    print("guarded")
            """
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "guarded"


class TestAnthropicGuards:
    @pytest.fixture(autouse=True)
    def _anthropic(self):
        pytest.importorskip("anthropic")

    def _adapter(self):
        from agentverify.cassette.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter()

    def test_streaming_raises(self):
        import anthropic

        client = anthropic.Anthropic(api_key="k", max_retries=0)
        aclient = anthropic.AsyncAnthropic(api_key="k", max_retries=0)
        with self._adapter().patch(_recorder(CassetteMode.RECORD)):
            with pytest.raises(CassetteUnsupportedCallError, match=r"create\(stream=True\)"):
                client.messages.create(model="m", messages=[], max_tokens=1, stream=True)
            with pytest.raises(CassetteUnsupportedCallError, match="messages.stream"):
                client.messages.stream(model="m", messages=[], max_tokens=1)
            with pytest.raises(CassetteUnsupportedCallError, match="AsyncMessages.stream"):
                aclient.messages.stream(model="m", messages=[], max_tokens=1)
            with pytest.raises(CassetteUnsupportedCallError, match=r"AsyncMessages.create\(stream=True\)"):
                asyncio.run(aclient.messages.create(model="m", messages=[], max_tokens=1, stream=True))

    @pytest.mark.parametrize(
        "call",
        [
            lambda c, a: c.beta.messages.create(model="m", messages=[], max_tokens=1),
            lambda c, a: c.messages.count_tokens(model="m", messages=[]),
        ],
        ids=["beta-create", "count-tokens"],
    )
    def test_unrecorded_calls_raise_in_replay(self, call):
        import anthropic

        client = anthropic.Anthropic(api_key="k", max_retries=0)
        aclient = anthropic.AsyncAnthropic(api_key="k", max_retries=0)
        with self._adapter().patch(_recorder(CassetteMode.REPLAY)):
            with pytest.raises(CassetteUnsupportedCallError):
                call(client, aclient)


class TestGeminiGuards:
    @pytest.fixture(autouse=True)
    def _genai(self):
        pytest.importorskip("google.genai")

    def test_stream_async_and_transport_guards(self):
        from google import genai

        from agentverify.cassette.adapters.gemini import GeminiAdapter

        client = genai.Client(api_key="k")
        with GeminiAdapter().patch(_recorder(CassetteMode.REPLAY)):
            with pytest.raises(CassetteUnsupportedCallError, match="models.generate_content_stream"):
                client.models.generate_content_stream(model="m", contents="hi")
            with pytest.raises(CassetteUnsupportedCallError, match="aio.models.generate_content_stream"):
                asyncio.run(client.aio.models.generate_content_stream(model="m", contents="hi"))
            with pytest.raises(CassetteUnsupportedCallError, match="does not reach the network"):
                client.models.embed_content(model="m", contents="hi")


class TestLiteLLMGuards:
    @pytest.fixture(autouse=True)
    def _litellm(self):
        pytest.importorskip("litellm")

    def test_stream_true_and_unintercepted_entry_points(self):
        import litellm

        from agentverify.cassette.adapters.litellm import LiteLLMAdapter

        with LiteLLMAdapter().patch(_recorder(CassetteMode.REPLAY)):
            with pytest.raises(CassetteUnsupportedCallError, match=r"completion\(stream=True\)"):
                litellm.completion(model="m", messages=[], stream=True)
            with pytest.raises(CassetteUnsupportedCallError, match="text_completion"):
                litellm.text_completion(model="m")
            for name in ("aresponses",):
                with pytest.raises(CassetteUnsupportedCallError, match=name):
                    asyncio.run(getattr(litellm, name)(model="m"))


class TestBedrockGuards:
    @pytest.fixture(autouse=True)
    def _boto(self):
        pytest.importorskip("botocore")

    def _call(self, mode, operation, service="bedrock-runtime"):
        import botocore.client

        from agentverify.cassette.adapters.bedrock import BedrockAdapter

        client = MagicMock()
        client.meta.service_model.service_name = service
        with unittest_mock_patch("botocore.client.BaseClient._make_api_call", return_value={"ok": True}):
            with BedrockAdapter().patch(_recorder(mode)):
                return botocore.client.BaseClient._make_api_call(client, operation, {})

    @pytest.mark.parametrize(
        "service, operation",
        [("bedrock-runtime", "ConverseStream"), ("bedrock-runtime", "InvokeModelWithResponseStream"), ("bedrock-agent-runtime", "RetrieveAndGenerateStream")],
    )
    def test_streaming_operations_raise_in_every_mode(self, service, operation):
        for mode in (CassetteMode.RECORD, CassetteMode.REPLAY):
            with pytest.raises(CassetteUnsupportedCallError, match=operation):
                self._call(mode, operation, service)

    @pytest.mark.parametrize(
        "service, operation",
        [("bedrock-runtime", "InvokeModel"), ("bedrock-agent-runtime", "InvokeAgent"), ("bedrock-agent-runtime", "RetrieveAndGenerate")],
    )
    def test_generating_operations_raise_in_replay_and_warn_in_record(self, service, operation):
        with pytest.raises(CassetteUnsupportedCallError, match=operation):
            self._call(CassetteMode.REPLAY, operation, service)
        with pytest.warns(UserWarning, match=f"{operation} is not recorded"):
            assert self._call(CassetteMode.RECORD, operation, service) == {"ok": True}

    @pytest.mark.parametrize(
        "service, operation",
        [("bedrock-runtime", "CountTokens"), ("bedrock-runtime", "ApplyGuardrail"), ("bedrock-agent-runtime", "Retrieve"), ("s3", "GetObject")],
    )
    def test_non_generating_operations_pass_through_in_replay(self, service, operation):
        assert self._call(CassetteMode.REPLAY, operation, service) == {"ok": True}


class TestStaleWrappers:
    """A wrapper captured inside a cassette defers to the real call after the cassette exits."""

    def test_openai_async_create(self):
        pytest.importorskip("openai")
        import openai.resources.chat.completions as completions

        from agentverify.cassette.adapters.openai import OpenAIAdapter

        async def real(*args, **kwargs):
            return "real"

        with unittest_mock_patch.object(completions.AsyncCompletions, "create", new=real):
            with OpenAIAdapter().patch(_recorder(CassetteMode.REPLAY)):
                captured = completions.AsyncCompletions.create
            assert asyncio.run(captured(MagicMock(), model="m", messages=[])) == "real"

    def test_anthropic_create(self):
        pytest.importorskip("anthropic")
        import anthropic.resources.messages as messages

        from agentverify.cassette.adapters.anthropic import AnthropicAdapter

        with unittest_mock_patch.object(messages.Messages, "create", return_value="real"):
            with AnthropicAdapter().patch(_recorder(CassetteMode.REPLAY)):
                captured = messages.Messages.create
            assert captured(MagicMock(), model="m", messages=[], max_tokens=1) == "real"


class TestCrossCassetteAndModes:
    def test_raw_response_wrapper_cached_in_one_cassette_replays_in_the_next(self):
        pytest.importorskip("openai")
        import openai

        from agentverify.cassette.adapters.base import NormalizedResponse
        from agentverify.cassette.adapters.openai import OpenAIAdapter

        client = openai.OpenAI(api_key="k", max_retries=0)
        first = _recorder(CassetteMode.REPLAY)
        first.lookup.return_value = NormalizedResponse(content="one")
        with OpenAIAdapter().patch(first):
            raw = client.chat.completions.with_raw_response  # cached with the first cassette's wrapper
            assert raw.create(model="m", messages=[]).parse().choices[0].message.content == "one"
        second = _recorder(CassetteMode.REPLAY)
        second.lookup.return_value = NormalizedResponse(content="two")
        with OpenAIAdapter().patch(second):
            assert raw.create(model="m", messages=[]).parse().choices[0].message.content == "two"

    def test_openai_stream_passes_through_in_auto_passthrough(self):
        pytest.importorskip("openai")
        import openai.resources.chat.completions as completions

        from agentverify.cassette.adapters.openai import OpenAIAdapter

        with unittest_mock_patch.object(completions.Completions, "create", return_value="live-stream"):
            with OpenAIAdapter().patch(_recorder(CassetteMode.AUTO)):
                assert completions.Completions.create(MagicMock(), model="m", messages=[], stream=True) == "live-stream"

    def test_bedrock_converse_is_intercepted_without_client_metadata(self):
        pytest.importorskip("botocore")
        import botocore.client

        from agentverify.cassette.adapters.base import NormalizedResponse
        from agentverify.cassette.adapters.bedrock import BedrockAdapter

        recorder = _recorder(CassetteMode.REPLAY)
        recorder.lookup.return_value = NormalizedResponse(content="replayed")
        with unittest_mock_patch("botocore.client.BaseClient._make_api_call", side_effect=AssertionError("live call")):
            with BedrockAdapter().patch(recorder):
                client = MagicMock(spec=botocore.client.BaseClient)  # no meta attribute
                result = botocore.client.BaseClient._make_api_call(
                    client, "Converse", {"modelId": "m", "messages": [{"role": "user", "content": [{"text": "hi"}]}]}
                )
        assert result["output"]["message"]["content"][0]["text"] == "replayed"

    def test_bedrock_streaming_passes_through_in_auto_passthrough(self):
        pytest.importorskip("botocore")
        import botocore.client

        from agentverify.cassette.adapters.bedrock import BedrockAdapter

        client = MagicMock()
        client.meta.service_model.service_name = "bedrock-runtime"
        with unittest_mock_patch("botocore.client.BaseClient._make_api_call", return_value={"stream": True}):
            with BedrockAdapter().patch(_recorder(CassetteMode.AUTO)):
                assert botocore.client.BaseClient._make_api_call(client, "ConverseStream", {}) == {"stream": True}

    def test_async_stream_passthrough_and_stale_async_streaming(self):
        pytest.importorskip("openai")
        import openai.resources.chat.completions as completions

        from agentverify.cassette.adapters.openai import OpenAIAdapter

        async def live(*args, **kwargs):
            return "live-stream"

        with unittest_mock_patch.object(completions.AsyncCompletions, "create", new=live):
            with OpenAIAdapter().patch(_recorder(CassetteMode.AUTO)):
                result = asyncio.run(completions.AsyncCompletions.create(MagicMock(), model="m", messages=[], stream=True))
        assert result == "live-stream"

        async def original():
            return "real"

        scope = PatchScope()
        auto = reject_streaming(scope, _recorder(CassetteMode.AUTO), "p", "x.astream", original, lambda: original)
        assert asyncio.run(auto()) == "real"
        scope.active = False
        assert asyncio.run(auto()) == "real"

    def test_anthropic_and_litellm_stream_pass_through_in_auto(self):
        pytest.importorskip("anthropic")
        pytest.importorskip("litellm")
        import anthropic.resources.messages as messages
        import litellm

        from agentverify.cassette.adapters.anthropic import AnthropicAdapter
        from agentverify.cassette.adapters.litellm import LiteLLMAdapter

        with unittest_mock_patch.object(messages.Messages, "create", return_value="live"):
            with AnthropicAdapter().patch(_recorder(CassetteMode.AUTO)):
                assert messages.Messages.create(MagicMock(), model="m", messages=[], max_tokens=1, stream=True) == "live"
        with unittest_mock_patch("litellm.completion", return_value="live"):
            with LiteLLMAdapter().patch(_recorder(CassetteMode.AUTO)):
                assert litellm.completion(model="m", messages=[], stream=True) == "live"
