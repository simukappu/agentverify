"""Record and replay through async SDK clients, and the guard against overlapping calls."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch as unittest_mock_patch

import pytest

from agentverify.cassette.adapters.base import NormalizedResponse
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder
from agentverify.errors import CassetteUnsupportedCallError


def _provider_cases():
    """(provider, patch target object, attribute, call) for each async entry point the adapters record."""
    cases = []
    try:
        import openai.resources.chat.completions as oc

        cases.append(
            pytest.param(
                "openai",
                oc.AsyncCompletions,
                "create",
                lambda fn: fn(MagicMock(), model="m", messages=[{"role": "user", "content": "hi"}]),
                id="openai",
            )
        )
    except ImportError:  # pragma: no cover
        pass
    try:
        import anthropic.resources.messages as am

        cases.append(
            pytest.param(
                "anthropic",
                am.AsyncMessages,
                "create",
                lambda fn: fn(MagicMock(), model="m", max_tokens=8, messages=[{"role": "user", "content": "hi"}]),
                id="anthropic",
            )
        )
    except ImportError:  # pragma: no cover
        pass
    try:
        import google.genai.models as gm

        cases.append(
            pytest.param(
                "gemini", gm.AsyncModels, "generate_content", lambda fn: fn(MagicMock(), model="m", contents="hi"), id="gemini"
            )
        )
    except ImportError:  # pragma: no cover
        pass
    try:
        import litellm

        cases.append(
            pytest.param(
                "litellm",
                litellm,
                "acompletion",
                lambda fn: fn(model="m", messages=[{"role": "user", "content": "hi"}]),
                id="litellm",
            )
        )
    except ImportError:  # pragma: no cover
        pass
    return cases


def _fake_response(provider: str, content: str):
    from agentverify.cassette.recorder import _resolve_provider

    return _resolve_provider(provider).denormalize_response(NormalizedResponse(content=content))


@pytest.mark.parametrize("provider, owner, attr, call", _provider_cases())
class TestAsyncRecordReplay:
    def test_record_then_replay(self, tmp_path, provider, owner, attr, call):
        path = tmp_path / "async.yaml"

        async def live(*args, **kwargs):
            await asyncio.sleep(0)
            return _fake_response(provider, "recorded answer")

        with unittest_mock_patch.object(owner, attr, new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider=provider):
                asyncio.run(call(getattr(owner, attr)))
        assert path.exists()

        async def must_not_run(*args, **kwargs):
            raise AssertionError("replay reached the real API")

        with unittest_mock_patch.object(owner, attr, new=must_not_run):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider=provider) as rec:
                asyncio.run(call(getattr(owner, attr)))
            result = rec.to_execution_result()
        assert result.final_output == "recorded answer"

    def test_sequential_awaits_record_in_order(self, tmp_path, provider, owner, attr, call):
        path = tmp_path / "seq.yaml"
        answers = iter(["first", "second"])

        async def live(*args, **kwargs):
            await asyncio.sleep(0)
            return _fake_response(provider, next(answers))

        async def run_twice():
            fn = getattr(owner, attr)
            await call(fn)
            await call(fn)

        with unittest_mock_patch.object(owner, attr, new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider=provider) as rec:
                asyncio.run(run_twice())
        assert [s.output for s in rec.to_execution_result().steps] == ["first", "second"]

    def test_overlapping_calls_are_rejected_while_recording(self, tmp_path, provider, owner, attr, call):
        async def slow(*args, **kwargs):
            await asyncio.sleep(0.01)
            return _fake_response(provider, "x")

        async def gather():
            fn = getattr(owner, attr)
            await asyncio.gather(call(fn), call(fn))

        with unittest_mock_patch.object(owner, attr, new=slow):
            with LLMCassetteRecorder(tmp_path / "c.yaml", mode=CassetteMode.RECORD, provider=provider):
                with pytest.raises(CassetteUnsupportedCallError, match="concurrent LLM call"):
                    asyncio.run(gather())


def test_replay_miss_falls_back_for_async_calls(tmp_path):
    pytest.importorskip("anthropic")
    import anthropic.resources.messages as am

    from agentverify.cassette.recorder import OnMissingRequest

    path = tmp_path / "empty.yaml"
    path.write_text("version: 1\nprovider: anthropic\ninteractions: []\n")

    async def live(*args, **kwargs):
        return _fake_response("anthropic", "live answer")

    with unittest_mock_patch.object(am.AsyncMessages, "create", new=live):
        with LLMCassetteRecorder(
            path, mode=CassetteMode.REPLAY, provider="anthropic", on_missing=OnMissingRequest.FALLBACK
        ) as rec:
            asyncio.run(am.AsyncMessages.create(MagicMock(), model="m", max_tokens=8, messages=[]))
    assert rec.to_execution_result().final_output == "live answer"


def test_async_stream_passthrough_in_auto_and_stale_async_wrapper():
    pytest.importorskip("anthropic")
    pytest.importorskip("litellm")
    import anthropic.resources.messages as am
    import litellm

    from agentverify.cassette.adapters.anthropic import AnthropicAdapter
    from agentverify.cassette.adapters.litellm import LiteLLMAdapter

    auto = MagicMock()
    auto.mode = CassetteMode.AUTO

    async def live(*args, **kwargs):
        return "live"

    with unittest_mock_patch.object(am.AsyncMessages, "create", new=live):
        with AnthropicAdapter().patch(auto):
            captured = am.AsyncMessages.create
            assert asyncio.run(captured(MagicMock(), model="m", messages=[], max_tokens=1, stream=True)) == "live"
        # Captured inside the cassette, called after it: defers to what is installed now.
        assert asyncio.run(captured(MagicMock(), model="m", messages=[], max_tokens=1)) == "live"
    with unittest_mock_patch("litellm.acompletion", new=live):
        with LiteLLMAdapter().patch(auto):
            assert asyncio.run(litellm.acompletion(model="m", messages=[], stream=True)) == "live"


def test_mock_llm_answers_async_clients():
    pytest.importorskip("anthropic")
    import anthropic

    from agentverify import MockLLM, mock_response

    async def agent():
        client = anthropic.AsyncAnthropic(api_key="test-key", max_retries=0)
        reply = await client.messages.create(model="m", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
        return reply.content[0].text

    with MockLLM([mock_response(content="mocked")], provider="anthropic"):
        assert asyncio.run(agent()) == "mocked"


class TestOverlapGuardScope:
    """The overlap guard applies only to genuinely concurrent calls while recording."""

    def _openai(self):
        pytest.importorskip("openai")
        import openai.resources.chat.completions as oc

        return oc

    def test_nested_call_runs_unrecorded(self, tmp_path):
        oc = self._openai()
        calls = []

        async def live(self_, *args, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "outer":
                # e.g. a tool that calls the model again inside the outer call
                await oc.AsyncCompletions.create(self_, model="inner", messages=[])
            return _fake_response("openai", kwargs["model"])

        with unittest_mock_patch.object(oc.AsyncCompletions, "create", new=live):
            with LLMCassetteRecorder(tmp_path / "n.yaml", mode=CassetteMode.RECORD, provider="openai") as rec:
                asyncio.run(oc.AsyncCompletions.create(MagicMock(), model="outer", messages=[]))
        assert calls == ["outer", "inner"]
        assert [s.output for s in rec.to_execution_result().steps] == ["outer"]

    @pytest.mark.parametrize("mode", [CassetteMode.AUTO, CassetteMode.REPLAY])
    def test_overlap_allowed_when_nothing_is_saved(self, tmp_path, mode):
        oc = self._openai()
        from agentverify.cassette.recorder import OnMissingRequest

        path = tmp_path / ("x.yaml" if mode == CassetteMode.AUTO else "empty.yaml")
        if mode == CassetteMode.REPLAY:
            path.write_text("version: 1\nprovider: openai\ninteractions: []\n")

        async def slow(*args, **kwargs):
            await asyncio.sleep(0.01)
            return _fake_response("openai", "x")

        async def gather():
            fn = oc.AsyncCompletions.create
            await asyncio.gather(fn(MagicMock(), model="m", messages=[]), fn(MagicMock(), model="m", messages=[]))

        with unittest_mock_patch.object(oc.AsyncCompletions, "create", new=slow):
            with LLMCassetteRecorder(path, mode=CassetteMode.AUTO, provider="openai", on_missing=OnMissingRequest.FALLBACK):
                asyncio.run(gather())

    def test_threads_are_detected_and_the_counter_recovers(self, tmp_path):
        import threading

        oc = self._openai()
        started, release = threading.Event(), threading.Event()
        errors = []

        def live(*args, **kwargs):
            if kwargs["model"] == "first":
                started.set()
                release.wait(5)
            return _fake_response("openai", kwargs["model"])

        def first():
            oc.Completions.create(MagicMock(), model="first", messages=[])

        with unittest_mock_patch.object(oc.Completions, "create", new=live):
            with LLMCassetteRecorder(tmp_path / "t.yaml", mode=CassetteMode.RECORD, provider="openai") as rec:
                t = threading.Thread(target=first)
                t.start()
                started.wait(5)
                try:
                    oc.Completions.create(MagicMock(), model="second", messages=[])
                except CassetteUnsupportedCallError as e:
                    errors.append(e)
                release.set()
                t.join(5)
                # Sequential calls work again once the first one finished.
                oc.Completions.create(MagicMock(), model="third", messages=[])
        assert len(errors) == 1
        assert [s.output for s in rec.to_execution_result().steps] == ["first", "third"]


def test_litellm_fallbacks_through_acompletion_on_a_worker_thread(tmp_path):
    pytest.importorskip("litellm")
    import concurrent.futures

    import litellm

    async def live_async(*args, **kwargs):
        return _fake_response("litellm", "fallback answer")

    def live_sync(*args, **kwargs):
        # litellm runs fallbacks through litellm.acompletion on a worker thread.
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, litellm.acompletion(**kwargs)).result()

    with unittest_mock_patch("litellm.acompletion", new=live_async), unittest_mock_patch("litellm.completion", new=live_sync):
        with LLMCassetteRecorder(tmp_path / "f.yaml", mode=CassetteMode.RECORD, provider="litellm") as rec:
            litellm.completion(model="m", messages=[{"role": "user", "content": "hi"}], fallbacks=["m2"])
    assert [s.output for s in rec.to_execution_result().steps] == ["fallback answer"]


def test_anthropic_raw_response_is_guarded_not_recorded():
    pytest.importorskip("anthropic")
    import anthropic.resources.messages as am

    from agentverify.cassette.adapters.anthropic import AnthropicAdapter

    raw = {"extra_headers": {"X-Stainless-Raw-Response": "raw"}}
    record, replay = MagicMock(), MagicMock()
    record.mode, replay.mode = CassetteMode.RECORD, CassetteMode.REPLAY
    with unittest_mock_patch.object(am.Messages, "create", return_value="raw-live"):
        with AnthropicAdapter().patch(record):
            with pytest.warns(UserWarning, match="with_raw_response.create is not recorded"):
                assert am.Messages.create(MagicMock(), model="m", messages=[], max_tokens=1, **raw) == "raw-live"
        record.record.assert_not_called()
    with AnthropicAdapter().patch(replay):
        with pytest.raises(CassetteUnsupportedCallError, match="with_raw_response"):
            asyncio.run(am.AsyncMessages.create(MagicMock(), model="m", messages=[], max_tokens=1, **raw))


def test_gemini_and_litellm_stale_references_redispatch():
    pytest.importorskip("google.genai")
    pytest.importorskip("litellm")
    import google.genai.models as gm
    import litellm

    from agentverify.cassette.adapters.gemini import GeminiAdapter
    from agentverify.cassette.adapters.litellm import LiteLLMAdapter

    rec = MagicMock()
    rec.mode = CassetteMode.REPLAY

    async def alive(*args, **kwargs):
        return "now"

    with unittest_mock_patch.object(gm.Models, "generate_content", return_value="now"), unittest_mock_patch.object(
        gm.AsyncModels, "generate_content", new=alive
    ):
        with GeminiAdapter().patch(rec):
            sync_ref, async_ref = gm.Models.generate_content, gm.AsyncModels.generate_content
        assert sync_ref(MagicMock(), model="m", contents="hi") == "now"
        assert asyncio.run(async_ref(MagicMock(), model="m", contents="hi")) == "now"
    with unittest_mock_patch("litellm.completion", return_value="now"), unittest_mock_patch("litellm.acompletion", new=alive):
        with LiteLLMAdapter().patch(rec):
            sync_ref, async_ref = litellm.completion, litellm.acompletion
        assert sync_ref(model="m", messages=[]) == "now"
        assert asyncio.run(async_ref(model="m", messages=[])) == "now"


def test_async_raw_response_record_and_sync_nested_call(tmp_path):
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    import anthropic.resources.messages as am
    import openai.resources.chat.completions as oc

    from agentverify.cassette.adapters.anthropic import AnthropicAdapter

    record = MagicMock()
    record.mode = CassetteMode.RECORD

    async def raw_live(*args, **kwargs):
        return "raw-live"

    with unittest_mock_patch.object(am.AsyncMessages, "create", new=raw_live):
        with AnthropicAdapter().patch(record):
            with pytest.warns(UserWarning, match="AsyncMessages.with_raw_response.create"):
                result = asyncio.run(
                    am.AsyncMessages.create(
                        MagicMock(), model="m", messages=[], max_tokens=1, extra_headers={"X-Stainless-Raw-Response": "raw"}
                    )
                )
    assert result == "raw-live"

    def live(self_, *args, **kwargs):
        if kwargs["model"] == "outer":
            oc.Completions.create(self_, model="inner", messages=[])
        return _fake_response("openai", kwargs["model"])

    with unittest_mock_patch.object(oc.Completions, "create", new=live):
        with LLMCassetteRecorder(tmp_path / "s.yaml", mode=CassetteMode.RECORD, provider="openai") as rec:
            oc.Completions.create(MagicMock(), model="outer", messages=[])
    assert [s.output for s in rec.to_execution_result().steps] == ["outer"]
