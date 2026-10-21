"""Tests for the OpenAI Responses API wire format and its interception by the OpenAI adapter."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch as unittest_mock_patch

import pytest

pytest.importorskip("openai")

import openai  # noqa: E402
import openai.resources.responses as responses_mod  # noqa: E402

from agentverify.cassette.adapters.base import NormalizedResponse  # noqa: E402
from agentverify.cassette.adapters.openai import OpenAIAdapter  # noqa: E402
from agentverify.cassette.adapters.openai_responses import (  # noqa: E402
    denormalize_response,
    normalize_request,
    normalize_response,
)
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder  # noqa: E402
from agentverify.errors import CassetteUnsupportedCallError  # noqa: E402
from agentverify.models import TokenUsage  # noqa: E402


def _response(content=None, calls=(), usage=(10, 5)):
    tool_calls = [{"name": n, "arguments": a} for n, a in calls] or None
    return denormalize_response(
        NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage=TokenUsage(*usage) if usage else None,
            raw_metadata={"model": "gpt-5"},
        )
    )


class TestNormalizeRequest:
    def test_instructions_and_string_input(self):
        req = normalize_request({"model": "gpt-5", "instructions": "Be brief.", "input": "Hi", "temperature": 0})
        assert req.model == "gpt-5"
        assert req.messages == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
        assert req.parameters == {"temperature": 0}

    def test_input_items_map_to_chat_conventions(self):
        items = [
            {"role": "user", "content": [{"type": "input_text", "text": "Weather in "}, {"type": "input_text", "text": "Tokyo?"}]},
            {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"role": "user", "content": [{"type": "input_image", "image_url": "data:..."}]},
        ]
        req = normalize_request({"model": "m", "input": items})
        assert req.messages == [
            {"role": "user", "content": "Weather in Tokyo?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"role": "user", "content": [{"type": "input_image", "image_url": "data:..."}]},
        ]

    def test_sdk_objects_and_non_dict_items(self):
        from openai.types.responses import EasyInputMessageParam  # noqa: F401 — TypedDict, a plain dict at runtime

        item = MagicMock()
        item.model_dump.return_value = {"role": "user", "content": "from an object"}
        req = normalize_request({"model": "m", "input": [item, "bare"]})
        assert req.messages == [{"role": "user", "content": "from an object"}, {"role": "user", "content": "bare"}]

    def test_flat_and_hosted_tools(self):
        tools = [
            {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
            {"type": "web_search"},
            "not-a-tool",
        ]
        req = normalize_request({"model": "m", "input": "x", "tools": tools})
        assert req.tools == [{"name": "get_weather", "parameters": {"type": "object"}}, {"name": "web_search", "parameters": {}}]

    def test_reasoning_blobs_and_volatile_parameters_are_dropped(self):
        items = [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA..."}]
        req = normalize_request({"model": "m", "input": items, "prompt_cache_key": "agents-sdk:run:1234", "store": False})
        assert req.messages == [{"type": "reasoning", "id": "rs_1", "summary": []}]
        assert req.parameters == {"store": False}

    def test_namespaced_and_named_typed_tools(self):
        tools = [
            {"type": "namespace", "name": "crm", "tools": [{"type": "function", "name": "lookup", "parameters": {}}]},
            {"type": "mcp", "server_label": "github"},
            {"type": "mcp", "server_label": "jira"},
            {"type": "custom", "name": "grammar_tool"},
        ]
        req = normalize_request({"model": "m", "input": "x", "tools": tools})
        assert [t["name"] for t in req.tools] == ["crm.lookup", "mcp:github", "mcp:jira", "custom:grammar_tool"]

    def test_sentinels_are_dropped(self):
        req = normalize_request({"model": "m", "input": openai.omit, "tools": openai.omit, "top_p": openai.NOT_GIVEN})
        assert req.messages == [] and req.tools is None and req.parameters == {}

    def test_stored_conversation_warns(self):
        with pytest.warns(UserWarning, match="previous_response_id"):
            normalize_request({"model": "m", "input": "x", "previous_response_id": "resp_1"})


class TestNormalizeResponse:
    def test_round_trip(self):
        resp = _response("Sunny.", calls=[("get_weather", {"city": "Tokyo"})])
        norm = normalize_response(resp)
        assert norm.content == "Sunny."
        [call] = norm.tool_calls
        assert (call["name"], call["arguments"]) == ("get_weather", '{"city": "Tokyo"}')
        assert call["id"].startswith("call_replayed_")
        assert norm.token_usage == TokenUsage(10, 5)
        assert norm.raw_metadata == {"model": "gpt-5"}

    def test_reasoning_items_refusals_and_missing_usage(self):
        resp = _response(None, usage=None)
        resp.output.append(MagicMock(type="reasoning"))
        resp.output.append(MagicMock(type="message", content=[MagicMock(type="refusal", refusal="I can't help.")]))
        resp.output.append(MagicMock(type="message", content=[MagicMock(type="output_audio")]))
        norm = normalize_response(resp)
        assert norm.content == "I can't help." and norm.tool_calls is None and norm.token_usage is None

    def test_client_executed_non_function_calls_are_rejected(self):
        resp = _response(None)
        resp.output.append(MagicMock(type="computer_call"))
        with pytest.raises(CassetteUnsupportedCallError, match="computer_call"):
            normalize_response(resp)

    def test_function_call_without_string_id(self):
        resp = _response(None)
        resp.output.append(MagicMock(type="function_call", call_id=None, arguments="{}"))
        resp.output[-1].name = "f"
        assert normalize_response(resp).tool_calls == [{"name": "f", "arguments": "{}"}]

    def test_raw_response_is_unwrapped(self):
        raw = MagicMock(spec=["parse"])
        raw.parse.return_value = _response("From raw.")
        assert normalize_response(raw).content == "From raw."


class TestDenormalizeResponse:
    def test_builds_sdk_response(self):
        resp = _response("Hi", calls=[("f", '{"a": 1}')])
        assert isinstance(resp, openai.types.responses.Response)
        assert resp.output_text == "Hi"
        assert resp.output[1].call_id.startswith("call_replayed_") and resp.output[1].arguments == '{"a": 1}'
        assert resp.usage.total_tokens == 15 and resp.usage.input_tokens_details.cached_tokens == 0
        assert resp.model_dump()["usage"]["output_tokens_details"]["reasoning_tokens"] == 0

    def test_defaults_without_metadata_or_usage(self):
        resp = denormalize_response(NormalizedResponse(content=None, tool_calls=None, token_usage=None))
        assert resp.model == "gpt-4o-mini" and resp.output == [] and resp.usage is None


class TestInterception:
    """responses.create is recorded and replayed, sync and async."""

    def _client(self, is_async=False):
        cls = openai.AsyncOpenAI if is_async else openai.OpenAI
        return cls(api_key="test-key", max_retries=0)

    def test_record_then_replay_sync(self, tmp_path):
        path = tmp_path / "r.yaml"
        with unittest_mock_patch.object(responses_mod.Responses, "create", return_value=_response("Recorded.")):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai"):
                self._client().responses.create(model="gpt-5", input="Hi")
        with unittest_mock_patch.object(responses_mod.Responses, "create", side_effect=AssertionError("live")):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="openai") as rec:
                out = self._client().responses.create(model="gpt-5", input="Hi")
        assert out.output_text == "Recorded."
        assert rec.to_execution_result().final_output == "Recorded."

    def test_record_then_replay_async(self, tmp_path):
        path = tmp_path / "a.yaml"

        async def live(*args, **kwargs):
            return _response("Async.")

        async def run():
            return await self._client(is_async=True).responses.create(model="gpt-5", input="Hi")

        with unittest_mock_patch.object(responses_mod.AsyncResponses, "create", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai"):
                asyncio.run(run())

        async def must_not_run(*args, **kwargs):
            raise AssertionError("live")

        with unittest_mock_patch.object(responses_mod.AsyncResponses, "create", new=must_not_run):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="openai"):
                assert asyncio.run(run()).output_text == "Async."

    def test_raw_response_records_and_replays(self, tmp_path):
        """langchain-openai's Responses path calls responses.with_raw_response.create."""
        path = tmp_path / "raw.yaml"
        legacy = MagicMock(spec=["parse"])
        legacy.parse.return_value = _response("Raw answer.")
        with unittest_mock_patch.object(responses_mod.Responses, "create", return_value=legacy):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai"):
                self._client().responses.with_raw_response.create(model="gpt-5", input="Hi")
        with unittest_mock_patch.object(responses_mod.Responses, "create", side_effect=AssertionError("live")):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="openai"):
                replayed = self._client().responses.with_raw_response.create(model="gpt-5", input="Hi")
        assert replayed.parse().output_text == "Raw answer."

    def test_stale_references_redispatch(self):
        rec = MagicMock()
        rec.mode = CassetteMode.REPLAY

        async def alive(*args, **kwargs):
            return "now"

        with unittest_mock_patch.object(responses_mod.Responses, "create", return_value="now"), unittest_mock_patch.object(
            responses_mod.AsyncResponses, "create", new=alive
        ):
            with OpenAIAdapter().patch(rec):
                sync_ref, async_ref = responses_mod.Responses.create, responses_mod.AsyncResponses.create
            assert sync_ref(MagicMock(), model="m", input="x") == "now"
            assert asyncio.run(async_ref(MagicMock(), model="m", input="x")) == "now"


class TestAgentsSdkOnResponses:
    """The OpenAI Agents SDK's default model uses the Responses API; a tool run records and replays with its data flow."""

    @pytest.fixture(autouse=True)
    def _agents(self, monkeypatch):
        pytest.importorskip("agents")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _run(self):
        from agents import Agent, Runner, function_tool, set_tracing_disabled

        set_tracing_disabled(True)

        @function_tool
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        agent = Agent(name="Assistant", instructions="Answer briefly.", tools=[get_weather])
        return asyncio.run(Runner.run(agent, "What's the weather in Tokyo?"))

    def test_record_and_replay_tool_run(self, tmp_path):
        from agentverify import ToolCall, assert_step_uses_result_from, assert_tool_calls

        path = tmp_path / "agents.yaml"
        seen_inputs = []

        async def live(*args, **kwargs):
            seen_inputs.append(kwargs["input"])
            if len(seen_inputs) == 1:
                return _response(None, calls=[("get_weather", '{"city": "Tokyo"}')])
            return _response("Sunny in Tokyo.")

        with unittest_mock_patch.object(responses_mod.AsyncResponses, "create", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai"):
                assert self._run().final_output == "Sunny in Tokyo."

        async def must_not_run(*args, **kwargs):
            raise AssertionError("live")

        with unittest_mock_patch.object(responses_mod.AsyncResponses, "create", new=must_not_run):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="openai") as rec:
                assert self._run().final_output == "Sunny in Tokyo."
        result = rec.to_execution_result()
        assert_tool_calls(result, expected=[ToolCall("get_weather", {"city": "Tokyo"})])
        # The function_call_output item in step 1's input is backfilled as step 0's tool result.
        assert result.steps[0].tool_results == ["The weather in Tokyo is sunny."]
        assert_step_uses_result_from(result, step=1, depends_on=0, via="tool_result")

    def test_multi_turn_tool_run_replays_with_distinct_call_ids(self, tmp_path):
        """Each replayed turn carries its recorded call ids; reusing one would make the SDK reject the run."""
        path = tmp_path / "multi.yaml"
        turns = [
            [("get_weather", '{"city": "Tokyo"}')],
            [("get_weather", '{"city": "Osaka"}'), ("get_weather", '{"city": "Kyoto"}')],
        ]
        count = {"n": 0}

        async def live(*args, **kwargs):
            n = count["n"]
            count["n"] += 1
            if n < len(turns):
                resp = _response(None, calls=turns[n])
                for i, item in enumerate(resp.output):
                    item.call_id = f"call_live_{n}_{i}"
                return resp
            return _response("Sunny everywhere.")

        with unittest_mock_patch.object(responses_mod.AsyncResponses, "create", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai"):
                self._run()

        async def must_not_run(*args, **kwargs):
            raise AssertionError("live")

        with unittest_mock_patch.object(responses_mod.AsyncResponses, "create", new=must_not_run):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="openai") as rec:
                assert self._run().final_output == "Sunny everywhere."
        steps = rec.to_execution_result().steps
        # Each step's tool results are the answers to its own calls, not the whole history.
        assert steps[0].tool_results == ["The weather in Tokyo is sunny."]
        assert steps[1].tool_results == ["The weather in Osaka is sunny.", "The weather in Kyoto is sunny."]

    def test_multi_turn_chat_completions_run_replays_too(self, tmp_path):
        """The same id reuse affected Chat Completions; the Agents SDK's chat model replays a two-turn run."""
        from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool

        import openai.resources.chat.completions as completions

        @function_tool
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        def run():
            model = OpenAIChatCompletionsModel(model="gpt-5", openai_client=openai.AsyncOpenAI(api_key="k", max_retries=0))
            agent = Agent(name="Assistant", instructions="Answer briefly.", tools=[get_weather], model=model)
            return asyncio.run(Runner.run(agent, "Weather in Tokyo and Osaka?"))

        chat = OpenAIAdapter()
        count = {"n": 0}

        async def live(*args, **kwargs):
            count["n"] += 1
            if count["n"] <= 2:
                city = "Tokyo" if count["n"] == 1 else "Osaka"
                return chat.denormalize_response(
                    NormalizedResponse(tool_calls=[{"name": "get_weather", "arguments": f'{{"city": "{city}"}}', "id": f"call_{count['n']}"}])
                )
            return chat.denormalize_response(NormalizedResponse(content="Both sunny."))

        path = tmp_path / "chat.yaml"
        with unittest_mock_patch.object(completions.AsyncCompletions, "create", new=live):
            with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, provider="openai"):
                run()

        async def must_not_run(*args, **kwargs):
            raise AssertionError("live")

        with unittest_mock_patch.object(completions.AsyncCompletions, "create", new=must_not_run):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, provider="openai"):
                assert run().final_output == "Both sunny."
