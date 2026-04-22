"""Tests for agentverify.cassette.recorder — CassetteMode, OnMissingRequest, LLMCassetteRecorder."""

import json

import pytest
import yaml

from agentverify.cassette.adapters.base import (
    NormalizedRequest,
    NormalizedResponse,
)
from agentverify.cassette.io import save_cassette
from agentverify.cassette.recorder import (
    CassetteMode,
    LLMCassetteRecorder,
    OnMissingRequest,
    _resolve_provider,
)
from agentverify.models import TokenUsage


class TestCassetteMode:
    def test_values(self):
        assert CassetteMode.RECORD.value == "record"
        assert CassetteMode.REPLAY.value == "replay"
        assert CassetteMode.AUTO.value == "auto"


class TestOnMissingRequest:
    def test_values(self):
        assert OnMissingRequest.ERROR.value == "error"
        assert OnMissingRequest.FALLBACK.value == "fallback"


class TestResolveProvider:
    def test_resolve_openai_string(self):
        adapter = _resolve_provider("openai")
        assert adapter.name == "openai"

    def test_resolve_unknown_string(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _resolve_provider("nonexistent_provider")

    def test_resolve_adapter_instance(self):
        from agentverify.cassette.adapters.openai import OpenAIAdapter

        adapter = OpenAIAdapter()
        resolved = _resolve_provider(adapter)
        assert resolved is adapter


# ---------------------------------------------------------------------------
# LLMCassetteRecorder
# ---------------------------------------------------------------------------


class TestRecorderInit:
    def test_auto_mode_no_file_stays_auto(self, tmp_path):
        path = tmp_path / "new.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.AUTO)
        assert rec.mode == CassetteMode.AUTO
        assert rec._auto_passthrough is True

    def test_auto_mode_existing_file_becomes_replay(self, tmp_path):
        path = tmp_path / "existing.yaml"
        # Create a valid cassette file
        save_cassette(path, [], provider="openai")
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.AUTO)
        assert rec.mode == CassetteMode.REPLAY

    def test_explicit_record_mode(self, tmp_path):
        path = tmp_path / "rec.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)
        assert rec.mode == CassetteMode.RECORD

    def test_explicit_replay_mode(self, tmp_path):
        path = tmp_path / "rep.yaml"
        save_cassette(path, [], provider="openai")
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.REPLAY)
        assert rec.mode == CassetteMode.REPLAY


class TestRecorderRecordAndLookup:
    def test_record_and_lookup(self, tmp_path):
        path = tmp_path / "test.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)

        req = NormalizedRequest(messages=[], model="gpt-4.1")
        resp = NormalizedResponse(content="hello")
        rec.record(req, resp)

        assert len(rec._interactions) == 1

    def test_lookup_sequential(self, tmp_path):
        path = tmp_path / "test.yaml"
        req1 = NormalizedRequest(messages=[], model="gpt-4.1")
        resp1 = NormalizedResponse(content="first")
        req2 = NormalizedRequest(messages=[], model="gpt-4.1")
        resp2 = NormalizedResponse(content="second")

        save_cassette(path, [(req1, resp1), (req2, resp2)], provider="openai")

        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.REPLAY)
        dummy_req = NormalizedRequest(messages=[], model="gpt-4.1")

        result1 = rec.lookup(dummy_req)
        assert result1.content == "first"

        result2 = rec.lookup(dummy_req)
        assert result2.content == "second"

        # Exhausted
        result3 = rec.lookup(dummy_req)
        assert result3 is None

    def test_lookup_exhausted_fallback(self, tmp_path):
        path = tmp_path / "test.yaml"
        save_cassette(path, [], provider="openai")
        rec = LLMCassetteRecorder(
            cassette_path=path,
            mode=CassetteMode.REPLAY,
            on_missing=OnMissingRequest.FALLBACK,
        )
        result = rec.lookup(NormalizedRequest(messages=[], model="m"))
        assert result is None


class TestRecorderToExecutionResult:
    def test_basic_execution_result(self, tmp_path):
        path = tmp_path / "test.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)

        req = NormalizedRequest(messages=[], model="gpt-4.1")
        resp = NormalizedResponse(
            content="final answer",
            tool_calls=[{"name": "search", "arguments": '{"q": "test"}'}],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        rec.record(req, resp)

        result = rec.to_execution_result()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}
        assert result.token_usage.input_tokens == 10
        assert result.token_usage.output_tokens == 5
        assert result.final_output == "final answer"

    def test_multiple_interactions(self, tmp_path):
        path = tmp_path / "test.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)

        req = NormalizedRequest(messages=[], model="gpt-4.1")
        resp1 = NormalizedResponse(
            tool_calls=[{"name": "a", "arguments": "{}"}],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        resp2 = NormalizedResponse(
            content="done",
            token_usage=TokenUsage(input_tokens=20, output_tokens=10),
        )
        rec.record(req, resp1)
        rec.record(req, resp2)

        result = rec.to_execution_result()
        assert len(result.tool_calls) == 1
        assert result.token_usage.input_tokens == 30
        assert result.token_usage.output_tokens == 15
        assert result.final_output == "done"

    def test_no_interactions(self, tmp_path):
        path = tmp_path / "test.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)
        result = rec.to_execution_result()
        assert result.tool_calls == []
        assert result.token_usage is None
        assert result.final_output is None
        assert result.duration_ms is None

    def test_duration_ms_propagated(self, tmp_path):
        """duration_ms captured in __exit__ is reflected in ExecutionResult."""
        path = tmp_path / "duration.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)
        rec._duration_ms = 123.45
        result = rec.to_execution_result()
        assert result.duration_ms == 123.45

    def test_tool_call_with_dict_arguments(self, tmp_path):
        path = tmp_path / "test.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(
            tool_calls=[{"name": "calc", "arguments": {"x": 1}}],
        )
        rec.record(req, resp)
        result = rec.to_execution_result()
        assert result.tool_calls[0].arguments == {"x": 1}

    def test_tool_call_with_invalid_json_arguments(self, tmp_path):
        path = tmp_path / "test.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(
            tool_calls=[{"name": "broken", "arguments": "not-json"}],
        )
        rec.record(req, resp)
        result = rec.to_execution_result()
        assert result.tool_calls[0].arguments == {}

    def test_tool_call_missing_arguments_key(self, tmp_path):
        path = tmp_path / "test.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(
            tool_calls=[{"name": "noargs"}],
        )
        rec.record(req, resp)
        result = rec.to_execution_result()
        assert result.tool_calls[0].arguments == {}


class TestRecorderContextManager:
    def test_record_saves_on_exit(self, tmp_path):
        path = tmp_path / "ctx.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)

        # Manually record an interaction (simulating what the adapter would do)
        req = NormalizedRequest(messages=[], model="gpt-4.1")
        resp = NormalizedResponse(content="hi")
        rec.record(req, resp)

        # Simulate __exit__ saving
        rec.mode = CassetteMode.RECORD
        rec.__exit__(None, None, None)

        assert path.exists()
        loaded = yaml.safe_load(path.read_text())
        assert len(loaded["interactions"]) == 1

    def test_replay_does_not_save(self, tmp_path):
        path = tmp_path / "replay.yaml"
        save_cassette(path, [], provider="openai")
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.REPLAY)
        original_content = path.read_text()
        rec.__exit__(None, None, None)
        assert path.read_text() == original_content

    def test_auto_no_file_does_not_save(self, tmp_path):
        """AUTO with no cassette file collects interactions but doesn't save."""
        path = tmp_path / "auto_passthrough.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.AUTO)
        req = NormalizedRequest(messages=[], model="gpt-4.1")
        resp = NormalizedResponse(content="hi")
        rec.record(req, resp)
        rec.__exit__(None, None, None)
        assert not path.exists()

    def test_enter_and_exit_with_context_manager(self, tmp_path):
        """Test the full __enter__/__exit__ cycle using the context manager protocol."""
        from unittest.mock import MagicMock, patch as mock_patch

        path = tmp_path / "full_ctx.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)

        # Mock the adapter's patch method to return a mock context manager
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=None)

        with mock_patch.object(rec._adapter, "patch", return_value=mock_cm):
            with rec as recorder:
                assert recorder is rec
                mock_cm.__enter__.assert_called_once()

        mock_cm.__exit__.assert_called_once()
        # File should be saved since mode is RECORD
        assert path.exists()
        # Duration should have been captured on exit.
        assert rec._duration_ms is not None
        assert rec._duration_ms >= 0

    def test_exit_without_enter(self, tmp_path):
        """__exit__ when _patch_ctx is None should not error."""
        path = tmp_path / "no_enter.yaml"
        save_cassette(path, [], provider="openai")
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.REPLAY)
        assert rec._patch_ctx is None
        rec.__exit__(None, None, None)  # should not raise


class TestRequestMatching:
    """Tests for cassette request matching (match_requests=True)."""

    def _make_tools_openai(self, names: list[str]) -> list[dict]:
        """Build OpenAI-format tool definitions."""
        return [
            {"type": "function", "function": {"name": n, "parameters": {}}}
            for n in names
        ]

    def _make_tools_bedrock(self, names: list[str]) -> list[dict]:
        """Build Bedrock-format tool definitions (top-level name)."""
        return [{"name": n, "description": f"Tool {n}"} for n in names]

    def test_matching_disabled_by_default(self, tmp_path):
        """Without match_requests, mismatched model passes silently."""
        path = tmp_path / "test.yaml"
        req = NormalizedRequest(messages=[], model="gpt-4.1")
        resp = NormalizedResponse(content="hello")
        save_cassette(path, [(req, resp)], provider="openai")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=False
        )
        different_req = NormalizedRequest(messages=[], model="gpt-3.5-turbo")
        result = rec.lookup(different_req)
        assert result.content == "hello"

    def test_model_match_passes(self, tmp_path):
        """Matching model name passes."""
        path = tmp_path / "test.yaml"
        req = NormalizedRequest(messages=[], model="gpt-4.1")
        resp = NormalizedResponse(content="ok")
        save_cassette(path, [(req, resp)], provider="openai")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=True
        )
        result = rec.lookup(NormalizedRequest(messages=[], model="gpt-4.1"))
        assert result.content == "ok"

    def test_model_mismatch_raises(self, tmp_path):
        """Mismatched model name raises CassetteRequestMismatchError."""
        from agentverify.errors import CassetteRequestMismatchError

        path = tmp_path / "test.yaml"
        req = NormalizedRequest(messages=[], model="gpt-4.1")
        resp = NormalizedResponse(content="ok")
        save_cassette(path, [(req, resp)], provider="openai")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=True
        )
        with pytest.raises(CassetteRequestMismatchError, match="model") as exc_info:
            rec.lookup(NormalizedRequest(messages=[], model="gpt-3.5-turbo"))
        assert exc_info.value.field == "model"
        assert exc_info.value.recorded == "gpt-4.1"
        assert exc_info.value.actual == "gpt-3.5-turbo"
        assert exc_info.value.index == 0

    def test_tools_match_passes(self, tmp_path):
        """Matching tool names passes."""
        path = tmp_path / "test.yaml"
        tools = self._make_tools_openai(["search", "calc"])
        req = NormalizedRequest(messages=[], model="m", tools=tools)
        resp = NormalizedResponse(content="ok")
        save_cassette(path, [(req, resp)], provider="openai")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=True
        )
        # Same tools, different order — should pass (sorted comparison)
        actual_tools = self._make_tools_openai(["calc", "search"])
        result = rec.lookup(NormalizedRequest(messages=[], model="m", tools=actual_tools))
        assert result.content == "ok"

    def test_tools_mismatch_raises(self, tmp_path):
        """Mismatched tool names raises CassetteRequestMismatchError."""
        from agentverify.errors import CassetteRequestMismatchError

        path = tmp_path / "test.yaml"
        tools = self._make_tools_openai(["search", "calc"])
        req = NormalizedRequest(messages=[], model="m", tools=tools)
        resp = NormalizedResponse(content="ok")
        save_cassette(path, [(req, resp)], provider="openai")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=True
        )
        different_tools = self._make_tools_openai(["search", "delete_user"])
        with pytest.raises(CassetteRequestMismatchError, match="tools"):
            rec.lookup(NormalizedRequest(messages=[], model="m", tools=different_tools))

    def test_bedrock_tool_format(self, tmp_path):
        """Tool names extracted from Bedrock format (top-level name key)."""
        path = tmp_path / "test.yaml"
        tools = self._make_tools_bedrock(["read_file", "list_dir"])
        req = NormalizedRequest(messages=[], model="m", tools=tools)
        resp = NormalizedResponse(content="ok")
        save_cassette(path, [(req, resp)], provider="bedrock")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=True
        )
        same_tools = self._make_tools_bedrock(["list_dir", "read_file"])
        result = rec.lookup(NormalizedRequest(messages=[], model="m", tools=same_tools))
        assert result.content == "ok"

    def test_empty_model_skips_check(self, tmp_path):
        """Empty model in recorded or actual request skips model check."""
        path = tmp_path / "test.yaml"
        req = NormalizedRequest(messages=[], model="")
        resp = NormalizedResponse(content="ok")
        save_cassette(path, [(req, resp)], provider="openai")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=True
        )
        result = rec.lookup(NormalizedRequest(messages=[], model="gpt-4.1"))
        assert result.content == "ok"

    def test_empty_tools_skips_check(self, tmp_path):
        """Empty tools in recorded or actual request skips tools check."""
        path = tmp_path / "test.yaml"
        req = NormalizedRequest(messages=[], model="m", tools=None)
        resp = NormalizedResponse(content="ok")
        save_cassette(path, [(req, resp)], provider="openai")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=True
        )
        actual_tools = self._make_tools_openai(["search"])
        result = rec.lookup(NormalizedRequest(messages=[], model="m", tools=actual_tools))
        assert result.content == "ok"

    def test_second_interaction_mismatch(self, tmp_path):
        """Mismatch at second interaction reports correct index."""
        from agentverify.errors import CassetteRequestMismatchError

        path = tmp_path / "test.yaml"
        req1 = NormalizedRequest(messages=[], model="gpt-4.1")
        resp1 = NormalizedResponse(content="first")
        req2 = NormalizedRequest(messages=[], model="gpt-4.1")
        resp2 = NormalizedResponse(content="second")
        save_cassette(path, [(req1, resp1), (req2, resp2)], provider="openai")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=True
        )
        # First lookup passes
        rec.lookup(NormalizedRequest(messages=[], model="gpt-4.1"))
        # Second lookup fails
        with pytest.raises(CassetteRequestMismatchError) as exc_info:
            rec.lookup(NormalizedRequest(messages=[], model="claude-3"))
        assert exc_info.value.index == 1

    def test_tool_without_name_ignored(self, tmp_path):
        """Tools without a name are ignored in matching."""
        path = tmp_path / "test.yaml"
        # Tool with no name in either format
        tools_with_nameless = [
            {"type": "function", "function": {"parameters": {}}},
            {"type": "function", "function": {"name": "search", "parameters": {}}},
        ]
        req = NormalizedRequest(messages=[], model="m", tools=tools_with_nameless)
        resp = NormalizedResponse(content="ok")
        save_cassette(path, [(req, resp)], provider="openai")

        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.REPLAY, match_requests=True
        )
        actual_tools = [
            {"type": "function", "function": {}},
            {"type": "function", "function": {"name": "search", "parameters": {}}},
        ]
        result = rec.lookup(NormalizedRequest(messages=[], model="m", tools=actual_tools))
        assert result.content == "ok"


class TestRecorderSanitize:
    """Tests for cassette sanitization on save."""

    def test_sanitize_on_record(self, tmp_path):
        """Sensitive data is redacted when saving cassette."""
        path = tmp_path / "sanitized.yaml"
        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.RECORD, sanitize=True
        )
        req = NormalizedRequest(
            messages=[{"content": "Key: sk-proj-abcdefghijklmnopqrstuvwxyz1234"}],
            model="gpt-4.1",
        )
        resp = NormalizedResponse(content="ok")
        rec.record(req, resp)
        rec.__exit__(None, None, None)

        content = path.read_text()
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz1234" not in content
        assert "sk-***REDACTED***" in content

    def test_sanitize_disabled(self, tmp_path):
        """Sensitive data is preserved when sanitize=False."""
        path = tmp_path / "raw.yaml"
        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.RECORD, sanitize=False
        )
        req = NormalizedRequest(
            messages=[{"content": "Key: sk-proj-abcdefghijklmnopqrstuvwxyz1234"}],
            model="gpt-4.1",
        )
        resp = NormalizedResponse(content="ok")
        rec.record(req, resp)
        rec.__exit__(None, None, None)

        content = path.read_text()
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz1234" in content

    def test_sanitize_custom_patterns(self, tmp_path):
        """Custom sanitize patterns are applied."""
        from agentverify.cassette.sanitize import SanitizePattern

        path = tmp_path / "custom.yaml"
        custom = [SanitizePattern(name="custom", pattern=r"secret-\d+", replacement="***")]
        rec = LLMCassetteRecorder(
            cassette_path=path, mode=CassetteMode.RECORD, sanitize=custom
        )
        req = NormalizedRequest(
            messages=[{"content": "value: secret-42"}],
            model="m",
        )
        resp = NormalizedResponse(content="ok")
        rec.record(req, resp)
        rec.__exit__(None, None, None)

        content = path.read_text()
        assert "secret-42" not in content
        assert "***" in content

    def test_sanitize_default_is_true(self, tmp_path):
        """Default sanitize is True (DEFAULT_PATTERNS applied)."""
        path = tmp_path / "default.yaml"
        rec = LLMCassetteRecorder(cassette_path=path, mode=CassetteMode.RECORD)
        assert rec._sanitize_patterns is not None
        assert len(rec._sanitize_patterns) > 0



# ---------------------------------------------------------------------------
# step_probe integration with LLMCassetteRecorder (v0.3.0)
# ---------------------------------------------------------------------------


from agentverify import step_probe
from agentverify.cassette.recorder import _build_execution_result, _request_to_context


class TestRecorderProbeIntegration:
    def _make_recorder(self, tmp_path, cassette_file, provider="openai"):
        return LLMCassetteRecorder(
            cassette_path=tmp_path / cassette_file,
            mode=CassetteMode.AUTO,
            provider=provider,
        )

    def test_probe_events_recorded(self, tmp_path):
        """Probe events go through the recorder's ProbeSession protocol."""
        rec = LLMCassetteRecorder(
            cassette_path=tmp_path / "unused.yaml",
            mode=CassetteMode.AUTO,
            provider="openai",
        )
        with rec:
            with step_probe("outer") as p:
                p.set_tool_result({"x": 1})
                with step_probe("inner"):
                    pass

        # Should have recorded 4 probe events (2 enter + 2 exit)
        assert len(rec._probe_events) == 4
        kinds = [e[0] for e in rec._probe_events]
        assert kinds == ["enter", "enter", "exit", "exit"]

    def test_probe_without_llm_results_in_standalone_step(self, tmp_path):
        """A probe with no LLM call emits a standalone probe step."""
        rec = LLMCassetteRecorder(
            cassette_path=tmp_path / "unused.yaml",
            mode=CassetteMode.AUTO,
            provider="openai",
        )
        with rec:
            with step_probe("cache_lookup", output="miss"):
                pass

        result = rec.to_execution_result()
        names = [s.name for s in result.steps]
        assert "cache_lookup" in names

    def test_request_to_context_none(self):
        """_request_to_context returns None for None."""
        assert _request_to_context(None) is None


class TestBuildExecutionResultEdgeCases:
    def test_build_with_no_probe_stack(self):
        """_build_execution_result works with an empty probe_stacks list."""
        from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse

        req = NormalizedRequest(messages=[], model="gpt-4")
        resp = NormalizedResponse(content="hi")

        result = _build_execution_result(
            interactions=[(req, resp)],
            probe_stacks=[],  # missing stacks — falls back to empty
            probe_events=[],
            probe_tool_results={},
            duration_ms=10.0,
        )
        assert len(result.steps) == 1
        assert result.steps[0].output == "hi"

    def test_build_merges_probe_token_usage(self):
        """Multiple responses inside same probe accumulate token usage."""
        from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
        from agentverify import TokenUsage

        req = NormalizedRequest(messages=[], model="gpt-4")
        r1 = NormalizedResponse(content=None, token_usage=TokenUsage(input_tokens=10, output_tokens=5))
        r2 = NormalizedResponse(content="done", token_usage=TokenUsage(input_tokens=20, output_tokens=7))

        # Both inside probe id=1 (innermost).
        result = _build_execution_result(
            interactions=[(req, r1), (req, r2)],
            probe_stacks=[[1], [1]],
            probe_events=[
                ("enter", 1, "thinking", None),
                ("exit", 1, None, None),
            ],
            probe_tool_results={},
            duration_ms=None,
        )
        assert len(result.steps) == 1
        assert result.steps[0].name == "thinking"
        assert result.steps[0].token_usage.input_tokens == 30
        assert result.steps[0].token_usage.output_tokens == 12

    def test_build_unfinished_probe_skipped(self):
        """A probe that never exited is not emitted as a standalone step."""
        result = _build_execution_result(
            interactions=[],
            probe_stacks=[],
            probe_events=[("enter", 1, "partial", None)],  # no matching exit
            probe_tool_results={},
            duration_ms=None,
        )
        assert result.steps == []

    def test_build_probe_not_in_any_stack_but_has_exit(self):
        """Probe with exit but handle never seen in any stack becomes standalone."""
        result = _build_execution_result(
            interactions=[],
            probe_stacks=[],
            probe_events=[
                ("enter", 1, "foo", None),
                ("exit", 1, None, "out"),
            ],
            probe_tool_results={1: ["tool-r"]},
            duration_ms=None,
        )
        assert len(result.steps) == 1
        assert result.steps[0].name == "foo"
        assert result.steps[0].output == "out"
        assert result.steps[0].tool_results == ["tool-r"]



class TestBuildExecutionResultProbeMerge:
    def test_probe_first_response_no_tokens_second_has_tokens(self):
        """Within a probe, first response has no tokens; second brings tokens."""
        from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
        from agentverify import TokenUsage

        req = NormalizedRequest(messages=[], model="gpt-4")
        r1 = NormalizedResponse(content=None, token_usage=None)
        r2 = NormalizedResponse(content="done", token_usage=TokenUsage(input_tokens=5, output_tokens=3))

        result = _build_execution_result(
            interactions=[(req, r1), (req, r2)],
            probe_stacks=[[1], [1]],
            probe_events=[
                ("enter", 1, "p", None),
                ("exit", 1, None, None),
            ],
            probe_tool_results={},
            duration_ms=None,
        )
        assert len(result.steps) == 1
        assert result.steps[0].token_usage.input_tokens == 5

    def test_probe_exit_without_enter_ignored(self, tmp_path):
        """Lone probe_exit without a matching enter doesn't crash."""
        rec = LLMCassetteRecorder(
            cassette_path=tmp_path / "nonexistent-av-test.yaml",
            mode=CassetteMode.AUTO,
            provider="openai",
        )
        # Simulate orphan exit event — should not raise.
        rec.probe_exit(handle_id=999, output=None)



class TestBuildExecutionResultProbeOrphanExit:
    def test_exit_event_without_enter_ignored(self):
        """An exit event for an unknown handle_id is ignored."""
        result = _build_execution_result(
            interactions=[],
            probe_stacks=[],
            probe_events=[("exit", 999, None, "orphan")],  # no matching enter
            probe_tool_results={},
            duration_ms=None,
        )
        assert result.steps == []

    def test_unknown_event_kind_ignored(self):
        """An unexpected event kind falls through the kind switch."""
        result = _build_execution_result(
            interactions=[],
            probe_stacks=[],
            probe_events=[("bogus", 1, "name", None)],
            probe_tool_results={},
            duration_ms=None,
        )
        assert result.steps == []
