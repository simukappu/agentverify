"""Tests for agentverify.probe — step_probe context manager."""

from __future__ import annotations

import pytest

from agentverify import (
    MockLLM,
    ProbeHandle,
    assert_step,
    assert_step_output,
    mock_response,
    step_probe,
)
from agentverify.probe import _active_session


class TestStepProbeNoOp:
    """step_probe must be a zero-cost no-op outside of recorder/MockLLM contexts."""

    def test_noop_without_session(self):
        assert _active_session.get() is None
        with step_probe("foo") as handle:
            assert isinstance(handle, ProbeHandle)
            assert handle.name == "foo"

    def test_noop_set_output_is_harmless(self):
        with step_probe("foo") as handle:
            handle.set_output("hello")
        assert handle._output == "hello"

    def test_noop_set_tool_result_is_harmless(self):
        with step_probe("foo") as handle:
            handle.set_tool_result({"x": 1})
        assert handle._tool_results == [{"x": 1}]

    def test_noop_nested(self):
        with step_probe("outer"):
            with step_probe("inner"):
                pass

    def test_noop_exception_propagates(self):
        with pytest.raises(RuntimeError):
            with step_probe("foo"):
                raise RuntimeError("boom")


class TestStepProbeWithMockLLM:
    """step_probe attaches names to steps inside a recording session."""

    def test_probe_names_attach_to_llm_step(self):
        with MockLLM(
            [
                mock_response(tool_calls=[("search", {"q": "Tokyo"})]),
                mock_response(content="Done"),
            ],
            provider="openai",
        ) as rec:
            from openai import OpenAI

            client = OpenAI(api_key="test")
            with step_probe("first_call"):
                client.chat.completions.create(model="gpt-4", messages=[])
            with step_probe("second_call"):
                client.chat.completions.create(model="gpt-4", messages=[])

        result = rec.to_execution_result()
        assert len(result.steps) == 2
        assert result.steps[0].name == "first_call"
        assert result.steps[1].name == "second_call"
        assert result.steps[0].source == "probe"

    def test_probe_merges_multiple_llm_calls(self):
        with MockLLM(
            [
                mock_response(tool_calls=[("search", {"q": "a"})]),
                mock_response(tool_calls=[("search", {"q": "b"})]),
                mock_response(content="Done"),
            ],
            provider="openai",
        ) as rec:
            from openai import OpenAI

            client = OpenAI(api_key="test")
            with step_probe("plan"):
                client.chat.completions.create(model="gpt-4", messages=[])
                client.chat.completions.create(model="gpt-4", messages=[])
            client.chat.completions.create(model="gpt-4", messages=[])

        result = rec.to_execution_result()
        # Two steps: one merged probe step + one non-probe step.
        assert len(result.steps) == 2
        assert result.steps[0].name == "plan"
        assert len(result.steps[0].tool_calls) == 2
        assert result.steps[1].name is None

    def test_probe_without_llm_call_produces_standalone_step(self):
        with MockLLM(
            [mock_response(content="hi")], provider="openai"
        ) as rec:
            from openai import OpenAI

            client = OpenAI(api_key="test")
            with step_probe("cache_lookup", output="cache miss") as h:
                # Simulate user workflow logic with no LLM call.
                h.set_tool_result({"cached": False})
            client.chat.completions.create(model="gpt-4", messages=[])

        result = rec.to_execution_result()
        # Expect a probe step for cache_lookup + an LLM step.
        names = [s.name for s in result.steps]
        assert "cache_lookup" in names

        probe_step = next(s for s in result.steps if s.name == "cache_lookup")
        assert probe_step.source == "probe"
        assert probe_step.output == "cache miss"
        assert probe_step.tool_results == [{"cached": False}]

    def test_probe_assertion_by_name(self):
        with MockLLM(
            [mock_response(tool_calls=[("search", {"q": "x"})])],
            provider="openai",
        ) as rec:
            from openai import OpenAI

            client = OpenAI(api_key="test")
            with step_probe("do_search"):
                client.chat.completions.create(model="gpt-4", messages=[])

        result = rec.to_execution_result()
        from agentverify import ToolCall

        assert_step(
            result,
            name="do_search",
            expected_tool=ToolCall("search", {"q": "x"}),
        )

    def test_probe_output_assertion(self):
        with MockLLM([mock_response(content="hi")], provider="openai") as rec:
            with step_probe("setup", output="ready"):
                pass

        result = rec.to_execution_result()
        assert_step_output(result, name="setup", equals="ready")


class TestStepProbeContextIsolation:
    """Probe state must not leak between sessions."""

    def test_session_resets_after_exit(self):
        assert _active_session.get() is None
        with MockLLM([mock_response(content="x")], provider="openai"):
            assert _active_session.get() is not None
        assert _active_session.get() is None

    def test_nested_sessions(self):
        with MockLLM([mock_response(content="outer")], provider="openai") as outer:
            outer_session = _active_session.get()
            with MockLLM([mock_response(content="inner")], provider="openai") as inner:
                inner_session = _active_session.get()
                assert inner_session is inner
                assert inner_session is not outer_session
            # Back to outer session after inner exits.
            assert _active_session.get() is outer_session
        assert _active_session.get() is None



class TestMockLLMProbeEdgeCases:
    def test_orphan_probe_exit_ignored(self):
        """Orphan probe_exit on MockLLM (no matching enter) does not raise."""
        mock = MockLLM([mock_response(content="x")], provider="openai")
        # Simulate orphan exit directly on the session object
        mock.probe_exit(handle_id=999, output=None)
