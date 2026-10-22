"""Replay fails when the agent's prompt or tool schemas changed since the cassette was recorded."""

from __future__ import annotations

from typing import Any

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.io import save_cassette
from agentverify.cassette.recorder import CassetteMode, LLMCassetteRecorder, _stale_cassettes
from agentverify.errors import CassetteRequestMismatchError


def _request(messages=None, tools=None, **parameters: Any) -> NormalizedRequest:
    return NormalizedRequest(messages=messages or [], model="m", tools=tools, parameters=parameters)


def _replay(tmp_path, recorded: NormalizedRequest, actual: NormalizedRequest, **kwargs):
    path = tmp_path / "c.yaml"
    save_cassette(path, [(recorded, NormalizedResponse(content="ok"))], provider="openai")
    rec = LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, **kwargs)
    return rec, rec.lookup(actual)


@pytest.fixture(autouse=True)
def _clear():
    _stale_cassettes.clear()
    yield
    _stale_cassettes.clear()


class TestWhatIsCompared:
    @pytest.mark.parametrize(
        "recorded, actual",
        [
            ({"messages": [{"role": "system", "content": "Be brief."}]}, {"messages": [{"role": "system", "content": "Be long."}]}),
            ({"messages": [{"role": "developer", "content": "a"}]}, {"messages": [{"role": "developer", "content": "b"}]}),
            ({"system": "Be brief."}, {"system": "Be long."}),  # Anthropic
            ({"system": [{"type": "text", "text": "a"}]}, {"system": [{"type": "text", "text": "b"}]}),
            ({"system": [{"text": "a"}]}, {"system": [{"text": "b"}]}),  # Bedrock
            ({"system_instruction": "a"}, {"system_instruction": "b"}),  # Gemini
        ],
        ids=["openai", "developer", "anthropic", "anthropic-blocks", "bedrock", "gemini"],
    )
    def test_system_prompt(self, tmp_path, recorded, actual):
        messages = lambda d: d.pop("messages", None)  # noqa: E731
        r, a = dict(recorded), dict(actual)
        with pytest.raises(CassetteRequestMismatchError, match="system prompt"):
            _replay(tmp_path, _request(messages(r), **r), _request(messages(a), **a))

    def test_user_text_but_not_tool_results(self, tmp_path):
        def history(question, tool_output):
            return [
                {"role": "user", "content": [{"text": question}, "plain block"]},
                {"role": "assistant", "content": [{"toolUse": {"name": "f", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"content": [{"text": tool_output}]}}]},
                {"role": "tool", "content": tool_output},
                {"role": "user", "parts": [{"text": question}, {"function_response": {"response": tool_output}}]},
                "not-a-message",
            ]

        # A tool's output may change between runs (tools running for real); that is not a stale prompt.
        _, response = _replay(tmp_path, _request(history("weather?", "sunny")), _request(history("weather?", "rainy")))
        assert response.content == "ok"
        with pytest.raises(CassetteRequestMismatchError, match="user messages"):
            _replay(tmp_path, _request(history("weather?", "sunny")), _request(history("forecast?", "sunny")))

    def test_tool_schemas(self, tmp_path):
        recorded = [{"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}]
        actual = [{"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}]
        with pytest.raises(CassetteRequestMismatchError, match="tool schema 'f'") as exc:
            _replay(tmp_path, _request(tools=recorded), _request(tools=actual))
        assert '-      "type": "string"' in str(exc.value) and '+      "type": "integer"' in str(exc.value)
        same = [{"name": "f", "parameters": {"required": ["a"], "properties": {"a": {"type": "string"}}, "type": "object"}}, {"no": "name"}]
        assert _replay(tmp_path, _request(tools=recorded), _request(tools=same))[1].content == "ok"

    def test_match_ignore(self, tmp_path):
        recorded = _request([{"role": "system", "content": "Today is 2026-10-21."}])
        actual = _request([{"role": "system", "content": "Today is 2026-10-22."}])
        assert _replay(tmp_path, recorded, actual, match_ignore=[r"\d{4}-\d{2}-\d{2}"])[1].content == "ok"
        with pytest.raises(CassetteRequestMismatchError):
            _replay(tmp_path, recorded, actual)

    def test_match_requests_false_skips_it(self, tmp_path):
        recorded, actual = _request([{"role": "system", "content": "a"}]), _request([{"role": "system", "content": "b"}])
        assert _replay(tmp_path, recorded, actual, match_requests=False)[1].content == "ok"


class TestReporting:
    def test_short_values_are_shown_inline_and_long_ones_as_a_diff(self):
        short = str(CassetteRequestMismatchError(0, "model", "a", "b"))
        assert "Recorded: 'a'" in short
        long = str(CassetteRequestMismatchError(1, "system prompt", "line one\nline two", "line one\nline 2"))
        assert "--- recorded" in long and "-line two" in long and "+line 2" in long and "match_ignore" in long

    def test_a_caught_mismatch_still_fails_when_the_cassette_closes(self, tmp_path):
        path = tmp_path / "c.yaml"
        save_cassette(path, [(_request([{"role": "user", "content": "a"}]), NormalizedResponse(content="ok"))])
        with pytest.raises(CassetteRequestMismatchError, match="user messages"):
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY) as rec:
                try:
                    rec.lookup(_request([{"role": "user", "content": "b"}]))
                except CassetteRequestMismatchError:
                    pass  # a framework catching it
        with pytest.raises(RuntimeError):  # the test's own error is not hidden
            with LLMCassetteRecorder(path, mode=CassetteMode.REPLAY) as rec:
                with pytest.raises(CassetteRequestMismatchError):
                    rec.lookup(_request([{"role": "user", "content": "b"}]))
                raise RuntimeError("boom")

    def test_summary_lists_stale_cassettes_and_the_record_command(self, tmp_path, cassette):
        from agentverify.plugin import pytest_terminal_summary

        path = tmp_path / "s.yaml"
        save_cassette(path, [(_request([{"role": "user", "content": "a"}]), NormalizedResponse(content="ok"))])
        rec = cassette("s.yaml", cassette_dir=tmp_path, mode="replay")
        with pytest.raises(CassetteRequestMismatchError):
            rec.lookup(_request([{"role": "user", "content": "b"}]))
        _stale_cassettes.append((str(path), None, "model"))  # the same cassette again, without a test id

        class Reporter:
            lines: list[str] = []
            config = type("Config", (), {"rootpath": tmp_path})

            def section(self, title):
                self.lines.append(f"== {title}")

            def write_line(self, line):
                self.lines.append(line)

        reporter = Reporter()
        pytest_terminal_summary(reporter)
        assert reporter.lines[0] == "== agentverify: stale cassettes"
        assert reporter.lines[1] == "s.yaml  (model, user messages changed)" and len(reporter.lines) == 5
        assert reporter.lines[-1].startswith("  pytest --cassette-mode=record ") and "test_summary_lists" in reporter.lines[-1]
        assert _stale_cassettes == []
        pytest_terminal_summary(reporter)  # nothing stale: no section
        assert len(reporter.lines) == 5

        _stale_cassettes.append(("/elsewhere/s.yaml", None, "model"))
        reporter.lines = []
        pytest_terminal_summary(reporter)
        assert reporter.lines == ["== agentverify: stale cassettes", "/elsewhere/s.yaml  (model changed)"]


class TestReviewFixes:
    def test_the_live_request_is_sanitized_before_comparing(self, tmp_path):
        from agentverify.cassette.sanitize import DEFAULT_PATTERNS, SanitizePattern

        email = [*DEFAULT_PATTERNS, SanitizePattern(name="email", pattern=r"[\w.]+@example\.com", replacement="<email>")]
        path = tmp_path / "s.yaml"
        question = [{"role": "user", "content": "My email is alice@example.com"}]
        with LLMCassetteRecorder(path, mode=CassetteMode.RECORD, sanitize=email) as rec:
            rec.record(_request(question), NormalizedResponse(content="ok"))
        assert "alice" not in path.read_text()
        assert LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, sanitize=email).lookup(_request(question)).content == "ok"
        unsanitized = LLMCassetteRecorder(path, mode=CassetteMode.REPLAY, sanitize=False)
        with pytest.raises(CassetteRequestMismatchError):
            unsanitized.lookup(_request(question))

    def test_cassettes_without_the_field_skip_the_check(self, tmp_path):
        # No schema recorded (an older cassette), or no system prompt recorded.
        tools = [{"name": "f", "parameters": {"type": "object"}}]
        assert _replay(tmp_path, _request(tools=[{"name": "f", "parameters": {}}]), _request(tools=tools))[1].content == "ok"
        assert _replay(tmp_path, _request(), _request(system="new prompt"))[1].content == "ok"
        # google-genai Content objects recorded as their repr by older versions.
        legacy = _request([{"role": "user", "content": "parts=[Part(text='hi')] role='user'"}])
        assert _replay(tmp_path, legacy, _request([{"role": "user", "parts": [{"text": "hi"}]}]))[1].content == "ok"

    def test_unset_schema_fields_do_not_count(self, tmp_path):
        recorded = [{"name": "f", "parameters": {"type": "OBJECT", "nullable": None}}]
        actual = [{"name": "f", "parameters": {"type": "OBJECT", "nullable": None, "new_field": None}}]
        assert _replay(tmp_path, _request(tools=recorded), _request(tools=actual))[1].content == "ok"

    def test_gemini_system_instruction_as_content(self, tmp_path):
        recorded = _request(system_instruction={"role": "system", "parts": [{"text": "Be terse."}]})
        actual = _request(system_instruction={"role": "system", "parts": [{"text": "Be chatty."}]})
        with pytest.raises(CassetteRequestMismatchError, match="system prompt"):
            _replay(tmp_path, recorded, actual)

    def test_hint_depends_on_the_field(self):
        assert "match_ignore" not in str(CassetteRequestMismatchError(0, "tool schema 'f'", "{}", "{}"))
        assert "match_ignore" in str(CassetteRequestMismatchError(0, "user messages", "a", "b"))

    def test_record_command_quotes_test_ids(self, tmp_path):
        from agentverify.plugin import pytest_terminal_summary

        _stale_cassettes.append(("c.yaml", "tests/test_x.py::test_y[gpt-4o]", "model"))

        class Reporter:
            lines: list[str] = []
            config = type("Config", (), {"rootpath": tmp_path})

            def section(self, title):
                pass

            def write_line(self, line):
                self.lines.append(line)

        reporter = Reporter()
        pytest_terminal_summary(reporter)
        assert reporter.lines[-1] == "  pytest --cassette-mode=record 'tests/test_x.py::test_y[gpt-4o]'"
        assert str(tmp_path) in reporter.lines[-2]
