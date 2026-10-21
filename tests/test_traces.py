"""Tests for building ExecutionResults from OpenTelemetry traces."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentverify import Handoff, ToolCall, assert_no_tool_errors, assert_step_uses_result_from, assert_tool_calls
from agentverify.traces import from_otel


def _span(name, span_id, parent=None, trace="t1", start=0, end=1, error=False, attributes=None, events=()):
    """A span as ReadableSpan.to_json() writes it (times in seconds past 2026-10-22T00:00:00Z)."""
    stamp = lambda s: f"2026-10-22T00:00:{s:06.3f}+00:00"  # noqa: E731
    return {
        "name": name,
        "context": {"trace_id": trace, "span_id": span_id},
        "parent_id": parent,
        "start_time": stamp(start),
        "end_time": stamp(end),
        "status": {"status_code": "ERROR" if error else "UNSET"},
        "attributes": attributes or {},
        "events": [{"name": n, "attributes": a} for n, a in events],
    }


def _latest_trace():
    """The current GenAI conventions: messages as attributes (Pydantic AI) or event attributes (Strands opt-in)."""
    out1 = [{"role": "assistant", "parts": [{"type": "tool_call", "name": "get_weather", "id": "c1", "arguments": {"city": "Tokyo"}}]}]
    out2 = [{"role": "assistant", "parts": [{"type": "reasoning", "content": "..."}, {"type": "text", "content": "Sunny, 22C."}]}]
    inp2 = [{"role": "user", "parts": [{"type": "text", "content": "Weather?"}]}, {"role": "tool", "parts": [{"type": "tool_call_response", "id": "c1", "response": "sunny 22C"}]}]
    return [
        _span("invoke_agent forecaster", "a", attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "forecaster"}, end=4),
        _span("chat m", "c1", "a", start=0.1, end=0.5, attributes={"gen_ai.operation.name": "chat", "gen_ai.output.messages": json.dumps(out1), "gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 4}),
        _span("execute_tool get_weather", "x1", "a", start=0.6, end=0.7, attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_weather", "gen_ai.tool.call.id": "c1", "gen_ai.tool.call.result": "sunny 22C"}),
        _span(
            "chat m",
            "c2",
            "a",
            start=1,
            end=2,
            attributes={"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 20, "gen_ai.usage.output_tokens": 5},
            events=[("gen_ai.client.inference.operation.details", {"gen_ai.input.messages": json.dumps(inp2), "gen_ai.output.messages": json.dumps(out2)})],
        ),
    ]


class TestConventions:
    def test_current_conventions(self):
        [result] = from_otel(_latest_trace())
        assert_tool_calls(result, expected=[ToolCall("get_weather", {"city": "Tokyo"})])
        assert result.steps[0].tool_results == ["sunny 22C"] and result.steps[0].agent == "forecaster"
        assert_step_uses_result_from(result, step=1, depends_on=0, via="tool_result")
        assert result.final_output == "Sunny, 22C." and result.token_usage.input_tokens == 30
        assert result.duration_ms == pytest.approx(4000)
        assert result.handoffs == []

    def test_earlier_conventions_as_strands_emits_them(self):
        choice = [{"toolUse": {"toolUseId": "t1", "name": "add", "input": {"a": 2}}}, "not-a-block", {"text": "thinking"}]
        spans = [
            _span("invoke_agent calc", "a", attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "calc"}),
            _span("execute_event_loop_cycle", "l", "a"),
            _span(
                "chat",
                "c",
                "l",
                start=0.1,
                attributes={"gen_ai.operation.name": "chat", "gen_ai.usage.prompt_tokens": 5, "gen_ai.usage.completion_tokens": 2},
                events=[("gen_ai.user.message", {"content": json.dumps([{"text": "2+?"}])}), ("gen_ai.choice", {"message": json.dumps(choice)})],
            ),
            _span(
                "execute_tool add",
                "x",
                "l",
                start=0.2,
                error=True,
                attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "add", "gen_ai.tool.call.id": "t1", "gen_ai.tool.status": "error"},
                events=[("gen_ai.choice", {"message": json.dumps([{"text": "Error: bad"}])}), ("gen_ai.tool.message", {"content": "{}"})],
            ),
        ]
        [result] = from_otel(spans)
        step = result.steps[0]
        assert [(c.name, c.arguments) for c in step.tool_calls] == [("add", {"a": 2})]
        assert step.tool_results == [[{"text": "Error: bad"}]] and step.tool_results_meta == [{"is_error": True}]
        assert step.output == "thinking" and step.agent == "calc" and step.token_usage.output_tokens == 2
        assert step.input_context == {"messages": [{"content": [{"text": "2+?"}]}]}
        with pytest.raises(AssertionError):
            assert_no_tool_errors(result)

    def test_openinference(self):
        attributes = {
            "openinference.span.kind": "LLM",
            "llm.token_count.prompt": 11,
            "llm.token_count.completion": 4,
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "2+3?",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "add",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"a": 2, "b": 3}',
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "c1",
            "llm.output_messages.1.message.role": "assistant",
            "llm.output_messages.1.message.content": "adding",
        }
        spans = [
            _span("Math", "a", attributes={"openinference.span.kind": "AGENT", "agent.name": "Math"}),
            _span("response", "c", "a", attributes=attributes),
            _span("add", "x", "a", start=0.5, attributes={"openinference.span.kind": "TOOL", "tool.name": "add", "output.value": "5"}),
            _span("handoff to Other", "h", "a", start=0.6, attributes={"openinference.span.kind": "TOOL"}),
        ]
        [result] = from_otel(spans)
        step = result.steps[0]
        assert [(c.name, c.arguments) for c in step.tool_calls] == [("add", {"a": 2, "b": 3})]
        assert step.tool_results == [5] and step.output == "adding" and step.agent == "Math"
        assert step.input_context == {"messages": [{"role": "user", "content": "2+3?"}]}

    def test_without_message_content_tool_spans_supply_the_calls(self):
        spans = [
            _span("chat", "c", attributes={"gen_ai.operation.name": "chat"}),
            _span("execute_tool lookup", "x", start=0.2, attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.call.arguments": '{"k": 1}'}),
            _span("execute_tool raw", "y", start=0.3, attributes={"gen_ai.operation.name": "execute_tool", "input.value": "plain text", "error.type": "Boom"}),
            _span("execute_tool (merged)", "z", start=0.4, attributes={"gen_ai.operation.name": "execute_tool"}),
        ]
        [result] = from_otel(spans)
        step = result.steps[0]
        assert [(c.name, c.arguments) for c in step.tool_calls] == [("lookup", {"k": 1}), ("raw", {"input": "plain text"})]
        assert step.tool_results == [None, None] and step.tool_results_meta == [{}, {"is_error": True}]
        assert step.token_usage is None and step.input_context is None and result.token_usage is None

    def test_nested_inference_spans_count_once_and_tool_names_match_without_ids(self):
        out = [{"role": "model", "parts": [{"type": "tool_call", "name": "f", "arguments": "not json"}, {"type": "tool_call", "name": "g"}]}, "bad"]
        spans = [
            _span("call_llm", "c", attributes={"gen_ai.operation.name": "generate_content", "gen_ai.output.messages": json.dumps(out)}),
            _span("generate_content m", "n", "c", attributes={"gen_ai.operation.name": "generate_content"}),
            _span("execute_tool f", "x", start=0.1, attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "f", "gcp.vertex.agent.tool_response": '{"ok": true}'}),
        ]
        [result] = from_otel(spans)
        assert len(result.steps) == 1
        assert [(c.name, c.arguments) for c in result.steps[0].tool_calls] == [("f", {"input": "not json"}), ("g", {})]
        assert result.steps[0].tool_results == [{"ok": True}, None]  # g has no tool span

    def test_handoffs_from_agent_changes(self):
        def agent(name, span_id, start):
            return [
                _span(f"invoke_agent {name}", span_id, start=start, end=start + 1, attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": name}),
                _span("chat", span_id + "c", span_id, start=start + 0.1, attributes={"gen_ai.operation.name": "chat"}),
            ]

        [result] = from_otel(agent("triage", "a", 0) + agent("billing", "b", 2))
        assert result.handoffs == [Handoff("triage", "billing", 0)]


class TestFormats:
    def test_sdk_spans(self):
        def sdk(name, span_id, parent=None, error=False):
            return SimpleNamespace(
                name=name,
                context=SimpleNamespace(trace_id=1, span_id=span_id),
                parent=SimpleNamespace(span_id=parent) if parent else None,
                status=SimpleNamespace(status_code=SimpleNamespace(name="ERROR" if error else "OK")),
                start_time=span_id * 1_000_000,
                end_time=None,
                attributes={"gen_ai.operation.name": "chat"} if name == "chat" else {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "t"},
                events=[SimpleNamespace(name="e", attributes=None)],
            )

        [result] = from_otel([sdk("chat", 1), sdk("tool", 2, parent=1, error=True)])
        assert result.steps[0].tool_calls[0].name == "t" and result.steps[0].tool_results_meta == [{"is_error": True}]

    def test_otlp_json_documents_and_files(self, tmp_path):
        def value(v):
            if isinstance(v, bool):
                return {"boolValue": v}
            if isinstance(v, int):
                return {"intValue": str(v)}
            if isinstance(v, float):
                return {"doubleValue": v}
            if isinstance(v, list):
                return {"arrayValue": {"values": [value(x) for x in v]}}
            if isinstance(v, dict):
                return {"kvlistValue": {"values": [{"key": k, "value": value(x)} for k, x in v.items()]}}
            if v is None:
                return {}
            return {"stringValue": v}

        def otlp(name, span_id, trace, start, attributes, status=0, parent=""):
            return {
                "traceId": trace,
                "spanId": span_id,
                "parentSpanId": parent,
                "name": name,
                "startTimeUnixNano": str(start * 1_000_000_000),
                "endTimeUnixNano": str((start + 1) * 1_000_000_000),
                "status": {"code": status},
                "attributes": [{"key": k, "value": value(v)} for k, v in attributes.items()],
                "events": [{"name": "e", "attributes": [{"key": "x", "value": value(1.5)}]}],
            }

        out = json.dumps([{"parts": [{"type": "tool_call", "name": "f", "arguments": {"q": "x"}}]}])
        document = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                otlp("chat", "c", "late", 5, {"gen_ai.operation.name": "chat"}),
                                otlp("chat", "d", "early", 1, {"gen_ai.operation.name": "chat", "gen_ai.output.messages": out, "gen_ai.usage.input_tokens": 3, "flag": True, "tags": ["a"], "meta": {"k": None}}),
                                otlp("execute_tool f", "e", "early", 2, {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "f"}, status="STATUS_CODE_ERROR", parent="d"),
                            ]
                        }
                    ]
                },
                {"instrumentationLibrarySpans": [{"spans": []}]},
            ]
        }
        early, late = from_otel(document)
        assert early.steps[0].tool_calls[0].arguments == {"q": "x"} and early.steps[0].tool_results_meta == [{"is_error": True}]
        assert early.token_usage.input_tokens == 3 and late.steps[0].tool_calls == []
        single = tmp_path / "trace.json"
        single.write_text(json.dumps(document))
        lines = tmp_path / "traces.jsonl"
        lines.write_text(json.dumps(document) + "\n\n" + json.dumps({"resourceSpans": []}) + "\n")
        assert len(from_otel(single)) == len(from_otel(str(lines))) == 2

    def test_odd_inputs(self):
        assert from_otel([]) == []
        spans = [_span("chat", "c", attributes={"gen_ai.operation.name": "chat"}), _span("orphan", "o", parent="missing", trace="t1")]
        spans[0]["start_time"] = None
        [result] = from_otel(spans)
        assert len(result.steps) == 1
        loop = [_span("chat", "a", parent="b", attributes={"gen_ai.operation.name": "chat"}), _span("x", "b", parent="a")]
        assert len(from_otel(loop)[0].steps) == 1  # a parent cycle does not hang


class TestReviewFixes:
    def test_google_adk_default_attributes(self):
        response = {"content": {"parts": [{"text": "plan", "thought": True}, {"function_call": {"id": "c1", "name": "add", "args": {"a": 1}}}, "bad"]}}
        final = {"content": {"parts": [{"text": "1"}]}}
        spans = [
            _span("call_llm", "p1", attributes={"gcp.vertex.agent.llm_response": json.dumps(response)}),
            _span("generate_content m", "g1", "p1", attributes={"gen_ai.operation.name": "generate_content"}),
            _span("execute_tool add", "x", start=0.5, attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "add", "gen_ai.tool.call.id": "c1", "gcp.vertex.agent.tool_response": '{"status": "error", "error_message": "no"}'}),
            _span("call_llm", "p2", start=1, attributes={"gcp.vertex.agent.llm_response": json.dumps(final)}),
            _span("generate_content m", "g2", "p2", start=1, attributes={"gen_ai.operation.name": "generate_content"}),
            _span("call_llm", "p3", start=2, attributes={"gcp.vertex.agent.llm_response": json.dumps({"content": "not a dict"})}),
            _span("generate_content m", "g3", "p3", start=2, attributes={"gen_ai.operation.name": "generate_content"}),
        ]
        [result] = from_otel(spans)
        assert [(c.name, c.arguments) for c in result.steps[0].tool_calls] == [("add", {"a": 1})]
        assert result.steps[0].output is None and result.steps[0].tool_results_meta == [{"is_error": True}]
        assert result.final_output == "1" and result.steps[2].output is None

    def test_failed_tool_response_in_output_messages(self):
        response = json.dumps([{"parts": [{"type": "text", "content": "x"}]}, {"parts": [{"type": "tool_call_response", "response": "Error: kaboom"}]}])
        spans = [
            _span("chat", "c", attributes={"gen_ai.operation.name": "chat"}),
            _span("execute_tool boom", "x", start=0.1, error=True, attributes={"gen_ai.operation.name": "execute_tool"}, events=[("gen_ai.client.inference.operation.details", {"gen_ai.output.messages": response})]),
        ]
        assert from_otel(spans)[0].steps[0].tool_results == ["Error: kaboom"]

    def test_tools_attach_by_id_and_agent_and_keep_their_place(self):
        out = lambda *calls: json.dumps([{"parts": [{"type": "tool_call", "name": n, "id": i} for n, i in calls]}])  # noqa: E731
        spans = [
            _span("invoke_agent a", "A", start=0, end=5, attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "a"}),
            _span("invoke_agent b", "B", start=0, end=5, attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "b"}),
            _span("chat", "ca", "A", start=0.1, attributes={"gen_ai.operation.name": "chat", "gen_ai.output.messages": out(("weather", "w1"), ("handoff", None))}),
            _span("chat", "cb", "B", start=0.2, attributes={"gen_ai.operation.name": "chat", "gen_ai.output.messages": out(("stock", None))}),
            # b's model call came last, but the id ties weather to a; the agent ties stock to b.
            _span("execute_tool weather", "tw", "A", start=0.3, attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "weather", "gen_ai.tool.call.id": "w1", "gen_ai.tool.call.result": "sunny"}),
            _span("execute_tool stock", "ts", "B", start=0.4, attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "stock", "gen_ai.tool.call.result": "up"}),
        ]
        [result] = from_otel(spans)
        assert result.steps[0].tool_results == ["sunny", None] and result.steps[1].tool_results == ["up"]
        assert result.handoffs == []  # a and b ran in parallel

    def test_an_agent_called_as_a_tool_is_not_a_handoff(self):
        spans = [
            _span("invoke_agent boss", "A", start=0, end=5, attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "boss"}),
            _span("chat", "c1", "A", start=0.1, attributes={"gen_ai.operation.name": "chat"}),
            _span("execute_tool helper", "T", "A", start=1, end=2, attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "helper"}),
            _span("invoke_agent helper", "H", "T", start=1.1, end=1.9, attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "helper"}),
            _span("chat", "c2", "H", start=1.2, attributes={"gen_ai.operation.name": "chat", "gen_ai.output.messages": json.dumps([{"parts": [{"type": "text", "content": "inner"}]}])}),
            _span("chat", "c3", "A", start=3, attributes={"gen_ai.operation.name": "chat"}),
        ]
        [result] = from_otel(spans)
        assert [s.agent for s in result.steps] == ["boss", "helper", "boss"]
        assert result.handoffs == [] and result.final_output is None

    def test_tools_before_any_model_call_and_without_one(self):
        tool = _span("execute_tool t", "x", start=0, attributes={"gen_ai.operation.name": "execute_tool"})
        [early] = from_otel([tool, _span("chat", "c", start=1, attributes={"gen_ai.operation.name": "chat"})])
        assert early.steps[0].tool_calls[0].name == "t"
        assert from_otel([tool])[0].steps == []

    def test_malformed_content_and_other_documents(self, tmp_path):
        spans = [
            _span("chat", "c", attributes={"gen_ai.operation.name": "chat", "gen_ai.output.messages": json.dumps([{"parts": None}, {"parts": ["x"]}])}, events=[("gen_ai.choice", {"message": json.dumps({"text": "object, not list"})})]),
        ]
        assert from_otel(spans)[0].final_output == "object, not list"
        lines = tmp_path / "mixed.jsonl"
        lines.write_text(json.dumps({"resourceLogs": []}) + "\n" + json.dumps({"resourceSpans": []}) + "\n")
        assert from_otel(lines) == []

    def test_unfinished_spans_have_no_negative_duration(self):
        span = _span("chat", "c", start=1, attributes={"gen_ai.operation.name": "chat"})
        span["end_time"] = None
        assert from_otel([span])[0].steps[0].duration_ms == 0

    def test_agents_named_on_the_model_calls_alone_still_hand_off(self):
        spans = [
            _span("chat", "c1", attributes={"gen_ai.operation.name": "chat", "gen_ai.agent.name": "a"}),
            _span("chat", "c2", start=1, attributes={"gen_ai.operation.name": "chat", "gen_ai.agent.name": "b"}),
        ]
        assert [(h.from_agent, h.to_agent) for h in from_otel(spans)[0].handoffs] == [("a", "b")]
