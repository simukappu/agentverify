"""Tests for agentverify.cassette.sanitize — cassette sanitization."""

from __future__ import annotations

import pytest

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.cassette.sanitize import (
    DEFAULT_PATTERNS,
    SanitizePattern,
    sanitize_interactions,
    _redact_value,
)
from agentverify.models import TokenUsage


class TestRedactValue:
    """Tests for the _redact_value helper."""

    def test_string_redaction(self):
        compiled = [(__import__("re").compile(r"sk-[A-Za-z0-9_-]{20,}"), "sk-***REDACTED***")]
        result = _redact_value("key is sk-abcdefghijklmnopqrstuvwxyz", compiled)
        assert "sk-***REDACTED***" in result
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result

    def test_dict_redaction(self):
        compiled = [(__import__("re").compile(r"secret123"), "***")]
        result = _redact_value({"key": "my secret123 value"}, compiled)
        assert result == {"key": "my *** value"}

    def test_list_redaction(self):
        compiled = [(__import__("re").compile(r"token"), "***")]
        result = _redact_value(["my token here", "no match"], compiled)
        assert result == ["my *** here", "no match"]

    def test_nested_structure(self):
        compiled = [(__import__("re").compile(r"secret"), "***")]
        data = {"a": [{"b": "my secret"}]}
        result = _redact_value(data, compiled)
        assert result == {"a": [{"b": "my ***"}]}

    def test_non_string_passthrough(self):
        compiled = [(__import__("re").compile(r"x"), "y")]
        assert _redact_value(42, compiled) == 42
        assert _redact_value(None, compiled) is None
        assert _redact_value(3.14, compiled) == 3.14


class TestSanitizeInteractions:
    """Tests for sanitize_interactions()."""

    def _make_interaction(
        self,
        messages: list | None = None,
        content: str | None = None,
        tools: list | None = None,
    ) -> tuple[NormalizedRequest, NormalizedResponse]:
        req = NormalizedRequest(
            messages=messages or [],
            model="gpt-4.1",
            tools=tools,
        )
        resp = NormalizedResponse(content=content)
        return (req, resp)

    def test_openai_key_redacted(self):
        interactions = [
            self._make_interaction(
                messages=[{"role": "system", "content": "Key: sk-proj-abcdefghijklmnopqrstuvwxyz1234"}],
                content="Response with sk-proj-abcdefghijklmnopqrstuvwxyz1234",
            )
        ]
        result = sanitize_interactions(interactions)
        req, resp = result[0]
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz1234" not in req.messages[0]["content"]
        assert "sk-***REDACTED***" in req.messages[0]["content"]
        assert "sk-***REDACTED***" in resp.content

    def test_anthropic_key_redacted(self):
        interactions = [
            self._make_interaction(
                messages=[{"content": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"}],
            )
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert "sk-ant-***REDACTED***" in req.messages[0]["content"]

    def test_aws_key_redacted(self):
        interactions = [
            self._make_interaction(
                messages=[{"content": "AKIAIOSFODNN7EXAMPLE"}],
            )
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert "AKIA***REDACTED***" in req.messages[0]["content"]

    def test_bearer_token_redacted(self):
        interactions = [
            self._make_interaction(
                messages=[{"content": "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc"}],
            )
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert "Bearer ***REDACTED***" in req.messages[0]["content"]

    def test_tools_sanitized(self):
        interactions = [
            self._make_interaction(
                tools=[{"description": "Uses sk-proj-abcdefghijklmnopqrstuvwxyz1234"}],
            )
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert "sk-***REDACTED***" in req.tools[0]["description"]

    def test_response_tool_calls_sanitized(self):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(
            tool_calls=[{"name": "auth", "arguments": '{"token": "sk-proj-abcdefghijklmnopqrstuvwxyz1234"}'}],
        )
        result = sanitize_interactions([(req, resp)])
        _, sanitized_resp = result[0]
        assert "sk-***REDACTED***" in sanitized_resp.tool_calls[0]["arguments"]

    def test_token_usage_preserved(self):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(
            content="ok",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        result = sanitize_interactions([(req, resp)])
        _, sanitized_resp = result[0]
        assert sanitized_resp.token_usage.input_tokens == 10
        assert sanitized_resp.token_usage.output_tokens == 5

    def test_none_content_preserved(self):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(content=None, tool_calls=None)
        result = sanitize_interactions([(req, resp)])
        _, sanitized_resp = result[0]
        assert sanitized_resp.content is None
        assert sanitized_resp.tool_calls is None

    def test_custom_patterns(self):
        custom = [SanitizePattern(name="custom", pattern=r"my-secret-\d+", replacement="***")]
        interactions = [
            self._make_interaction(
                messages=[{"content": "value is my-secret-12345"}],
            )
        ]
        result = sanitize_interactions(interactions, patterns=custom)
        req, _ = result[0]
        assert "***" in req.messages[0]["content"]
        assert "my-secret-12345" not in req.messages[0]["content"]

    def test_empty_patterns_returns_unchanged(self):
        interactions = [
            self._make_interaction(messages=[{"content": "sk-proj-abcdefghijklmnopqrstuvwxyz1234"}])
        ]
        result = sanitize_interactions(interactions, patterns=[])
        req, _ = result[0]
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz1234" in req.messages[0]["content"]

    def test_no_sensitive_data_unchanged(self):
        interactions = [
            self._make_interaction(
                messages=[{"content": "Hello, world!"}],
                content="Response text",
            )
        ]
        result = sanitize_interactions(interactions)
        req, resp = result[0]
        assert req.messages[0]["content"] == "Hello, world!"
        assert resp.content == "Response text"

    def test_model_preserved(self):
        interactions = [
            self._make_interaction(messages=[])
        ]
        result = sanitize_interactions(interactions)
        req, _ = result[0]
        assert req.model == "gpt-4.1"

    def test_parameters_sanitized(self):
        req = NormalizedRequest(
            messages=[],
            model="m",
            parameters={"user": "sk-proj-abcdefghijklmnopqrstuvwxyz1234"},
        )
        resp = NormalizedResponse(content="ok")
        result = sanitize_interactions([(req, resp)])
        sanitized_req, _ = result[0]
        assert "sk-***REDACTED***" in sanitized_req.parameters["user"]

    def test_raw_metadata_sanitized(self):
        req = NormalizedRequest(messages=[], model="m")
        resp = NormalizedResponse(
            content="ok",
            raw_metadata={"header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghijklmnop"},
        )
        result = sanitize_interactions([(req, resp)])
        _, sanitized_resp = result[0]
        assert "Bearer ***REDACTED***" in sanitized_resp.raw_metadata["header"]


class TestDefaultPatterns:
    """Verify DEFAULT_PATTERNS structure."""

    def test_has_patterns(self):
        assert len(DEFAULT_PATTERNS) > 0

    def test_all_are_sanitize_pattern(self):
        for p in DEFAULT_PATTERNS:
            assert isinstance(p, SanitizePattern)
            assert p.name
            assert p.pattern
            assert p.replacement


class TestSensitiveKeys:
    """Values under credential keys are redacted whatever their format."""

    def _sanitize_params(self, parameters):
        req = NormalizedRequest(messages=[], model="m", parameters=parameters)
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        return out[0][0].parameters

    def test_api_key_parameter_redacted(self):
        params = self._sanitize_params({"api_key": "plain-secret", "temperature": 0})
        assert params == {"api_key": "***REDACTED***", "temperature": 0}

    def test_header_names_redacted_case_and_dash_insensitive(self):
        params = self._sanitize_params(
            {"extra_headers": {"X-Api-Key": "abc", "Authorization": "Basic xyz", "X-Trace": "keep"}}
        )
        assert params == {
            "extra_headers": {"X-Api-Key": "***REDACTED***", "Authorization": "***REDACTED***", "X-Trace": "keep"}
        }

    def test_non_string_secret_redacted_and_empty_kept(self):
        params = self._sanitize_params({"password": 12345, "api_key": None, "client_secret": ""})
        assert params == {"password": "***REDACTED***", "api_key": None, "client_secret": ""}

    def test_tool_schema_property_named_like_a_credential_is_kept(self):
        schema = {"type": "object", "properties": {"password": {"type": "string"}, "api_key": {"type": "string"}}}
        req = NormalizedRequest(messages=[], model="m", tools=[{"name": "login", "parameters": schema}])
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert out[0][0].tools == [{"name": "login", "parameters": schema}]

    def test_token_counts_not_redacted(self):
        params = self._sanitize_params({"max_tokens": 100, "max_output_tokens": 5, "token": "t"})
        assert params == {"max_tokens": 100, "max_output_tokens": 5, "token": "***REDACTED***"}

    def test_key_redaction_applies_with_custom_patterns(self):
        req = NormalizedRequest(messages=[], model="m", parameters={"api_key": "s"})
        custom = [SanitizePattern(name="x", pattern="never-matches", replacement="")]
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))], patterns=custom)
        assert out[0][0].parameters == {"api_key": "***REDACTED***"}


class TestNewKeyFormats:
    @pytest.mark.parametrize(
        "secret, marker",
        [
            ("ASIA" + "A" * 16, "ASIA***REDACTED***"),
            ("ABSK" + "QmVkcm9ja0FQSUtleS1leGFtcGxl", "ABSK***REDACTED***"),
            ("AIza" + "S" * 35, "AIza***REDACTED***"),
            ("ghp_" + "a" * 36, "gh***REDACTED***"),
            ("github_pat_" + "b" * 30, "gh***REDACTED***"),
        ],
    )
    def test_redacted_in_message_content(self, secret, marker):
        req = NormalizedRequest(messages=[{"role": "user", "content": f"key {secret} end"}], model="m")
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        content = out[0][0].messages[0]["content"]
        assert secret not in content and marker in content


class TestAdapterParametersEndToEnd:
    """Credentials passed through provider kwargs do not reach the cassette."""

    def test_openai_extra_headers(self):
        pytest.importorskip("openai")
        from agentverify.cassette.adapters.openai import OpenAIAdapter

        req = OpenAIAdapter().normalize_request(
            {"model": "gpt", "messages": [], "extra_headers": {"x-api-key": "plainsecretvalue"}}
        )
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert "plainsecretvalue" not in repr(out[0][0])


class TestParameterShapes:
    """Key redaction reaches credentials inside any container shape an adapter passes through."""

    def _params(self, parameters, patterns=None):
        req = NormalizedRequest(messages=[], model="m", parameters=parameters)
        return sanitize_interactions([(req, NormalizedResponse(content="x"))], patterns=patterns)[0][0].parameters

    def test_non_scalar_values_under_credential_keys(self):
        params = self._params({"extra_headers": {"Authorization": ["Basic dXNlcjpwYXNz"]}, "api_key": b"secret"})
        assert params == {"extra_headers": {"Authorization": "***REDACTED***"}, "api_key": "***REDACTED***"}

    def test_camel_case_and_header_style_keys(self):
        params = self._params({"accessToken": "t", "X-Amz-Security-Token": "s", "vertex_credentials": "{...}"})
        assert set(params.values()) == {"***REDACTED***"}

    def test_mapping_and_tuple_containers(self):
        from types import MappingProxyType

        params = self._params({"headers": MappingProxyType({"x-api-key": "k", "accept": "json"}), "stop": ("a", "b")})
        assert params == {"headers": {"x-api-key": "***REDACTED***", "accept": "json"}, "stop": ["a", "b"]}

    def test_pydantic_object_is_dumped_and_redacted(self):
        pydantic = pytest.importorskip("pydantic")

        class HttpOptions(pydantic.BaseModel):
            headers: dict

        params = self._params({"http_options": HttpOptions(headers={"x-goog-api-key": "k"})})
        assert params == {"http_options": {"headers": {"x-goog-api-key": "***REDACTED***"}}}

    def test_list_of_dicts(self):
        params = self._params({"items": [{"password": "p"}, {"name": "n"}]})
        assert params == {"items": [{"password": "***REDACTED***"}, {"name": "n"}]}

    def test_key_redaction_runs_with_empty_pattern_list(self):
        assert self._params({"api_key": "s"}, patterns=[]) == {"api_key": "***REDACTED***"}

    def test_messages_are_not_key_redacted(self):
        req = NormalizedRequest(messages=[{"role": "tool", "content": {"password": True}}], model="m")
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert out[0][0].messages == [{"role": "tool", "content": {"password": True}}]


class TestPatternBoundaries:
    def test_absk_inside_base64_is_not_redacted(self):
        data = "data:image/png;base64,AAAAABSKQmVkcm9ja0FQSUtleS1leGFtcGxlabc=="
        req = NormalizedRequest(messages=[{"role": "user", "content": data}], model="m")
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert out[0][0].messages[0]["content"] == data

    def test_bedrock_short_term_key(self):
        secret = "bedrock-api-key-YmVkcm9jay5hbWF6b25hd3MuY29tLz9BY3Rpb249Q2FsbFdpdGhCZWFyZXJUb2tlbg=="
        req = NormalizedRequest(messages=[{"role": "user", "content": secret}], model="m")
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert out[0][0].messages[0]["content"] == "bedrock-api-key-***REDACTED***"


class TestAdapterCredentialPaths:
    """Each adapter's credential-carrying kwargs end up redacted in the saved request."""

    def _saved(self, req):
        return repr(sanitize_interactions([(req, NormalizedResponse(content="x"))])[0][0])

    def test_litellm_api_key_and_session_token(self):
        pytest.importorskip("litellm")
        from agentverify.cassette.adapters.litellm import LiteLLMAdapter

        req = LiteLLMAdapter().normalize_request(
            {"model": "gemini/g", "messages": [], "api_key": "plain-key-1", "aws_session_token": "plain-tok-2"}
        )
        saved = self._saved(req)
        assert "plain-key-1" not in saved and "plain-tok-2" not in saved

    def test_anthropic_extra_headers(self):
        pytest.importorskip("anthropic")
        from agentverify.cassette.adapters.anthropic import AnthropicAdapter

        req = AnthropicAdapter().normalize_request(
            {"model": "claude", "messages": [], "max_tokens": 10, "extra_headers": {"x-api-key": "plain-key-3"}}
        )
        assert "plain-key-3" not in self._saved(req)

    def test_gemini_http_options_object(self):
        genai_types = pytest.importorskip("google.genai.types")
        from agentverify.cassette.adapters.gemini import GeminiAdapter

        config = {"http_options": genai_types.HttpOptions(headers={"x-goog-api-key": "plain-key-4"})}
        req = GeminiAdapter().normalize_request({"model": "gemini", "contents": "hi", "config": config})
        assert "plain-key-4" not in self._saved(req)


class TestCredentialKeyRules:
    @pytest.mark.parametrize(
        "key",
        ["authorization_token", "azure_password", "Ocp-Apim-Subscription-Key", "x-auth-token", "api_token", "client_secret", "aws_session_token", "Cookie"],
    )
    def test_suffix_and_exact_keys(self, key):
        req = NormalizedRequest(messages=[], model="m", parameters={key: "plain"})
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert out[0][0].parameters == {key: "***REDACTED***"}

    def test_header_pairs(self):
        req = NormalizedRequest(
            messages=[], model="m", parameters={"extra_headers": [("x-api-key", "plainhdr"), ("accept", "json")]}
        )
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert out[0][0].parameters == {"extra_headers": [["x-api-key", "***REDACTED***"], ["accept", "json"]]}

    def test_mcp_servers_authorization_token(self):
        servers = [{"type": "url", "url": "https://mcp.example", "authorization_token": "opaque"}]
        req = NormalizedRequest(messages=[], model="m", parameters={"mcp_servers": servers})
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert out[0][0].parameters["mcp_servers"][0]["authorization_token"] == "***REDACTED***"

    def test_array_like_value_does_not_break_saving(self):
        class ArrayLike:
            def __eq__(self, other):
                raise ValueError("ambiguous")

            __hash__ = object.__hash__

        req = NormalizedRequest(messages=[], model="m", parameters={"api_key": ArrayLike()})
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert out[0][0].parameters == {"api_key": "***REDACTED***"}


class TestObjectUnwrapping:
    def _params(self, parameters):
        req = NormalizedRequest(messages=[], model="m", parameters=parameters)
        return sanitize_interactions([(req, NormalizedResponse(content="x"))])[0][0].parameters

    def test_dataclass(self):
        import dataclasses

        @dataclasses.dataclass
        class Opts:
            headers: dict

        assert self._params({"o": Opts(headers={"x-api-key": "k"})}) == {"o": {"headers": {"x-api-key": "***REDACTED***"}}}

    def test_pydantic_v1_style_dict_method(self):
        class Legacy:
            def dict(self):
                return {"api_key": "k"}

        assert self._params({"o": Legacy()}) == {"o": {"api_key": "***REDACTED***"}}

    def test_class_value_is_left_alone(self):
        pydantic = pytest.importorskip("pydantic")

        class Answer(pydantic.BaseModel):
            text: str

        assert self._params({"response_format": Answer}) == {"response_format": Answer}

    def test_failing_or_non_mapping_dump_is_left_alone(self):
        class Broken:
            def model_dump(self):
                raise TypeError("nope")

            def dict(self):
                return ["not", "a", "mapping"]

        obj = Broken()
        assert self._params({"o": obj}) == {"o": obj}

    def test_sdk_message_object_in_messages_is_pattern_redacted(self):
        pydantic = pytest.importorskip("pydantic")

        class Message(pydantic.BaseModel):
            role: str
            content: str

        secret = "sk-" + "a" * 30
        req = NormalizedRequest(messages=[Message(role="assistant", content=f"key {secret}")], model="m")
        out = sanitize_interactions([(req, NormalizedResponse(content="x"))])
        assert out[0][0].messages == [{"role": "assistant", "content": "key sk-***REDACTED***"}]
