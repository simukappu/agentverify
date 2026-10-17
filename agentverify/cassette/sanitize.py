"""Cassette sanitization — mask sensitive data before saving.

Provides pattern-based redaction of API keys, tokens, and other sensitive values in cassette request/response data.

Usage::

    from agentverify.cassette.sanitize import sanitize_interactions, DEFAULT_PATTERNS

    sanitized = sanitize_interactions(interactions, patterns=DEFAULT_PATTERNS)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
import dataclasses
from collections.abc import Mapping
from typing import Any

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse


@dataclass(frozen=True)
class SanitizePattern:
    """A pattern to match and replace sensitive data."""

    name: str
    pattern: str  # regex pattern
    replacement: str  # replacement string


#: Built-in patterns for common API keys and tokens.
#: More specific patterns must come before general ones (e.g., sk-ant- before sk-).
DEFAULT_PATTERNS: list[SanitizePattern] = [
    SanitizePattern(
        name="anthropic_api_key",
        pattern=r"sk-ant-[A-Za-z0-9_-]{20,}",
        replacement="sk-ant-***REDACTED***",
    ),
    SanitizePattern(
        name="openai_api_key",
        pattern=r"sk-[A-Za-z0-9_-]{20,}",
        replacement="sk-***REDACTED***",
    ),
    SanitizePattern(
        name="aws_access_key",
        pattern=r"AKIA[A-Z0-9]{16}",
        replacement="AKIA***REDACTED***",
    ),
    SanitizePattern(
        name="aws_temporary_access_key",
        pattern=r"ASIA[A-Z0-9]{16}",
        replacement="ASIA***REDACTED***",
    ),
    # Bedrock long-term and short-term API keys. The lookbehind keeps most matches
    # from starting inside base64 data such as an inline image.
    SanitizePattern(
        name="bedrock_api_key",
        pattern=r"(?<![A-Za-z0-9])ABSK[A-Za-z0-9+/=]{20,}",
        replacement="ABSK***REDACTED***",
    ),
    SanitizePattern(
        name="bedrock_short_term_api_key",
        pattern=r"bedrock-api-key-[A-Za-z0-9+/=_-]{20,}",
        replacement="bedrock-api-key-***REDACTED***",
    ),
    SanitizePattern(
        name="google_api_key",
        pattern=r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}",
        replacement="AIza***REDACTED***",
    ),
    SanitizePattern(
        name="github_token",
        pattern=r"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})",
        replacement="gh***REDACTED***",
    ),
    SanitizePattern(
        name="bearer_token",
        pattern=r"Bearer\s+[A-Za-z0-9_.\-/+=]{20,}",
        replacement="Bearer ***REDACTED***",
    ),
]


#: Request parameter keys whose values are credentials whatever they look like, e.g. ``api_key=``
#: passed through to a provider call or an ``extra_headers`` entry. Keys are compared after
#: lower-casing and dropping ``-`` and ``_``, so ``x-api-key``, ``api_key``, and ``apiKey`` match.
SENSITIVE_KEYS: frozenset[str] = frozenset({"authorization", "cookie", "awsaccesskeyid", "vertexcredentials"})

#: A normalized key ending in one of these is also a credential (``azure_password``,
#: ``client_secret``, ``Ocp-Apim-Subscription-Key``, ``aws_session_token``, ...).
#: Keys ending in ``tokens`` are counts such as ``max_tokens`` and are not affected.
SENSITIVE_KEY_SUFFIXES: tuple[str, ...] = (
    "apikey",
    "password",
    "secret",
    "secretkey",
    "privatekey",
    "subscriptionkey",
    "credentials",
    "token",
)

_REDACTED = "***REDACTED***"


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    norm = key.lower().replace("-", "").replace("_", "")
    return norm in SENSITIVE_KEYS or norm.endswith(SENSITIVE_KEY_SUFFIXES)


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value)


def _as_plain(value: Any) -> Any:
    """Return a dict or list view of a mapping, sequence, dataclass, or model instance, or ``value`` itself."""
    if isinstance(value, (dict, list, str)):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, type):
        return value
    if dataclasses.is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    for method in ("model_dump", "dict"):  # pydantic v2, then v1
        dump = getattr(value, method, None)
        if callable(dump):
            try:
                dumped = dump()
            except Exception:
                continue
            if isinstance(dumped, Mapping):
                return dict(dumped)
    return value


def _redact_value(
    value: Any, compiled: list[tuple[re.Pattern[str], str]], redact_keys: bool = False
) -> Any:
    """Recursively redact sensitive patterns in a value.

    Mappings, tuples, dataclasses, and pydantic models are turned into dicts and lists first, so
    nothing reaches the cassette unscanned. With *redact_keys*, every value under a credential key
    (:data:`SENSITIVE_KEYS`, :data:`SENSITIVE_KEY_SUFFIXES`) is replaced too, including header pairs
    given as ``[name, value]`` sequences.
    """
    value = _as_plain(value)
    if isinstance(value, str):
        for regex, replacement in compiled:
            value = regex.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {
            k: _REDACTED
            if redact_keys and _is_sensitive_key(k) and not _is_empty(v)
            else _redact_value(v, compiled, redact_keys)
            for k, v in value.items()
        }
    if isinstance(value, list):
        if redact_keys and len(value) == 2 and _is_sensitive_key(value[0]) and not _is_empty(value[1]):
            return [value[0], _REDACTED]
        return [_redact_value(item, compiled, redact_keys) for item in value]
    return value


def _sanitize_request(
    req: NormalizedRequest,
    compiled: list[tuple[re.Pattern[str], str]],
) -> NormalizedRequest:
    """Return a new NormalizedRequest with sensitive data redacted."""
    return NormalizedRequest(
        messages=_redact_value(req.messages, compiled),
        model=req.model,
        tools=_redact_value(req.tools, compiled) if req.tools is not None else None,
        # Credentials arrive through call arguments and headers, so key-based
        # redaction covers parameters only; messages and tool schemas keep their
        # data (a schema may legitimately name a property "password").
        parameters=_redact_value(req.parameters, compiled, redact_keys=True),
    )


def _sanitize_response(
    resp: NormalizedResponse,
    compiled: list[tuple[re.Pattern[str], str]],
) -> NormalizedResponse:
    """Return a new NormalizedResponse with sensitive data redacted."""
    return NormalizedResponse(
        content=_redact_value(resp.content, compiled) if resp.content is not None else None,
        tool_calls=_redact_value(resp.tool_calls, compiled) if resp.tool_calls is not None else None,
        token_usage=resp.token_usage,
        raw_metadata=_redact_value(resp.raw_metadata, compiled),
    )


def sanitize_interactions(
    interactions: list[tuple[NormalizedRequest, NormalizedResponse]],
    patterns: list[SanitizePattern] | None = None,
) -> list[tuple[NormalizedRequest, NormalizedResponse]]:
    """Sanitize all interactions by redacting sensitive patterns.

    Args:
        interactions: List of request/response pairs.
        patterns: Patterns to redact. Defaults to :data:`DEFAULT_PATTERNS`. Values under credential keys in request parameters are redacted regardless, even with an empty list.

    Returns:
        A new list of sanitized request/response pairs.
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    # Key-based redaction of parameters runs even with an empty pattern list;
    # only sanitize=False on the recorder turns sanitization off entirely.
    compiled = [(re.compile(p.pattern), p.replacement) for p in patterns]

    return [
        (_sanitize_request(req, compiled), _sanitize_response(resp, compiled))
        for req, resp in interactions
    ]
