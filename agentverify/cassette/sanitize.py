"""Cassette sanitization — mask sensitive data before saving.

Provides pattern-based redaction of API keys, tokens, and other
sensitive values in cassette request/response data.

Usage::

    from agentverify.cassette.sanitize import sanitize_interactions, DEFAULT_PATTERNS

    sanitized = sanitize_interactions(interactions, patterns=DEFAULT_PATTERNS)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
        name="bearer_token",
        pattern=r"Bearer\s+[A-Za-z0-9_.\-/+=]{20,}",
        replacement="Bearer ***REDACTED***",
    ),
]


def _redact_value(value: Any, compiled: list[tuple[re.Pattern[str], str]]) -> Any:
    """Recursively redact sensitive patterns in a value."""
    if isinstance(value, str):
        for regex, replacement in compiled:
            value = regex.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {k: _redact_value(v, compiled) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, compiled) for item in value]
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
        parameters=_redact_value(req.parameters, compiled),
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
        patterns: Patterns to redact. Defaults to :data:`DEFAULT_PATTERNS`.

    Returns:
        A new list of sanitized request/response pairs.
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    if not patterns:
        return interactions

    compiled = [(re.compile(p.pattern), p.replacement) for p in patterns]

    return [
        (_sanitize_request(req, compiled), _sanitize_response(resp, compiled))
        for req, resp in interactions
    ]
