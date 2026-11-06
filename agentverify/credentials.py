"""``assert_no_credential_urls``: no URL the model was sent carries a key, token, or signature."""

from __future__ import annotations

import re
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlsplit

from agentverify.cassette.texts import Source, iter_request_texts
from agentverify.errors import CredentialURLError

#: Query names that make a URL a credential (compared case-insensitively). ``key`` gives false alarms on APIs that use it for an ordinary value: pass ``ignore=["key"]``.
CREDENTIAL_QUERY_NAMES = frozenset(
    {
        "token", "access_token", "id_token", "refresh_token", "security_token", "session_token", "sessionid",
        "api_key", "apikey", "key", "client_secret", "secret", "password", "passwd", "auth", "authorization", "jwt",
        "sig", "signature", "x-amz-signature", "x-amz-security-token", "x-amz-credential", "x-goog-signature", "x-goog-credential",
    }
)  # fmt: skip
_URL = re.compile(r"https?://[^\s\"'<>\\]+", re.IGNORECASE)
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


def _reasons(url: str, names: frozenset[str], extra: list[re.Pattern[str]]) -> list[str]:
    parts = urlsplit(url)
    reasons = sorted({name for name, _ in parse_qsl(parts.query, keep_blank_values=True) if name.lower() in names})
    if _JWT.search(parts.path) or _JWT.search(parts.query):
        reasons.append("JWT")
    reasons += [f"pattern {pattern.pattern!r}" for pattern in extra if pattern.search(url)]
    return reasons


def assert_no_credential_urls(source: Source, extra_patterns: Optional[Iterable[str]] = None, ignore: Optional[Iterable[str]] = None) -> None:
    """Verify that no URL in the requests the model was sent carries a credential.

    A URL carries one when a query name is in :data:`CREDENTIAL_QUERY_NAMES` (presigned download links, ``?access_token=``, ``X-Amz-Signature``, Azure SAS ``sig``), or a JWT sits in its path or query. Every text of the requests is searched, as :func:`iter_request_texts` walks them: tool results, tool definitions, and the system prompt included.

    Args:
        source: A cassette path, an ``LLMCassetteRecorder`` or ``MockLLM`` after the run, or one ``NormalizedRequest``.
        extra_patterns: Regular expressions; a URL any of them matches also fails (for example ``r"[?&]code="`` for OAuth codes).
        ignore: Query names to drop from the defaults (for example ``["key"]``).

    Raises:
        CredentialURLError: Listing where each URL was, its host, and why, never the values.
    """
    names = CREDENTIAL_QUERY_NAMES - {name.lower() for name in ignore or []}
    extra = [re.compile(pattern) for pattern in extra_patterns or []]
    found = []
    for where, text in iter_request_texts(source):
        for url in _URL.findall(text):
            reasons = _reasons(url, names, extra)
            if reasons:
                found.append({"where": where, "host": urlsplit(url).hostname or "?", "reasons": reasons})
    if found:
        raise CredentialURLError(violations=found)
