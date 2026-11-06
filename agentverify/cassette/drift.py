"""Drift: how a replayed request differs from its recording, beyond what request matching already compares.

Request matching fails on a changed model, tool names and schemas, system prompt, or user text. Drift covers the rest: tool results, the number and roles of the messages, and the provider's other parameters. It is off by default, since a tool that runs for real may return something new on every run.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

from agentverify.cassette.adapters.base import NormalizedRequest
from agentverify.cassette.texts import _NULLS, iter_request_texts

DRIFT = ("off", "warn", "fail")
_ANNOTATION = re.compile(r"\([^()]*\)")


def check_drift(value: str) -> str:
    if value not in DRIFT:
        raise ValueError(f"drift must be one of {', '.join(map(repr, DRIFT))}; got {value!r}")
    return value


def compile_ignore(patterns: Iterable[str] | None) -> list[re.Pattern[str]]:
    """``drift_ignore`` globs over ``where``: ``*`` stands for anything, everything else is literal (brackets included). A glob may leave out the ``(role)`` and ``(tool)`` parts: ``request[*].message[*].tool_result[*].now``."""
    return [re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$") for pattern in patterns or []]


def _compared(request: NormalizedRequest, index: int, normalize: Callable[[str], str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    token = _NULLS.set(True)
    try:
        for where, text in iter_request_texts(request):
            where = f"request[{index}]" + where[len("request[0]") :]
            if ".tool_result[" in where or where.startswith(f"request[{index}].parameters."):
                texts[where] = normalize(text)
    finally:
        _NULLS.reset(token)
    return texts


def _roles(request: NormalizedRequest) -> list[str]:
    return [str(m.get("role") or m.get("type") or "message") if isinstance(m, dict) else "message" for m in request.messages or []]


def drift(recorded: NormalizedRequest, actual: NormalizedRequest, index: int, ignore: list[re.Pattern[str]], normalize: Callable[[str], str] = lambda s: s) -> list[str]:
    """The places where *actual* differs from *recorded*, as ``where`` strings (no values)."""
    places: list[str] = []
    recorded_roles, actual_roles = _roles(recorded), _roles(actual)
    if recorded_roles != actual_roles:
        places.append(f"request[{index}].messages (roles {','.join(recorded_roles)} → {','.join(actual_roles)})")
    before, after = _compared(recorded, index, normalize), _compared(actual, index, normalize)
    for where in sorted(set(before) | set(after)):
        plain = _ANNOTATION.sub("", where)  # globs may leave out the (role) and (tool) parts
        if before.get(where) != after.get(where) and not any(p.match(where) or p.match(plain) for p in ignore):
            places.append(where)
    return places


def describe(places: list[str], limit: int = 5) -> str:
    shown = ", ".join(places[:limit])
    return shown + (f", and {len(places) - limit} more" if len(places) > limit else "")
