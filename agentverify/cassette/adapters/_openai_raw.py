"""Stand-ins for the OpenAI SDK's raw responses (``with_raw_response`` / ``with_streaming_response``).

Both wrappers call ``create`` with the ``X-Stainless-Raw-Response`` header and expect a response object back whose ``parse()`` gives the parsed result: the ``ChatCompletion`` / ``Response``, or the ``Stream`` for a streamed call. ``with_streaming_response`` (header value ``"stream"``) on the async client awaits ``parse()`` and ``close()``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"


def raw_mode(kwargs: dict[str, Any]) -> str | None:
    """The raw-response header value of a ``create`` call, if any.

    The value has varied across SDK versions (``"raw"`` / ``"true"`` / ``"stream"``), so any non-empty value counts.
    """
    extra = kwargs.get("extra_headers") or {}
    value = extra.get(RAW_RESPONSE_HEADER) if isinstance(extra, dict) else None
    return value or None


class _RawResponseWrapper:
    """Minimal stand-in for ``openai._legacy_response.LegacyAPIResponse`` (and the sync ``APIResponse``)."""

    def __init__(self, parsed: Any) -> None:
        self._parsed = parsed
        self.headers: dict[str, str] = {}
        self.http_response = None
        self.request_id = None

    def parse(self) -> Any:  # noqa: D401 — mimics openai SDK
        return self._parsed

    def close(self) -> None:
        pass


class _AsyncRawResponseWrapper(_RawResponseWrapper):
    """Stand-in for ``openai._response.AsyncAPIResponse``, whose ``parse()`` and ``close()`` are awaited."""

    async def parse(self) -> Any:  # type: ignore[override]
        return self._parsed

    async def close(self) -> None:  # type: ignore[override]
        pass


def wrap_raw(parsed: Any, kwargs: dict[str, Any], is_async: bool) -> Any:
    """Return a replayed result in the shape the caller asked for: as it is, or in a raw-response stand-in."""
    mode = raw_mode(kwargs)
    if mode is None:
        return parsed
    if mode == "stream" and is_async:
        return _AsyncRawResponseWrapper(parsed)
    return _RawResponseWrapper(parsed)


class TappedRawResponse:
    """A live raw response whose parsed stream is tapped for recording when the caller parses it."""

    def __init__(self, response: Any, tap: Callable[[Any], Any]) -> None:
        self._response = response
        self._tap = tap
        self._tapped: Any = _UNSET

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        if self._tapped is not _UNSET:
            return self._cached()
        parsed = self._response.parse(*args, **kwargs)
        if inspect.isawaitable(parsed):

            async def tapped() -> Any:
                self._tapped = self._tap(await parsed)
                return self._tapped

            return tapped()
        self._tapped = self._tap(parsed)
        return self._tapped

    def _cached(self) -> Any:
        """The tapped stream again, awaitable when the response's own ``parse()`` is."""
        if inspect.iscoroutinefunction(self._response.parse):

            async def cached() -> Any:
                return self._tapped

            return cached()
        return self._tapped


_UNSET: Any = object()


def tap_raw_or_stream(response: Any, tap: Callable[[Any], Any]) -> Any:
    """Tap a live stream, or the stream inside a live raw response."""
    if raw_response(response):
        return TappedRawResponse(response, tap)
    return tap(response)


def raw_response(response: Any) -> bool:
    return hasattr(response, "parse") and not hasattr(response, "__iter__") and not hasattr(response, "__aiter__")
