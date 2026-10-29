"""How old a cassette may get before replaying it fails: ``max_age`` and ``on_expired``."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Union

MaxAge = Union[timedelta, str, None]

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([dh])\s*$")
ON_EXPIRED = ("error", "warn")


def parse_max_age(value: MaxAge) -> timedelta | None:
    """``"30d"`` / ``"12h"`` / a ``timedelta`` / ``None`` (no limit)."""
    if value is None or isinstance(value, timedelta):
        return value
    if isinstance(value, str):
        match = _DURATION.match(value)
        if match:
            amount, unit = float(match.group(1)), match.group(2)
            return timedelta(days=amount) if unit == "d" else timedelta(hours=amount)
    raise ValueError(f"max_age must be a duration such as '30d' or '12h', or a timedelta; got {value!r}")


def check_on_expired(value: str) -> str:
    if value not in ON_EXPIRED:
        raise ValueError(f"on_expired must be one of {', '.join(map(repr, ON_EXPIRED))}; got {value!r}")
    return value


def recorded_at(metadata: dict[str, Any]) -> datetime | None:
    """When the cassette was recorded, from its ``metadata.recorded_at``. A value without a time zone is UTC."""
    value = metadata.get("recorded_at")
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


def expiry(metadata: dict[str, Any], max_age: timedelta, now: datetime | None = None) -> str | None:
    """Why the cassette is past *max_age*, or ``None`` while it is not. A cassette without a recording time is past any limit."""
    stamp = recorded_at(metadata)
    if stamp is None:
        return "expired: no recorded_at in its metadata"
    now = now or datetime.now(timezone.utc)
    age = now - stamp
    if age <= max_age:
        return None
    due = (stamp + max_age).date().isoformat()
    return f"expired: recorded {stamp.date().isoformat()} ({age.days} days ago), due for re-recording by {due}"
