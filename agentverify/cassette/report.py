"""``pytest --cassette-report``: the cassettes next to the collected tests, when and with which model each was recorded, and whether it is past ``max_age``."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentverify.cassette.expiry import expiry, parse_max_age, recorded_at
from agentverify.cassette.io import load_document

_SUFFIXES = (".yaml", ".yml", ".json")


def cassette_files(session: Any) -> list[Path]:
    """The files in the ``cassettes/`` directory next to each collected test file, where the ``cassette`` fixture keeps them."""
    directories = dict.fromkeys(Path(str(item.path)).parent / "cassettes" for item in session.items)
    return sorted(p for d in directories if d.is_dir() for p in d.iterdir() if p.suffix in _SUFFIXES)


def report(session: Any) -> None:
    config = session.config
    from agentverify.fixtures import _ini

    max_age = parse_max_age(config.getoption("--cassette-max-age", default=None) or _ini(config, "agentverify_cassette_max_age"))
    writer = config.pluginmanager.get_plugin("terminalreporter")
    writer.section("agentverify: cassettes")
    files = cassette_files(session)
    if not files:
        hint = " Under pytest-xdist the report needs a run without workers (-p no:xdist or -n 0)." if config.pluginmanager.hasplugin("dsession") else ""
        writer.write_line("No cassettes next to the collected tests." + hint)
        return
    now = datetime.now(timezone.utc)
    rows = []
    for path in files:
        try:
            metadata = load_document(path).get("metadata") or {}
            if not isinstance(metadata, dict):
                raise TypeError("metadata is not a mapping")
        except Exception as error:  # a file that is not a cassette
            rows.append((path, "-", "-", "-", f"unreadable: {type(error).__name__}"))
            continue
        stamp = recorded_at(metadata)
        recorded = f"{stamp.date().isoformat()} ({(now - stamp).days}d ago)" if stamp else "-"
        status = "-" if max_age is None else (expiry(metadata, max_age, now) or "ok")
        rows.append((path, recorded, str(metadata.get("provider", "-")), str(metadata.get("model", "-")), status))
    root = config.rootpath
    shown = [(_relative(p, root), *rest) for p, *rest in rows]
    header = ("cassette", "recorded", "provider", "model", "status" if max_age is None else f"status (max age {_format(max_age)})")
    widths = [max(len(str(row[i])) for row in [header, *shown]) for i in range(4)]
    for row in [header, *shown]:
        writer.write_line("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row[:4])) + "  " + str(row[4]))


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _format(max_age: Any) -> str:
    days = max_age.total_seconds() / 86400
    return f"{days:g}d" if days >= 1 else f"{max_age.total_seconds() / 3600:g}h"
