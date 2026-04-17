"""Cassette file I/O for recording and replaying LLM interactions.

Supports both YAML and JSON formats, auto-detected by file extension:
- ``.yaml`` / ``.yml`` → YAML (requires PyYAML)
- ``.json`` → JSON (stdlib only)

YAML is the default and recommended format for human readability and
git diff friendliness.  JSON is useful when you want to avoid the
PyYAML dependency or need machine-readable output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentverify.cassette.adapters.base import NormalizedRequest, NormalizedResponse
from agentverify.models import TokenUsage

from importlib.metadata import version as _pkg_version

#: Library version stamped into cassette metadata.
_AGENTVERIFY_VERSION = _pkg_version("agentverify")

_YAML_EXTENSIONS = {".yaml", ".yml"}
_JSON_EXTENSIONS = {".json"}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_request(request: NormalizedRequest) -> dict[str, Any]:
    """Convert a *NormalizedRequest* to a plain dict."""
    data: dict[str, Any] = {
        "messages": request.messages,
        "model": request.model,
    }
    if request.tools is not None:
        data["tools"] = request.tools
    if request.parameters:
        data["parameters"] = request.parameters
    return data


def _serialize_response(response: NormalizedResponse) -> dict[str, Any]:
    """Convert a *NormalizedResponse* to a plain dict."""
    data: dict[str, Any] = {
        "content": response.content,
        "tool_calls": response.tool_calls,
    }
    if response.token_usage is not None:
        data["token_usage"] = {
            "input_tokens": response.token_usage.input_tokens,
            "output_tokens": response.token_usage.output_tokens,
        }
    else:
        data["token_usage"] = None
    return data


def _deserialize_request(data: dict[str, Any]) -> NormalizedRequest:
    """Reconstruct a *NormalizedRequest* from a loaded dict."""
    return NormalizedRequest(
        messages=data.get("messages", []),
        model=data.get("model", ""),
        tools=data.get("tools"),
        parameters=data.get("parameters", {}),
    )


def _deserialize_response(data: dict[str, Any]) -> NormalizedResponse:
    """Reconstruct a *NormalizedResponse* from a loaded dict."""
    token_usage: TokenUsage | None = None
    tu = data.get("token_usage")
    if tu is not None:
        token_usage = TokenUsage(
            input_tokens=tu.get("input_tokens", 0),
            output_tokens=tu.get("output_tokens", 0),
        )
    return NormalizedResponse(
        content=data.get("content"),
        tool_calls=data.get("tool_calls"),
        token_usage=token_usage,
    )


def _parse_interactions(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[NormalizedRequest, NormalizedResponse]]]:
    """Extract metadata and interactions from a parsed cassette document."""
    metadata: dict[str, Any] = raw.get("metadata", {})
    interactions: list[tuple[NormalizedRequest, NormalizedResponse]] = []
    for entry in raw.get("interactions", []):
        req = _deserialize_request(entry.get("request", {}))
        resp = _deserialize_response(entry.get("response", {}))
        interactions.append((req, resp))
    return metadata, interactions


def _build_document(
    interactions: list[tuple[NormalizedRequest, NormalizedResponse]],
    provider: str,
    model: str,
) -> dict[str, Any]:
    """Build the cassette document dict ready for serialization."""
    if not model and interactions:
        model = interactions[0][0].model

    metadata: dict[str, Any] = {
        "agentverify_version": _AGENTVERIFY_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
    }
    serialized: list[dict[str, Any]] = [
        {
            "request": _serialize_request(req),
            "response": _serialize_response(resp),
        }
        for req, resp in interactions
    ]
    return {"metadata": metadata, "interactions": serialized}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_cassette(
    path: Path | str,
) -> tuple[dict[str, Any], list[tuple[NormalizedRequest, NormalizedResponse]]]:
    """Load a cassette file (YAML or JSON, auto-detected by extension).

    Args:
        path: Path to the cassette file.

    Returns:
        A tuple of ``(metadata, interactions)``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    with path.open("r", encoding="utf-8") as fh:
        if suffix in _YAML_EXTENSIONS:
            import yaml

            raw = yaml.safe_load(fh) or {}
        elif suffix in _JSON_EXTENSIONS:
            raw = json.load(fh)
        else:
            raise ValueError(
                f"Unsupported cassette file extension '{suffix}'. "
                f"Use .yaml, .yml, or .json."
            )

    return _parse_interactions(raw)


def save_cassette(
    path: Path | str,
    interactions: list[tuple[NormalizedRequest, NormalizedResponse]],
    provider: str = "",
    model: str = "",
) -> None:
    """Save interactions to a cassette file (YAML or JSON, auto-detected by extension).

    Args:
        path: Destination path. Extension determines format.
        interactions: List of ``(NormalizedRequest, NormalizedResponse)`` pairs.
        provider: Provider name for metadata (e.g. ``"openai"``).
        model: Model name for metadata (e.g. ``"gpt-4.1"``).

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    document = _build_document(interactions, provider, model)

    with path.open("w", encoding="utf-8") as fh:
        if suffix in _YAML_EXTENSIONS:
            import yaml

            yaml.dump(
                document,
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        elif suffix in _JSON_EXTENSIONS:
            json.dump(document, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        else:
            raise ValueError(
                f"Unsupported cassette file extension '{suffix}'. "
                f"Use .yaml, .yml, or .json."
            )
