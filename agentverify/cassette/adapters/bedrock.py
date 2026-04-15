"""Amazon Bedrock Converse API adapter for LLM cassette recording and replay.

Monkey-patches ``botocore.client.BaseClient._make_api_call`` to intercept
``Converse`` operations for recording and replaying via the cassette recorder.

boto3 is an **optional** dependency.  Import this module only when the
boto3 extra is installed (``pip install agentverify[bedrock]``).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import patch

from agentverify.cassette.adapters.base import (
    LLMProviderAdapter,
    NormalizedRequest,
    NormalizedResponse,
)
from agentverify.models import TokenUsage

if TYPE_CHECKING:
    from agentverify.cassette.recorder import LLMCassetteRecorder

try:
    import botocore.client  # noqa: F401 – presence check

    _botocore_available = True
except ImportError:  # pragma: no cover
    _botocore_available = False

_PATCH_TARGET = "botocore.client.BaseClient._make_api_call"


def _ensure_boto3_installed() -> None:
    """Raise a clear error when the boto3/botocore package is missing."""
    if not _botocore_available:  # pragma: no cover
        raise ImportError(
            "The boto3 package is required to use BedrockAdapter. "
            "Install it with: pip install agentverify[bedrock]"
        )


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class BedrockAdapter(LLMProviderAdapter):
    """Adapter for the Amazon Bedrock Converse API (via boto3)."""

    @property
    def name(self) -> str:
        return "bedrock"

    # -- normalisation -------------------------------------------------------

    def normalize_request(self, raw_request: dict[str, Any]) -> NormalizedRequest:
        """Convert Bedrock ``converse()`` kwargs to *NormalizedRequest*.

        Bedrock messages use ``[{"role": "user", "content": [{"text": "..."}]}]``.
        We flatten single-text content blocks to plain strings for the
        normalised format.
        """
        raw_messages = raw_request.get("messages", [])
        messages: list[dict[str, Any]] = []
        for msg in raw_messages:
            role = msg.get("role", "")
            content_blocks = msg.get("content", [])
            # Flatten: if all blocks are text, join them into a single string
            if isinstance(content_blocks, list) and all(
                isinstance(b, dict) and "text" in b for b in content_blocks
            ):
                text_parts = [b["text"] for b in content_blocks]
                content = text_parts[0] if len(text_parts) == 1 else "\n".join(text_parts)
                messages.append({"role": role, "content": content})
            else:
                # Keep as-is for complex content (images, tool results, etc.)
                messages.append({"role": role, "content": content_blocks})

        model = raw_request.get("modelId", "")

        # Normalise tools from toolConfig
        tool_config = raw_request.get("toolConfig")
        tools: list[dict[str, Any]] | None = None
        if tool_config is not None:
            raw_tools = tool_config.get("tools", [])
            tools = []
            for t in raw_tools:
                spec = t.get("toolSpec", {})
                tools.append(
                    {
                        "name": spec.get("name", ""),
                        "parameters": spec.get("inputSchema", {}).get("json", {}),
                    }
                )

        # Everything else goes into parameters
        _reserved = {"messages", "modelId", "toolConfig"}
        parameters: dict[str, Any] = {}
        # inferenceConfig and other top-level keys
        for k, v in raw_request.items():
            if k not in _reserved:
                parameters[k] = v

        return NormalizedRequest(
            messages=messages,
            model=model,
            tools=tools,
            parameters=parameters,
        )

    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        """Convert a Bedrock Converse response dict to *NormalizedResponse*.

        Bedrock responses are plain dicts with structure:
        ``{"output": {"message": {"role": "assistant", "content": [...]}}, "usage": {...}}``
        """
        output = raw_response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        content: str | None = None
        tool_calls: list[dict[str, Any]] | None = None

        for block in content_blocks:
            if "text" in block:
                # Concatenate text blocks (usually just one)
                if content is None:
                    content = block["text"]
                else:
                    content += "\n" + block["text"]
            elif "toolUse" in block:
                if tool_calls is None:
                    tool_calls = []
                tu = block["toolUse"]
                tool_calls.append(
                    {
                        "name": tu.get("name", ""),
                        "arguments": json.dumps(tu.get("input", {})),
                    }
                )

        # Token usage
        token_usage: TokenUsage | None = None
        usage = raw_response.get("usage")
        if usage is not None:
            token_usage = TokenUsage(
                input_tokens=usage.get("inputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
            )

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
        )

    def denormalize_response(self, normalized: NormalizedResponse) -> dict[str, Any]:
        """Build a dict that mimics a Bedrock Converse response structure."""
        content_blocks: list[dict[str, Any]] = []

        if normalized.content is not None:
            content_blocks.append({"text": normalized.content})

        if normalized.tool_calls:
            for tc in normalized.tool_calls:
                arguments = tc["arguments"]
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, ValueError):
                        arguments = {}
                content_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": f"tooluse_{id(tc)}",
                            "name": tc["name"],
                            "input": arguments,
                        }
                    }
                )

        stop_reason = "tool_use" if normalized.tool_calls else "end_turn"

        result: dict[str, Any] = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": content_blocks,
                }
            },
            "stopReason": stop_reason,
        }

        if normalized.token_usage is not None:
            result["usage"] = {
                "inputTokens": normalized.token_usage.input_tokens,
                "outputTokens": normalized.token_usage.output_tokens,
                "totalTokens": normalized.token_usage.total_tokens,
            }
        else:
            result["usage"] = None

        return result

    # -- monkey-patching -----------------------------------------------------

    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:
        """Monkey-patch ``botocore.client.BaseClient._make_api_call``.

        Only intercepts the ``Converse`` operation; all other API calls
        pass through to the original implementation.

        In **record** mode the real API is called, the response is normalised,
        recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the
        cassette, denormalised, and returned without calling the real API.
        """
        _ensure_boto3_installed()

        adapter = self

        import botocore.client

        original_fn = botocore.client.BaseClient._make_api_call

        def _patched_make_api_call(self_client: Any, operation_name: str, api_params: dict[str, Any]) -> Any:
            """Intercept Converse calls; pass through everything else."""
            if operation_name != "Converse":
                return original_fn(self_client, operation_name, api_params)

            from agentverify.cassette.recorder import CassetteMode, OnMissingRequest

            norm_req = adapter.normalize_request(api_params)

            if recorder.mode in (CassetteMode.RECORD, CassetteMode.AUTO):
                response = original_fn(self_client, operation_name, api_params)
                norm_resp = adapter.normalize_response(response)
                recorder.record(norm_req, norm_resp)
                return response

            # REPLAY mode
            norm_resp = recorder.lookup(norm_req)
            if norm_resp is None:
                if recorder.on_missing == OnMissingRequest.FALLBACK:
                    response = original_fn(self_client, operation_name, api_params)
                    norm_resp = adapter.normalize_response(response)
                    recorder.record(norm_req, norm_resp)
                    return response
                from agentverify.errors import CassetteMissingRequestError

                raise CassetteMissingRequestError(
                    f"No matching cassette entry for request: model={norm_req.model}"
                )
            return adapter.denormalize_response(norm_resp)

        with patch(_PATCH_TARGET, new=_patched_make_api_call):
            yield
