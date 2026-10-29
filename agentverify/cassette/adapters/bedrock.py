"""Amazon Bedrock Converse API adapter for LLM cassette recording and replay.

Monkey-patches ``botocore.client.BaseClient._make_api_call`` to intercept ``Converse`` operations for recording and replaying via the cassette recorder.

boto3 is an **optional** dependency.  Import this module only when the boto3 extra is installed (``pip install agentverify[bedrock]``).
"""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import patch

from agentverify.cassette.adapters.base import (
    _guards_off,
    call_async,
    call_sync,
    fallback_call_id,
    patch_scope,
    check_unrecorded,
    unsupported_call_error,
    LLMProviderAdapter,
    NormalizedRequest,
    NormalizedResponse,
    StreamAccumulator,
    StreamingWire,
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

# Operations that ask a model to generate, other than the recorded Converse. Other
# operations on these services (CountTokens, ApplyGuardrail, Retrieve, ...) do not
# generate and pass through untouched. Not exhaustive: AWS adds operations.
_GENERATING_OPERATIONS: dict[str, frozenset[str]] = {
    "bedrock-runtime": frozenset({"InvokeModel", "InvokeModelWithResponseStream", "StartAsyncInvoke"}),
    "bedrock-agent-runtime": frozenset(
        {
            "InvokeAgent",
            "InvokeInlineAgent",
            "InvokeFlow",
            "RetrieveAndGenerate",
            "RetrieveAndGenerateStream",
            "GenerateQuery",
            "OptimizePrompt",
        }
    ),
    "bedrock-agentcore": frozenset({"InvokeAgentRuntime"}),
}


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


class ConverseStreamAccumulator(StreamAccumulator):
    """Collects ``ConverseStream`` events into one response."""

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._usage: TokenUsage | None = None

    def add(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        if "contentBlockStart" in event:
            start = event["contentBlockStart"]
            tool_use = (start.get("start") or {}).get("toolUse")
            if tool_use is not None:
                self._blocks[start.get("contentBlockIndex", len(self._blocks))] = {
                    "toolUse": {"toolUseId": tool_use.get("toolUseId"), "name": tool_use.get("name", "")},
                    "json": "",
                }
        elif "contentBlockDelta" in event:
            body = event["contentBlockDelta"]
            block = self._blocks.setdefault(body.get("contentBlockIndex", len(self._blocks)), {"text": ""})
            delta = body.get("delta") or {}
            if "text" in delta:
                block["text"] = block.get("text", "") + delta["text"]
            elif "toolUse" in delta:
                block["json"] = block.get("json", "") + (delta["toolUse"].get("input") or "")
        elif "metadata" in event:
            usage = event["metadata"].get("usage")
            if usage is not None:
                self._usage = TokenUsage(input_tokens=usage.get("inputTokens", 0), output_tokens=usage.get("outputTokens", 0))

    def result(self) -> NormalizedResponse:
        content: list[dict[str, Any]] = []
        for index in sorted(self._blocks):
            block = self._blocks[index]
            if "toolUse" in block:
                try:
                    arguments = json.loads(block["json"]) if block["json"] else {}
                except ValueError:
                    arguments = {}
                content.append({"toolUse": {**block["toolUse"], "input": arguments}})
            else:
                content.append({"text": block["text"]})
        response: dict[str, Any] = {"output": {"message": {"role": "assistant", "content": content}}}
        normalized = BedrockAdapter().normalize_response(response)
        normalized.token_usage = self._usage
        return normalized


class BedrockAdapter(LLMProviderAdapter, StreamingWire):
    """Adapter for the Amazon Bedrock Converse API (via boto3)."""

    call_id_prefix = "tooluse"

    @property
    def name(self) -> str:
        return "bedrock"

    # -- normalisation -------------------------------------------------------

    def normalize_request(self, raw_request: dict[str, Any]) -> NormalizedRequest:
        """Convert Bedrock ``converse()`` kwargs to *NormalizedRequest*.

        Bedrock messages use ``[{"role": "user", "content": [{"text": "..."}]}]``. We flatten single-text content blocks to plain strings for the normalised format.
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

        Bedrock responses are plain dicts with structure: ``{"output": {"message": {"role": "assistant", "content": [...]}}, "usage": {...}}``
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
                call = {"name": tu.get("name", ""), "arguments": json.dumps(tu.get("input", {}))}
                if isinstance(tu.get("toolUseId"), str):
                    call["id"] = tu["toolUseId"]  # replayed turns keep the recorded ids
                tool_calls.append(call)

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
                            "toolUseId": tc.get("id") or fallback_call_id("tooluse"),
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
        # Without recorded usage the key is left out, as in a real response without usage.

        return result

    # -- streaming -------------------------------------------------------------

    def stream_accumulator(self) -> StreamAccumulator:
        return ConverseStreamAccumulator()

    def tap_response(self, response: Any, tap: Any) -> Any:
        """The event stream is the ``stream`` key of the ``ConverseStream`` response."""
        response["stream"] = tap(response["stream"])
        return response

    def replay_stream(self, normalized: NormalizedResponse, kwargs: dict[str, Any], is_async: bool) -> Any:
        """Build a ``ConverseStream`` response whose ``stream`` yields the recorded response as events."""
        message = self.denormalize_response(normalized)
        events: list[dict[str, Any]] = [{"messageStart": {"role": "assistant"}}]
        for index, block in enumerate(message["output"]["message"]["content"]):
            if "toolUse" in block:
                tool_use = block["toolUse"]
                start = {"toolUse": {"toolUseId": tool_use["toolUseId"], "name": tool_use["name"]}}
                events.append({"contentBlockStart": {"start": start, "contentBlockIndex": index}})
                delta = {"toolUse": {"input": json.dumps(tool_use["input"], ensure_ascii=False)}}
            else:
                delta = {"text": block["text"]}
            events.append({"contentBlockDelta": {"delta": delta, "contentBlockIndex": index}})
            events.append({"contentBlockStop": {"contentBlockIndex": index}})
        events.append({"messageStop": {"stopReason": message["stopReason"]}})
        if "usage" in message:
            events.append({"metadata": {"usage": message["usage"], "metrics": {"latencyMs": 0}}})
        if is_async:

            async def stream() -> Any:
                for event in events:
                    yield event

            return {"stream": stream(), "ResponseMetadata": {"HTTPStatusCode": 200}}
        return {"stream": iter(events), "ResponseMetadata": {"HTTPStatusCode": 200}}

    # -- monkey-patching -----------------------------------------------------

    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:
        """Monkey-patch ``botocore.client.BaseClient._make_api_call``, and aiobotocore's ``AioBaseClient._make_api_call`` when it is installed.

        Records and replays the ``bedrock-runtime`` ``Converse`` and ``ConverseStream`` operations. Other operations that make a model generate (``InvokeModel``, ``InvokeAgent``, ``RetrieveAndGenerate``, ...) are guarded: streaming ones raise :class:`~agentverify.errors.CassetteUnsupportedCallError` in every mode, the rest raise in replay mode and pass through with a warning in record mode. Every other AWS call passes through untouched.

        In **record** mode the real API is called, the response is normalised, recorded into the cassette, and the original response is returned.

        In **replay** mode the request is normalised, looked up in the cassette, denormalised, and returned without calling the real API.
        """
        _ensure_boto3_installed()

        adapter = self

        import botocore.client

        original_fn = botocore.client.BaseClient._make_api_call

        def _guard_other(self_client: Any, operation_name: str, api_params: dict[str, Any], stacklevel: int) -> None:
            """Guard bedrock-runtime and agent-runtime operations that make a model generate."""
            service = getattr(getattr(getattr(self_client, "meta", None), "service_model", None), "service_name", None)
            if operation_name in _GENERATING_OPERATIONS.get(service, ()) and not _guards_off(recorder):
                if "Stream" in operation_name:
                    raise unsupported_call_error("bedrock", operation_name, "streaming is not recorded")
                check_unrecorded(recorder, "bedrock", operation_name, api_params, stacklevel=stacklevel)

        def _recorded(operation_name: str, api_params: dict[str, Any]) -> dict[str, Any]:
            # ConverseStream is marked as streamed so the shared flow records and replays it as a stream.
            return {**api_params, "stream": True} if operation_name == "ConverseStream" else api_params

        def _patched_make_api_call(self_client: Any, operation_name: str, api_params: dict[str, Any]) -> Any:
            """Intercept Converse calls; guard other bedrock-runtime calls; pass through everything else."""
            # Converse and ConverseStream exist only on bedrock-runtime, so the operation name alone selects the recorded path.
            if operation_name not in ("Converse", "ConverseStream"):
                # this helper, this wrapper, botocore's generated client method, the caller
                _guard_other(self_client, operation_name, api_params, stacklevel=4)
                return original_fn(self_client, operation_name, api_params)
            return call_sync(
                scope,
                recorder,
                adapter,
                operation_name,
                _recorded(operation_name, api_params),
                lambda: original_fn(self_client, operation_name, api_params),
            )

        with ExitStack() as stack:
            scope = stack.enter_context(patch_scope())
            stack.enter_context(patch(_PATCH_TARGET, new=_patched_make_api_call))
            try:
                from aiobotocore.client import AioBaseClient
            except ImportError:
                AioBaseClient = None
            if AioBaseClient is not None:
                async_original = AioBaseClient._make_api_call

                async def _patched_async(self_client: Any, operation_name: str, api_params: dict[str, Any]) -> Any:
                    """The same for aiobotocore / aioboto3 clients."""
                    if operation_name not in ("Converse", "ConverseStream"):
                        # aiobotocore's client method returns this coroutine, awaited by the caller
                        _guard_other(self_client, operation_name, api_params, stacklevel=3)
                        return await async_original(self_client, operation_name, api_params)
                    return await call_async(
                        scope,
                        recorder,
                        adapter,
                        operation_name,
                        _recorded(operation_name, api_params),
                        lambda: async_original(self_client, operation_name, api_params),
                    )

                stack.enter_context(patch.object(AioBaseClient, "_make_api_call", new=_patched_async))
            yield
