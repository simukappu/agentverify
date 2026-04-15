"""Base classes for LLM provider adapters.

Defines the provider-agnostic normalized request/response formats
and the abstract base class that all provider adapters must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator

from agentverify.models import TokenUsage

if TYPE_CHECKING:
    from agentverify.cassette.recorder import LLMCassetteRecorder


@dataclass
class NormalizedRequest:
    """Provider-agnostic request format."""

    messages: list[dict[str, Any]]
    model: str
    tools: list[dict[str, Any]] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedResponse:
    """Provider-agnostic response format."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    token_usage: TokenUsage | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class LLMProviderAdapter(ABC):
    """Abstract base class for LLM SDK chat completion adapters."""

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover — abstract method
        """Provider name (e.g. "openai", "anthropic")."""
        ...

    @abstractmethod
    @contextmanager
    def patch(self, recorder: LLMCassetteRecorder) -> Generator[None, None, None]:  # pragma: no cover — abstract method
        """Monkey-patch the LLM SDK's chat completion method.

        Records or replays requests/responses through the recorder.
        The patch is automatically removed when the context manager exits.
        """
        ...

    @abstractmethod
    def normalize_request(self, raw_request: dict[str, Any]) -> NormalizedRequest:  # pragma: no cover — abstract method
        """Normalize an SDK-specific request into the common format."""
        ...

    @abstractmethod
    def normalize_response(self, raw_response: Any) -> NormalizedResponse:  # pragma: no cover — abstract method
        """Normalize an SDK-specific response into the common format."""
        ...

    @abstractmethod
    def denormalize_response(self, normalized: NormalizedResponse) -> Any:  # pragma: no cover — abstract method
        """Restore a normalized response back to the SDK-specific format."""
        ...
