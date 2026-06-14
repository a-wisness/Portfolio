"""Internal LLM types + provider protocol."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_log = logging.getLogger(__name__)


def retrying(*exc_types: type[BaseException]):
    """Shared retry policy for transient LLM API errors (429 / 5xx / connection).

    Exponential backoff, up to 4 attempts, then re-raise so callers still see the
    final error. Each provider passes its own SDK's transient exception types.
    """
    return retry(
        reraise=True,
        retry=retry_if_exception_type(exc_types),
        wait=wait_exponential(multiplier=0.5, max=8),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(_log, logging.WARNING),
    )


@dataclass
class Message:
    role: Literal["user", "assistant"]
    # str for plain text; list[dict] for multi-part content (tool use/result blocks)
    content: str | list[Any]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Response:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    # Provider-native reasoning blocks that must be echoed back on the next turn
    # (Anthropic extended thinking). Opaque to the agent; only the provider reads them.
    thinking_blocks: list[Any] = field(default_factory=list)


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
        model: str,
        max_tokens: int = 4096,
        extended_thinking: bool = True,
    ) -> Response: ...

    def build_assistant_turn(self, resp: Response) -> Message:
        """Serialize a tool-calling assistant turn into a Message for the next call.

        Each provider emits its own wire format; the agent stays provider-agnostic.
        """
        ...

    def build_tool_results_turn(self, results: list[tuple[ToolCall, str]]) -> Message:
        """Serialize tool outputs into the follow-up Message the provider expects."""
        ...


def get_provider(provider: str) -> LLMProvider:
    if provider == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider()
    if provider == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {provider!r}")


# Allowlist of provider -> selectable models. Keeps `/config` from storing
# arbitrary strings that only blow up later inside the agentic loop.
ALLOWED_MODELS: dict[str, tuple[str, ...]] = {
    "anthropic": (
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-fable-5",
    ),
    "openai": (
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o3",
        "o4-mini",
    ),
}


def is_valid_provider(provider: str) -> bool:
    return provider in ALLOWED_MODELS


def is_valid_model(provider: str, model: str) -> bool:
    return model in ALLOWED_MODELS.get(provider, ())
