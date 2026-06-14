"""Anthropic / Claude LLM provider."""
from __future__ import annotations

import logging
import time

import anthropic

from ..config import settings
from .provider import Message, Response, Tool, ToolCall, retrying

log = logging.getLogger(__name__)

# Transient errors worth retrying with backoff (rate limits, 5xx, network).
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)

# Thinking needs room to produce a reasoning block *and* the actual answer. Below
# this budget we skip it so reasoning tokens can't crowd out the response entirely
# (e.g. the automod classifier runs with a few hundred tokens).
_THINKING_MIN_TOKENS = 1024


def _block_to_dict(block: object) -> object:
    """Anthropic SDK content blocks are pydantic models; serialize for re-send."""
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return block


class AnthropicProvider:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    @retrying(*_RETRYABLE)
    async def _create(self, **kwargs):
        return await self._client.messages.create(**kwargs)

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
        model: str,
        max_tokens: int = 4096,
        extended_thinking: bool = True,
    ) -> Response:
        api_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        api_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": api_messages,
        }
        if (
            extended_thinking
            and settings.anthropic_extended_thinking
            and max_tokens >= _THINKING_MIN_TOKENS
        ):
            kwargs["thinking"] = {"type": "adaptive"}
        if api_tools:
            kwargs["tools"] = api_tools

        start = time.monotonic()
        resp = await self._create(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        usage = getattr(resp, "usage", None)
        log.info(
            "LLM call complete",
            extra={
                "provider": "anthropic",
                "model": model,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "latency_ms": latency_ms,
            },
        )

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        thinking_blocks: list[object] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
            elif block.type in ("thinking", "redacted_thinking"):
                thinking_blocks.append(block)

        stop = "tool_use" if tool_calls else (resp.stop_reason or "end_turn")
        return Response(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            thinking_blocks=thinking_blocks,
        )

    def build_assistant_turn(self, resp: Response) -> Message:
        """Assistant turn echoing thinking + text + tool_use blocks, in API order.

        Anthropic requires any thinking blocks be sent back ahead of the tool_use
        blocks they preceded; dropping them 400s multi-turn extended-thinking calls.
        """
        parts: list[object] = [_block_to_dict(b) for b in resp.thinking_blocks]
        if resp.content:
            parts.append({"type": "text", "text": resp.content})
        for tc in resp.tool_calls:
            parts.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
        return Message(role="assistant", content=parts)

    def build_tool_results_turn(self, results: list[tuple[ToolCall, str]]) -> Message:
        blocks = [
            {"type": "tool_result", "tool_use_id": tc.id, "content": output}
            for tc, output in results
        ]
        return Message(role="user", content=blocks)
