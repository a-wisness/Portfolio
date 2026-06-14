"""OpenAI LLM provider."""
from __future__ import annotations

import json
import logging
import time

import openai as openai_sdk

from ..config import settings
from .provider import Message, Response, Tool, ToolCall, retrying

log = logging.getLogger(__name__)

# Transient errors worth retrying with backoff (rate limits, 5xx, network).
_RETRYABLE = (
    openai_sdk.RateLimitError,
    openai_sdk.InternalServerError,
    openai_sdk.APIConnectionError,
    openai_sdk.APITimeoutError,
)


def _to_api_messages(messages: list[Message]) -> list[dict]:
    """Translate internal Messages (incl. neutral tool blocks) to OpenAI wire format.

    Plain-text turns map directly. A list-of-blocks turn carries the neutral
    representation produced by `build_assistant_turn` / `build_tool_results_turn`:
    `tool_use` blocks become an assistant `tool_calls` array, and `tool_result`
    blocks become separate `role="tool"` messages with their `tool_call_id`.
    """
    out: list[dict] = []
    for m in messages:
        if isinstance(m.content, str):
            out.append({"role": m.role, "content": m.content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block in m.content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block["text"])
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    },
                })
            elif btype == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": block["content"],
                })

        if tool_calls or text_parts:
            msg: dict = {"role": m.role, "content": "\n".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        out.extend(tool_results)
    return out


class OpenAIProvider:
    def __init__(self) -> None:
        self._client = openai_sdk.AsyncOpenAI(api_key=settings.openai_api_key)

    @retrying(*_RETRYABLE)
    async def _create(self, **kwargs):
        return await self._client.chat.completions.create(**kwargs)

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
        model: str,
        max_tokens: int = 4096,
        extended_thinking: bool = True,  # OpenAI has no equivalent; accepted for protocol parity
    ) -> Response:
        api_messages: list[dict] = [{"role": "system", "content": system}]
        api_messages.extend(_to_api_messages(messages))

        api_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

        kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": api_messages}
        if api_tools:
            kwargs["tools"] = api_tools
            kwargs["tool_choice"] = "auto"

        start = time.monotonic()
        resp = await self._create(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        usage = getattr(resp, "usage", None)
        log.info(
            "LLM call complete",
            extra={
                "provider": "openai",
                "model": model,
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "latency_ms": latency_ms,
            },
        )
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    # Malformed tool arguments from the API — degrade to empty args
                    # rather than crashing the whole agent run.
                    log.warning("OpenAI returned non-JSON tool arguments for %s: %r",
                                tc.function.name, tc.function.arguments)
                    arguments = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        stop = "tool_use" if tool_calls else (choice.finish_reason or "end_turn")
        return Response(content=msg.content or "", tool_calls=tool_calls, stop_reason=stop)

    def build_assistant_turn(self, resp: Response) -> Message:
        """Assistant turn in the neutral block format; `_to_api_messages` maps it
        to an OpenAI message carrying a `tool_calls` array."""
        parts: list[dict] = []
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
