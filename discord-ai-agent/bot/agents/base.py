"""Agentic loop: call LLM → dispatch tools → repeat until end_turn."""
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from ..llm.provider import LLMProvider, Message, Response, Tool, ToolCall

log = logging.getLogger(__name__)

ToolFn = Callable[..., Any]


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[tuple[Tool, ToolFn]],
        system: str,
        model: str,
        max_iterations: int = 10,
    ) -> None:
        self._provider = provider
        self._tool_map = {t.name: (t, fn) for t, fn in tools}
        self._system = system
        self._model = model
        self._max_iterations = max_iterations

    async def run(self, history: list[Message]) -> str:
        messages = list(history)
        tool_defs = [t for t, _ in self._tool_map.values()]

        for _ in range(self._max_iterations):
            resp: Response = await self._provider.complete(
                messages=messages,
                tools=tool_defs,
                system=self._system,
                model=self._model,
            )

            if resp.stop_reason in ("end_turn", "max_tokens") or not resp.tool_calls:
                return resp.content

            # Each provider serializes the tool-use continuation in its own wire
            # format (Anthropic tool_use/tool_result blocks; OpenAI tool_calls +
            # role="tool" messages). The agent stays provider-agnostic.
            messages.append(self._provider.build_assistant_turn(resp))
            results = [(tc, await self._call(tc)) for tc in resp.tool_calls]
            messages.append(self._provider.build_tool_results_turn(results))

        log.warning("Reached max iterations (%d) without end_turn", self._max_iterations)
        return resp.content

    async def _call(self, tc: ToolCall) -> str:
        if tc.name not in self._tool_map:
            return f"Unknown tool: {tc.name}"
        _, fn = self._tool_map[tc.name]
        try:
            result = fn(**tc.arguments)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as exc:
            log.exception("Tool %s raised", tc.name)
            return f"Tool error: {exc}"
