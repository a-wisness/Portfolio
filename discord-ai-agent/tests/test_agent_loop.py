"""CR23 — agentic loop coverage: single-turn, multi-turn, max-iterations, errors."""
from bot.agents.base import Agent
from bot.llm.provider import Message, Response, Tool, ToolCall

_TOOL = Tool(name="echo", description="echo", parameters={})


def _tool_use(name="echo", tid="t1", args=None):
    return Response(content="", tool_calls=[ToolCall(id=tid, name=name, arguments=args or {})],
                    stop_reason="tool_use")


class ScriptedProvider:
    """Returns queued Responses on successive complete() calls; records last input."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.last_messages: list[Message] = []

    async def complete(self, messages, tools, system, model, max_tokens=4096, extended_thinking=True):
        self.calls += 1
        self.last_messages = list(messages)
        if self._responses:
            return self._responses.pop(0)
        return Response(content="end", stop_reason="end_turn")

    def build_assistant_turn(self, resp):
        return Message(role="assistant", content=[
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            for tc in resp.tool_calls
        ])

    def build_tool_results_turn(self, results):
        return Message(role="user", content=[
            {"type": "tool_result", "tool_use_id": tc.id, "content": out} for tc, out in results
        ])


def _last_tool_results(provider) -> list[str]:
    """Tool-result contents from the most recent turn the provider received."""
    out = []
    for m in provider.last_messages:
        if isinstance(m.content, list):
            out += [b["content"] for b in m.content if b.get("type") == "tool_result"]
    return out


# ------------------------------------------------------------------ #

async def test_single_turn_no_tools():
    provider = ScriptedProvider([Response(content="just an answer", stop_reason="end_turn")])
    agent = Agent(provider=provider, tools=[], system="s", model="m")
    assert await agent.run([Message(role="user", content="hi")]) == "just an answer"
    assert provider.calls == 1


async def test_multi_turn_tool_then_answer():
    provider = ScriptedProvider([_tool_use(args={"value": "x"}),
                                 Response(content="final", stop_reason="end_turn")])

    async def echo(value: str) -> str:
        return f"echoed:{value}"

    agent = Agent(provider=provider, tools=[(_TOOL, echo)], system="s", model="m")
    assert await agent.run([Message(role="user", content="go")]) == "final"
    assert provider.calls == 2
    assert _last_tool_results(provider) == ["echoed:x"]


async def test_max_iterations_fallback():
    # Provider that never stops requesting tools.
    provider = ScriptedProvider([_tool_use() for _ in range(10)])

    async def echo(**kw) -> str:
        return "ok"

    agent = Agent(provider=provider, tools=[(_TOOL, echo)], system="s", model="m", max_iterations=3)
    result = await agent.run([Message(role="user", content="go")])
    assert provider.calls == 3          # stopped at the cap
    assert result == ""                 # returns last (toolful) response's content


async def test_tool_exception_is_swallowed_into_result():
    provider = ScriptedProvider([_tool_use(name="boom"),
                                 Response(content="recovered", stop_reason="end_turn")])

    async def boom(**kw):
        raise ValueError("kaboom")

    boom_tool = Tool(name="boom", description="boom", parameters={})
    agent = Agent(provider=provider, tools=[(boom_tool, boom)], system="s", model="m")
    result = await agent.run([Message(role="user", content="go")])
    assert result == "recovered"
    assert any("Tool error" in r and "kaboom" in r for r in _last_tool_results(provider))


async def test_unknown_tool_reported_to_model():
    provider = ScriptedProvider([_tool_use(name="missing"),
                                 Response(content="done", stop_reason="end_turn")])
    agent = Agent(provider=provider, tools=[], system="s", model="m")
    result = await agent.run([Message(role="user", content="go")])
    assert result == "done"
    assert any("Unknown tool: missing" in r for r in _last_tool_results(provider))


async def test_sync_tool_function_supported():
    # _call must handle a plain (non-async) function via inspect.isawaitable.
    provider = ScriptedProvider([_tool_use(args={"value": "y"}),
                                 Response(content="ok", stop_reason="end_turn")])

    def echo_sync(value: str) -> str:
        return f"sync:{value}"

    agent = Agent(provider=provider, tools=[(_TOOL, echo_sync)], system="s", model="m")
    assert await agent.run([Message(role="user", content="go")]) == "ok"
    assert _last_tool_results(provider) == ["sync:y"]
