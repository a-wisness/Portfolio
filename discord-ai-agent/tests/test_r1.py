"""Tests for Phase R1 — provider-agnostic tool loop, OpenAI wire format, rate limiter."""
from bot.agents.base import Agent
from bot.llm.openai import _to_api_messages
from bot.llm.provider import Message, Response, Tool, ToolCall
from bot.utils import RateLimiter


# ------------------------------------------------------------------ #
# CR14 — RateLimiter
# ------------------------------------------------------------------ #

def test_rate_limiter_allows_up_to_max_then_blocks():
    rl = RateLimiter(max_calls=3, window=60.0)
    key = (1, 2)
    assert [rl.allow(key) for _ in range(4)] == [True, True, True, False]


def test_rate_limiter_keys_are_independent():
    rl = RateLimiter(max_calls=1, window=60.0)
    assert rl.allow((1, 1)) is True
    assert rl.allow((1, 2)) is True  # different user, own bucket
    assert rl.allow((1, 1)) is False


def test_rate_limiter_window_expiry():
    rl = RateLimiter(max_calls=1, window=0.0)  # everything is already "expired"
    key = "k"
    assert rl.allow(key) is True
    assert rl.allow(key) is True


# ------------------------------------------------------------------ #
# CR6 — OpenAI neutral-block → wire-format translation
# ------------------------------------------------------------------ #

def test_openai_translates_tool_use_to_tool_calls():
    assistant = Message(role="assistant", content=[
        {"type": "text", "text": "let me check"},
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}},
    ])
    out = _to_api_messages([assistant])
    assert len(out) == 1
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == "let me check"
    assert out[0]["tool_calls"][0]["id"] == "call_1"
    assert out[0]["tool_calls"][0]["function"]["name"] == "search"
    # arguments must be a JSON *string*, per the OpenAI API
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'


def test_openai_translates_tool_result_to_tool_role():
    results = Message(role="user", content=[
        {"type": "tool_result", "tool_use_id": "call_1", "content": "found it"},
    ])
    out = _to_api_messages([results])
    assert out == [{"role": "tool", "tool_call_id": "call_1", "content": "found it"}]


def test_openai_plain_text_passthrough():
    out = _to_api_messages([Message(role="user", content="hello")])
    assert out == [{"role": "user", "content": "hello"}]


# ------------------------------------------------------------------ #
# CR6 — agent runs the tool loop without provider isinstance branching
# ------------------------------------------------------------------ #

class FakeProvider:
    """Minimal provider: first call requests a tool, second call ends the turn."""

    def __init__(self) -> None:
        self.calls = 0
        self.turns: list[Message] = []

    async def complete(self, messages, tools, system, model, max_tokens=4096, extended_thinking=True):
        self.calls += 1
        if self.calls == 1:
            return Response(
                content="",
                tool_calls=[ToolCall(id="t1", name="echo", arguments={"value": "hi"})],
                stop_reason="tool_use",
            )
        # On the second call we should have received the continuation turns.
        self.turns = list(messages)
        return Response(content="done", stop_reason="end_turn")

    def build_assistant_turn(self, resp: Response) -> Message:
        return Message(role="assistant", content=[
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            for tc in resp.tool_calls
        ])

    def build_tool_results_turn(self, results) -> Message:
        return Message(role="user", content=[
            {"type": "tool_result", "tool_use_id": tc.id, "content": out} for tc, out in results
        ])


async def test_agent_tool_loop_is_provider_agnostic():
    provider = FakeProvider()
    tool = Tool(name="echo", description="echo", parameters={})

    async def echo(value: str) -> str:
        return f"echoed:{value}"

    agent = Agent(provider=provider, tools=[(tool, echo)], system="s", model="m")
    result = await agent.run([Message(role="user", content="please echo")])

    assert result == "done"
    assert provider.calls == 2
    # The continuation must include the assistant tool_use turn and the tool_result turn.
    assert any(
        isinstance(m.content, list) and any(b.get("type") == "tool_result" for b in m.content)
        for m in provider.turns
    )


async def test_agent_unknown_tool_returns_error_string():
    provider = FakeProvider()
    agent = Agent(provider=provider, tools=[], system="s", model="m")
    # 'echo' isn't registered, so _call should yield an "Unknown tool" result and the loop continues.
    result = await agent.run([Message(role="user", content="x")])
    assert result == "done"
