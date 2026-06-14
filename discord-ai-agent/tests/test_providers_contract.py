"""CR24 — contract tests over both LLMProvider implementations with mocked HTTP."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.llm.anthropic import AnthropicProvider
from bot.llm.openai import OpenAIProvider
from bot.llm.provider import Message, Response, ToolCall


@pytest.fixture(params=["anthropic", "openai"])
def provider(request):
    return AnthropicProvider() if request.param == "anthropic" else OpenAIProvider()


# ------------------------------------------------------------------ #
# Shared continuation-format contract (operates on a Response)
# ------------------------------------------------------------------ #

def test_build_assistant_turn_carries_tool_use(provider):
    resp = Response(
        content="let me look",
        tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "x"})],
        stop_reason="tool_use",
    )
    msg = provider.build_assistant_turn(resp)
    assert msg.role == "assistant"
    assert any(
        b.get("type") == "tool_use" and b["id"] == "t1" and b["name"] == "search"
        for b in msg.content
    )


def test_build_tool_results_turn_carries_results(provider):
    tc = ToolCall(id="t1", name="search", arguments={})
    msg = provider.build_tool_results_turn([(tc, "the answer")])
    block = msg.content[0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "t1"
    assert block["content"] == "the answer"


async def test_complete_returns_response_with_tool_call(provider, monkeypatch):
    """Each provider parses its own SDK response shape into the internal Response."""
    if isinstance(provider, AnthropicProvider):
        fake = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="hi"),
                SimpleNamespace(type="tool_use", id="t1", name="foo", input={"a": 1}),
            ],
            stop_reason="tool_use",
        )
    else:
        fake = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi", tool_calls=[SimpleNamespace(
                id="t1", function=SimpleNamespace(name="foo", arguments='{"a": 1}'),
            )]),
            finish_reason="tool_calls",
        )])

    monkeypatch.setattr(provider, "_create", AsyncMock(return_value=fake))
    resp = await provider.complete(messages=[Message(role="user", content="x")],
                                   tools=[], system="s", model="m")

    assert isinstance(resp, Response)
    assert resp.content == "hi"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "foo"
    assert resp.tool_calls[0].arguments == {"a": 1}
    assert resp.stop_reason == "tool_use"


async def test_complete_plain_text_response(provider, monkeypatch):
    if isinstance(provider, AnthropicProvider):
        fake = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="just text")],
            stop_reason="end_turn",
        )
    else:
        fake = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="just text", tool_calls=None),
            finish_reason="stop",
        )])

    monkeypatch.setattr(provider, "_create", AsyncMock(return_value=fake))
    resp = await provider.complete(messages=[Message(role="user", content="x")],
                                   tools=[], system="s", model="m")
    assert resp.content == "just text"
    assert resp.tool_calls == []
    assert resp.stop_reason != "tool_use"
