"""Tests for bot/history.py — DB-backed per-channel conversation history (CR30)."""
from bot.history import add_turn, clear, get
from bot.llm.provider import Message

G = 0


async def test_empty():
    assert await get(G, 10) == []


async def test_add_and_retrieve():
    await add_turn(G, 11, "hello", "world")
    h = await get(G, 11)
    assert h == [Message(role="user", content="hello"), Message(role="assistant", content="world")]


async def test_multiple_turns_ordered():
    await add_turn(G, 12, "q1", "a1")
    await add_turn(G, 12, "q2", "a2")
    h = await get(G, 12)
    assert [m.content for m in h] == ["q1", "a1", "q2", "a2"]


async def test_clear_resets():
    await add_turn(G, 13, "a", "b")
    await clear(G, 13)
    assert await get(G, 13) == []


async def test_channels_isolated():
    await add_turn(G, 14, "in-14", "r14")
    await add_turn(G, 15, "in-15", "r15")
    assert [m.content for m in await get(G, 14)] == ["in-14", "r14"]
    assert [m.content for m in await get(G, 15)] == ["in-15", "r15"]


async def test_persists_across_calls():
    """A fresh get() (no in-memory state) still returns prior turns — i.e. persisted."""
    await add_turn(G, 16, "remembered", "yes")
    # Simulate a "restart": nothing cached in process; read straight from the DB.
    h = await get(G, 16)
    assert [m.content for m in h] == ["remembered", "yes"]


async def test_bounded_by_max_history(monkeypatch):
    from bot import config

    monkeypatch.setattr(config.settings, "max_conversation_history", 4)
    await add_turn(G, 17, "q1", "a1")  # 2 messages
    await add_turn(G, 17, "q2", "a2")  # 4 — full
    await add_turn(G, 17, "q3", "a3")  # 6 added, only last 4 kept

    h = await get(G, 17)
    assert len(h) == 4
    assert h[0].content == "q2"  # q1/a1 pruned
