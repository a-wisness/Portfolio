"""Per-channel conversation history, persisted to the database.

Keyed by (guild_id, channel_id). Backed by the `ConversationMessage` table so
history survives bot restarts (previously an in-memory dict that was wiped on
every reconnect). Each channel is bounded to the most recent
`settings.max_conversation_history` messages.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlmodel import select

from .config import settings
from .database import ConversationMessage, get_session
from .llm.provider import Message


async def get(guild_id: int, channel_id: int) -> list[Message]:
    """Return the recent history for a channel in chronological order."""
    async with get_session() as session:
        result = await session.exec(
            select(ConversationMessage)
            .where(
                ConversationMessage.guild_id == guild_id,
                ConversationMessage.channel_id == channel_id,
            )
            .order_by(ConversationMessage.id.desc())
            .limit(settings.max_conversation_history)
        )
        rows = list(result.all())
    rows.reverse()  # newest-first query → chronological for the agent
    return [Message(role=r.role, content=r.content) for r in rows]


async def add_turn(
    guild_id: int, channel_id: int, user_content: str, assistant_content: str
) -> None:
    """Append a completed user→assistant exchange and prune to the bound."""
    async with get_session() as session:
        session.add(ConversationMessage(
            guild_id=guild_id, channel_id=channel_id, role="user", content=user_content,
        ))
        session.add(ConversationMessage(
            guild_id=guild_id, channel_id=channel_id, role="assistant", content=assistant_content,
        ))
        await session.commit()
        # Keep only the most recent N rows for this channel. Use the raw connection
        # (as qa.py does) since this is a bulk DELETE, not a SQLModel select.
        conn = await session.connection()
        await conn.execute(
            text("""
                DELETE FROM conversationmessage
                WHERE guild_id = :g AND channel_id = :c AND id NOT IN (
                    SELECT id FROM conversationmessage
                    WHERE guild_id = :g AND channel_id = :c
                    ORDER BY id DESC LIMIT :n
                )
            """),
            {"g": guild_id, "c": channel_id, "n": settings.max_conversation_history},
        )
        await session.commit()


async def clear(guild_id: int, channel_id: int) -> None:
    """Delete all stored history for a channel."""
    async with get_session() as session:
        conn = await session.connection()
        await conn.execute(
            text(
                "DELETE FROM conversationmessage WHERE guild_id = :g AND channel_id = :c"
            ),
            {"g": guild_id, "c": channel_id},
        )
        await session.commit()
