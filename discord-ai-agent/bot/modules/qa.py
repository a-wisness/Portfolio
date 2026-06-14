"""Q&A Knowledge Base — CRUD and FTS5 full-text search."""
from __future__ import annotations

from sqlalchemy import text
from sqlmodel import select

from ..database import KBEntry, get_session


async def add_entry(guild_id: int, title: str, content: str) -> KBEntry:
    async with get_session() as session:
        entry = KBEntry(guild_id=guild_id, title=title, content=content)
        session.add(entry)
        await session.flush()  # assigns entry.id before we need it for FTS
        # Use the underlying connection to bypass SQLModel's session.execute() warning.
        conn = await session.connection()
        await conn.execute(
            text(
                "INSERT INTO kbentry_fts(rowid, title, content)"
                " VALUES (:id, :title, :content)"
            ),
            {"id": entry.id, "title": title, "content": content},
        )
        await session.commit()
        return entry


async def delete_entry(entry_id: int, guild_id: int) -> bool:
    """Delete a KB entry and remove it from the FTS index. Returns False if not found."""
    async with get_session() as session:
        result = await session.exec(
            select(KBEntry).where(KBEntry.id == entry_id, KBEntry.guild_id == guild_id)
        )
        entry = result.first()
        if entry is None:
            return False
        conn = await session.connection()
        # FTS5 content-table delete protocol: supply old values so the index can be updated.
        await conn.execute(
            text(
                "INSERT INTO kbentry_fts(kbentry_fts, rowid, title, content)"
                " VALUES ('delete', :id, :title, :content)"
            ),
            {"id": entry.id, "title": entry.title, "content": entry.content},
        )
        await session.delete(entry)
        await session.commit()
        return True


async def search(guild_id: int, query: str, limit: int = 5) -> list[dict]:
    """FTS5 search scoped to the guild. Returns list of {id, title, content} dicts."""
    # Wrap each token in quotes so FTS5 treats them as exact terms, not operators.
    clean = " ".join(f'"{w.replace(chr(34), "")}"' for w in query.split() if w)
    if not clean:
        return []
    async with get_session() as session:
        conn = await session.connection()
        # Join FTS to kbentry so the guild filter and relevance ordering are applied
        # *before* LIMIT — otherwise the limit could discard the target guild's rows.
        rows = await conn.execute(
            text("""
                SELECT ke.id, ke.title, ke.content
                FROM kbentry_fts
                JOIN kbentry AS ke ON ke.id = kbentry_fts.rowid
                WHERE kbentry_fts MATCH :q AND ke.guild_id = :guild_id
                ORDER BY kbentry_fts.rank
                LIMIT :limit
            """),
            {"q": clean, "guild_id": guild_id, "limit": limit},
        )
        return [{"id": r.id, "title": r.title, "content": r.content} for r in rows.fetchall()]


async def list_entries(guild_id: int, limit: int = 25) -> list[KBEntry]:
    async with get_session() as session:
        result = await session.exec(
            select(KBEntry)
            .where(KBEntry.guild_id == guild_id)
            .order_by(KBEntry.created_at.desc())
            .limit(limit)
        )
        return list(result.all())
