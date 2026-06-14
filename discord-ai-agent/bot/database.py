from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import Field, SQLModel, text
from sqlmodel.ext.asyncio.session import AsyncSession

from .config import settings

# Engine is created lazily (not at import time) so tests can inject a DB URL via
# init_db() without monkey-patching before import.
_engine: Optional[AsyncEngine] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.database_url, echo=False)
    return _engine


class GuildConfig(SQLModel, table=True):
    """Per-guild bot configuration."""
    id: Optional[int] = Field(default=None, primary_key=True)
    guild_id: int = Field(unique=True, index=True)
    provider: str = Field(default_factory=lambda: settings.default_provider)
    model: str = Field(default_factory=lambda: settings.default_model)
    system_prompt: str = "You are a helpful assistant for this Discord server."
    automod_enabled: bool = False
    manager_enabled: bool = False
    qa_enabled: bool = True
    allow_role_assign: bool = False
    allow_role_remove: bool = False
    allow_channel_create: bool = False
    allow_channel_archive: bool = False
    automod_threshold: float = 0.8
    mod_log_channel_id: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # onupdate fires on every UPDATE so updated_at tracks the last mutation.
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


class KBEntry(SQLModel, table=True):
    """Knowledge base entry for the Q&A module."""
    id: Optional[int] = Field(default=None, primary_key=True)
    guild_id: int = Field(index=True)
    title: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationMessage(SQLModel, table=True):
    """A persisted conversation turn, so history survives bot restarts."""
    id: Optional[int] = Field(default=None, primary_key=True)
    guild_id: int = Field(index=True)
    channel_id: int = Field(index=True)
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModLog(SQLModel, table=True):
    """Moderation action log."""
    id: Optional[int] = Field(default=None, primary_key=True)
    guild_id: int = Field(index=True)
    user_id: int
    action: str
    reason: str
    message_content: Optional[str] = None
    moderator: str = "bot"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


async def init_db(database_url: Optional[str] = None) -> None:
    """Create tables (incl. the FTS5 table). Pass *database_url* to (re)bind the
    engine to a specific database — used by tests to inject an isolated DB."""
    global _engine
    if database_url is not None:
        _engine = create_async_engine(database_url, echo=False)
    async with get_engine().begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        # FTS5 virtual table is not creatable via SQLModel metadata; create explicitly.
        await conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS kbentry_fts
            USING fts5(title, content, content='kbentry', content_rowid='id')
        """))


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    # expire_on_commit=False keeps column values readable after commit without a refresh round-trip.
    async with AsyncSession(get_engine(), expire_on_commit=False) as session:
        yield session
