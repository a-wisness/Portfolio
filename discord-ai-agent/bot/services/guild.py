"""Guild service — load per-guild config from DB and build agents."""
from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from ..agents.base import Agent, ToolFn
from ..database import GuildConfig, get_session
from ..llm.provider import Tool, get_provider


async def get_or_create_config(guild_id: int) -> GuildConfig:
    """Return the guild's config row, creating it with defaults on first access."""
    async with get_session() as session:
        result = await session.exec(
            select(GuildConfig).where(GuildConfig.guild_id == guild_id)
        )
        cfg = result.first()
        if cfg is None:
            cfg = GuildConfig(guild_id=guild_id)
            session.add(cfg)
            try:
                await session.commit()
            except IntegrityError:
                # A concurrent request created the row first (unique guild_id).
                # Roll back and re-read the winner instead of failing.
                await session.rollback()
                result = await session.exec(
                    select(GuildConfig).where(GuildConfig.guild_id == guild_id)
                )
                cfg = result.first()
        return cfg


def make_agent(
    cfg: GuildConfig,
    tools: list[tuple[Tool, ToolFn]] | None = None,
    system: str | None = None,
) -> Agent:
    """Build a fresh agent from an already-loaded guild config.

    Agents are cheap to construct (no I/O). Creating one per request keeps tool
    lists correct without any cache-invalidation logic.

    Pass *system* to override the guild's configured system prompt (e.g. the
    management agent uses its own purpose-built prompt instead of the guild default).
    """
    return Agent(
        provider=get_provider(cfg.provider),
        tools=tools or [],
        system=system if system is not None else cfg.system_prompt,
        model=cfg.model,
    )
