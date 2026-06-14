"""Server management slash commands — role and channel management via agent."""
from __future__ import annotations

import logging
from typing import Literal

import discord
from discord import app_commands
from discord.ext import commands
from sqlmodel import select

from ..database import GuildConfig, get_session
from ..llm.provider import Message
from ..logging_config import request_context
from ..modules.manager import (
    _MANAGER_SYSTEM,
    make_channel_tools,
    make_moderation_tools,
    make_role_tools,
)
from ..services.guild import get_or_create_config, make_agent
from ..utils import require_admin, split_message

log = logging.getLogger(__name__)

ActionType = Literal["role_assign", "role_remove", "channel_create", "channel_archive"]

_ACTION_FIELDS: dict[str, tuple[str, str]] = {
    "role_assign": ("allow_role_assign", "Role Assignment"),
    "role_remove": ("allow_role_remove", "Role Removal"),
    "channel_create": ("allow_channel_create", "Channel Creation"),
    "channel_archive": ("allow_channel_archive", "Channel Archive"),
}


async def _invoke_manage(
    cfg: GuildConfig,
    request: str,
    tools: list,
) -> str:
    agent = make_agent(cfg, tools, system=_MANAGER_SYSTEM)
    return await agent.run([Message(role="user", content=request)])


class ManageCog(commands.GroupCog, name="manage"):
    """Agent-driven role and channel management with per-action opt-in."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ------------------------------------------------------------------ #
    # Agent-driven management commands
    # ------------------------------------------------------------------ #

    @app_commands.command(
        name="roles",
        description="Ask the management agent to suggest or apply a role change.",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(request="Describe the role change (e.g. 'Give @alice the Moderator role').")
    async def manage_roles(self, interaction: discord.Interaction, request: str) -> None:
        if not await require_admin(interaction):
            return
        await interaction.response.defer()
        cfg = await get_or_create_config(interaction.guild_id)
        tools = make_role_tools(interaction.guild, cfg)
        try:
            with request_context(interaction.guild_id, interaction.id):
                response = await _invoke_manage(cfg, request, tools)
        except Exception:
            log.exception("Manager error in /manage roles")
            await interaction.followup.send(
                "Sorry — the management agent hit an error. Please try again.", ephemeral=True
            )
            return
        for chunk in split_message(response):
            await interaction.followup.send(chunk)

    @app_commands.command(
        name="channels",
        description="Ask the management agent to suggest or apply a channel change.",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(request="Describe the channel action (e.g. 'Create an #announcements channel').")
    async def manage_channels(self, interaction: discord.Interaction, request: str) -> None:
        if not await require_admin(interaction):
            return
        await interaction.response.defer()
        cfg = await get_or_create_config(interaction.guild_id)
        tools = make_channel_tools(interaction.guild, cfg)
        try:
            with request_context(interaction.guild_id, interaction.id):
                response = await _invoke_manage(cfg, request, tools)
        except Exception:
            log.exception("Manager error in /manage channels")
            await interaction.followup.send(
                "Sorry — the management agent hit an error. Please try again.", ephemeral=True
            )
            return
        for chunk in split_message(response):
            await interaction.followup.send(chunk)

    @app_commands.command(
        name="moderate",
        description="Ask the agent to warn a user or delete a message.",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(request="Describe the action (e.g. 'Warn user 123 for spamming links').")
    async def manage_moderate(self, interaction: discord.Interaction, request: str) -> None:
        if not await require_admin(interaction):
            return
        await interaction.response.defer()
        cfg = await get_or_create_config(interaction.guild_id)
        tools = make_moderation_tools(interaction.guild)
        try:
            with request_context(interaction.guild_id, interaction.id):
                response = await _invoke_manage(cfg, request, tools)
        except Exception:
            log.exception("Manager error in /manage moderate")
            await interaction.followup.send(
                "Sorry — the management agent hit an error. Please try again.", ephemeral=True
            )
            return
        for chunk in split_message(response):
            await interaction.followup.send(chunk)

    # ------------------------------------------------------------------ #
    # Opt-in / opt-out commands
    # ------------------------------------------------------------------ #

    @app_commands.command(
        name="enable",
        description="Opt in to a management action type so the agent can execute it.",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(action="The action type to enable.")
    async def enable(
        self, interaction: discord.Interaction, action: ActionType
    ) -> None:
        await self._set_action(interaction, action, True)

    @app_commands.command(
        name="disable",
        description="Opt out of a management action type (agent will suggest only).",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(action="The action type to disable.")
    async def disable(
        self, interaction: discord.Interaction, action: ActionType
    ) -> None:
        await self._set_action(interaction, action, False)

    async def _set_action(
        self, interaction: discord.Interaction, action: str, value: bool
    ) -> None:
        if not await require_admin(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        field_name, label = _ACTION_FIELDS[action]
        async with get_session() as session:
            result = await session.exec(
                select(GuildConfig).where(GuildConfig.guild_id == interaction.guild_id)
            )
            cfg = result.first()
            if cfg is None:
                cfg = GuildConfig(guild_id=interaction.guild_id)
                session.add(cfg)
            setattr(cfg, field_name, value)
            await session.commit()
        state = "enabled" if value else "disabled"
        await interaction.followup.send(f"{label} {state}.", ephemeral=True)

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    @app_commands.command(
        name="status",
        description="Show which management action types are currently enabled.",
    )
    @app_commands.guild_only()
    async def status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        cfg = await get_or_create_config(interaction.guild_id)
        embed = discord.Embed(title="Management Status", color=discord.Color.blue())
        for action, (field_name, label) in _ACTION_FIELDS.items():
            enabled = getattr(cfg, field_name)
            embed.add_field(
                name=label,
                value="enabled" if enabled else "disabled",
                inline=True,
            )
        embed.set_footer(
            text="Use /manage enable <action> to allow the agent to execute an action type."
        )
        await interaction.followup.send(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ManageCog(bot))
