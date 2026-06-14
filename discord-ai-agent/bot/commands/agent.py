"""Agent slash-command group — /agent prompt | show | reset | clear."""
from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands
from sqlmodel import select

from .. import history as conversation
from ..database import GuildConfig, get_session
from ..utils import require_admin, require_permissions

DEFAULT_PROMPT = "You are a helpful assistant for this Discord server."


class AgentCog(commands.GroupCog, name="agent"):
    """Configure and inspect the AI agent for this server."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="prompt", description="Set the system prompt that shapes the agent's behaviour.")
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(prompt="The new system prompt (max ~4000 chars).")
    async def set_prompt(self, interaction: discord.Interaction, prompt: str) -> None:
        if not await require_admin(interaction):
            return
        await interaction.response.defer(ephemeral=True)

        async with get_session() as session:
            result = await session.exec(
                select(GuildConfig).where(GuildConfig.guild_id == interaction.guild_id)
            )
            cfg = result.first()
            if cfg is None:
                cfg = GuildConfig(guild_id=interaction.guild_id)
                session.add(cfg)
            cfg.system_prompt = prompt
            await session.commit()

        await interaction.followup.send("System prompt updated.", ephemeral=True)

    @app_commands.command(name="show", description="Show the current agent configuration for this server.")
    @app_commands.guild_only()
    async def show(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)

        async with get_session() as session:
            result = await session.exec(
                select(GuildConfig).where(GuildConfig.guild_id == interaction.guild_id)
            )
            cfg = result.first()

        if cfg is None:
            await interaction.followup.send(
                "No configuration yet — use `/config` to pick a provider/model.", ephemeral=True
            )
            return

        prompt_preview = cfg.system_prompt[:800] + ("…" if len(cfg.system_prompt) > 800 else "")
        embed = discord.Embed(title="Agent Configuration", color=discord.Color.green())
        embed.add_field(name="Provider", value=cfg.provider, inline=True)
        embed.add_field(name="Model", value=cfg.model, inline=True)
        embed.add_field(name="Q&A", value="on" if cfg.qa_enabled else "off", inline=True)
        embed.add_field(name="AutoMod", value="on" if cfg.automod_enabled else "off", inline=True)
        embed.add_field(name="System Prompt", value=f"```\n{prompt_preview}\n```", inline=False)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="reset", description="Reset the system prompt to the default.")
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    async def reset(self, interaction: discord.Interaction) -> None:
        if not await require_admin(interaction):
            return
        await interaction.response.defer(ephemeral=True)

        async with get_session() as session:
            result = await session.exec(
                select(GuildConfig).where(GuildConfig.guild_id == interaction.guild_id)
            )
            cfg = result.first()
            if cfg is None:
                cfg = GuildConfig(guild_id=interaction.guild_id)
                session.add(cfg)
            cfg.system_prompt = DEFAULT_PROMPT
            await session.commit()

        await interaction.followup.send("System prompt reset to default.", ephemeral=True)

    @app_commands.command(name="clear", description="Clear the conversation history for this channel.")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_messages=True)
    async def clear_history(self, interaction: discord.Interaction) -> None:
        if not await require_permissions(interaction, manage_messages=True):
            return
        await conversation.clear(interaction.guild_id, interaction.channel_id)
        await interaction.response.send_message("Conversation history cleared.", ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AgentCog(bot))
