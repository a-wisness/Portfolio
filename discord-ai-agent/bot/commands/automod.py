"""Auto-moderation slash commands + on_message classifier listener."""
from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands
from sqlmodel import select

from ..config import settings
from ..database import GuildConfig, get_session
from ..logging_config import request_context
from ..modules import automod as mod
from ..services.guild import get_or_create_config
from ..utils import RateLimiter, require_admin, require_permissions

log = logging.getLogger(__name__)

# Caps automod LLM classifications per user so a message flood can't blow up cost.
_classify_limiter = RateLimiter(settings.automod_rate_limit, settings.automod_rate_window)


class AutomodCog(commands.GroupCog, name="automod"):
    """LLM-based message moderation: configure, enable, and review actions."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ------------------------------------------------------------------ #
    # Configuration commands
    # ------------------------------------------------------------------ #

    @app_commands.command(name="enable", description="Enable LLM-based auto-moderation for this server.")
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    async def enable(self, interaction: discord.Interaction) -> None:
        await self._set_enabled(interaction, True)

    @app_commands.command(name="disable", description="Disable auto-moderation for this server.")
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    async def disable(self, interaction: discord.Interaction) -> None:
        await self._set_enabled(interaction, False)

    async def _set_enabled(self, interaction: discord.Interaction, value: bool) -> None:
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
            cfg.automod_enabled = value
            await session.commit()
        await interaction.followup.send(
            f"Auto-moderation {'enabled' if value else 'disabled'}.", ephemeral=True
        )

    @app_commands.command(
        name="threshold",
        description="Set the severity threshold (0.0–1.0) above which action is taken.",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(value="0.0 = act on everything · 0.8 = only clear violations · 1.0 = only the most severe.")
    async def threshold(self, interaction: discord.Interaction, value: float) -> None:
        if not await require_admin(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        if not 0.0 <= value <= 1.0:
            await interaction.followup.send("Threshold must be between 0.0 and 1.0.", ephemeral=True)
            return
        async with get_session() as session:
            result = await session.exec(
                select(GuildConfig).where(GuildConfig.guild_id == interaction.guild_id)
            )
            cfg = result.first()
            if cfg is None:
                cfg = GuildConfig(guild_id=interaction.guild_id)
                session.add(cfg)
            cfg.automod_threshold = value
            await session.commit()
        await interaction.followup.send(f"Severity threshold set to **{value:.2f}**.", ephemeral=True)

    @app_commands.command(name="logchannel", description="Set the channel where moderation actions are posted.")
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(channel="Text channel to receive mod notices (omit to clear).")
    async def logchannel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel | None = None,
    ) -> None:
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
            cfg.mod_log_channel_id = channel.id if channel else None
            await session.commit()
        msg = f"Mod log channel set to {channel.mention}." if channel else "Mod log channel cleared."
        await interaction.followup.send(msg, ephemeral=True)

    @app_commands.command(name="status", description="Show the current auto-moderation configuration.")
    @app_commands.guild_only()
    async def status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        cfg = await get_or_create_config(interaction.guild_id)
        log_ch = f"<#{cfg.mod_log_channel_id}>" if cfg.mod_log_channel_id else "not set"
        embed = discord.Embed(title="Auto-Moderation Status", color=discord.Color.orange())
        embed.add_field(name="Enabled", value="yes" if cfg.automod_enabled else "no", inline=True)
        embed.add_field(name="Threshold", value=f"{cfg.automod_threshold:.2f}", inline=True)
        embed.add_field(name="Log Channel", value=log_ch, inline=True)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="log", description="View recent moderation actions for this server.")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_messages=True)
    @app_commands.describe(limit="Number of entries to show (1–20, default 10).")
    async def view_log(self, interaction: discord.Interaction, limit: int = 10) -> None:
        if not await require_permissions(interaction, manage_messages=True):
            return
        await interaction.response.defer(ephemeral=True)
        limit = max(1, min(limit, 20))
        entries = await mod.get_mod_logs(interaction.guild_id, limit)
        if not entries:
            await interaction.followup.send("No moderation actions logged yet.", ephemeral=True)
            return
        embed = discord.Embed(title="Moderation Log", color=discord.Color.red())
        for e in entries:
            ts = e.timestamp.strftime("%Y-%m-%d %H:%M UTC")
            embed.add_field(
                name=f"#{e.id} · <@{e.user_id}> · {ts}",
                value=f"**{e.action}**\n{e.reason[:200]}",
                inline=False,
            )
        embed.set_footer(text=f"Showing {len(entries)} most recent actions")
        await interaction.followup.send(embed=embed, ephemeral=True)

    # ------------------------------------------------------------------ #
    # Message classifier
    # ------------------------------------------------------------------ #

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild or not message.content.strip():
            return

        # Administrators are exempt from automod.
        if (
            isinstance(message.author, discord.Member)
            and message.author.guild_permissions.administrator
        ):
            return

        cfg = await get_or_create_config(message.guild.id)
        if not cfg.automod_enabled:
            return

        # Skip messages too short to classify reliably.
        if len(message.content.strip()) < 8:
            return

        # Throttle per user. A flood is capped; normal chatters never hit this.
        if not _classify_limiter.allow((message.guild.id, message.author.id)):
            log.debug("Automod rate limit hit for user %s in guild %s", message.author.id, message.guild.id)
            return

        try:
            with request_context(message.guild.id, message.id):
                result = await mod.classify_message(message.content, message.guild.id)
        except Exception:
            log.exception("Automod classification failed for message %s", message.id)
            return

        log.debug(
            "Automod [guild=%s msg=%s]: violation=%s severity=%.2f",
            message.guild.id, message.id, result.violation, result.severity,
        )

        if result.severity < cfg.automod_threshold:
            return

        action_label = result.violation if result.violation != "none" else "policy_violation"
        try:
            await message.delete()
            action_taken = f"delete_message ({action_label})"
        except discord.Forbidden:
            action_taken = f"flagged ({action_label})"
        except discord.NotFound:
            action_taken = f"delete_message ({action_label})"  # already gone

        reason = f"[{result.violation}] {result.reason} (severity={result.severity:.2f})"
        await mod.log_action(
            guild_id=message.guild.id,
            user_id=message.author.id,
            action=action_taken,
            reason=reason,
            message_content=message.content[:500],
        )

        if cfg.mod_log_channel_id:
            log_ch = message.guild.get_channel(cfg.mod_log_channel_id)
            if isinstance(log_ch, discord.TextChannel):
                embed = discord.Embed(
                    title="Message Removed",
                    color=discord.Color.red(),
                    description=f"**User:** <@{message.author.id}> · **Channel:** {message.channel.mention}",
                )
                embed.add_field(name="Violation", value=result.violation, inline=True)
                embed.add_field(name="Severity", value=f"{result.severity:.2f}", inline=True)
                embed.add_field(name="Reason", value=result.reason, inline=False)
                if message.content:
                    snippet = message.content[:300] + ("…" if len(message.content) > 300 else "")
                    embed.add_field(name="Content", value=f"```{snippet}```", inline=False)
                try:
                    await log_ch.send(embed=embed)
                except discord.Forbidden:
                    log.warning("No permission to post in mod log channel %s", cfg.mod_log_channel_id)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AutomodCog(bot))
