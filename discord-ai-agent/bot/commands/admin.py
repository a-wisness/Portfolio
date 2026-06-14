"""Admin slash commands — configure the bot per guild."""
from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands
from sqlmodel import select

from ..database import GuildConfig, get_session
from ..llm.provider import ALLOWED_MODELS, is_valid_model, is_valid_provider
from ..utils import require_admin


class AdminCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="ping", description="Check if the bot is alive.")
    async def ping(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            f"Pong! Latency: {round(self.bot.latency * 1000)}ms"
        )

    @app_commands.command(name="config", description="View or update bot configuration for this server.")
    @app_commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    @app_commands.choices(
        provider=[
            app_commands.Choice(name="Anthropic", value="anthropic"),
            app_commands.Choice(name="OpenAI", value="openai"),
        ]
    )
    async def config_cmd(
        self,
        interaction: discord.Interaction,
        provider: str | None = None,
        model: str | None = None,
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

            # Validate against the allowlist before persisting anything.
            if provider is not None and not is_valid_provider(provider):
                allowed = ", ".join(ALLOWED_MODELS)
                await interaction.followup.send(
                    f"Unknown provider `{provider}`. Choose one of: {allowed}.", ephemeral=True
                )
                return

            target_provider = provider or cfg.provider
            if model is not None and not is_valid_model(target_provider, model):
                allowed = ", ".join(ALLOWED_MODELS.get(target_provider, ())) or "(none)"
                await interaction.followup.send(
                    f"Unknown model `{model}` for provider `{target_provider}`. "
                    f"Valid models: {allowed}.",
                    ephemeral=True,
                )
                return

            if provider:
                cfg.provider = provider
            if model:
                cfg.model = model

            await session.commit()

        embed = discord.Embed(title="Bot Configuration", color=discord.Color.blue())
        embed.add_field(name="Provider", value=cfg.provider, inline=True)
        embed.add_field(name="Model", value=cfg.model, inline=True)
        embed.add_field(name="Q&A", value="on" if cfg.qa_enabled else "off", inline=True)
        embed.add_field(name="AutoMod", value="on" if cfg.automod_enabled else "off", inline=True)
        await interaction.followup.send(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AdminCog(bot))
