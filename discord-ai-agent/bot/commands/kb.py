"""Knowledge base slash-command group — /kb add | search | delete | list."""
from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from ..modules import qa


class KBCog(commands.GroupCog, name="kb"):
    """Manage and search the server's AI knowledge base."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="add", description="Add an entry to the knowledge base.")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_messages=True)
    @app_commands.describe(
        title="Short title for this entry.",
        content="The knowledge to store (up to ~4000 chars; split longer docs into multiple entries).",
    )
    async def add(self, interaction: discord.Interaction, title: str, content: str) -> None:
        await interaction.response.defer(ephemeral=True)
        entry = await qa.add_entry(interaction.guild_id, title, content)
        await interaction.followup.send(
            f"Added KB entry **#{entry.id}**: {title}", ephemeral=True
        )

    @app_commands.command(name="search", description="Search the knowledge base.")
    @app_commands.guild_only()
    @app_commands.describe(query="Keywords or phrase to search for.")
    async def search(self, interaction: discord.Interaction, query: str) -> None:
        await interaction.response.defer()
        results = await qa.search(interaction.guild_id, query, limit=5)

        if not results:
            await interaction.followup.send("No matching entries found.")
            return

        embed = discord.Embed(
            title=f'KB results for "{query}"',
            color=discord.Color.blurple(),
        )
        for r in results:
            snippet = r["content"][:300] + ("…" if len(r["content"]) > 300 else "")
            embed.add_field(name=f"#{r['id']} — {r['title']}", value=snippet, inline=False)
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="delete", description="Delete a knowledge base entry by its ID.")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_messages=True)
    @app_commands.describe(entry_id="The numeric ID shown next to the entry title.")
    async def delete(self, interaction: discord.Interaction, entry_id: int) -> None:
        await interaction.response.defer(ephemeral=True)
        deleted = await qa.delete_entry(entry_id, interaction.guild_id)
        if deleted:
            await interaction.followup.send(f"Deleted KB entry #{entry_id}.", ephemeral=True)
        else:
            await interaction.followup.send(
                f"Entry #{entry_id} not found in this server's knowledge base.", ephemeral=True
            )

    @app_commands.command(name="list", description="List the most recent knowledge base entries.")
    @app_commands.guild_only()
    async def list_entries(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        entries = await qa.list_entries(interaction.guild_id, limit=15)

        if not entries:
            await interaction.followup.send("The knowledge base is empty. Use `/kb add` to add entries.")
            return

        embed = discord.Embed(title="Knowledge Base", color=discord.Color.green())
        for e in entries:
            snippet = e.content[:120] + ("…" if len(e.content) > 120 else "")
            embed.add_field(name=f"#{e.id} — {e.title}", value=snippet, inline=False)
        embed.set_footer(text=f"{len(entries)} most recent entries shown")
        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(KBCog(bot))
