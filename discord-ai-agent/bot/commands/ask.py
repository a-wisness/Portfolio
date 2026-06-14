"""Conversational agent — /ask command and @mention handling."""
from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from .. import history as conversation
from ..config import settings
from ..llm.provider import Message
from ..logging_config import request_context
from ..modules import qa
from ..services.guild import get_or_create_config, make_agent
from ..tools.kb import search_kb_tool
from ..utils import RateLimiter, split_message

log = logging.getLogger(__name__)

# Per-user throttle shared by /ask and the @mention handler.
_limiter = RateLimiter(settings.ask_rate_limit, settings.ask_rate_window)


def _make_kb_fn(guild_id: int):
    """Return an async tool function that searches the guild's KB."""
    async def _search(query: str) -> str:
        results = await qa.search(guild_id, query, limit=5)
        if not results:
            return "No relevant knowledge base entries found."
        return "\n\n---\n\n".join(f"**{r['title']}**\n{r['content']}" for r in results)
    return _search


async def _invoke(guild_id: int, channel_id: int, prompt: str) -> str:
    """Core ask logic: load guild config, arm tools, run agent, update history."""
    cfg = await get_or_create_config(guild_id)

    tools = [(search_kb_tool, _make_kb_fn(guild_id))] if cfg.qa_enabled else []
    agent = make_agent(cfg, tools)

    history = await conversation.get(guild_id, channel_id)
    history.append(Message(role="user", content=prompt))

    response = await agent.run(history)
    if response:
        await conversation.add_turn(guild_id, channel_id, prompt, response)
    return response or "(no response)"


class AskCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="ask", description="Ask the AI agent a question.")
    @app_commands.guild_only()
    @app_commands.describe(prompt="Your question or message for the agent.")
    async def ask(self, interaction: discord.Interaction, prompt: str) -> None:
        if not _limiter.allow((interaction.guild_id, interaction.user.id)):
            await interaction.response.send_message(
                "You're sending requests too quickly — please wait a moment and try again.",
                ephemeral=True,
            )
            return
        await interaction.response.defer()
        try:
            with request_context(interaction.guild_id, interaction.id):
                response = await _invoke(interaction.guild_id, interaction.channel_id, prompt)
        except Exception:
            log.exception("Agent error in /ask")
            await interaction.followup.send(
                "Sorry — the agent hit an error while handling your request. Please try again.",
                ephemeral=True,
            )
            return

        chunks = split_message(response)
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return
        if self.bot.user not in message.mentions:
            return

        prompt = message.content
        for pat in (f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"):
            prompt = prompt.replace(pat, "")
        prompt = prompt.strip()

        if not prompt:
            await message.reply("What can I help you with?")
            return

        if not _limiter.allow((message.guild.id, message.author.id)):
            log.debug("Rate-limited mention from user %s in guild %s", message.author.id, message.guild.id)
            return

        async with message.channel.typing():
            try:
                with request_context(message.guild.id, message.id):
                    response = await _invoke(message.guild.id, message.channel.id, prompt)
            except Exception:
                log.exception("Agent error in mention handler")
                await message.reply(
                    "Sorry — I hit an error while handling that. Please try again."
                )
                return

        chunks = split_message(response)
        await message.reply(chunks[0])
        for chunk in chunks[1:]:
            await message.channel.send(chunk)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AskCog(bot))
