import logging
import math
import sys
import time

import discord
from discord.ext import commands, tasks

from .config import settings
from .db_migrate import run_migrations
from .healthcheck import HEARTBEAT_FILE
from .logging_config import setup_logging

setup_logging()
log = logging.getLogger("discord-ai-agent")


class AIAgentBot(commands.Bot):
    def __init__(self, sync_commands: bool = False) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self._sync_commands = sync_commands

    async def setup_hook(self) -> None:
        # Apply schema via Alembic (not create_all) so existing DBs evolve safely.
        run_migrations()
        await self.load_extension("bot.commands.admin")
        await self.load_extension("bot.commands.agent")
        await self.load_extension("bot.commands.ask")
        await self.load_extension("bot.commands.kb")
        await self.load_extension("bot.commands.automod")
        await self.load_extension("bot.commands.manage")
        # Discord rate-limits global command sync to ~2/day, so don't sync on every
        # startup. Sync explicitly with `--sync` when command signatures change.
        if self._sync_commands:
            await self.tree.sync()
            log.info("Slash commands synced.")
        else:
            log.info("Skipping command sync (pass --sync to register slash commands).")
        self._heartbeat.start()

    @tasks.loop(seconds=30)
    async def _heartbeat(self) -> None:
        """Touch the heartbeat file only while actually connected to Discord, so
        the container HEALTHCHECK reflects gateway connectivity, not just liveness."""
        if self.is_ready() and math.isfinite(self.latency):
            HEARTBEAT_FILE.write_text(str(time.time()))

    @_heartbeat.before_loop
    async def _before_heartbeat(self) -> None:
        await self.wait_until_ready()

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id)


def main() -> None:
    bot = AIAgentBot(sync_commands="--sync" in sys.argv)
    bot.run(settings.discord_bot_token, log_handler=None)


if __name__ == "__main__":
    main()
