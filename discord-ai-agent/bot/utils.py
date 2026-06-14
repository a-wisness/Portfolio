"""Shared helpers for command cogs."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Hashable

import discord

MSG_LIMIT = 1990  # 10-char buffer below Discord's 2000-char hard limit


def split_message(text: str, limit: int = MSG_LIMIT) -> list[str]:
    """Split text into chunks that fit Discord's per-message character limit."""
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


class RateLimiter:
    """Sliding-window rate limiter keyed on an arbitrary hashable (e.g. (guild, user)).

    Bounds how many LLM-triggering actions a single key may perform per window so
    one user can't spam `/ask`, mentions, or automod into a flood of API calls.
    """

    def __init__(self, max_calls: int, window: float) -> None:
        self.max_calls = max_calls
        self.window = window
        self._hits: dict[Hashable, deque[float]] = defaultdict(deque)

    def allow(self, key: Hashable) -> bool:
        """Record an attempt; return True if it's within the limit, False if throttled."""
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.max_calls:
            return False
        hits.append(now)
        return True


async def require_permissions(
    interaction: discord.Interaction, **required: bool
) -> bool:
    """Runtime permission guard for privileged slash commands.

    `default_permissions(...)` only sets the *default* gate shown in Discord's UI;
    server admins can override it in Integrations settings. This checks the
    invoking user's *actual* resolved permissions and cannot be bypassed that way.

    Returns True if the user holds every requested permission, otherwise sends an
    ephemeral denial and returns False. Pass e.g. ``administrator=True``.
    """
    perms = interaction.permissions
    missing = [name for name, needed in required.items() if needed and not getattr(perms, name, False)]
    if not missing:
        return True

    pretty = ", ".join(name.replace("_", " ").title() for name in missing)
    message = f"You need the following permission(s) to use this command: {pretty}."
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)
    return False


async def require_admin(interaction: discord.Interaction) -> bool:
    """Convenience guard for administrator-only commands."""
    return await require_permissions(interaction, administrator=True)
