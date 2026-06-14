"""Role & Channel Management — tool closures for the management agent."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord

from ..database import GuildConfig
from ..llm.provider import Tool
from ..tools.management import (
    archive_channel_tool,
    assign_role_tool,
    create_channel_tool,
    find_member_tool,
    list_channels_tool,
    list_roles_tool,
    remove_role_tool,
)
from ..tools.moderation import delete_message as delete_message_tool
from ..tools.moderation import warn_user as warn_user_tool

if TYPE_CHECKING:
    from ..agents.base import ToolFn

log = logging.getLogger(__name__)

_MANAGER_SYSTEM = (
    "You are a server management assistant for a Discord guild. "
    "Use the available tools to look up members, roles, and channels, "
    "then carry out the administrator's request.\n\n"
    "If a tool returns a line starting with [SUGGESTION], the action is not yet "
    "enabled. Report the suggestion to the administrator and tell them which "
    "`/manage enable` subcommand would activate it.\n\n"
    "Be concise: confirm what was done, or explain clearly why it could not be done."
)

# Shown when full member list may be unavailable (GUILD_MEMBERS privileged intent is off).
_MEMBERS_HINT = (
    "Note: member search requires the GUILD_MEMBERS privileged intent. "
    "Enable it in the Discord developer portal for full results, "
    "or search by numeric user ID."
)


def make_role_tools(
    guild: discord.Guild, cfg: GuildConfig
) -> list[tuple[Tool, ToolFn]]:
    """Return (Tool, async_fn) pairs for role management, capturing guild + cfg."""

    async def list_roles() -> str:
        roles = sorted(
            [r for r in guild.roles if not r.is_default() and not r.managed],
            key=lambda r: r.position,
            reverse=True,
        )
        if not roles:
            return "No assignable roles found."
        return "\n".join(f"ID: {r.id} | @{r.name}" for r in roles[:50])

    async def find_member(query: str) -> str:
        q = query.strip().lower()
        matches = [
            f"ID: {m.id} | {m.display_name}"
            for m in guild.members
            if q in m.display_name.lower() or q == str(m.id)
        ]
        if not matches:
            return f"No members found matching '{query}'. {_MEMBERS_HINT}"
        return "\n".join(matches[:10])

    async def assign_role(user_id: int, role_id: int, reason: str) -> str:
        member = guild.get_member(user_id)
        role = guild.get_role(role_id)
        role_name = role.name if role else str(role_id)
        member_name = member.display_name if member else str(user_id)

        if not cfg.allow_role_assign:
            return (
                f"[SUGGESTION] Would assign @{role_name} to {member_name}. "
                "Enable this action with `/manage enable role_assign`."
            )
        if not member:
            return f"Member {user_id} not found in this guild."
        if not role:
            return f"Role {role_id} not found in this guild."
        try:
            await member.add_roles(role, reason=f"[AI Manager] {reason}")
            return f"Assigned @{role_name} to {member_name}."
        except discord.Forbidden:
            return "Bot lacks permission to assign this role (check role hierarchy)."

    async def remove_role(user_id: int, role_id: int, reason: str) -> str:
        member = guild.get_member(user_id)
        role = guild.get_role(role_id)
        role_name = role.name if role else str(role_id)
        member_name = member.display_name if member else str(user_id)

        if not cfg.allow_role_remove:
            return (
                f"[SUGGESTION] Would remove @{role_name} from {member_name}. "
                "Enable this action with `/manage enable role_remove`."
            )
        if not member:
            return f"Member {user_id} not found in this guild."
        if not role:
            return f"Role {role_id} not found in this guild."
        try:
            await member.remove_roles(role, reason=f"[AI Manager] {reason}")
            return f"Removed @{role_name} from {member_name}."
        except discord.Forbidden:
            return "Bot lacks permission to remove this role (check role hierarchy)."

    return [
        (list_roles_tool, list_roles),
        (find_member_tool, find_member),
        (assign_role_tool, assign_role),
        (remove_role_tool, remove_role),
    ]


def make_channel_tools(
    guild: discord.Guild, cfg: GuildConfig
) -> list[tuple[Tool, ToolFn]]:
    """Return (Tool, async_fn) pairs for channel management, capturing guild + cfg."""

    async def list_channels() -> str:
        lines: list[str] = []
        for cat in guild.categories:
            lines.append(f"Category: {cat.name} (ID: {cat.id})")
            for ch in cat.text_channels:
                lines.append(f"  #{ch.name} (ID: {ch.id})")
        uncategorized = [ch for ch in guild.text_channels if ch.category is None]
        for ch in uncategorized:
            lines.append(f"#{ch.name} (ID: {ch.id}) [no category]")
        return "\n".join(lines) if lines else "No text channels found."

    async def create_channel(
        name: str, topic: str = "", category_name: str = ""
    ) -> str:
        if not cfg.allow_channel_create:
            return (
                f"[SUGGESTION] Would create text channel #{name}. "
                "Enable this action with `/manage enable channel_create`."
            )
        category = None
        if category_name:
            category = discord.utils.get(guild.categories, name=category_name)
        try:
            ch = await guild.create_text_channel(
                name=name,
                topic=topic or "",
                category=category,
                reason="[AI Manager] Channel created by management agent.",
            )
            return f"Created #{ch.name} (ID: {ch.id})."
        except discord.Forbidden:
            return "Bot lacks permission to create channels."
        except discord.HTTPException as exc:
            return f"Failed to create channel: {exc}"

    async def archive_channel(channel_id: int, reason: str) -> str:
        ch = guild.get_channel(channel_id)
        if ch is None:
            return f"Channel {channel_id} not found."

        if not cfg.allow_channel_archive:
            return (
                f"[SUGGESTION] Would archive #{ch.name} "
                f"(rename to archive-{ch.name} and lock for @everyone). "
                "Enable this action with `/manage enable channel_archive`."
            )
        old_name = ch.name
        new_name = f"archive-{old_name}"
        try:
            await ch.edit(name=new_name, reason=f"[AI Manager] {reason}")
            overwrite = ch.overwrites_for(guild.default_role)
            overwrite.send_messages = False
            await ch.set_permissions(
                guild.default_role,
                overwrite=overwrite,
                reason="[AI Manager] Channel archived.",
            )
            return f"Archived #{old_name} → #{new_name} and locked for @everyone."
        except discord.Forbidden:
            return "Bot lacks permission to edit this channel."
        except discord.HTTPException as exc:
            return f"Failed to archive channel: {exc}"

    return [
        (list_channels_tool, list_channels),
        (create_channel_tool, create_channel),
        (archive_channel_tool, archive_channel),
    ]


def make_moderation_tools(guild: discord.Guild) -> list[tuple[Tool, ToolFn]]:
    """Return (Tool, async_fn) pairs for moderation actions, capturing the guild.

    Gated by the admin-only `/manage moderate` command rather than a per-action
    opt-in flag; every action is recorded to the ModLog.
    """
    from .automod import log_action

    async def warn_user(user_id: int, reason: str) -> str:
        member = guild.get_member(user_id)
        name = member.display_name if member else str(user_id)
        await log_action(guild.id, user_id, "warn (agent)", reason, None)
        if member is not None:
            try:
                await member.send(f"⚠️ You've received a warning in **{guild.name}**: {reason}")
            except (discord.Forbidden, discord.HTTPException):
                pass  # DMs closed — the warning is still logged.
        return f"Warned {name}: {reason}"

    async def delete_message(channel_id: int, message_id: int, reason: str) -> str:
        channel = guild.get_channel(channel_id)
        if channel is None:
            return f"Channel {channel_id} not found."
        try:
            msg = await channel.fetch_message(message_id)
        except discord.NotFound:
            return f"Message {message_id} not found in #{getattr(channel, 'name', channel_id)}."
        except discord.Forbidden:
            return "Bot lacks permission to read that channel."
        try:
            await msg.delete()
        except discord.Forbidden:
            return "Bot lacks permission to delete that message."
        await log_action(
            guild.id, msg.author.id, "delete_message (agent)", reason,
            msg.content[:500] if msg.content else None,
        )
        return f"Deleted message {message_id} in #{getattr(channel, 'name', channel_id)}."

    return [
        (warn_user_tool, warn_user),
        (delete_message_tool, delete_message),
    ]
