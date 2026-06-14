"""Tests for bot/modules/manager.py — tool suggestion mode and DB config operations."""
from bot.database import GuildConfig
from bot.modules.manager import make_channel_tools, make_role_tools
from bot.services.guild import get_or_create_config

G = 900


# ------------------------------------------------------------------ #
# Minimal fake Discord objects (no real Discord connection needed)
# ------------------------------------------------------------------ #

class FakeRole:
    def __init__(self, id, name, managed=False, default=False, position=1):
        self.id = id
        self.name = name
        self.managed = managed
        self._default = default
        self.position = position

    def is_default(self) -> bool:
        return self._default


class FakeMember:
    def __init__(self, id, display_name):
        self.id = id
        self.display_name = display_name


class FakeChannel:
    def __init__(self, id, name, category=None):
        self.id = id
        self.name = name
        self.category = category


class FakeCategory:
    def __init__(self, id, name, channels=None):
        self.id = id
        self.name = name
        self.text_channels = channels or []


class FakeGuild:
    def __init__(self):
        self.members = [FakeMember(1, "Alice"), FakeMember(2, "Bob")]
        self.roles = [
            FakeRole(100, "Moderator", position=2),
            FakeRole(101, "Member", position=1),
        ]
        self.categories = [
            FakeCategory(200, "General", [FakeChannel(500, "general")])
        ]
        self.text_channels = []  # channels listed separately from categories
        self.default_role = object()

    def get_member(self, uid):
        return next((m for m in self.members if m.id == uid), None)

    def get_role(self, rid):
        return next((r for r in self.roles if r.id == rid), None)

    def get_channel(self, cid):
        for cat in self.categories:
            for ch in cat.text_channels:
                if ch.id == cid:
                    return ch
        return next((ch for ch in self.text_channels if ch.id == cid), None)


# ------------------------------------------------------------------ #
# Role tools — suggestion mode (opt-in disabled)
# ------------------------------------------------------------------ #

async def test_assign_role_suggestion_when_disabled():
    cfg = GuildConfig(guild_id=G, allow_role_assign=False)
    tool_map = {t.name: f for t, f in make_role_tools(FakeGuild(), cfg)}
    result = await tool_map["assign_role"](user_id=1, role_id=100, reason="They helped a lot")
    assert "[SUGGESTION]" in result
    assert "role_assign" in result
    assert "Moderator" in result
    assert "Alice" in result


async def test_remove_role_suggestion_when_disabled():
    cfg = GuildConfig(guild_id=G, allow_role_remove=False)
    tool_map = {t.name: f for t, f in make_role_tools(FakeGuild(), cfg)}
    result = await tool_map["remove_role"](user_id=2, role_id=101, reason="Inactive")
    assert "[SUGGESTION]" in result
    assert "role_remove" in result
    assert "Member" in result
    assert "Bob" in result


async def test_list_roles_returns_assignable_roles():
    cfg = GuildConfig(guild_id=G)
    tool_map = {t.name: f for t, f in make_role_tools(FakeGuild(), cfg)}
    result = await tool_map["list_roles"]()
    assert "Moderator" in result
    assert "Member" in result
    assert "100" in result
    assert "101" in result


async def test_list_roles_excludes_default_and_managed():
    guild = FakeGuild()
    guild.roles = [
        FakeRole(1, "@everyone", default=True),
        FakeRole(2, "BotRole", managed=True),
        FakeRole(3, "Admin", position=3),
    ]
    cfg = GuildConfig(guild_id=G)
    tool_map = {t.name: f for t, f in make_role_tools(guild, cfg)}
    result = await tool_map["list_roles"]()
    assert "@everyone" not in result
    assert "BotRole" not in result
    assert "Admin" in result


async def test_find_member_by_name():
    cfg = GuildConfig(guild_id=G)
    tool_map = {t.name: f for t, f in make_role_tools(FakeGuild(), cfg)}
    result = await tool_map["find_member"](query="alice")
    assert "Alice" in result
    assert "1" in result


async def test_find_member_by_id():
    cfg = GuildConfig(guild_id=G)
    tool_map = {t.name: f for t, f in make_role_tools(FakeGuild(), cfg)}
    result = await tool_map["find_member"](query="2")
    assert "Bob" in result


async def test_find_member_not_found():
    cfg = GuildConfig(guild_id=G)
    tool_map = {t.name: f for t, f in make_role_tools(FakeGuild(), cfg)}
    result = await tool_map["find_member"](query="nobody")
    assert "No members found" in result


async def test_assign_role_member_not_found():
    cfg = GuildConfig(guild_id=G, allow_role_assign=True)
    tool_map = {t.name: f for t, f in make_role_tools(FakeGuild(), cfg)}
    result = await tool_map["assign_role"](user_id=9999, role_id=100, reason="Test")
    assert "not found" in result


async def test_assign_role_role_not_found():
    cfg = GuildConfig(guild_id=G, allow_role_assign=True)
    tool_map = {t.name: f for t, f in make_role_tools(FakeGuild(), cfg)}
    result = await tool_map["assign_role"](user_id=1, role_id=9999, reason="Test")
    assert "not found" in result


# ------------------------------------------------------------------ #
# Channel tools — suggestion mode (opt-in disabled)
# ------------------------------------------------------------------ #

async def test_create_channel_suggestion_when_disabled():
    cfg = GuildConfig(guild_id=G, allow_channel_create=False)
    tool_map = {t.name: f for t, f in make_channel_tools(FakeGuild(), cfg)}
    result = await tool_map["create_channel"](name="new-channel")
    assert "[SUGGESTION]" in result
    assert "channel_create" in result
    assert "new-channel" in result


async def test_archive_channel_suggestion_when_disabled():
    cfg = GuildConfig(guild_id=G, allow_channel_archive=False)
    tool_map = {t.name: f for t, f in make_channel_tools(FakeGuild(), cfg)}
    result = await tool_map["archive_channel"](channel_id=500, reason="Inactive")
    assert "[SUGGESTION]" in result
    assert "channel_archive" in result
    assert "general" in result


async def test_archive_channel_not_found():
    cfg = GuildConfig(guild_id=G, allow_channel_archive=True)
    tool_map = {t.name: f for t, f in make_channel_tools(FakeGuild(), cfg)}
    result = await tool_map["archive_channel"](channel_id=9999, reason="Test")
    assert "not found" in result


async def test_list_channels_returns_channel_info():
    cfg = GuildConfig(guild_id=G)
    tool_map = {t.name: f for t, f in make_channel_tools(FakeGuild(), cfg)}
    result = await tool_map["list_channels"]()
    assert "General" in result
    assert "general" in result
    assert "500" in result


async def test_list_channels_empty_guild():
    guild = FakeGuild()
    guild.categories = []
    guild.text_channels = []
    cfg = GuildConfig(guild_id=G)
    tool_map = {t.name: f for t, f in make_channel_tools(guild, cfg)}
    result = await tool_map["list_channels"]()
    assert "No text channels found" in result


# ------------------------------------------------------------------ #
# DB: new opt-in fields default to False
# ------------------------------------------------------------------ #

async def test_new_guild_config_defaults():
    cfg = await get_or_create_config(G + 50)
    assert cfg.allow_role_assign is False
    assert cfg.allow_role_remove is False
    assert cfg.allow_channel_create is False
    assert cfg.allow_channel_archive is False


async def test_opt_in_fields_persist():
    from bot.database import get_session
    from sqlmodel import select

    gid = G + 51
    await get_or_create_config(gid)
    async with get_session() as session:
        result = await session.exec(select(GuildConfig).where(GuildConfig.guild_id == gid))
        cfg = result.first()
        cfg.allow_role_assign = True
        cfg.allow_channel_create = True
        await session.commit()

    reloaded = await get_or_create_config(gid)
    assert reloaded.allow_role_assign is True
    assert reloaded.allow_channel_create is True
    assert reloaded.allow_role_remove is False
    assert reloaded.allow_channel_archive is False
