"""Tests for Phase R2 — utils, moderation tools, updated_at, default-model sourcing."""
from sqlmodel import select

from bot.config import settings
from bot.database import GuildConfig, get_session
from bot.modules.automod import get_mod_logs
from bot.modules.manager import make_moderation_tools
from bot.services.guild import get_or_create_config
from bot.utils import split_message

G = 700


# ------------------------------------------------------------------ #
# CR15 — split_message
# ------------------------------------------------------------------ #

def test_split_short_text_single_chunk():
    assert split_message("hello") == ["hello"]


def test_split_long_text_respects_limit():
    text = "x" * 4100
    chunks = split_message(text)
    assert len(chunks) == 3
    assert all(len(c) <= 1990 for c in chunks)
    assert "".join(chunks) == text


# ------------------------------------------------------------------ #
# CR21 — GuildConfig defaults come from settings (single source)
# ------------------------------------------------------------------ #

def test_guildconfig_defaults_from_settings():
    cfg = GuildConfig(guild_id=1)
    assert cfg.provider == settings.default_provider
    assert cfg.model == settings.default_model


# ------------------------------------------------------------------ #
# CR17 — updated_at advances on mutation
# ------------------------------------------------------------------ #

async def test_updated_at_changes_on_mutation():
    gid = G + 80
    await get_or_create_config(gid)
    # Read the persisted timestamp, mutate, then re-read from a fresh session so both
    # values come back from SQLite the same way (naive) and are comparable.
    async with get_session() as session:
        cfg = (await session.exec(select(GuildConfig).where(GuildConfig.guild_id == gid))).first()
        before = cfg.updated_at
        cfg.system_prompt = "a different prompt"
        await session.commit()
    async with get_session() as session:
        reloaded = (await session.exec(select(GuildConfig).where(GuildConfig.guild_id == gid))).first()
        after = reloaded.updated_at
    assert after > before


# ------------------------------------------------------------------ #
# CR16 — moderation tools are wired and perform/log actions
# ------------------------------------------------------------------ #

class FakeMember:
    def __init__(self, id, display_name):
        self.id = id
        self.display_name = display_name
        self.dms: list[str] = []

    async def send(self, content):
        self.dms.append(content)


class FakeMessage:
    def __init__(self, author, content):
        self.author = author
        self.content = content
        self.deleted = False

    async def delete(self):
        self.deleted = True


class FakeChannel:
    def __init__(self, id, name, message=None):
        self.id = id
        self.name = name
        self._message = message

    async def fetch_message(self, message_id):
        return self._message


class FakeGuild:
    def __init__(self, members=None, channels=None):
        self.id = G + 90
        self.name = "Test Guild"
        self._members = {m.id: m for m in (members or [])}
        self._channels = {c.id: c for c in (channels or [])}

    def get_member(self, uid):
        return self._members.get(uid)

    def get_channel(self, cid):
        return self._channels.get(cid)


def _tool_map(pairs):
    return {t.name: fn for t, fn in pairs}


async def test_warn_user_logs_and_dms():
    member = FakeMember(4242, "Spammer")
    guild = FakeGuild(members=[member])
    tools = _tool_map(make_moderation_tools(guild))

    result = await tools["warn_user"](user_id=4242, reason="posting spam")

    assert "Warned Spammer" in result
    assert member.dms and "posting spam" in member.dms[0]
    logs = await get_mod_logs(guild.id, limit=5)
    assert any(e.action == "warn (agent)" and e.user_id == 4242 for e in logs)


async def test_delete_message_deletes_and_logs():
    author = FakeMember(777, "Author")
    msg = FakeMessage(author=author, content="bad content")
    channel = FakeChannel(id=555, name="general", message=msg)
    guild = FakeGuild(channels=[channel])
    tools = _tool_map(make_moderation_tools(guild))

    result = await tools["delete_message"](channel_id=555, message_id=999, reason="rule break")

    assert msg.deleted is True
    assert "Deleted message 999" in result
    logs = await get_mod_logs(guild.id, limit=5)
    assert any(e.action == "delete_message (agent)" for e in logs)


async def test_delete_message_channel_not_found():
    guild = FakeGuild(channels=[])
    tools = _tool_map(make_moderation_tools(guild))
    result = await tools["delete_message"](channel_id=123, message_id=456, reason="x")
    assert "not found" in result
