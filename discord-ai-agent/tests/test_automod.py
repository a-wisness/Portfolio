"""Tests for bot/modules/automod.py — classification parsing and DB operations."""
from bot.modules.automod import (
    _parse_result,
    get_mod_logs,
    log_action,
)

G = 800


# ------------------------------------------------------------------ #
# _parse_result — pure, no I/O
# ------------------------------------------------------------------ #

def test_parse_clean_json():
    r = _parse_result('{"severity": 0.9, "violation": "harassment", "reason": "Direct insult."}')
    assert r.severity == 0.9
    assert r.violation == "harassment"
    assert r.reason == "Direct insult."
    assert r.is_violation


def test_parse_benign():
    r = _parse_result('{"severity": 0.0, "violation": "none", "reason": "Normal message."}')
    assert r.severity == 0.0
    assert not r.is_violation


def test_parse_severity_clamped_above_one():
    r = _parse_result('{"severity": 2.5, "violation": "spam", "reason": "Too much."}')
    assert r.severity == 1.0


def test_parse_severity_clamped_below_zero():
    r = _parse_result('{"severity": -0.5, "violation": "none", "reason": "Negative."}')
    assert r.severity == 0.0


def test_parse_json_in_code_fence():
    raw = '```json\n{"severity": 0.8, "violation": "hate_speech", "reason": "Slur used."}\n```'
    r = _parse_result(raw)
    assert r.severity == 0.8
    assert r.violation == "hate_speech"


def test_parse_plain_code_fence():
    raw = '```\n{"severity": 0.3, "violation": "none", "reason": "Borderline."}\n```'
    r = _parse_result(raw)
    assert r.severity == 0.3


def test_parse_invalid_json_returns_safe_default():
    r = _parse_result("I cannot classify this message.")
    assert r.severity == 0.0
    assert r.violation == "none"
    assert not r.is_violation


def test_parse_missing_fields_defaults():
    r = _parse_result('{"severity": 0.5}')
    assert r.violation == "none"
    assert r.reason == ""


# ------------------------------------------------------------------ #
# log_action / get_mod_logs — requires test DB (via _init_db fixture)
# ------------------------------------------------------------------ #

async def test_log_action_persists():
    entry = await log_action(G, 99001, "delete_message (spam)", "Spam detected.", "buy now!!!")
    assert entry.id is not None
    assert entry.guild_id == G
    assert entry.user_id == 99001
    assert entry.action == "delete_message (spam)"
    assert entry.message_content == "buy now!!!"


async def test_get_mod_logs_most_recent_first():
    await log_action(G + 1, 111, "flagged (harassment)", "Insult", "bad word")
    await log_action(G + 1, 222, "delete_message (spam)", "Spam", "buy now")
    logs = await get_mod_logs(G + 1, limit=10)
    assert logs[0].user_id == 222  # most recent
    assert logs[1].user_id == 111


async def test_get_mod_logs_limit():
    for i in range(5):
        await log_action(G + 2, i, "flagged (spam)", f"reason {i}", None)
    logs = await get_mod_logs(G + 2, limit=3)
    assert len(logs) == 3


async def test_get_mod_logs_empty_guild():
    logs = await get_mod_logs(G + 3, limit=10)
    assert logs == []


async def test_log_action_no_message_content():
    entry = await log_action(G + 4, 55555, "flagged (nsfw)", "Image detected.", None)
    assert entry.message_content is None
