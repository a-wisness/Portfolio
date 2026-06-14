"""Tests for bot/logging_config.py — CR29 structured logging + request context."""
import json
import logging

from bot.logging_config import ContextFilter, JsonFormatter, request_context


def _format(record: logging.LogRecord) -> dict:
    ContextFilter().filter(record)
    return json.loads(JsonFormatter().format(record))


def _record(msg="hi", **extra) -> logging.LogRecord:
    rec = logging.LogRecord("test", logging.INFO, __file__, 1, msg, (), None)
    rec.__dict__.update(extra)
    return rec


def test_json_formatter_emits_core_fields():
    out = _format(_record("hello"))
    assert out["message"] == "hello"
    assert out["level"] == "INFO"
    assert out["logger"] == "test"


def test_context_defaults_to_none_outside_request():
    out = _format(_record())
    assert out["guild_id"] is None
    assert out["request_id"] is None


def test_request_context_is_injected():
    with request_context(guild_id=42, request_id=999):
        out = _format(_record())
    assert out["guild_id"] == 42
    assert out["request_id"] == 999


def test_request_context_resets_after_exit():
    with request_context(guild_id=42, request_id=999):
        pass
    out = _format(_record())
    assert out["guild_id"] is None


def test_extra_fields_are_serialized():
    out = _format(_record("LLM call", provider="anthropic", input_tokens=10, latency_ms=120))
    assert out["provider"] == "anthropic"
    assert out["input_tokens"] == 10
    assert out["latency_ms"] == 120
