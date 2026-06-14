"""Structured JSON logging with per-request guild/request context.

A flat `%(message)s` format makes multi-guild stack traces impossible to
correlate. This emits one JSON object per line, automatically tagged with the
`guild_id` / `request_id` of the in-flight interaction (via context vars), and
includes any structured `extra=` fields (e.g. LLM token usage).
"""
from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_guild_id: ContextVar[Optional[int]] = ContextVar("guild_id", default=None)
_request_id: ContextVar[Optional[int]] = ContextVar("request_id", default=None)

# Standard LogRecord attributes — everything else on a record is a custom `extra`.
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class ContextFilter(logging.Filter):
    """Attach the current guild_id / request_id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.guild_id = _guild_id.get()
        record.request_id = _request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "guild_id": getattr(record, "guild_id", None),
            "request_id": getattr(record, "request_id", None),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in data:
                data[key] = value
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


@contextmanager
def request_context(
    guild_id: Optional[int] = None, request_id: Optional[int] = None
) -> Iterator[None]:
    """Bind guild/request identifiers for the duration of a handler."""
    g_token = _guild_id.set(guild_id)
    r_token = _request_id.set(request_id)
    try:
        yield
    finally:
        _guild_id.reset(g_token)
        _request_id.reset(r_token)
