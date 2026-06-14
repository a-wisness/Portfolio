"""Container healthcheck — verifies the bot is connected to Discord, not just alive.

The bot touches HEARTBEAT_FILE every ~30s, but only while its gateway connection
is ready. This script exits 0 if that file was updated recently, else 1 — so a
process that is running but disconnected reports unhealthy.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HEARTBEAT_FILE = Path(os.environ.get("HEARTBEAT_FILE", "/tmp/discord_bot_heartbeat"))
# Heartbeat is written every 30s; allow a couple of missed beats before failing.
MAX_AGE_SECONDS = 90


def is_healthy() -> bool:
    try:
        last = float(HEARTBEAT_FILE.read_text().strip())
    except (OSError, ValueError):
        return False
    return (time.time() - last) <= MAX_AGE_SECONDS


def main() -> int:
    healthy = is_healthy()
    if not healthy:
        print("unhealthy: heartbeat missing or stale", file=sys.stderr)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
