import os
import tempfile

# Must be set before any bot module is imported — Settings() runs at import time.
os.environ.setdefault("DISCORD_BOT_TOKEN", "test_token_placeholder")
os.environ.setdefault("ANTHROPIC_API_KEY", "test_key_placeholder")
os.environ.setdefault("OPENAI_API_KEY", "test_key_placeholder")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def _isolated_db():
    """Give every test its own fresh on-disk SQLite DB.

    Function-scoped (not session-scoped) so tests can't pollute each other via
    shared rows / reused guild IDs. mkstemp() (not the deprecated, race-prone
    mktemp()) creates the file; it's disposed and removed after each test.
    """
    from bot import database

    fd, path = tempfile.mkstemp(suffix="-test.db")
    os.close(fd)
    await database.init_db(f"sqlite+aiosqlite:///{path}")
    try:
        yield
    finally:
        await database.get_engine().dispose()
        os.remove(path)
