"""Run Alembic migrations programmatically (used at bot startup)."""
from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALEMBIC_INI = _PROJECT_ROOT / "alembic.ini"


def run_migrations() -> None:
    """Upgrade the database to the latest revision. Safe to run on every startup."""
    cfg = Config(str(_ALEMBIC_INI))
    # Pin paths absolutely so this works regardless of the process CWD.
    cfg.set_main_option("script_location", str(_PROJECT_ROOT / "migrations"))
    log.info("Applying database migrations (alembic upgrade head)")
    command.upgrade(cfg, "head")
