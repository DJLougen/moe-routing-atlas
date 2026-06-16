"""ASGI entrypoint for uvicorn — reads DB path from environment."""

from __future__ import annotations

import os
from pathlib import Path

from .backend import create_app

_default_db = Path.home() / ".moe-atlas" / "atlas.db"
_dev_db = Path(__file__).resolve().parents[2] / "dev.db"
if "MOE_ATLAS_DB_PATH" in os.environ:
    _db_path = os.environ["MOE_ATLAS_DB_PATH"]
elif _dev_db.exists():
    _db_path = str(_dev_db)
else:
    _db_path = str(_default_db)

app = create_app(db_path=_db_path)