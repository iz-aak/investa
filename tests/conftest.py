"""Shared pytest fixtures.

Isolates each test into a fresh temp data directory so SQLite/DuckDB files
never bleed between tests or into the developer's real ``./data`` folder.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("SQLITE_PATH", str(data_dir / "investa.sqlite"))
    monkeypatch.setenv("DUCKDB_PATH", str(data_dir / "signals.duckdb"))
    monkeypatch.setenv("UNIVERSES_DIR", str(data_dir / "universes"))
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    # Reset cached singletons so each test picks up the fresh env.
    from investa import config as config_mod
    from investa import database as database_mod

    config_mod.get_settings.cache_clear()
    database_mod._engine = None
    database_mod._SessionLocal = None

    # Prevent leaking test env into the actual working dir's .env
    os.chdir(tmp_path)
    return data_dir
