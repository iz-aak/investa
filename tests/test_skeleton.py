"""Smoke tests for the step-1 skeleton.

These verify only that the plumbing is correct: settings load, both DBs
initialise, the FastAPI app boots, and /api/health returns a well-formed
envelope. Domain logic (indicators, signals, scoring) is tested in later
steps.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_settings_load_from_env(_isolated_data_dir):
    from investa.config import get_settings

    settings = get_settings()
    assert settings.app_env.value == "test"
    assert str(settings.sqlite_path).endswith("investa.sqlite")
    assert str(settings.duckdb_path).endswith("signals.duckdb")


def test_cors_origins_parsed_from_comma_string(_isolated_data_dir, monkeypatch):
    """pydantic-settings v2 must not JSON-decode CORS_ORIGINS — regression test."""
    from investa import config as config_mod

    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    config_mod.get_settings.cache_clear()

    settings = config_mod.get_settings()
    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_init_db_creates_sqlite_file(_isolated_data_dir):
    from investa.config import get_settings
    from investa.database import init_db

    init_db()
    assert get_settings().sqlite_path.exists()


def test_init_duckdb_creates_file_and_table(_isolated_data_dir):
    from investa.config import get_settings
    from investa.data.duckdb_store import connect, init_duckdb

    init_duckdb()
    assert get_settings().duckdb_path.exists()

    with connect(read_only=True) as con:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        assert "signal_history" in tables


def test_health_endpoint_returns_envelope(_isolated_data_dir):
    from investa.main import create_app

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200

        body = resp.json()
        assert body["success"] is True
        assert body["data"]["sqlite"] is True
        assert body["data"]["duckdb"] is True
        assert body["data"]["last_scan_at"] is None
        assert "generated_at" in body["meta"]
