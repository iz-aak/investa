"""Health check endpoint. Reports DB reachability and last scan time."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from ..data.duckdb_store import connect as duckdb_connect
from ..data.models import ScanJob
from ..database import session_scope
from .envelope import ok

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> dict:
    sqlite_ok = False
    duckdb_ok = False
    last_scan_at = None

    # SQLite check
    try:
        with session_scope() as db:
            stmt = select(ScanJob).order_by(ScanJob.started_at.desc()).limit(1)
            row = db.execute(stmt).scalar_one_or_none()
            if row is not None:
                last_scan_at = (row.finished_at or row.started_at).isoformat()
        sqlite_ok = True
    except Exception:  # noqa: BLE001 — health check must never raise
        sqlite_ok = False

    # DuckDB check
    try:
        with duckdb_connect(read_only=True) as con:
            con.execute("SELECT 1").fetchone()
        duckdb_ok = True
    except Exception:  # noqa: BLE001
        duckdb_ok = False

    return ok(
        {
            "sqlite": sqlite_ok,
            "duckdb": duckdb_ok,
            "last_scan_at": last_scan_at,
        }
    )
