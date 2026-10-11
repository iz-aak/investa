"""SQLite (SQLAlchemy) engine + session management.

Kept intentionally small — no async engine, no connection pool tuning. The
scan worker writes; the API mostly reads. If we outgrow SQLite, swap the URL
in ``.env`` for Postgres and nothing else needs to change.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .data.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _build_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.sqlite_url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )


def init_db(settings: Settings | None = None) -> Engine:
    """Initialise the SQLite engine and create all tables. Idempotent."""
    global _engine, _SessionLocal

    settings = settings or get_settings()
    settings.ensure_dirs()

    if _engine is None:
        _engine = _build_engine(settings)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)

    Base.metadata.create_all(_engine)
    # Sanity: confirm the file is writable. Some environments (read-only
    # mount, etc.) silently present a read-only file after first attempt.
    try:
        with _engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA quick_check")
    except Exception:
        pass
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        init_db()
    assert _engine is not None
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session scope; commits on success, rolls back on error."""
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None

    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
