"""SQLAlchemy 2.x ORM models for the relational (SQLite) store.

Only relational, low-volume state lives here: watchlists, scan/backtest job
metadata, failed-ticker records, and the ticker-validation cache. Time-series
signal history is stored separately in DuckDB (see :mod:`investa.data.duckdb`).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Watchlist(Base):
    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tickers: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ScanJob(Base):
    __tablename__ = "scan_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    status: Mapped[str] = mapped_column(String(16), index=True)  # queued|running|done|failed
    mode: Mapped[str] = mapped_column(String(32))
    interval: Mapped[str] = mapped_column(String(8))
    universe: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    total_tickers: Mapped[int] = mapped_column(Integer, default=0)
    processed_tickers: Mapped[int] = mapped_column(Integer, default=0)
    candidates_found: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    failed_tickers: Mapped[list["FailedTicker"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan"
    )


class FailedTicker(Base):
    __tablename__ = "failed_tickers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scan_jobs.id", ondelete="CASCADE"), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    stage: Mapped[str] = mapped_column(String(32))  # fetcher|indicator|signal|scorer
    error: Mapped[str] = mapped_column(Text)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    scan: Mapped[ScanJob] = relationship(back_populates="failed_tickers")


class TickerValidation(Base):
    """Cached results of yfinance-backed ticker validation lookups."""

    __tablename__ = "ticker_validations"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    is_valid: Mapped[bool]
    resolved_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cached_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class BacktestJob(Base):
    __tablename__ = "backtest_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class MarketContextSnapshot(Base):
    """Weekly-refreshed market regime + sector relative-strength blob."""

    __tablename__ = "market_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    regime: Mapped[str] = mapped_column(String(16))  # risk_on|risk_off|neutral
    payload: Mapped[dict] = mapped_column(JSON)
    score: Mapped[float] = mapped_column(Float, default=0.0)


class UniverseSnapshot(Base):
    """Current eligible US-listed common stock universe.

    Each rebuild deletes the prior row and inserts a fresh snapshot (single-row
    design for Stage 1 v1). The full ticker list + per-stage stats live in
    JSON columns so we can rebuild the entire universe in one round-trip.
    """

    __tablename__ = "universe_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rebuilt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    exchange_scope: Mapped[str] = mapped_column(String(64))
    tickers: Mapped[list[dict]] = mapped_column(JSON)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
