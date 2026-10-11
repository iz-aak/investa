"""SQLAlchemy ORM model for backtest-v2 jobs.

Hosted in the SAME Base as v1's models (via inheritance from
``investa.data.models.Base``) so all tables live in one SQLite database
file. v1 models are untouched - we only ADD a new row class here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..data.models import Base


class BacktestV2Job(Base):
    """One reversal-validation run - persistence + status tracking."""

    __tablename__ = "backtest_v2_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    status: Mapped[str] = mapped_column(String(16), index=True)  # queued|running|done|failed
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.utcnow()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_tickers_processed: Mapped[int] = mapped_column(Integer, default=0)
    progress_events_found: Mapped[int] = mapped_column(Integer, default=0)
