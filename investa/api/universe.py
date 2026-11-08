"""Universe API endpoints - Stage 1 rebuild + read-only snapshot access.

* ``POST /api/universe/rebuild`` - run the full pipeline + persist.
* ``GET  /api/universe``          - return tickers from latest snapshot.
* ``GET  /api/universe/stats``    - return stats from latest snapshot.

The rebuild runs synchronously via ``asyncio.to_thread`` so it does not
block the FastAPI event loop. If the NASDAQ FTP/HTTP server is unreachable
after retries, the endpoint returns 503 with a clear error message.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..data.universe import (
    UniverseResult,
    build_universe,
    get_current_universe,
    get_universe_stats,
    save_universe,
)
from .envelope import fail, ok

router = APIRouter(tags=["universe"])


class RebuildBody(BaseModel):
    """Optional overrides for ``POST /api/universe/rebuild``."""

    min_avg_volume_usd: float | None = Field(default=None, ge=0)
    min_share_price: float | None = Field(default=None, ge=0)
    min_market_cap: float | None = Field(default=None, ge=0)
    min_history_days: int | None = Field(default=None, ge=0)
    max_sector_pct: float | None = Field(default=None, gt=0.0, le=1.0)
    batch_size: int | None = Field(default=None, ge=1, le=500)
    batch_delay_seconds: float | None = Field(default=None, ge=0.0)
    nasdaq_url: str | None = None
    other_url: str | None = None


def _run_rebuild(body: RebuildBody) -> UniverseResult:
    """Sync helper: run the full pipeline + persist. Returns the result."""
    result = build_universe(
        min_avg_volume_usd=body.min_avg_volume_usd,
        min_share_price=body.min_share_price,
        min_market_cap=body.min_market_cap,
        min_history_days=body.min_history_days,
        max_sector_pct=body.max_sector_pct,
        batch_size=body.batch_size,
        batch_delay_seconds=body.batch_delay_seconds,
        nasdaq_url=body.nasdaq_url,
        other_url=body.other_url,
    )
    save_universe(result)
    return result


@router.post("/api/universe/rebuild")
async def rebuild_universe(body: RebuildBody | None = None) -> Any:
    """Rebuild the universe via NASDAQ directory + yfinance + filters.

    Runs synchronously on a worker thread (``asyncio.to_thread``) so the
    FastAPI event loop stays responsive. Returns 503 if the NASDAQ directory
    cannot be downloaded after all retries.
    """
    from fastapi.responses import JSONResponse

    body = body or RebuildBody()
    try:
        result = await asyncio.to_thread(_run_rebuild, body)
    except RuntimeError as exc:
        # NasdaqDirectoryError is a RuntimeError; surface 503.
        message = "nasdaq directory unavailable" if "download" in str(exc).lower() else str(exc)
        return JSONResponse(
            status_code=503,
            content=fail(error=message, detail=str(exc)),
        )
    result_dict = result.as_dict()
    return ok(
        {
            "rebuilt_at": result_dict["rebuilt_at"],
            "exchange_scope": result_dict["exchange_scope"],
            "n_tickers": result_dict["n_tickers"],
            "stats": result_dict["stats"],
        }
    )


@router.get("/api/universe")
async def get_universe() -> dict[str, Any]:
    """Return all tickers from the most recently persisted snapshot."""
    snap = get_current_universe()
    if snap is None:
        raise HTTPException(status_code=404, detail="no universe snapshot yet; POST /api/universe/rebuild")
    return ok(
        {
            "rebuilt_at": snap.rebuilt_at.isoformat(),
            "exchange_scope": snap.exchange_scope,
            "n_tickers": len(snap.tickers),
            "tickers": list(snap.tickers),
        }
    )


@router.get("/api/universe/stats")
async def get_stats() -> dict[str, Any]:
    """Return only the stats from the most recently persisted snapshot."""
    snap = get_current_universe()
    if snap is None:
        raise HTTPException(status_code=404, detail="no universe snapshot yet; POST /api/universe/rebuild")
    return ok(
        {
            "rebuilt_at": snap.rebuilt_at.isoformat(),
            "exchange_scope": snap.exchange_scope,
            "stats": dict(snap.stats),
        }
    )