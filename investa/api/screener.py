"""Screener API routes.

Walking-skeleton version: a single ``GET /api/screener`` endpoint that runs
an on-demand scan over a user-supplied ticker list and returns scored
candidates. In step 7-proper this file grows to include:

* ``GET /api/screener`` reading the latest persisted scan from DuckDB
* ``GET /api/screener/{ticker}`` — full signal breakdown for one ticker
* Query params for min_score, limit, offset, sector, sort_by
* Watchlist / preset-universe integration
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..config import Interval, ScanMode
from ..data.provider import YFinanceProvider
from ..pipeline.scanner import Scanner
from .envelope import ok

router = APIRouter(tags=["screener"])

# Single shared provider instance so the TTLCache is actually useful.
_provider = YFinanceProvider()
_scanner = Scanner(_provider)


def _parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


@router.get("/api/screener")
async def screener(
    tickers: str = Query(
        ...,
        description="Comma-separated ticker symbols, e.g. 'AAPL,MSFT,NVDA'",
        min_length=1,
    ),
    mode: ScanMode = Query(ScanMode.deep_reversal),
    interval: Interval = Query(Interval.daily),
    min_score: float = Query(0.0, ge=0.0, le=100.0),
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    symbols = _parse_tickers(tickers)
    if not symbols:
        raise HTTPException(status_code=400, detail="no valid tickers supplied")
    if len(symbols) > 100:
        raise HTTPException(
            status_code=400, detail="max 100 tickers per request in walking-skeleton mode"
        )

    # Run the (blocking) scanner off the event loop.
    import asyncio

    result = await asyncio.to_thread(
        _scanner.scan_sync,
        symbols,
        mode=mode,
        interval=interval,
        min_score=min_score,
        persist=True,
    )

    payload = [c.as_dict() for c in result.candidates[:limit]]

    return ok(
        payload,
        scanned_at=result.scanned_at.isoformat(),
        total=len(result.candidates),
        returned=len(payload),
        requested=len(symbols),
        failed=result.failed,
        mode=result.mode.value,
        interval=result.interval.value,
        scan_id=result.scan_id,
    )
