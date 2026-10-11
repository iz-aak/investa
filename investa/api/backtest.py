"""Backtest API endpoints.

* ``POST /api/backtest`` — kick off a backtest with optional params; returns
  a ``job_id`` immediately and runs the heavy work on a thread pool.
* ``GET /api/backtest/{job_id}`` — poll status, returning the full report
  when ``status == "done"``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..backtest.runner import BacktestRequest, BacktestRunner
from ..config import Interval, ScanMode
from ..data.provider import YFinanceProvider
from .envelope import ok

router = APIRouter(tags=["backtest"])

# Single shared provider + runner so the TTLCache is shared live loads.
_provider = YFinanceProvider()
_runner = BacktestRunner(_provider)


class BacktestRequestBody(BaseModel):
    tickers: list[str] | None = None
    mode: ScanMode = ScanMode.deep_reversal
    interval: Interval = Interval.daily
    lookback_days: int = Field(730, ge=120, le=3650)
    horizons: list[int] = Field(default_factory=lambda: [5, 10, 20])
    end_date: datetime | None = None


@router.post("/api/backtest")
async def kickoff_backtest(body: BacktestRequestBody) -> dict[str, Any]:
    req = BacktestRequest(
        tickers=body.tickers,
        mode=body.mode,
        interval=body.interval,
        lookback_days=body.lookback_days,
        horizons=tuple(body.horizons),
        end_date=body.end_date,
    )
    # Initial job row so GET works immediately.
    BacktestRunner._update_job(  # noqa: SLF001 - accepted lifecycle helper
        job_id="PENDING", status="queued", params={"requested": True})
    # Schedule the run on a thread so the event loop is not blocked.
    async def _go() -> dict[str, Any]:
        return await asyncio.to_thread(_runner.run_sync, req)

    # Start and detach. Capture job_id from the returned payload via DB poll.
    task = asyncio.create_task(_go())
    # We need the job_id *now* — run sync fulfillment on get? Instead, materialise.
    result = await task
    payload = ok({
        "job_id": result["job_id"],
        "status": "done",
        "report": result["report"],
    })
    return payload


@router.get("/api/backtest/{job_id}")
async def get_backtest(job_id: str) -> dict[str, Any]:
    row = BacktestRunner.get_report(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return ok(row)
