"""Backtest v2 (reversal-validation) API endpoints.

Run on the FastAPI app. Reuses the existing single-threaded YFinanceProvider
+ global TTLCache via the singleton below.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..api.envelope import ok
from ..backtest_v2.runner import V2Request, V2Runner
from ..config import Interval, ScanMode
from ..data.provider import YFinanceProvider

router = APIRouter(tags=["backtest-v2"])

_provider = YFinanceProvider()
_runner = V2Runner(_provider)


def _hm_to_labels(ints):
    """Coerce horizons from ints (3,6,9) to label strings ('3m','6m','9m')."""
    out = []
    for h in ints:
        if isinstance(h, str) and h.endswith("m"):
            out.append(h)
        else:
            out.append(f"{int(h)}m")
    return tuple(out)


class V2RequestBody(BaseModel):
    tickers: list[str] | None = None
    mode: ScanMode = ScanMode.deep_reversal
    interval: Interval = Interval.daily
    lookback_days: int = 1646
    replay_start: datetime | None = None
    replay_end: datetime | None = None
    horizons_months: list[int] = [3, 6, 9]
    sustain_days: int = 5


@router.post("/api/backtest-v2")
async def kickoff(body: V2RequestBody) -> dict[str, Any]:
    from ..config import get_settings as _gs
    from ..database import init_db as _init_sdb
    from .duckdb_schema import init_v2_schema as _init_v2
    s = _gs()
    _init_sdb(s)
    _init_v2(s.duckdb_path)
    labels = _hm_to_labels(body.horizons_months)
    req = V2Request(tickers=body.tickers, mode=body.mode,
                    interval=body.interval, lookback_days=body.lookback_days,
                    replay_start=body.replay_start, replay_end=body.replay_end,
                    horizons_months=labels,
                    sustain_days=body.sustain_days)
    result = await asyncio.to_thread(_runner.run_sync, req)
    return ok({
        "job_id": result["job_id"],
        "status": "done",
        "n_events": len(result.get("events", [])),
        "report_summary": {
            "n_events": result["report"]["n_events"],
            "primary_horizon": result["report"]["primary_horizon"],
            "section_a": result["report"]["section_a_verdict_distribution_by_horizon"],
            "section_b": result["report"]["section_b_reversal_rate_by_horizon"],
            "metadata": result["report"]["metadata"],
        },
        "report": result["report"],
    })


@router.get("/api/backtest-v2/{job_id}")
async def poll(job_id: str) -> dict[str, Any]:
    row = V2Runner.get_report(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return ok(row)
