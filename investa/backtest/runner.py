"""Backtest orchestrator."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import Interval, ScanMode
from ..data.baskets import default_basket
from ..data.models import BacktestJob
from ..data.provider import DataFetchError, DataProvider
from ..database import session_scope
from ..logging_setup import get_logger
from .replay import SignalEvent, replay_ticker
from .report import DEFAULT_HORIZONS, build_report, join as join_streams
from .returns import TradeOutcome, attach_forward_returns, simulate_trade

log = get_logger(__name__)


@dataclass
class BacktestRequest:
    tickers: list[str] | None = None
    mode: ScanMode = ScanMode.deep_reversal
    interval: Interval = Interval.daily
    lookback_days: int = 730
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    end_date: datetime | None = None


class BacktestRunner:
    def __init__(self, provider: DataProvider) -> None:
        self._provider = provider

    def run_sync(self, req: BacktestRequest, *, persist: bool = True) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        tickers = list(req.tickers) if req.tickers else default_basket()
        end_date = req.end_date or datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0)

        if persist:
            self._update_job(job_id, status="running",
                             params={"tickers": tickers, "mode": req.mode.value,
                                     "interval": req.interval.value,
                                     "lookback_days": req.lookback_days,
                                     "horizons": list(req.horizons),
                                     "end_date": end_date.isoformat()})

        t0 = time.monotonic()
        log.info("backtest.start", job_id=job_id, tickers=len(tickers),
                 mode=req.mode.value, lookback_days=req.lookback_days)

        events_all: list[SignalEvent] = []
        fails: list[dict[str, str]] = []
        ohlcv_by_ticker: dict[str, Any] = {}
        for i, ticker in enumerate(tickers):
            try:
                df = self._provider.fetch_ohlcv_sync(
                    ticker, interval=req.interval.value, lookback_days=req.lookback_days)
                if df.empty or len(df) < 220:
                    fails.append({"ticker": ticker, "stage": "fetcher",
                                  "error": "insufficient_history"})
                    continue
                ohlcv_by_ticker[ticker.upper()] = df
                events = replay_ticker(df, ticker=ticker, mode=req.mode,
                                        interval=req.interval)
                events_all.extend(events)
            except DataFetchError as e:
                fails.append({"ticker": ticker, "stage": "fetcher", "error": str(e)})
            except Exception as e:
                log.exception("backtest.ticker_error", ticker=ticker, error=str(e))
                fails.append({"ticker": ticker, "stage": "replay", "error": str(e)})

            if persist:
                self._update_job(job_id, status="running",
                                 processed_tickers=i + 1,
                                 candidates_found=len(events_all))

        events_all.sort(key=lambda e: (e.ticker, e.sim_date))

        forward_per_event: list[dict[str, Any]] = []
        trades_per_event: list[TradeOutcome] = []
        for ev in events_all:
            df = ohlcv_by_ticker.get(ev.ticker)
            fwd = attach_forward_returns([ev], df, horizons=req.horizons)[0]
            forward_per_event.append(fwd)
            trades_per_event.append(simulate_trade(ev, df))

        joined = join_streams(events_all, forward_per_event, trades_per_event)
        start_date = end_date - timedelta(days=req.lookback_days)
        report = build_report(
            joined, horizons=req.horizons, start_date=start_date, end_date=end_date,
            metadata={
                "tickers_attempted": len(tickers),
                "tickers_succeeded": len(ohlcv_by_ticker),
                "tickers_failed": len(fails),
                "duration_seconds": round(time.monotonic() - t0, 2),
                "interval": req.interval.value,
                "mode": req.mode.value,
                "warnings": [
                    "N per band is modest; treat as suggestive, not conclusive.",
                    "Survivorship & delisting bias: yfinance returns current only.",
                    "Right-edge censoring: horizons exceeding future are None.",
                    "Ambiguous bars: stop hit before target (pessimistic).",
                    "No costs/slippage modeled — raw forward-return stats.",
                ],
            })
        report["failed_tickers"] = fails

        if persist:
            self._update_job(job_id, status="done", report=report,
                             candidates_found=len(events_all),
                             processed_tickers=len(tickers),
                             finished_at=datetime.now(timezone.utc))

        log.info("backtest.done", job_id=job_id, n_events=len(events_all),
                 n_failed=len(fails),
                 duration_seconds=round(time.monotonic() - t0, 2))
        return {"job_id": job_id, "report": report}

    @staticmethod
    def _update_job(job_id, *, status=None, report=None, processed_tickers=None,
                    candidates_found=None, finished_at=None, params=None):
        with session_scope() as db:
            row = db.get(BacktestJob, job_id)
            if row is None:
                row = BacktestJob(id=job_id, status=status or "queued",
                                  params=params or {})
                db.add(row)
            else:
                if status is not None:
                    row.status = status
                if params is not None:
                    row.params = params
            if report is not None:
                row.report = report
            if processed_tickers is not None:
                row.params = {**(row.params or {}), "processed_tickers": processed_tickers}
            if candidates_found is not None:
                row.params = {**(row.params or {}), "candidates_found": candidates_found}
            if finished_at is not None:
                row.finished_at = finished_at

    @staticmethod
    def get_report(job_id):
        with session_scope() as db:
            row = db.get(BacktestJob, job_id)
            if row is None:
                return None
            return {
                "job_id": row.id,
                "status": row.status,
                "params": row.params,
                "report": row.report,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            }

        if persist:
            self._update_job(job_id, status="done", report=report,
                             candidates_found=len(events_all),
                             processed_tickers=len(tickers),
                             finished_at=datetime.now(timezone.utc))

        log.info("backtest.done", job_id=job_id, n_events=len(events_all),
                 n_failed=len(fails),
                 duration_seconds=round(time.monotonic() - t0, 2))
        return {"job_id": job_id, "report": report}
