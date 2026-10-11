"""V2 reversal-validation runner - composite orchestrator."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..config import Interval, ScanMode, get_settings
from ..data.baskets import default_basket
from ..data import models as _base_models  # noqa: F401 — registers Base table metadata
from .models import BacktestV2Job
from ..data.provider import DataFetchError, DataProvider
from ..database import session_scope
from ..logging_setup import get_logger
from . import duckdb_schema
from .models import BacktestV2Job
from .report import build_v2_report
from .validator import ValidationRequest, validate_ticker, row_to_persist_tuple

log = get_logger(__name__)


@dataclass
class V2Request:
    tickers: list[str] | None = None
    mode: ScanMode = ScanMode.deep_reversal
    interval: Interval = Interval.daily
    lookback_days: int = 1646
    replay_start: datetime | None = None
    replay_end: datetime | None = None
    horizons_months: tuple = (3, 6, 9)
    sustain_days: int = 5
class V2Runner:
    def __init__(self, provider: DataProvider) -> None:
        self._provider = provider

    def run_sync(self, req, *, persist=True):
        job_id = str(uuid.uuid4())
        tickers = list(req.tickers) if req.tickers else default_basket()
        replay_start = req.replay_start or datetime(2023, 1, 1)
        replay_end = req.replay_end or datetime(2025, 12, 31)

        if persist:
            self._update_job(job_id, status="running",
                             params={"tickers": tickers, "mode": req.mode.value,
                                     "interval": req.interval.value,
                                     "lookback_days": req.lookback_days,
                                     "horizons_months": list(req.horizons_months),
                                     "sustain_days": req.sustain_days,
                                     "replay_start": replay_start.isoformat(),
                                     "replay_end": replay_end.isoformat()})

        duckdb_path = get_settings().duckdb_path if persist else None
        if duckdb_path is not None:
            duckdb_schema.init_v2_schema(duckdb_path)

        t0 = time.monotonic()
        log.info("v2.start", job_id=job_id, tickers=len(tickers),
                 replay_start=replay_start.isoformat(),
                 replay_end=replay_end.isoformat())

        all_rows = []
        failed = []
        ohlcv = {}
        for i, ticker in enumerate(tickers):
            try:
                df = self._provider.fetch_ohlcv_sync(
                    ticker, interval=req.interval.value,
                    lookback_days=req.lookback_days)
                if df.empty or len(df) < 220:
                    failed.append({"ticker": ticker, "stage": "fetcher",
                                    "error": "insufficient_history"})
                    continue
                ohlcv[ticker.upper()] = df
                vreq = ValidationRequest(
                    ticker=ticker, mode=req.mode, interval=req.interval,
                    lookback_days=req.lookback_days,
                    replay_start=replay_start, replay_end=replay_end,
                    horizons_months=tuple(
                        (f"{int(m)}m" if not (isinstance(m, str) and m.endswith("m")) else m)
                        for m in (req.horizons_months or ("3m", "6m", "9m"))),
                    sustain_days=req.sustain_days)
                rows = validate_ticker(df, vreq)
                all_rows.extend(rows)
            except DataFetchError as e:
                failed.append({"ticker": ticker, "stage": "fetcher", "error": str(e)})
            except Exception as e:
                log.exception("v2.ticker_error", ticker=ticker, error=str(e))
                failed.append({"ticker": ticker, "stage": "validate", "error": str(e)})
            if persist:
                self._update_job(job_id, status="running",
                                 processed_tickers=i + 1,
                                 events_found=len(all_rows))

        all_rows.sort(key=lambda r: (r.ticker, r.sim_date))
        report = build_v2_report(
            all_rows, horizons=tuple(req.horizons_months),
            start_date=replay_start, end_date=replay_end,
            metadata={"tickers_attempted": len(tickers),
                     "tickers_succeeded": len(ohlcv),
                     "tickers_failed": len(failed),
                     "duration_seconds": round(time.monotonic() - t0, 2),
                     "interval": req.interval.value,
                     "mode": req.mode.value,
                     "sustain_days": req.sustain_days,
                     "warnings": [
                         "N per band modest.",
                         "Survivorship bias.",
                         "Right-edge censoring possible.",
                         "No costs / slippage modeled.",
                         "Pure-technical tiers, no fundamentals/news.",
                     ]})
        if report is None:
            report = build_v2_report([], horizons=tuple(req.horizons_months),
                                     start_date=replay_start,
                                     end_date=replay_end,
                                     metadata={"warning": "no events"})
        report["failed_tickers"] = failed

        if persist:
            self._persist_rows(job_id, all_rows, duckdb_path)
            self._update_job(job_id, status="done", report=report,
                             processed_tickers=len(tickers),
                             events_found=len(all_rows),
                             finished_at=datetime.now(timezone.utc))

        log.info("v2.done", job_id=job_id, n_events=len(all_rows),
                 n_failed=len(failed),
                 duration_seconds=round(time.monotonic() - t0, 2))
        return {"job_id": job_id, "report": report,
                "events": [r.as_dict() for r in all_rows]}

    @staticmethod
    def _persist_rows(job_id, rows, duckdb_path):
        if not rows or duckdb_path is None:
            return
        import duckdb
        tuples = [row_to_persist_tuple(job_id, r) for r in rows]
        try:
            with duckdb.connect(str(duckdb_path)) as con:
                con.executemany(
                    '''INSERT INTO reversal_events VALUES (
                        ?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?, ?,?,?, ?,?,?, ?,?,?)''',
                    tuples)
        except Exception as exc:
            log.error("v2.persist_failed", error=str(exc))

    @staticmethod
    def _update_job(job_id, *, status=None, report=None,
                    processed_tickers=None, events_found=None,
                    finished_at=None, params=None):
        with session_scope() as db:
            row = db.get(BacktestV2Job, job_id)
            if row is None:
                row = BacktestV2Job(id=job_id, status=status or "queued",
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
                row.progress_tickers_processed = processed_tickers
            if events_found is not None:
                row.progress_events_found = events_found
            if finished_at is not None:
                row.finished_at = finished_at

    @staticmethod
    def get_report(job_id):
        with session_scope() as db:
            row = db.get(BacktestV2Job, job_id)
            if row is None:
                return None
            return {
                "job_id": row.id, "status": row.status,
                "params": row.params, "report": row.report,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "finished_at": row.finished_at.isoformat() if row.finished_at else None,
                "progress": {"tickers_processed": row.progress_tickers_processed,
                             "events_found": row.progress_events_found},
            }
