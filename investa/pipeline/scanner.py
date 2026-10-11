"""Scan orchestrator.

Walks a list of tickers through the pipeline:
    fetch → indicators → gate → signals → score → risk annotation → persist

For the walking-skeleton slice we run **sequentially** and **on demand**
(no batching, no scheduler). Batching + APScheduler jobs land in step 6-proper.

Failures on any single ticker are logged with ``ticker`` + ``stage`` bindings
and never abort the whole scan — the contract in the plan.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from ..config import Interval, ScanMode, get_settings
from ..data.duckdb_store import connect as duckdb_connect
from ..data.provider import DataFetchError, DataProvider
from ..logging_setup import get_logger
from .gate import evaluate_gate
from .indicators import compute_indicators, resample_weekly
from .risk import annotate_risk
from .scorer import band_for, compute_score
from .signals import detect_all

log = get_logger(__name__)


@dataclass
class Candidate:
    ticker: str
    scanned_at: datetime
    close: float
    score: float
    band: str
    passed_gate: bool
    gate_reason: str
    signals: dict[str, dict[str, Any]]
    risk: dict[str, float | None]
    indicators: dict[str, float | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scanned_at"] = self.scanned_at.isoformat()
        return d


@dataclass
class ScanResult:
    scan_id: str
    scanned_at: datetime
    mode: ScanMode
    interval: Interval
    candidates: list[Candidate]
    failed: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "scanned_at": self.scanned_at.isoformat(),
            "mode": self.mode.value,
            "interval": self.interval.value,
            "candidates": [c.as_dict() for c in self.candidates],
            "failed": self.failed,
        }


def _nullable_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


class Scanner:
    def __init__(self, provider: DataProvider) -> None:
        self._provider = provider

    def scan_sync(
        self,
        tickers: list[str],
        *,
        mode: ScanMode | None = None,
        interval: Interval | None = None,
        min_score: float = 0.0,
        persist: bool = True,
    ) -> ScanResult:
        settings = get_settings()
        mode = mode or settings.default_mode
        interval = interval or settings.default_interval
        scan_id = str(uuid.uuid4())
        scanned_at = datetime.now(timezone.utc)

        candidates: list[Candidate] = []
        failed: list[dict[str, str]] = []

        for raw_ticker in tickers:
            ticker = raw_ticker.strip().upper()
            if not ticker:
                continue
            binder = structlog.contextvars.bound_contextvars(ticker=ticker, scan_id=scan_id)
            with binder:
                try:
                    cand = self._process_one(ticker, mode=mode, interval=interval)
                except DataFetchError as exc:
                    log.warning("scan.ticker_failed", stage="fetcher", error=str(exc))
                    failed.append({"ticker": ticker, "stage": "fetcher", "error": str(exc)})
                    continue
                except Exception as exc:  # noqa: BLE001
                    log.exception("scan.ticker_error", stage="pipeline", error=str(exc))
                    failed.append({"ticker": ticker, "stage": "pipeline", "error": str(exc)})
                    continue

                if cand is None:
                    continue
                if cand.score < min_score:
                    continue
                candidates.append(cand)

        candidates.sort(key=lambda c: c.score, reverse=True)

        result = ScanResult(
            scan_id=scan_id,
            scanned_at=scanned_at,
            mode=mode,
            interval=interval,
            candidates=candidates,
            failed=failed,
        )
        if persist and candidates:
            self._persist(result)

        log.info(
            "scan.completed",
            scan_id=scan_id,
            requested=len(tickers),
            candidates=len(candidates),
            failed=len(failed),
        )
        return result

    def _process_one(
        self, ticker: str, *, mode: ScanMode, interval: Interval
    ) -> Candidate | None:
        settings = get_settings()
        df = self._provider.fetch_ohlcv_sync(  # type: ignore[attr-defined]
            ticker,
            interval=interval.value,
            lookback_days=settings.scan_lookback_days,
        )
        if df.empty:
            return None

        enriched = compute_indicators(df)
        latest = enriched.iloc[-1]

        gate = evaluate_gate(latest, mode)
        if not gate.passed:
            log.debug("scan.gate_rejected", reason=gate.reason)
            return None

        # Weekly confirmation frame (resampled from the daily OHLCV).
        weekly_df = resample_weekly(df) if interval is Interval.daily else None

        signals = detect_all(enriched, weekly_df)
        score = compute_score(signals)
        risk = annotate_risk(enriched)

        risk_dict = (
            risk.as_dict()
            if risk is not None
            else {"suggested_stop": None, "suggested_target": None, "risk_reward": None}
        )

        return Candidate(
            ticker=ticker,
            scanned_at=datetime.now(timezone.utc),
            close=float(latest["close"]),
            score=score,
            band=band_for(score).label,
            passed_gate=gate.passed,
            gate_reason=gate.reason,
            signals={k: v.as_dict() for k, v in signals.items()},
            risk=risk_dict,
            indicators={
                "rsi_14": _nullable_float(latest.get("rsi_14")),
                "ema_50": _nullable_float(latest.get("ema_50")),
                "ema_200": _nullable_float(latest.get("ema_200")),
                "atr_14": _nullable_float(latest.get("atr_14")),
                "macd": _nullable_float(latest.get("macd")),
                "macd_signal": _nullable_float(latest.get("macd_signal")),
                "macd_hist": _nullable_float(latest.get("macd_hist")),
                "bb_lower": _nullable_float(latest.get("bb_lower")),
                "bb_upper": _nullable_float(latest.get("bb_upper")),
                "bb_pos": _nullable_float(latest.get("bb_pos")),
                "obv": _nullable_float(latest.get("obv")),
                "obv_slope": _nullable_float(latest.get("obv_slope")),
                "stoch_k": _nullable_float(latest.get("stoch_k")),
                "stoch_d": _nullable_float(latest.get("stoch_d")),
                "volume_ratio": _nullable_float(latest.get("volume_ratio")),
            },
        )

    # ── persistence ────────────────────────────────────────────────
    def _persist(self, result: ScanResult) -> None:
        rows = [
            (
                result.scan_id,
                result.scanned_at,
                c.ticker,
                result.interval.value,
                result.mode.value,
                c.close,
                c.indicators.get("atr_14"),
                c.indicators.get("rsi_14"),
                c.indicators.get("macd"),
                c.indicators.get("macd_signal"),
                c.indicators.get("macd_hist"),
                c.indicators.get("bb_lower"),
                c.indicators.get("bb_upper"),
                c.indicators.get("bb_pos"),
                c.indicators.get("ema_50"),
                c.indicators.get("ema_200"),
                c.indicators.get("obv"),
                c.indicators.get("obv_slope"),
                c.indicators.get("stoch_k"),
                c.indicators.get("stoch_d"),
                None,                   # volume  (raw not persisted in v1)
                None,                   # volume_avg_20 (raw not persisted in v1)
                c.indicators.get("volume_ratio"),
                c.passed_gate,
                json.dumps({k: v["fired"] for k, v in c.signals.items()}),
                json.dumps({k: v["strength"] for k, v in c.signals.items()}),
                bool((c.signals.get("weekly") or {}).get("fired", False)),
                c.score,
                c.band,
                c.risk.get("suggested_stop"),
                c.risk.get("suggested_target"),
                c.risk.get("risk_reward"),
            )
            for c in result.candidates
        ]
        if not rows:
            return
        try:
            with duckdb_connect() as con:
                con.executemany(
                    """
                    INSERT INTO signal_history VALUES (
                        ?, ?, ?, ?, ?, ?, ?,
                        ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?
                    )
                    """,
                    rows,
                )
        except Exception as exc:  # noqa: BLE001 — persistence must not fail scan
            log.error("scan.persist_failed", error=str(exc))
