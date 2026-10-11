"""Historical replay engine.

Walks a multi-year OHLCV frame week-by-week (or any other frequency) and
runs the existing Investa pipeline at each sim date, producing
:class:`SignalEvent` records.

**Correctness contract:** every pipeline call must observe only the slice
``df.loc[:sim_date]`` — never data after ``sim_date``. Any look-ahead leak
invalidates the report. The dedicated unit test
``test_replay_respects_lookahead`` is the trust anchor — if it ever fails,
every historical backtest ran by this engine is suspect.

The replay is *purely deterministic*: same input frame, same sim dates, same
config → identical events. No subprocesses, no mocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..config import Interval, ScanMode
from ..pipeline.gate import evaluate_gate
from ..pipeline.indicators import compute_indicators, resample_weekly
from ..pipeline.risk import annotate_risk
from ..pipeline.scorer import band_for, compute_score
from ..pipeline.signals import detect_all

# Minimum rows required to satisfy EMA-200 warm-up. With 260 daily bars the
# EMA-200 is defined for bars 200..259, so we require 210 to leave a small
# safety margin.
MIN_ROWS_REQUIRED = 210
# Default replay frequency. Anything finer risks adjacent-signal correlation;
# anything coarser wastes statistical power.
DEFAULT_REPLAY_FREQ = "W-FRI"


@dataclass
class SignalEvent:
    """One replayed signal at one sim date for one ticker."""

    ticker: str
    sim_date: datetime
    close: float
    score: float
    band: str
    passed_gate: bool
    gate_reason: str
    signal_strengths: dict[str, float]
    signal_flags: dict[str, bool]
    weekly_confirmed: bool
    risk_stop: float | None
    risk_target: float | None
    risk_reward: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "sim_date": self.sim_date.isoformat(),
            "close": self.close,
            "score": self.score,
            "band": self.band,
            "passed_gate": self.passed_gate,
            "gate_reason": self.gate_reason,
            "signal_strengths": {k: round(v, 4) for k, v in self.signal_strengths.items()},
            "signal_flags": dict(self.signal_flags),
            "weekly_confirmed": self.weekly_confirmed,
            "risk": {
                "suggested_stop": self.risk_stop,
                "suggested_target": self.risk_target,
                "risk_reward": self.risk_reward,
            },
        }


def _sim_dates(daily_index: pd.DatetimeIndex,
               *, freq: str = DEFAULT_REPLAY_FREQ) -> pd.DatetimeIndex:
    """Generate sim dates BETWEEN (exclusive) the index range.

    Excluding the first day (no EMA-200 warm-up yet) and the last day (no
    forward data for return computation) keeps every recorded signal bookended
    by usable market context.
    """
    if len(daily_index) < 2:
        return daily_index
    start = daily_index[1]
    end = daily_index[-2]
    return pd.date_range(start=start, end=end, freq=freq)


def replay_ticker(
    df_daily: pd.DataFrame,
    *,
    ticker: str,
    mode: ScanMode,
    interval: Interval = Interval.daily,
) -> list[SignalEvent]:
    """Re-run the Investa pipeline for one ticker across historical sim dates.

    ``df_daily`` must be sorted ascending by date with a DatetimeIndex and
    ``open/high/low/close/volume`` (lowercase) columns. The full frame must
    contain at least :data:`MIN_ROWS_REQUIRED` rows for EMA-200 warm-up.

    Returns events only for sim dates that pass the gate. Gate-rejected
    sim dates are NOT recorded — ``report.py`` treats the absence of an event
    at a sim date as 'no signal fired then', which is the correct semantics.
    """
    if len(df_daily) < MIN_ROWS_REQUIRED:
        return []

    sim_dates = _sim_dates(df_daily.index)
    events: list[SignalEvent] = []

    for sim_date in sim_dates:
        # df.loc[:sim_date] is THE look-ahead guardrail.
        slice_loc = df_daily.loc[:sim_date]
        if len(slice_loc) < MIN_ROWS_REQUIRED:
            continue
        try:
            event = _evaluate_at(slice_loc, ticker=ticker, mode=mode, interval=interval)
        except Exception:
            # Per-date errors must skip cleanly — never abort the replay.
            continue
        if event is None:
            continue
        events.append(event)

    return events


def _evaluate_at(
    df_slice: pd.DataFrame, *, ticker: str, mode: ScanMode, interval: Interval
) -> SignalEvent | None:
    """Run the full pipeline on a historical slice."""
    enriched = compute_indicators(df_slice)
    latest = enriched.iloc[-1]

    gate = evaluate_gate(latest, mode)
    if not gate.passed:
        return None

    weekly_df = resample_weekly(df_slice) if interval is Interval.daily else None
    signals = detect_all(enriched, weekly_df)
    score = compute_score(signals)
    risk = annotate_risk(enriched)

    weekly_reading = signals.get("weekly")
    weekly_confirmed = bool(weekly_reading and weekly_reading.fired)

    sim_ts = pd.Timestamp(latest.name)
    if sim_ts.tzinfo is not None:
        sim_dt = sim_ts.to_pydatetime()
    else:
        sim_dt = sim_ts.replace(tzinfo=timezone.utc).to_pydatetime()

    return SignalEvent(
        ticker=ticker,
        sim_date=sim_dt,
        close=float(latest["close"]),
        score=score,
        band=band_for(score).label,
        passed_gate=True,
        gate_reason=gate.reason,
        signal_strengths={k: float(v.strength) for k, v in signals.items()},
        signal_flags={k: bool(v.fired) for k, v in signals.items()},
        weekly_confirmed=weekly_confirmed,
        risk_stop=risk.suggested_stop if risk else None,
        risk_target=risk.suggested_target if risk else None,
        risk_reward=risk.risk_reward if risk else None,
    )
