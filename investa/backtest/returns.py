"""Forward return + trade-outcome computation.

Given a list of :class:`~investa.backtest.replay.SignalEvent` and the raw
OHLCV frames they were generated from, produce for each event:

* Forward returns at 5 / 10 / 20 trading days (USD pct).
* Max intra-period drawdown and run-up over the longest window.
* Trade outcome using the suggested stop/target with pessimistic ordering.

Returns are decorated onto the events as plain attributes — the report layer
consumes them without needing to re-touch OHLCV.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import math
import pandas as pd

from ..config import Interval
from .replay import SignalEvent

DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
HOLD_PERIOD_FOR_TRADE_SIM = 20  # bars from the signal bar (excluded)


@dataclass(frozen=True)
class TradeOutcome:
    label: str                       # "stop_hit" | "target_hit" | "timeout"
    realized_rr: float | None        # realised R-multiple vs stop distance (None on degenerate)
    bars_held: int | None            # None on timeout → max_hold


# ─── forward returns ────────────────────────────────────────────────

def attach_forward_returns(
    events: list[SignalEvent],
    df_daily: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> list[dict[str, float | None]]:
    """Return one forward-returns dict per event in the same order.

    ``df_daily`` is the unsliced ticker frame. Each event's forward window
    starts at the bar *strictly after* the event's sim date (excludes the
    signal bar to avoid using signal-bar close for both entry and exit).
    """
    out: list[dict[str, float | None]] = []
    if df_daily is None or df_daily.empty or not events:
        return [{"returns": {h: None for h in horizons},
                 "max_drawdown": None, "max_runup": None} for _ in events]

    df = df_daily.sort_index()
    for ev in events:
        out.append(_one_lookahead(df, ev, horizons=horizons))
    return out


def _one_lookahead(df: pd.DataFrame, ev: SignalEvent,
                   *, horizons: tuple[int, ...]) -> dict[str, float | None]:
    # Locate the sim date as a datetime; bars strictly after are the future.
    sim_ts = pd.Timestamp(ev.sim_date)
    # Include the row that *is* the signal bar (for entry price = close)
    # but slice to position+1 for fwd-window indexing.
    try:
        signal_loc = df.index.get_indexer([sim_ts], method="ffill")[0]
    except Exception:
        return {"returns": {h: None for h in horizons}, "max_drawdown": None, "max_runup": None}
    future = df.iloc[signal_loc + 1:]
    entry = float(ev.close)
    returns: dict[int, float | None] = {}
    for h in horizons:
        if len(future) >= h:
            end_close = float(future["close"].iloc[h - 1])
            returns[h] = (end_close - entry) / entry
        else:
            returns[h] = None

    if len(future) == 0:
        return {"returns": returns, "max_drawdown": None, "max_runup": None}
    window = future.head(max(horizons))
    if entry <= 0:
        return {"returns": returns, "max_drawdown": None, "max_runup": None}
    max_dd = (float(window["low"].min()) - entry) / entry
    max_ru = (float(window["high"].max()) - entry) / entry
    return {"returns": returns, "max_drawdown": max_dd, "max_runup": max_ru}


# ─── trade simulation ───────────────────────────────────────────────

def simulate_trade(ev: SignalEvent, df_daily: pd.DataFrame,
                   *, max_hold: int = HOLD_PERIOD_FOR_TRADE_SIM) -> TradeOutcome:
    """Step bar-by-bar through the first ``max_hold`` future bars.

    Pessimistic ordering: ``low <= stop`` is checked *before* ``high >= target``
    on each bar, so an ambiguous bar is recorded as a stop hit. This bias is
    documented in the report's metadata.
    """
    if ev.risk_stop is None or ev.risk_target is None or df_daily is None:
        return TradeOutcome(label="no_trade", realized_rr=None, bars_held=None)

    df = df_daily.sort_index()
    try:
        signal_loc = df.index.get_indexer([pd.Timestamp(ev.sim_date)], method="ffill")[0]
    except Exception:
        return TradeOutcome(label="no_trade", realized_rr=None, bars_held=None)
    future = df.iloc[signal_loc + 1 : signal_loc + 1 + max_hold]
    if future.empty:
        return TradeOutcome(label="no_trade", realized_rr=None, bars_held=None)

    stop = float(ev.risk_stop)
    target = float(ev.risk_target)
    entry = float(ev.close)
    if entry <= stop:
        return TradeOutcome(label="no_trade", realized_rr=None, bars_held=None)

    stop_distance = entry - stop
    if stop_distance <= 0 or math.isclose(stop_distance, 0):
        return TradeOutcome(label="no_trade", realized_rr=None, bars_held=None)

    for i, (_ts, bar) in enumerate(future.iterrows()):
        low = float(bar["low"])
        high = float(bar["high"])
        if low <= stop:
            return TradeOutcome(label="stop_hit", realized_rr=-1.0, bars_held=i + 1)
        if high >= target:
            realised = (target - entry) / stop_distance
            return TradeOutcome(label="target_hit", realized_rr=realised, bars_held=i + 1)

    # Neither hit within hold — mark-to-close on the last available bar.
    last_n = min(max_hold, len(future))
    last_close = float(future["close"].iloc[last_n - 1])
    realised = (last_close - entry) / stop_distance
    return TradeOutcome(label="timeout", realized_rr=realised, bars_held=last_n)


# ─── look-ups ──────────────────────────────────────────────────────

def lookup_ohlcv(full_lookup: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame | None:
    return full_lookup.get(ticker.upper())
