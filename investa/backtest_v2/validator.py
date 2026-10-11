"""Per-event orchestrator for v2 reversal-validation. Reuses v1."""


from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from ..backtest.replay import SignalEvent, replay_ticker
from ..backtest.returns import attach_forward_returns
from ..config import Interval, ScanMode
from .reversal import (
    TIER_EMA50_RECLAIM, TIER_GOLDEN_CROSS, TIER_SWING,
    TierResult, collapse_verdict, evaluate_at_horizon,
    unverifiable_tier_names,
)
@dataclass
class ReversalRow:
    """One event x (3 tiers x 3 horizons) - fully joined."""

    ticker: str
    sim_date: datetime
    score: float
    band: str
    signal_flags: dict[str, bool]

    tier_golden_cross_3m: bool
    tier_ema50_reclaim_3m: bool
    tier_swing_3m: bool
    tier_golden_cross_6m: bool
    tier_ema50_reclaim_6m: bool
    tier_swing_6m: bool
    tier_golden_cross_9m: bool
    tier_ema50_reclaim_9m: bool
    tier_swing_9m: bool

    verdict_3m: str
    verdict_6m: str
    verdict_9m: str
    n_tiers_fired_6m: int
    fwd_return_3m: float | None
    fwd_return_6m: float | None
    fwd_return_9m: float | None
    days_to_first_fire_3m: int | None = None
    days_to_first_fire_6m: int | None = None
    days_to_first_fire_9m: int | None = None
    unverifiable_tiers_3m: list[str] = field(default_factory=list)
    unverifiable_tiers_6m: list[str] = field(default_factory=list)
    unverifiable_tiers_9m: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            "ticker": self.ticker,
            "sim_date": self.sim_date.isoformat(),
            "score": round(self.score, 4),
            "band": self.band,
            "signal_flags": self.signal_flags,
            "tiers": {
                "3m": {"golden_cross": self.tier_golden_cross_3m,
                        "ema50_reclaim": self.tier_ema50_reclaim_3m,
                        "swing": self.tier_swing_3m},
                "6m": {"golden_cross": self.tier_golden_cross_6m,
                        "ema50_reclaim": self.tier_ema50_reclaim_6m,
                        "swing": self.tier_swing_6m},
                "9m": {"golden_cross": self.tier_golden_cross_9m,
                        "ema50_reclaim": self.tier_ema50_reclaim_9m,
                        "swing": self.tier_swing_9m},
            },
            "verdicts": {"3m": self.verdict_3m, "6m": self.verdict_6m,
                          "9m": self.verdict_9m},
            "n_tiers_fired_6m": self.n_tiers_fired_6m,
            "days_to_first_fire": {
                "3m": self.days_to_first_fire_3m,
                "6m": self.days_to_first_fire_6m,
                "9m": self.days_to_first_fire_9m,
            },
            "unverifiable_tiers": {
                "3m": self.unverifiable_tiers_3m,
                "6m": self.unverifiable_tiers_6m,
                "9m": self.unverifiable_tiers_9m,
            },
            "forward_returns": {
                "3m": self.fwd_return_3m, "6m": self.fwd_return_6m,
                "9m": self.fwd_return_9m,
            },
        }


@dataclass
class ValidationRequest:
    ticker: str
    mode: ScanMode = ScanMode.deep_reversal
    interval: Interval = Interval.daily
    lookback_days: int = 1646
    replay_start: datetime | None = None
    replay_end: datetime | None = None
    horizons_months: tuple = ("3m", "6m", "9m")
    sustain_days: int = 5


PRIMARY_HORIZON_LABEL = "6m"
HORIZON_LABELS = ("3m", "6m", "9m")
def _filter_sim_window(events, start, end):
    from datetime import timezone as _tz
    if start is not None and start.tzinfo is None:
        start = start.replace(tzinfo=_tz.utc)
    if end is not None and end.tzinfo is None:
        end = end.replace(tzinfo=_tz.utc)
    out = []
    for e in events:
        s = e.sim_date
        if s is not None and s.tzinfo is None:
            s = s.replace(tzinfo=_tz.utc)
        if start is not None and s < start:
            continue
        if end is not None and s > end:
            continue
        out.append(e)
    return out

def _eval_event(df_daily, ev, horizons_months, sustain_days):
    """horizons_months must be a tuple of LABEL strings like ('3m','6m','9m')."""
    fwd_horizons = tuple(HORIZON_DAYS[h] for h in horizons_months)
    fwd = attach_forward_returns([ev], df_daily, horizons=fwd_horizons)[0]
    fwd_map = {h: fwd["returns"][n] for h, n in zip(horizons_months, fwd_horizons)}
    tier_by_horizon = {}
    verdict_by_horizon = {}
    nv_by_horizon = {}
    n_fired_by_horizon = {}
    days_by_horizon = {}
    for h_label in horizons_months:
        h_days = HORIZON_DAYS.get(h_label)
        if h_days is None:
            continue
        tiers = evaluate_at_horizon(df_daily, ev.sim_date,
                                     window_days=h_days, sustain_days=sustain_days)
        tier_by_horizon[h_label] = tiers
        v, n_fired, _nv, days = collapse_verdict(tiers)
        verdict_by_horizon[h_label] = v
        n_fired_by_horizon[h_label] = n_fired
        nv_by_horizon[h_label] = unverifiable_tier_names(tiers)
        days_by_horizon[h_label] = days

    def t(h_label, tier_name):
        return bool(tier_by_horizon[h_label][tier_name].fired)

    return ReversalRow(
        ticker=ev.ticker, sim_date=ev.sim_date, score=ev.score, band=ev.band,
        signal_flags=dict(ev.signal_flags),
        tier_golden_cross_3m=t("3m", TIER_GOLDEN_CROSS),
        tier_ema50_reclaim_3m=t("3m", TIER_EMA50_RECLAIM),
        tier_swing_3m=t("3m", TIER_SWING),
        tier_golden_cross_6m=t("6m", TIER_GOLDEN_CROSS),
        tier_ema50_reclaim_6m=t("6m", TIER_EMA50_RECLAIM),
        tier_swing_6m=t("6m", TIER_SWING),
        tier_golden_cross_9m=t("9m", TIER_GOLDEN_CROSS),
        tier_ema50_reclaim_9m=t("9m", TIER_EMA50_RECLAIM),
        tier_swing_9m=t("9m", TIER_SWING),
        verdict_3m=verdict_by_horizon.get("3m", "unverifiable"),
        verdict_6m=verdict_by_horizon.get("6m", "unverifiable"),
        verdict_9m=verdict_by_horizon.get("9m", "unverifiable"),
        n_tiers_fired_6m=n_fired_by_horizon.get("6m", 0),
        days_to_first_fire_3m=days_by_horizon.get("3m"),
        days_to_first_fire_6m=days_by_horizon.get("6m"),
        days_to_first_fire_9m=days_by_horizon.get("9m"),
        unverifiable_tiers_3m=nv_by_horizon.get("3m", []),
        unverifiable_tiers_6m=nv_by_horizon.get("6m", []),
        unverifiable_tiers_9m=nv_by_horizon.get("9m", []),
        fwd_return_3m=fwd_map.get("3m"),
        fwd_return_6m=fwd_map.get("6m"),
        fwd_return_9m=fwd_map.get("9m"),
    )


def validate_ticker(df_daily, req):
    raw = replay_ticker(df_daily, ticker=req.ticker,
                          mode=req.mode, interval=req.interval)
    keep = _filter_sim_window(raw, req.replay_start, req.replay_end)
    rows = []
    for ev in keep:
        try:
            rows.append(_eval_event(df_daily, ev,
                                     req.horizons_months, req.sustain_days))
        except Exception:
            continue
    return rows


def row_to_persist_tuple(job_id, row):
    return (
        job_id, row.ticker, row.sim_date, row.score, row.band,
        json.dumps(row.signal_flags),
        row.tier_golden_cross_3m, row.tier_ema50_reclaim_3m, row.tier_swing_3m,
        row.tier_golden_cross_6m, row.tier_ema50_reclaim_6m, row.tier_swing_6m,
        row.tier_golden_cross_9m, row.tier_ema50_reclaim_9m, row.tier_swing_9m,
        row.verdict_3m, row.verdict_6m, row.verdict_9m,
        row.n_tiers_fired_6m,
        row.days_to_first_fire_3m, row.days_to_first_fire_6m, row.days_to_first_fire_9m,
        json.dumps(row.unverifiable_tiers_3m),
        json.dumps(row.unverifiable_tiers_6m),
        json.dumps(row.unverifiable_tiers_9m),
        row.fwd_return_3m, row.fwd_return_6m, row.fwd_return_9m,
    )

HORIZON_DAYS = {"3m": 63, "6m": 126, "9m": 189}



def _coerce_horizon_spec(spec) -> tuple[str, ...]:
    """Accept either tuple of ints (3,6,9) or strings ('3m','6m','9m')."""
    out = []
    for h in spec:
        if isinstance(h, str) and h.endswith("m"):
            out.append(h)
        else:
            out.append(f"{int(h)}m")
    return tuple(out)
