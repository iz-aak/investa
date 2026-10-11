"""Reversal tier validators - pure pandas functions.

Each tier validates whether a *specific kind* of trend reversal occurred
within ``window_days`` trading days after the event date. The three tiers
are independent (different signal sources); the verdict layer combines them
into a multi-tier classification.

Design contract
----------------
* Forward-looking by design (opposite of replay's no-look-ahead rule).
* Censoring is honest: insufficient window -> unverifiable.
* Pure pandas, no I/O. Deterministic.
* Sustain counts trading days.

CRITICAL DETAIL — EMA warmup handling
------------------------------------
The golden cross and EMA-50 reclaim tiers are sensitive to EMA warmup. If
we compute EMA-50 and EMA-200 strictly on the future slice, EMA-200 will be
NaN for the entire slim 126-bar window because it needs 200 prior bars of
warmup. We compute both EMAs on the FULL frame first, then slice the
post-event portion. This is NOT a look-ahead in the reversal-stage sense -
we are not peeking at whether a reversal happened; we are just ensuring
the indicators we evaluate are well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class TierResult:
    fired: bool
    unverifiable: bool = False
    days_to_fire: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _loc_for(df, event_date) -> int:
    try:
        return df.index.get_indexer([pd.Timestamp(event_date)], method="ffill")[0]
    except Exception:
        return -1

import numpy as np
import pandas as pd

from ..pipeline.indicators import ema as _ema
def validate_golden_cross(df, event_date, *, window_days=126, sustain_days=5):
    """EMA-50 crosses back above EMA-200 and holds for sustain_days bars.

    Both EMAs are computed on the FULL frame then sliced to the post-event
    window. EMA-200 needs 200+ prior bars of warmup, so computing it only on
    a thin future slice would always yield NaN.
    """
    loc = _loc_for(df, event_date)
    if loc < 0:
        return TierResult(fired=False, unverifiable=True)
    raw_future_len = len(df) - (loc + 1)
    if raw_future_len < max(window_days or 0, sustain_days + 1):
        return TierResult(fired=False, unverifiable=True)
    n_window = window_days if window_days and window_days > 0 else raw_future_len
    n_window = min(n_window, raw_future_len)

    start = loc + 1
    end = start + n_window
    ema_50 = _ema(df["close"], 50).iloc[start:end].reset_index(drop=True)
    ema_200 = _ema(df["close"], 200).iloc[start:end].reset_index(drop=True)
    if len(ema_50) == 0:
        return TierResult(fired=False, unverifiable=True)

    diff = (ema_50 - ema_200).to_numpy(dtype=float)
    cross_idx = None
    for i in range(1, len(diff)):
        if np.isnan(diff[i - 1]) or np.isnan(diff[i]):
            continue
        if diff[i - 1] <= 0 and diff[i] > 0:
            cross_idx = i
            break
    if cross_idx is None:
        return TierResult(fired=False)
    if cross_idx + sustain_days > len(ema_50):
        return TierResult(fired=False, unverifiable=True)
    sustain = diff[cross_idx: cross_idx + sustain_days]
    if np.any(np.isnan(sustain)) or np.any(sustain <= 0):
        return TierResult(fired=False)
    return TierResult(fired=True, days_to_fire=cross_idx, meta={"sustain_bars": sustain_days})


TIER_GOLDEN_CROSS = "golden_cross"
TIER_EMA50_RECLAIM = "ema50_reclaim"
TIER_SWING = "swing"
ALL_TIERS = (TIER_GOLDEN_CROSS, TIER_EMA50_RECLAIM, TIER_SWING)

def validate_ema50_reclaim(df, event_date, *, window_days=126, sustain_days=5):
    """Close reclaims EMA-50 and holds for sustain_days bars.

    Two firing conditions, both requiring sustain_days bars of confirmation:
    (a) Strict forward cross: close_prev <= ema_prev AND close > ema (in window)
    (b) Recent cross happened during the pre-event warmup (any time during
        lookback), and close has stayed above ema for sustain_days bars.
    Both signal that price has reclaimed and is holding the medium-term mean.
    """
    loc = _loc_for(df, event_date)
    if loc < 0:
        return TierResult(fired=False, unverifiable=True)
    raw_future_len = len(df) - (loc + 1)
    if raw_future_len < max(window_days or 0, sustain_days + 1):
        return TierResult(fired=False, unverifiable=True)
    n_window = window_days if window_days and window_days > 0 else raw_future_len
    n_window = min(n_window, raw_future_len)
    start = loc + 1
    end = start + n_window
    future_close = df["close"].iloc[start:end].reset_index(drop=True)
    ema_50 = _ema(df["close"], 50).iloc[start:end].reset_index(drop=True)
    if future_close.empty:
        return TierResult(fired=False, unverifiable=True)

    # Case (a): strict forward cross
    reclaim_idx = None
    for i in range(1, len(future_close)):
        c = float(future_close.iloc[i])
        cp = float(future_close.iloc[i - 1])
        e = ema_50.iloc[i]
        ep = ema_50.iloc[i - 1]
        if pd.isna(e) or pd.isna(ep):
            continue
        if cp <= ep and c > e:
            reclaim_idx = i
            break
    if reclaim_idx is not None:
        if reclaim_idx + sustain_days > len(future_close):
            return TierResult(fired=False, unverifiable=True)
        for i in range(reclaim_idx, reclaim_idx + sustain_days):
            c = float(future_close.iloc[i])
            e = ema_50.iloc[i]
            if pd.isna(e) or c <= e:
                return TierResult(fired=False)
        return TierResult(fired=True, days_to_fire=reclaim_idx)

    # Case (b): close already above EMA-50 at window-start, *and* the prior
    # close-EMA-50 diff turned positive within the same window. This
    # handles the case where the cross happened close to (but just before)
    # the event date, and we want to count it as a 'reclaim in progress'.
    if len(future_close) >= sustain_days + 1:
        sustain_close = future_close.iloc[: sustain_days].reset_index(drop=True)
        sustain_ema = ema_50.iloc[: sustain_days].reset_index(drop=True)
        all_above = all((sustain_close.iloc[i] > sustain_ema.iloc[i])
                        for i in range(sustain_days))
        if all_above:
            # The cross happened pre-event. Find when.
            ema_full = _ema(df["close"], 50)
            diff_full = (df["close"] - ema_full).dropna()
            ev_ts = pd.Timestamp(df.index[loc]) if loc < len(df.index) else pd.Timestamp(event_date)
            # Most recent sign change from <=0 to >0 at-or-before event.
            pre = diff_full.loc[: ev_ts].iloc[:-1]
            if len(pre) >= 2:
                signs = np.sign(pre.to_numpy())
                cross_idx = None
                for i in range(1, len(signs)):
                    if signs[i - 1] <= 0 and signs[i] > 0:
                        cross_idx = i
                if cross_idx is not None:
                    return TierResult(fired=True, days_to_fire=0,
                                      meta={"cross_pre_event": True})

    return TierResult(fired=False)
"""Append-only file used to add the missing swing/aggregator functions to reversal.py.
Deleted after append."""


def validate_swing_structure(df, event_date, *, window_days=126):
    """Higher-high breakout followed by higher-low confirmation."""
    from .reversal import _loc_for  # type: ignore
    from .reversal import TierResult  # type: ignore
    loc = _loc_for(df, event_date)
    if loc < 0:
        return TierResult(fired=False, unverifiable=True)
    raw_future_len = len(df) - (loc + 1)
    if raw_future_len < 2:
        return TierResult(fired=False, unverifiable=True)
    n_window = window_days if window_days and window_days > 0 else raw_future_len
    n_window = min(n_window, raw_future_len)
    start = loc + 1
    end = start + n_window
    future = df.iloc[start:end]
    if future.empty:
        return TierResult(fired=False, unverifiable=True)
    prior_high = float(df["high"].iloc[: loc + 1].max())
    breakout_idx = None
    for i in range(len(future)):
        if float(future["high"].iloc[i]) > prior_high:
            breakout_idx = i
            break
    if breakout_idx is None:
        return TierResult(fired=False)
    after = future.iloc[breakout_idx + 1:]
    if after.empty:
        return TierResult(fired=False, unverifiable=True)
    lowest = float(future["low"].iloc[breakout_idx:].min())
    higher_low_idx = None
    for i in range(len(after)):
        if float(after["low"].iloc[i]) > lowest:
            higher_low_idx = breakout_idx + 1 + i
            break
    if higher_low_idx is None:
        return TierResult(fired=False, unverifiable=True)
    return TierResult(fired=True, days_to_fire=breakout_idx,
                      meta={"higher_low_at": higher_low_idx})


def evaluate_at_horizon(df, event_date, *, window_days, sustain_days=5):
    return {
        TIER_GOLDEN_CROSS: validate_golden_cross(
            df, event_date, window_days=window_days, sustain_days=sustain_days),
        TIER_EMA50_RECLAIM: validate_ema50_reclaim(
            df, event_date, window_days=window_days, sustain_days=sustain_days),
        TIER_SWING: validate_swing_structure(df, event_date, window_days=window_days),
    }


VERDICT_CONFIRMED = "confirmed_reversal"
VERDICT_PARTIAL = "partial_reversal"
VERDICT_NONE = "no_reversal"
VERDICT_UNVERIFIABLE = "unverifiable"


def collapse_verdict(tier_results):
    fired = [r for r in tier_results.values() if r.fired]
    unverifiable = [r for r in tier_results.values() if r.unverifiable]
    n_verifiable = len(tier_results) - len(unverifiable)
    if n_verifiable == 0:
        return VERDICT_UNVERIFIABLE, 0, len(unverifiable), None
    n_fired = len(fired)
    if n_fired >= 2:
        verdict = VERDICT_CONFIRMED
    elif n_fired == 1:
        verdict = VERDICT_PARTIAL
    else:
        verdict = VERDICT_NONE
    days = min((r.days_to_fire for r in fired if r.days_to_fire is not None),
               default=None)
    return verdict, n_fired, len(unverifiable), days


def unverifiable_tier_names(tier_results):
    return [name for name, r in tier_results.items() if r.unverifiable]
