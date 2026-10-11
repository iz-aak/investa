"""Downtrend gate filters.

The gate is a hard prerequisite: a stock that does not pass the selected
gate is excluded from scoring entirely. Investa is specifically a *reversal*
screener; wasting compute on stocks not in the intended setup dilutes the
output.

Two modes:

* ``deep_reversal`` — the 'falling-knife' setup: price below both EMAs, with
  EMA-50 already below EMA-200 (post-death-cross configuration).
* ``pullback`` — 'buy-the-dip' in a longer-term uptrend: close above EMA-200
  (long-term uptrend intact) but close below EMA-50 (pulled back to the
  mean), and RSI not yet deeply oversold on the *latest* bar (we want the
  dip, not a falling knife inside an uptrend — the deep-oversold reading is
  left to the signal layer to score).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import ScanMode


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str  # human-readable, useful for /api/screener/{ticker} explainability


def evaluate_gate(latest: pd.Series, mode: ScanMode) -> GateResult:
    """Evaluate the gate against the latest indicator-enriched row.

    ``latest`` must contain: ``close``, ``ema_50``, ``ema_200``.
    Missing values (indicator warm-up not satisfied) always fail the gate —
    we never assume trend direction we cannot measure.
    """
    if mode is ScanMode.deep_reversal:
        return _deep_reversal(latest)
    if mode is ScanMode.pullback:
        return _pullback(latest)
    raise ValueError(f"unknown gate mode: {mode!r}")


def _deep_reversal(latest: pd.Series) -> GateResult:
    close = latest.get("close")
    ema_50 = latest.get("ema_50")
    ema_200 = latest.get("ema_200")

    if pd.isna(close) or pd.isna(ema_50) or pd.isna(ema_200):
        return GateResult(False, "warmup:ema_not_ready")

    if not (close < ema_50):
        return GateResult(False, "close_not_below_ema50")
    if not (ema_50 < ema_200):
        return GateResult(False, "ema50_not_below_ema200")

    return GateResult(True, "deep_reversal_confirmed")


def _pullback(latest: pd.Series) -> GateResult:
    """Buy-the-dip in an established uptrend.

    Conditions:
      * close > ema_200          — long-term uptrend intact
      * close < ema_50           — pulled back below the medium-term mean
      * ema_50 > ema_200         — medium-term still above long-term (no death cross)
    """
    close = latest.get("close")
    ema_50 = latest.get("ema_50")
    ema_200 = latest.get("ema_200")

    if pd.isna(close) or pd.isna(ema_50) or pd.isna(ema_200):
        return GateResult(False, "warmup:ema_not_ready")

    if not (close > ema_200):
        return GateResult(False, "close_not_above_ema200")
    if not (close < ema_50):
        return GateResult(False, "close_not_below_ema50")
    if not (ema_50 > ema_200):
        return GateResult(False, "ema50_not_above_ema200")

    return GateResult(True, "pullback_in_uptrend_confirmed")
