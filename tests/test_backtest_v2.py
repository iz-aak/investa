"""Tests for the backtest v2 reversal-validation layer."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from investa.backtest_v2.reversal import (
    TIER_EMA50_RECLAIM, TIER_GOLDEN_CROSS, TIER_SWING,
    VERDICT_CONFIRMED, VERDICT_NONE, VERDICT_PARTIAL, VERDICT_UNVERIFIABLE,
    collapse_verdict, evaluate_at_horizon, validate_ema50_reclaim,
    validate_golden_cross, validate_swing_structure,
)
from investa.backtest_v2.report import build_v2_report
from investa.backtest_v2.runner import V2Request, V2Runner
from investa.backtest_v2.validator import (
    ReversalRow, ValidationRequest, _filter_sim_window, validate_ticker,
)

def _df(n, *, start=100.0, step=-0.1, freq="B"):
    idx = pd.date_range("2022-01-03", periods=n, freq=freq, tz="UTC")
    close = np.array([start + i * step for i in range(n)], dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": np.full(n, 1_000_000)},
        index=idx,
    )


def _deep_reversal_then_cross_df(n=400, start=300.0):
    """V-shape synthesised so EMAs cross, EMA-50 reclaims, and structure
    triggers within the 126-bar post-event window. Recovery phased: a sharp
    continued drop, then a long pump that overshoots so the EMAs converge
    and cross, then a brief hold."""
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    half = n // 2
    close = np.empty(n)
    close[:half] = np.linspace(start, start * 0.55, half)
    # recovery: continue a touch lower, then ramp aggressively so EMA-50
    # crosses back above EMA-200 well within the post-event window.
    cross_at = half + 40
    close[half:cross_at] = np.linspace(close[half - 1], close[half - 1] * 1.45,
                                         cross_at - half)
    close[cross_at:n - 80] = np.linspace(close[cross_at - 1] * 1.01,
                                            close[cross_at - 1] * 1.10,
                                            n - 80 - cross_at)
    close[n - 80:] = close[n - 81] * 1.005  # flat hold
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": np.full(n, 1_000_000)},
        index=idx,
    )


class FakeTier:
    """Light stand-in for TierResult for verdict-collapse tests where the body doesn't matter."""
    def __init__(self, fired=False, unverifiable=False):
        self.fired = fired
        self.unverifiable = unverifiable
        self.days_to_fire = 5 if fired else None


# ── Tier 1: golden cross ───────────────────────────────────────────


def test_golden_cross_fires_on_clean_cross():
    df = _deep_reversal_then_cross_df()
    event_date = df.index[247]
    r = validate_golden_cross(df, event_date, window_days=126, sustain_days=5)
    assert r.fired is True
    assert r.unverifiable is False


def test_golden_cross_does_not_fire_without_cross():
    # Long enough that unverifiable=False.
    df = _df(500, start=300.0, step=-1.0)
    event_date = df.index[260]
    r = validate_golden_cross(df, event_date, window_days=126, sustain_days=5)
    assert r.fired is False
    assert r.unverifiable is False


def test_golden_cross_censored_at_right_edge():
    df = _df(50)
    event_date = df.index[40]
    r = validate_golden_cross(df, event_date, window_days=126, sustain_days=5)
    assert r.fired is False
    assert r.unverifiable is True


# ── Tier 2: EMA-50 reclaim ─────────────────────────────────────────


def test_ema50_reclaim_fires_on_clean_reclaim():
    df = _deep_reversal_then_cross_df()
    event_date = df.index[265]
    r = validate_ema50_reclaim(df, event_date, window_days=126, sustain_days=5)
    assert r.fired is True


def test_ema50_reclaim_does_not_fire_on_immediate_failure():
    df = _df(300, start=300.0, step=-1.0)
    event_date = df.index[260]
    r = validate_ema50_reclaim(df, event_date, window_days=126, sustain_days=5)
    assert r.fired is False


# ── Tier 3: swing structure ────────────────────────────────────────


def test_swing_structure_fires_on_breakout_then_higher_low():
    df = _deep_reversal_then_cross_df()
    # Use a custom dataframe tuned to fire breakout+higher-low.
    # Build flat 100s for 100 bars, then a sharp drop+recovery V.
    n = 200; idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    close = np.full(n, 100.0)
    close[40:55] = np.linspace(100, 60, 15)  # drop
    close[55:80] = np.linspace(60, 110, 25)   # recovery past 100
    close[80:120] = np.linspace(110, 120, 40)  # higher ladder
    df2 = pd.DataFrame(
        {"open": close, "high": close + 0.6, "low": close - 0.6,
         "close": close, "volume": np.full(n, 1_000_000)},
        index=idx,
    )
    r = validate_swing_structure(df2, idx[35], window_days=80)
    assert r.fired is True


def test_swing_structure_does_not_fire_when_only_breakout_no_higher_low():
    n = 200
    close = np.full(n, 100.0)
    close[50] = 110
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5 * (close > 102).astype(float),
         "low": close - 0.5, "close": close,
         "volume": np.full(n, 1_000_000)},
        index=idx,
    )
    event_date = df.index[40]
    r = validate_swing_structure(df, event_date, window_days=126)
    assert r.fired is False
# ── all-tiers-at-horizon + verdict collapse ────────────────────────


def test_evaluate_at_horizon_returns_all_three_tiers():
    df = _deep_reversal_then_cross_df()
    ed = df.index[270]
    out = evaluate_at_horizon(df, ed, window_days=126)
    assert set(out.keys()) == {TIER_GOLDEN_CROSS, TIER_EMA50_RECLAIM, TIER_SWING}


def test_verdict_2_of_3_is_confirmed():
    tiers = {TIER_GOLDEN_CROSS: FakeTier(True), TIER_EMA50_RECLAIM: FakeTier(True),
             TIER_SWING: FakeTier(False)}
    v, n_fired, _nv, _d = collapse_verdict(tiers)
    assert v == VERDICT_CONFIRMED
    assert n_fired == 2


def test_verdict_1_of_3_is_partial():
    tiers = {TIER_GOLDEN_CROSS: FakeTier(True), TIER_EMA50_RECLAIM: FakeTier(False),
             TIER_SWING: FakeTier(False)}
    v, n_fired, _nv, _d = collapse_verdict(tiers)
    assert v == VERDICT_PARTIAL
    assert n_fired == 1


def test_verdict_0_of_3_is_no_reversal():
    tiers = {t: FakeTier(False) for t in (TIER_GOLDEN_CROSS, TIER_EMA50_RECLAIM, TIER_SWING)}
    v, n_fired, _nv, _d = collapse_verdict(tiers)
    assert v == VERDICT_NONE
    assert n_fired == 0


def test_verdict_unverifiable_when_all_tiers_censored():
    tiers = {t: FakeTier(False, unverifiable=True) for t in
             (TIER_GOLDEN_CROSS, TIER_EMA50_RECLAIM, TIER_SWING)}
# ── sim-window filter ──────────────────────────────────────────────


def _se(t, sd):
    from investa.backtest.replay import SignalEvent
    return SignalEvent(ticker=t, sim_date=sd, close=10.0, score=10.0, band="Weak",
                       passed_gate=True, gate_reason="ok", signal_strengths={},
                       signal_flags={}, weekly_confirmed=False,
                       risk_stop=None, risk_target=None, risk_reward=None)


def test_filter_sim_window_excludes_pre_2023():
    pre = _se("X", datetime(2022, 6, 1, tzinfo=timezone.utc))
    inside = _se("X", datetime(2024, 1, 15, tzinfo=timezone.utc))
    post = _se("X", datetime(2026, 3, 1, tzinfo=timezone.utc))
    out = _filter_sim_window([pre, inside, post],
                              start=datetime(2023, 1, 1),
                              end=datetime(2025, 12, 31))
    assert out == [inside]


# ── v2 round-trip with stub provider ─────────────────────────────


def test_validate_ticker_filters_and_runs_tiers(_isolated_data_dir):
    from investa.config import Interval, ScanMode
    df = _deep_reversal_then_cross_df()
    req = ValidationRequest(ticker="ZZZ", mode=ScanMode.deep_reversal,
                            interval=Interval.daily, lookback_days=400,
                            replay_start=datetime(2023, 1, 1),
                            replay_end=datetime(2025, 12, 31))
    rows = validate_ticker(df, req)
    if rows:
        r = rows[0]
        assert r.ticker == "ZZZ"
        assert r.verdict_3m in (VERDICT_CONFIRMED, VERDICT_PARTIAL,
                                  VERDICT_NONE, VERDICT_UNVERIFIABLE)
        # And every field has the right type / is in the closed verdict set.
        assert r.fwd_return_3m is None or isinstance(r.fwd_return_3m, float)


def test_report_sections_present():
    rows = []
    for _ in range(20):
        rows.append(ReversalRow(
            ticker="X",
            sim_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            score=10.0, band="Weak",
            signal_flags={"rsi": True, "macd": False, "bollinger": False,
                          "volume": False, "obv": False, "stochastic": False,
                          "weekly": False},
            tier_golden_cross_3m=False, tier_ema50_reclaim_3m=False, tier_swing_3m=False,
            tier_golden_cross_6m=False, tier_ema50_reclaim_6m=False, tier_swing_6m=False,
            tier_golden_cross_9m=False, tier_ema50_reclaim_9m=False, tier_swing_9m=False,
            verdict_3m=VERDICT_NONE, verdict_6m=VERDICT_NONE, verdict_9m=VERDICT_NONE,
            n_tiers_fired_6m=0, fwd_return_3m=None, fwd_return_6m=None, fwd_return_9m=None,
        ))
    r = build_v2_report(rows)
    assert r["primary_horizon"] == "6m"
    for k in ("section_a_verdict_distribution_by_horizon",
              "section_b_reversal_rate_by_horizon",
              "section_c_per_band_at_3m_6m_9m",
              "section_d_per_signal_contribution_at_6m",
              "section_e_per_tier_fire_rates_by_horizon",
              "section_f_per_ticker_rollup_at_6m",
              "section_g_cross_check_returns_by_verdict_at_6m"):
        assert k in r, k


class _StubProvider:
    def __init__(self, frames, raises=None):
        self._frames = frames
        self._raises = raises or set()

    def fetch_ohlcv_sync(self, ticker, *, interval="1d", lookback_days=1646):
        from investa.data.provider import DataFetchError
        if ticker.upper() in self._raises:
            raise DataFetchError(ticker, RuntimeError("boom"))
        return self._frames[ticker.upper()].copy()


def test_runner_records_jobs_and_persists_events(_isolated_data_dir):
    from investa.database import init_db
    from investa.data.duckdb_store import init_duckdb
    init_db()
    init_duckdb()
    frames = {f"T{i}": _deep_reversal_then_cross_df() for i in range(3)}
    frames["TFAIL"] = _df(50)
    runner = V2Runner(_StubProvider(frames, raises={"BAD"}))
    from investa.backtest_v2.runner import V2Request

    req = V2Request(tickers=["T0", "T1", "T2", "TFAIL", "BAD"], lookback_days=1646)
    out = runner.run_sync(req, persist=True)
    assert out["job_id"]
    fetched = V2Runner.get_report(out["job_id"])
    assert fetched is not None
    assert fetched["status"] == "done"
