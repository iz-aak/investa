"""Tests for the backtest layer."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from investa.backtest.replay import (
    SignalEvent, _evaluate_at, _sim_dates, replay_ticker,
)
from investa.backtest.report import build_report, join as join_streams
from investa.backtest.returns import attach_forward_returns, simulate_trade
from investa.backtest.runner import BacktestRequest, BacktestRunner
from investa.config import Interval, ScanMode
from investa.data.provider import DataFetchError


def _linear_df(n, start=100.0, step=-0.1):
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    close = np.array([start + i * step for i in range(n)], dtype=float)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {"open": close, "high": high, "low": low,
         "close": close, "volume": np.full(n, 1_000_000)},
        index=idx,
    )


def _uptrend_df(n=260):
    return _linear_df(n, start=100.0, step=0.1)


def _mock_flat_df(sim_date, n=250):
    idx = pd.date_range(sim_date - pd.Timedelta(days=n), periods=n + 30,
                        freq="D", tz="UTC")
    close = np.full(len(idx), 100.0)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {"open": close, "high": high, "low": low,
         "close": close, "volume": np.full(len(idx), 1_000_000)},
        index=idx,
    )


def _signal_event(score, band, fired, ret_10d=None):
    return SignalEvent(
        ticker="X",
        sim_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
        close=100.0, score=score, band=band, passed_gate=True,
        gate_reason="deep_reversal_confirmed",
        signal_strengths={k: 0.5 for k in fired},
        signal_flags={k: (k in fired) for k in (
            "rsi", "macd", "bollinger", "volume", "obv", "stochastic", "weekly")},
        weekly_confirmed="weekly" in fired,
        risk_stop=98.0, risk_target=104.0, risk_reward=2.0,
    )

# --- Replay correctness ---


def test_sim_dates_exclude_first_and_last_day():
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    sims = _sim_dates(idx, freq="D")
    assert sims[0] > idx[0]
    assert sims[-1] < idx[-1]


def test_replay_strict_no_lookahead():
    """Plant large future moves AFTER a sim date and assert the replay at the
    earlier sim date does NOT see them. The trust anchor of the engine."""
    df = _linear_df(n=300, start=300.0, step=-0.05)
    early_sim = df.index[220]
    # Modify strictly FUTURE bars (after early_sim) by +300%.
    df.iloc[-30:, df.columns.get_loc("close")] *= 4.0
    df.iloc[-30:, df.columns.get_loc("high")] *= 4.0
    df.iloc[-30:, df.columns.get_loc("low")] *= 4.0

    slice_loc = df.loc[:early_sim]
    ev = _evaluate_at(slice_loc, ticker="X", mode=ScanMode.deep_reversal,
                       interval=Interval.daily)
    assert ev is not None, f"slice produced no event (gate: {slice_loc['close'].iloc[-1]})"
    # The recorded close MUST equal the bar at early_sim, untouched by mutation.
    assert ev.close == float(slice_loc["close"].iloc[-1])


def test_replay_returns_events_sorted_and_unique():
    df = _linear_df(n=300, start=200.0, step=-0.05)
    events = replay_ticker(df, ticker="ZZZ", mode=ScanMode.deep_reversal)
    assert events
    sim_dates = [e.sim_date for e in events]
    assert sim_dates == sorted(sim_dates)
    assert len(sim_dates) == len(set(sim_dates))


def test_replay_skips_too_short_frame():
    assert replay_ticker(_linear_df(n=100), ticker="ZZZ",
                         mode=ScanMode.deep_reversal) == []


def test_replay_uptrend_no_events_in_deep_reversal_mode():
    assert replay_ticker(_uptrend_df(), ticker="UP",
                         mode=ScanMode.deep_reversal) == []


# --- Forward returns ---


def test_forward_returns_exact_on_known_frame():
    n = 300
    df = _linear_df(n=n, start=120.0, step=-0.05)
    sim_loc = 205
    ev = SignalEvent(
        ticker="X", sim_date=df.index[sim_loc].to_pydatetime(),
        close=float(df["close"].iloc[sim_loc]),
        score=10.0, band="Moderate", passed_gate=True,
        gate_reason="deep_reversal_confirmed",
        signal_strengths={}, signal_flags={}, weekly_confirmed=False,
        risk_stop=None, risk_target=None, risk_reward=None,
    )
    fwd = attach_forward_returns([ev], df, horizons=(5, 10, 20))[0]
    r5 = fwd["returns"][5]
    expected_close_t5 = float(df["close"].iloc[sim_loc + 5])
    assert r5 == pytest.approx((expected_close_t5 - ev.close) / ev.close)


def test_forward_returns_censored_at_right_edge():
    n = 250
    df = _linear_df(n=n, start=100.0, step=-0.05)
    sim_loc = n - 2
    ev = SignalEvent(
        ticker="X", sim_date=df.index[sim_loc].to_pydatetime(),
        close=float(df["close"].iloc[sim_loc]),
        score=10.0, band="Moderate", passed_gate=True,
        gate_reason="deep_reversal_confirmed",
        signal_strengths={}, signal_flags={}, weekly_confirmed=False,
        risk_stop=None, risk_target=None, risk_reward=None,
    )
    fwd = attach_forward_returns([ev], df, horizons=(5, 10, 20))[0]
    assert fwd["returns"][20] is None


# --- Trade simulation ---


def test_simulate_trade_ambiguous_bar_prefers_stop():
    n = 250
    df = _linear_df(n=n, start=120.0, step=-0.05)
    sim_loc = 200
    ev_close = float(df["close"].iloc[sim_loc])
    stop = ev_close - 1.0
    target = ev_close + 1.0
    future_loc = sim_loc + 1
    df.iat[future_loc, df.columns.get_loc("low")] = stop - 0.01
    df.iat[future_loc, df.columns.get_loc("high")] = target + 0.01
    df.iat[future_loc, df.columns.get_loc("open")] = ev_close
    df.iat[future_loc, df.columns.get_loc("close")] = ev_close
    ev = SignalEvent(
        ticker="X", sim_date=df.index[sim_loc].to_pydatetime(),
        close=ev_close, score=10.0, band="Moderate",
        passed_gate=True, gate_reason="x",
        signal_strengths={}, signal_flags={}, weekly_confirmed=False,
        risk_stop=stop, risk_target=target, risk_reward=None,
    )
    trade = simulate_trade(ev, df, max_hold=20)
    assert trade.label == "stop_hit"
    assert trade.realized_rr == pytest.approx(-1.0)


def test_simulate_trade_timeout_when_neither_hit():
    n = 250
    df = _linear_df(n=n, start=120.0, step=-0.05)
    sim_loc = 200
    ev_close = float(df["close"].iloc[sim_loc])
    stop = ev_close * 0.5
    target = ev_close * 2.0
    ev = SignalEvent(
        ticker="X", sim_date=df.index[sim_loc].to_pydatetime(),
        close=ev_close, score=10.0, band="Moderate",
        passed_gate=True, gate_reason="x",
        signal_strengths={}, signal_flags={}, weekly_confirmed=False,
        risk_stop=stop, risk_target=target, risk_reward=None,
    )
    trade = simulate_trade(ev, df, max_hold=5)
    assert trade.label == "timeout"

# --- Report aggregation ---


def test_report_includes_baseline_band_alpha_and_per_signal():
    e1 = _signal_event(score=70.0, band="Strong", fired={"rsi", "macd"})
    e2 = _signal_event(score=10.0, band="Weak", fired=set())
    fwd1 = {"returns": {5: 0.02, 10: 0.05, 20: 0.06},
            "max_drawdown": -0.03, "max_runup": 0.07}
    fwd2 = {"returns": {5: -0.01, 10: -0.02, 20: -0.04},
            "max_drawdown": -0.05, "max_runup": 0.01}
    tr1 = simulate_trade(e1, _mock_flat_df(e1.sim_date))
    tr2 = simulate_trade(e2, _mock_flat_df(e2.sim_date))
    joined = join_streams([e1, e2], [fwd1, fwd2], [tr1, tr2])
    report = build_report(joined, horizons=(5, 10, 20))
    assert "overall_baseline" in report
    assert "per_band" in report and len(report["per_band"]) == 5
    assert "alpha_vs_baseline" in report
    assert "per_signal_contribution" in report
    keys = set(report["per_signal_contribution"].keys())
    assert {"rsi", "macd", "volume", "obv"}.issubset(keys)


def test_report_alpha_equals_band_minus_baseline():
    e1 = _signal_event(score=80.0, band="Very Strong", fired={"rsi"})
    e2 = _signal_event(score=10.0, band="Weak", fired=set())
    fwd1 = {"returns": {5: 0.02, 10: 0.08, 20: 0.09},
            "max_drawdown": -0.04, "max_runup": 0.10}
    fwd2 = {"returns": {5: 0.00, 10: 0.01, 20: 0.02},
            "max_drawdown": -0.05, "max_runup": 0.03}
    tr1 = simulate_trade(e1, _mock_flat_df(e1.sim_date))
    tr2 = simulate_trade(e2, _mock_flat_df(e2.sim_date))
    joined = join_streams([e1, e2], [fwd1, fwd2], [tr1, tr2])
    r = build_report(joined, horizons=(5, 10, 20))
    baseline = r["overall_baseline"]["mean_return_10d"]["mean"]
    band_val = next(b for b in r["per_band"] if b["band"] == "Very Strong")["mean_return_10d"]["mean"]
    assert r["alpha_vs_baseline"]["Very Strong"]["return_10d_alpha"]["alpha_return"] == pytest.approx(
        band_val - baseline, abs=1e-6)


# --- Runner ---


class _StubProvider:
    def __init__(self, frames, raises=None):
        self._frames = frames
        self._raises = raises or set()

    def fetch_ohlcv_sync(self, ticker, *, interval="1d", lookback_days=260):
        if ticker.upper() in self._raises:
            raise DataFetchError(ticker, RuntimeError("boom"))
        return self._frames[ticker.upper()].copy()


def test_runner_records_jobs_and_persists_report(_isolated_data_dir):
    from investa.data.duckdb_store import init_duckdb
    from investa.database import init_db
    init_db()
    init_duckdb()
    frames = {f"T{i}": _linear_df(260, start=200.0, step=-0.05) for i in range(3)}
    frames["TFAIL"] = _linear_df(50)
    runner = BacktestRunner(_StubProvider(frames, raises={"BAD"}))
    req = BacktestRequest(tickers=["T0", "T1", "T2", "TFAIL", "BAD"],
                          lookback_days=260)
    out = runner.run_sync(req, persist=True)
    assert out["job_id"]
    assert isinstance(out["report"], dict)
    assert out["report"]["n_events"] >= 2
    assert out["report"]["metadata"]["tickers_attempted"] == 5
    assert out["report"]["metadata"]["tickers_failed"] >= 1
    assert "failed_tickers" in out["report"]
    fetched = BacktestRunner.get_report(out["job_id"])
    assert fetched is not None
    assert fetched["status"] == "done"


def test_api_body_validation():
    from investa.api.backtest import BacktestRequestBody

    body = BacktestRequestBody(lookback_days=120)
    assert body.lookback_days == 120
    with pytest.raises(Exception):
        BacktestRequestBody(lookback_days=10)
