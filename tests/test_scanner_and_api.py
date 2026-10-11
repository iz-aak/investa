"""Scanner + API tests using a fake DataProvider — no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from investa.config import Interval, ScanMode
from investa.pipeline.scanner import Scanner


class FakeProvider:
    """DataProvider stub returning deterministic synthetic OHLCV."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames

    def fetch_ohlcv_sync(
        self, ticker: str, *, interval: str = "1d", lookback_days: int = 260
    ) -> pd.DataFrame:
        try:
            return self._frames[ticker.upper()].copy()
        except KeyError as e:
            from investa.data.provider import DataFetchError

            raise DataFetchError(ticker, KeyError(str(e))) from e

    async def fetch_ohlcv(self, ticker, *, interval="1d", lookback_days=260):
        return self.fetch_ohlcv_sync(ticker, interval=interval, lookback_days=lookback_days)

    def validate_ticker_sync(self, ticker):
        return (ticker.upper() in self._frames, ticker.upper())

    async def validate_ticker(self, ticker):
        return self.validate_ticker_sync(ticker)


def _downtrend_frame(n: int = 250, start: float = 200.0):
    """Long-term downtrend guaranteed to pass the deep_reversal gate."""
    rng = np.random.default_rng(42)
    trend = np.linspace(start, start * 0.55, n)
    noise = rng.normal(0, start * 0.005, n)
    close = trend + noise
    # Push last bars sharply lower to drive RSI into oversold territory.
    close[-10:] = np.linspace(close[-10], close[-10] * 0.90, 10)
    high = close + 0.5
    low = close - 0.5
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": np.full(n, 1_000_000)},
        index=idx,
    )


def _uptrend_frame(n: int = 250, start: float = 100.0):
    """Strong uptrend that must fail the deep_reversal gate."""
    close = np.linspace(start, start * 1.5, n)
    high = close + 0.5
    low = close - 0.5
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": np.full(n, 1_000_000)},
        index=idx,
    )


def test_scanner_produces_candidate_for_downtrending_ticker(_isolated_data_dir):
    from investa.data.duckdb_store import init_duckdb
    from investa.database import init_db

    init_db()
    init_duckdb()

    provider = FakeProvider({"DOWN": _downtrend_frame(), "UP": _uptrend_frame()})
    result = Scanner(provider).scan_sync(
        ["DOWN", "UP"], mode=ScanMode.deep_reversal, interval=Interval.daily, persist=True
    )

    tickers = [c.ticker for c in result.candidates]
    assert "DOWN" in tickers, "downtrend should pass deep_reversal gate"
    assert "UP" not in tickers, "uptrend must be filtered by gate"

    down = next(c for c in result.candidates if c.ticker == "DOWN")
    assert down.passed_gate is True
    assert down.score > 0.0
    assert "rsi" in down.signals
    assert down.risk["suggested_stop"] < down.close


def test_scanner_persists_to_duckdb(_isolated_data_dir):
    from investa.data.duckdb_store import connect, init_duckdb
    from investa.database import init_db

    init_db()
    init_duckdb()

    provider = FakeProvider({"DOWN": _downtrend_frame()})
    Scanner(provider).scan_sync(["DOWN"], mode=ScanMode.deep_reversal, persist=True)

    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT ticker, score, band, suggested_stop FROM signal_history"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "DOWN"
    assert rows[0][1] > 0
    assert rows[0][3] is not None


def test_scanner_records_failed_tickers_without_aborting(_isolated_data_dir):
    from investa.data.duckdb_store import init_duckdb
    from investa.database import init_db

    init_db()
    init_duckdb()

    provider = FakeProvider({"DOWN": _downtrend_frame()})  # "BAD" unknown → failure
    result = Scanner(provider).scan_sync(["DOWN", "BAD"], mode=ScanMode.deep_reversal)

    assert any(c.ticker == "DOWN" for c in result.candidates)
    assert any(f["ticker"] == "BAD" and f["stage"] == "fetcher" for f in result.failed)


def test_screener_endpoint_returns_envelope(_isolated_data_dir, monkeypatch):
    from investa.api import screener as screener_module
    from investa.main import create_app

    fake = FakeProvider({"DOWN": _downtrend_frame(), "UP": _uptrend_frame()})
    monkeypatch.setattr(screener_module, "_scanner", Scanner(fake))

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/screener", params={"tickers": "DOWN,UP", "min_score": 0})
        assert resp.status_code == 200

        body = resp.json()
        assert body["success"] is True
        assert body["meta"]["requested"] == 2
        assert body["meta"]["mode"] == "deep_reversal"
        tickers = [c["ticker"] for c in body["data"]]
        assert "DOWN" in tickers
        assert "UP" not in tickers


def test_screener_endpoint_rejects_empty_ticker_list(_isolated_data_dir):
    from investa.main import create_app

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/screener", params={"tickers": ""})
        assert resp.status_code in (400, 422)


# ── Integration (real yfinance) — opt-in with `-m integration` ─────

@pytest.mark.integration
def test_scanner_against_real_yfinance(_isolated_data_dir):
    from investa.data.duckdb_store import init_duckdb
    from investa.data.provider import YFinanceProvider
    from investa.database import init_db

    init_db()
    init_duckdb()

    scanner = Scanner(YFinanceProvider())
    result = scanner.scan_sync(
        ["AAPL", "MSFT", "NVDA", "META", "GOOGL"],
        mode=ScanMode.deep_reversal,
    )
    assert result.scan_id
    accounted = len(result.candidates) + len(result.failed)
    assert 0 <= accounted <= 5
