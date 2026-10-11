"""Curated ticker baskets for the backtest layer.

The default 50-ticker basket spans all 11 GICS sectors with an emphasis on
**diversity over representativeness** — we explicitly do *not* use the S&P 500
top 50 because that would over-weight mega-cap tech and effectively turn any
backtest into a sector bet. Every name is large-cap and liquid so yfinance
will return clean, complete history.

Override at API time by passing ``tickers=[...]`` in the POST body.
"""

from __future__ import annotations

DEFAULT_BACKTEST_BASKET: tuple[str, ...] = (
    # ── Technology (8) ─────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "ORCL", "CRM", "ADBE", "CSCO", "IBM",
    # ── Communication Services (4) ────────────────────────────────
    "GOOGL", "META", "NFLX", "DIS",
    # ── Consumer Discretionary (5) ────────────────────────────────
    "AMZN", "TSLA", "HD", "NKE", "MCD",
    # ── Consumer Staples (4) ──────────────────────────────────────
    "PG", "KO", "PEP", "WMT",
    # ── Financials (6) ─────────────────────────────────────────────
    "JPM", "BAC", "WFC", "GS", "MS", "BLK",
    # ── Health Care (5) ───────────────────────────────────────────
    "JNJ", "PFE", "UNH", "LLY", "ABBV",
    # ── Industrials (5) ───────────────────────────────────────────
    "CAT", "HON", "GE", "UPS", "RTX",
    # ── Materials (3) ──────────────────────────────────────────────
    "LIN", "FCX", "APD",
    # ── Energy (3) ────────────────────────────────────────────────
    "XOM", "CVX", "COP",
    # ── Utilities (3) ──────────────────────────────────────────────
    "NEE", "DUK", "SO",
    # ── Real Estate (4) ───────────────────────────────────────────
    "PLD", "AMT", "CCI", "EQIX",
)


def default_basket() -> list[str]:
    """Return a fresh list (callers can mutate safely)."""
    return list(DEFAULT_BACKTEST_BASKET)
