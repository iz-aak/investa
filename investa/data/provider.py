"""Data provider abstraction + yfinance implementation.

The rest of the pipeline talks to :class:`DataProvider`, never to yfinance
directly. This is the single seam that lets us swap in Polygon, Tiingo, or
Alpaca later without touching the indicator/signal/scoring code.

Design notes
------------
* **Synchronous underneath, awaitable on top.** yfinance is a blocking library.
  We expose an ``async`` API (``fetch_ohlcv``) that offloads the blocking call
  to a thread pool via ``asyncio.to_thread`` so it doesn't stall the FastAPI
  event loop. Callers in synchronous contexts (tests, worker) can use
  :meth:`YFinanceProvider.fetch_ohlcv_sync` directly.
* **Retry with exponential backoff + jitter.** Yahoo throttles aggressively;
  transient failures are the norm, not the exception.
* **TTLCache on raw OHLCV frames.** Same (ticker, interval, lookback) tuple
  called twice within ``YF_CACHE_TTL_SECONDS`` returns the cached frame.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from typing import Protocol, runtime_checkable

import pandas as pd
import yfinance as yf
from cachetools import TTLCache

from ..config import get_settings
from ..logging_setup import get_logger

log = get_logger(__name__)

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class DataFetchError(RuntimeError):
    """Raised when a ticker cannot be fetched after all retries."""

    def __init__(self, ticker: str, cause: Exception | None = None) -> None:
        msg = f"failed to fetch {ticker}: {cause!r}" if cause else f"failed to fetch {ticker}"
        super().__init__(msg)
        self.ticker = ticker
        self.cause = cause


@runtime_checkable
class DataProvider(Protocol):
    """Protocol every data source must satisfy.

    Implementations must return a DataFrame with a UTC ``DatetimeIndex`` and
    columns ``open, high, low, close, volume`` — lowercase, in that order.
    Rows with NaN close must already be dropped.
    """

    async def fetch_ohlcv(
        self, ticker: str, *, interval: str = "1d", lookback_days: int = 260
    ) -> pd.DataFrame: ...

    async def validate_ticker(self, ticker: str) -> tuple[bool, str | None]:
        """Return (is_valid, resolved_long_name)."""
        ...



class YFinanceProvider:
    """Concrete :class:`DataProvider` backed by the yfinance library."""

    def __init__(
        self,
        *,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        cache_ttl_seconds: int | None = None,
        cache_maxsize: int = 1024,
    ) -> None:
        settings = get_settings()
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds
        ttl = cache_ttl_seconds if cache_ttl_seconds is not None else settings.yf_cache_ttl_seconds
        self._cache: TTLCache = TTLCache(maxsize=cache_maxsize, ttl=ttl)
        # cachetools is not thread-safe; FastAPI serves requests concurrently.
        self._cache_lock = threading.Lock()

    # ── public async API ───────────────────────────────────────────
    async def fetch_ohlcv(
        self, ticker: str, *, interval: str = "1d", lookback_days: int = 260
    ) -> pd.DataFrame:
        return await asyncio.to_thread(
            self.fetch_ohlcv_sync, ticker, interval=interval, lookback_days=lookback_days
        )

    async def validate_ticker(self, ticker: str) -> tuple[bool, str | None]:
        return await asyncio.to_thread(self.validate_ticker_sync, ticker)

    # ── public sync API (worker / tests) ───────────────────────────
    def fetch_ohlcv_sync(
        self, ticker: str, *, interval: str = "1d", lookback_days: int = 260
    ) -> pd.DataFrame:
        key = (ticker.upper(), interval, lookback_days)
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()

        raw = self._download_with_retry(ticker, interval=interval, lookback_days=lookback_days)
        cleaned = self._normalise(raw, ticker)

        with self._cache_lock:
            self._cache[key] = cleaned
        return cleaned.copy()

    def validate_ticker_sync(self, ticker: str) -> tuple[bool, str | None]:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as exc:  # noqa: BLE001 — yfinance raises broad errors
            log.warning("provider.validate_failed", ticker=ticker, error=str(exc))
            return False, None

        # yfinance returns a mostly-empty dict for unknown tickers.
        name = info.get("longName") or info.get("shortName")
        symbol = info.get("symbol") or info.get("underlyingSymbol")
        return (bool(symbol) and bool(name)), name

    # ── internals ──────────────────────────────────────────────────
    def _download_with_retry(
        self, ticker: str, *, interval: str, lookback_days: int
    ) -> pd.DataFrame:
        period = f"{lookback_days}d"
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                df = yf.download(
                    tickers=ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by="column",
                )
                if df is None or df.empty:
                    raise DataFetchError(ticker, RuntimeError("empty frame"))
                return df
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                sleep_s = self._backoff_base * (2 ** (attempt - 1))
                sleep_s += random.uniform(0, self._backoff_base)  # jitter
                log.warning(
                    "provider.retry",
                    ticker=ticker,
                    attempt=attempt,
                    sleep_s=round(sleep_s, 3),
                    error=str(exc),
                )
                time.sleep(sleep_s)

        raise DataFetchError(ticker, last_exc)

    @staticmethod
    def _normalise(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Return a lowercase-OHLCV frame with NaN close rows dropped."""
        # yfinance often returns a MultiIndex on columns even for a single
        # ticker (level 0 = field, level 1 = ticker). Flatten it.
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker.upper(), axis=1, level=1, drop_level=True)
            except (KeyError, ValueError):
                df = df.copy()
                df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=str.lower)
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataFetchError(
                ticker, RuntimeError(f"missing columns after normalise: {missing}")
            )

        df = df[list(_REQUIRED_COLUMNS)].copy()
        df = df.dropna(subset=["close"])
        df.index = pd.to_datetime(df.index, utc=True)
        return df
