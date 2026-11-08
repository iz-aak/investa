"""NASDAQ official symbol directory download + parse.

Pulls pipe-delimited CSVs from the NASDAQ Trader FTP site to enumerate all
US-listed securities tradable via NASDAQ (NASDAQ itself + NYSE + AMEX + ARCA).
This is the single source of truth for Stage 1 raw ticker list -- not an
index membership list.

The two files are refreshed daily and contain Test Issue / Financial Status /
ETF flags so we can filter down to common stocks only.
"""

from __future__ import annotations

import io
import random
import time
import urllib.request
from urllib.error import URLError

import pandas as pd

from ..logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_NASDAQ_LISTED_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
)
DEFAULT_OTHER_LISTED_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
)

# Map NASDAQ exchange codes -> canonical listing names used by the rest of
# the codebase. Unknown codes surface as OTHER(<code>) rather than dropped so
# we keep an audit trail when NASDAQ changes its encoding scheme.
EXCHANGE_CODE_MAP: dict[str, str] = {
    "Q": "NASDAQ",
    "N": "NYSE",
    "A": "AMEX",
    "P": "NYSE ARCA",
    "Z": "CBOE",
    "V": "IEX",
    # Market Category codes used by nasdaqlisted.txt (no Exchange column).
    "S": "NASDAQ",
    "G": "NASDAQ",
}

# Ticker symbol suffixes indicating non-common-stock securities: preferred
# shares, warrants, units, rights etc. We want plain common stock only.
_NON_COMMON_STOCK_SUFFIXES: tuple[str, ...] = (".PR", ".W", ".U", ".R", ".P")

# Required columns we expect in each source file. Surfaced as a loud error
# when NASDAQ changes schema rather than silently dropped rows.
_NASDAQ_REQUIRED = {"Symbol", "Security Name", "Market Category", "Test Issue", "ETF"}
_OTHER_REQUIRED = {"ACT Symbol", "Security Name", "Exchange", "Test Issue", "ETF"}


class NasdaqDirectoryError(RuntimeError):
    """Raised when the NASDAQ directory cannot be downloaded or parsed."""


def _download_with_retry(
    url: str,
    *,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
    backoff_base_seconds: float = 1.0,
) -> bytes:
    """Download url to bytes with retry + exponential backoff + jitter.

    Mirrors the retry philosophy used by YFinanceProvider.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
                return resp.read()
        except (URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_s = backoff_base_seconds * (2 ** (attempt - 1))
            sleep_s += random.uniform(0, backoff_base_seconds)
            log.warning(
                "nasdaq.download.retry",
                url=url,
                attempt=attempt,
                sleep_s=round(sleep_s, 3),
                error=str(exc),
            )
            time.sleep(sleep_s)
    raise NasdaqDirectoryError(
        f"failed to download {url} after {max_retries} attempts: {last_exc!r}"
    )


def _is_common_stock_symbol(symbol: str) -> bool:
    upper = symbol.upper().strip()
    return not any(upper.endswith(sfx) for sfx in _NON_COMMON_STOCK_SUFFIXES)


def _coerce_flag(value: object) -> str:
    return (str(value) if value is not None else "").strip().upper()


def download_nasdaq_directory(
    *,
    nasdaq_url: str = DEFAULT_NASDAQ_LISTED_URL,
    other_url: str = DEFAULT_OTHER_LISTED_URL,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download both NASDAQ directory files as DataFrames.

    The CSVs are pipe-delimited and end with a single non-pipe footer line
    (File Creation Time: ...) which we coerce pandas to skip via
    on_bad_lines="skip".
    """
    nasdaq_bytes = _download_with_retry(
        nasdaq_url, max_retries=max_retries, timeout_seconds=timeout_seconds
    )
    other_bytes = _download_with_retry(
        other_url, max_retries=max_retries, timeout_seconds=timeout_seconds
    )
    nasdaq_df = pd.read_csv(
        io.BytesIO(nasdaq_bytes),
        sep="|",
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
    )
    other_df = pd.read_csv(
        io.BytesIO(other_bytes),
        sep="|",
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
    )
    return nasdaq_df, other_df


def parse_listings(
    nasdaq_df: pd.DataFrame, other_df: pd.DataFrame
) -> list[dict[str, str]]:
    """Merge + filter the two CSVs down to common-stock listings.

    Returns a list of dicts with {symbol, exchange, name}. Raises
    NasdaqDirectoryError if a required column is missing.
    """
    missing_n = _NASDAQ_REQUIRED - set(nasdaq_df.columns)
    missing_o = _OTHER_REQUIRED - set(other_df.columns)
    if missing_n:
        raise NasdaqDirectoryError(
            f"nasdaqlisted.txt missing columns: {sorted(missing_n)}"
        )
    if missing_o:
        raise NasdaqDirectoryError(
            f"otherlisted.txt missing columns: {sorted(missing_o)}"
        )

    # Symbol + exchange-code columns differ between the two files.
    #   nasdaqlisted.txt uses "Symbol" + "Market Category" (no Exchange col).
    #   otherlisted.txt   uses "ACT Symbol" + "Exchange".
    nasdaq_rows = nasdaq_df.rename(
        columns={
            "Symbol": "symbol",
            "Security Name": "name",
            "Market Category": "exchange",
        }
    )
    other_rows = other_df.rename(
        columns={
            "ACT Symbol": "symbol",
            "Security Name": "name",
            "Exchange": "exchange",
        }
    )

    base = pd.concat(
        [
            nasdaq_rows[["symbol", "name", "exchange"]],
            other_rows[["symbol", "name", "exchange"]],
        ],
        ignore_index=True,
    )
    base["test_issue"] = (
        pd.concat([nasdaq_df["Test Issue"], other_df["Test Issue"]], ignore_index=True)
        .map(_coerce_flag)
    )
    base["etf"] = (
        pd.concat([nasdaq_df["ETF"], other_df["ETF"]], ignore_index=True)
        .map(_coerce_flag)
    )

    # Apply filter masks.
    mask = (base["test_issue"] == "N") & (base["etf"] == "N")
    base = base[mask].copy()

    for col in ("symbol", "name", "exchange"):
        base[col] = base[col].astype(str).str.strip()

    # Drop empty-symbol rows (paranoia -- garbled rows in the wild).
    base = base[base["symbol"].astype(bool)]

    # Drop non-common-stock suffixes.
    base = base[base["symbol"].apply(_is_common_stock_symbol)]

    # Map exchange codes -> canonical names.  Some nasdaqlisted.txt entries
    # arrive WITHOUT a Market Category (e.g. SPACs that just listed).  Treat
    # blank as "NASDAQ" so we don't lose them via the OTHER() fallback.
    def _canonical(code: object) -> str:
        c = (str(code) if code is not None else "").strip()
        if not c:
            return "NASDAQ"
        return EXCHANGE_CODE_MAP.get(c.upper(), f"OTHER({c})")

    base["exchange_canonical"] = base["exchange"].map(_canonical)

    # De-dupe on symbol (rare but possible if cross-listed twice).
    base = base.drop_duplicates(subset=["symbol"], keep="first")

    return [
        {
            "symbol": row["symbol"],
            "exchange": row["exchange_canonical"],
            "name": row["name"],
        }
        for _, row in base.iterrows()
    ]


def fetch_common_stock_listings(
    *,
    nasdaq_url: str = DEFAULT_NASDAQ_LISTED_URL,
    other_url: str = DEFAULT_OTHER_LISTED_URL,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
) -> list[dict[str, str]]:
    """Convenience wrapper: download + parse in one call.

    Returns [{symbol, exchange, name}, ...] for common stocks listed on
    NASDAQ / NYSE / AMEX / NYSE ARCA. Raises NasdaqDirectoryError when one
    or both files fail to download after max_retries.
    """
    nasdaq_df, other_df = download_nasdaq_directory(
        nasdaq_url=nasdaq_url,
        other_url=other_url,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    return parse_listings(nasdaq_df, other_df)