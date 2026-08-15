"""DuckDB-backed signal history store.

DuckDB is embedded (single file), column-oriented, and extremely fast on the
kind of time-series aggregations the backtest layer performs (``GROUP BY score
band``, forward-return joins, etc.). Kept deliberately thin: one table, one
connection helper, one schema-migration function.

The schema uses typed columns rather than JSON blobs so backtests can filter
and aggregate without deserialising Python dicts row-by-row.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from ..config import Settings, get_settings

SIGNAL_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS signal_history (
    scan_id           VARCHAR,
    scanned_at        TIMESTAMP,
    ticker            VARCHAR,
    interval          VARCHAR,        -- '1d' | '1wk'
    mode              VARCHAR,        -- 'deep_reversal' | 'pullback'
    close             DOUBLE,
    atr               DOUBLE,

    -- Raw indicator readings (nullable during warm-up periods)
    rsi               DOUBLE,
    macd              DOUBLE,
    macd_signal       DOUBLE,
    macd_hist         DOUBLE,
    bb_lower          DOUBLE,
    bb_upper          DOUBLE,
    bb_pos            DOUBLE,         -- (close - lower) / (upper - lower)
    ema_50            DOUBLE,
    ema_200           DOUBLE,
    obv               DOUBLE,
    obv_slope         DOUBLE,
    stoch_k           DOUBLE,
    stoch_d           DOUBLE,
    volume            BIGINT,
    volume_avg_20     DOUBLE,
    volume_ratio      DOUBLE,

    -- Signal outputs
    passed_gate       BOOLEAN,
    signal_flags      JSON,           -- {"rsi": true, "macd": false, ...}
    signal_strengths  JSON,           -- {"rsi": 0.8, "macd": 0.0, ...}
    weekly_confirm    BOOLEAN,
    score             DOUBLE,
    band              VARCHAR,

    -- Risk annotation
    suggested_stop    DOUBLE,
    suggested_target  DOUBLE,
    risk_reward       DOUBLE
);
"""

INDEXES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_signal_history_scan_id ON signal_history(scan_id);",
    "CREATE INDEX IF NOT EXISTS idx_signal_history_ticker ON signal_history(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_signal_history_scanned_at ON signal_history(scanned_at);",
    "CREATE INDEX IF NOT EXISTS idx_signal_history_score ON signal_history(score);",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def init_duckdb(settings: Settings | None = None) -> None:
    """Create the DuckDB file and signal_history schema if missing. Idempotent."""
    settings = settings or get_settings()
    _ensure_parent(settings.duckdb_path)

    with connect(settings) as con:
        con.execute(SIGNAL_HISTORY_DDL)
        for stmt in INDEXES_DDL:
            con.execute(stmt)


@contextmanager
def connect(settings: Settings | None = None, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a short-lived DuckDB connection. Callers should not hold connections
    across long-running operations — open, use, close."""
    settings = settings or get_settings()
    _ensure_parent(settings.duckdb_path)
    con = duckdb.connect(database=str(settings.duckdb_path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()
