"""DuckDB schema for reversal-validation events.

A new ``reversal_events`` table in the SAME DuckDB file used by v1's signal
history. ``investa/data/duckdb_store.py`` is NOT modified.
"""

from __future__ import annotations

import duckdb

from .models import Base  # noqa: F401 - import here so init_db() picks up BacktestV2Job

REVERSAL_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS reversal_events (
    job_id                  VARCHAR,
    ticker                  VARCHAR,
    sim_date                TIMESTAMP,
    score                   DOUBLE,
    band                    VARCHAR,
    signal_flags            JSON,

    -- Tier firing at each horizon (3m / 6m / 9m)
    tier_golden_cross_3m    BOOLEAN,
    tier_ema50_reclaim_3m   BOOLEAN,
    tier_swing_3m           BOOLEAN,
    tier_golden_cross_6m    BOOLEAN,
    tier_ema50_reclaim_6m   BOOLEAN,
    tier_swing_6m           BOOLEAN,
    tier_golden_cross_9m    BOOLEAN,
    tier_ema50_reclaim_9m   BOOLEAN,
    tier_swing_9m           BOOLEAN,

    -- Verdicts at each horizon
    verdict_3m              VARCHAR,
    verdict_6m              VARCHAR,        -- PRIMARY
    verdict_9m              VARCHAR,        -- nullable if censored

    n_tiers_fired_6m        INTEGER,
    days_to_first_fire_6m   INTEGER,        -- NULL if no tier fired at 6m
    days_to_first_fire_3m   INTEGER,
    days_to_first_fire_9m   INTEGER,

    unverifiable_tiers_3m   JSON,           -- ["golden_cross"] etc
    unverifiable_tiers_6m   JSON,
    unverifiable_tiers_9m   JSON,

    -- Forward returns at each horizon (USD pct)
    fwd_return_3m           DOUBLE,
    fwd_return_6m           DOUBLE,
    fwd_return_9m           DOUBLE
);
"""

INDEXES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_reversal_events_job   ON reversal_events(job_id);",
    "CREATE INDEX IF NOT EXISTS idx_reversal_events_ticker ON reversal_events(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_reversal_events_sim    ON reversal_events(sim_date);",
    "CREATE INDEX IF NOT EXISTS idx_reversal_events_score   ON reversal_events(score);",
    "CREATE INDEX IF NOT EXISTS idx_reversal_events_verdict ON reversal_events(verdict_6m);",
]


def init_v2_schema(duckdb_path) -> None:
    """Create reversal_events table + indexes if missing. Idempotent."""
    from pathlib import Path
    if isinstance(duckdb_path, str):
        duckdb_path = Path(duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(duckdb_path)) as con:
        con.execute(REVERSAL_EVENTS_DDL)
        for stmt in INDEXES_DDL:
            con.execute(stmt)
