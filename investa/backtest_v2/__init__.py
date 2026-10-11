"""Investa Backtest v2 - reversal-validation layer over v1's replay engine.

Composed:
    backtest_v2.reversal    - 3 tier validators (golden cross / EMA-50 reclaim / swing)
    backtest_v2.validator  - per-event orchestrator
    backtest_v2.report     - aggregation
    backtest_v2.runner     - job lifecycle + persistence
    backtest_v2.api_routes - HTTP endpoints (in api_routes.py)

Reuses v1's :func:`replay_ticker` + :func:`attach_forward_returns`.
v1 files are NEVER modified - this is an additive layer.
"""
