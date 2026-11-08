"""Stage 1 universe construction pipeline.

Builds an eligible US-listed common stock universe from the NASDAQ symbol
directory, enriches it with yfinance metadata, applies liquidity / price /
market-cap / sector-cap filters, and persists the snapshot to SQLite.

Architecture
------------
* :func:`enrich_tickers` - batched yfinance ``.info`` calls.
* :func:`apply_filters` - cheap filters run after enrichment.
* :func:`apply_sector_cap` - deterministic random sampling when a sector
  exceeds ``max_sector_pct`` of the eligible universe.
* :func:`build_universe` - orchestrates download -> parse -> enrich ->
  filter -> cap -> persist.
* :func:`save_universe` / :func:`get_current_universe` -
  SQLite persistence (single-row overwrite per rebuild).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from sqlalchemy import select

from ..config import get_settings
from ..database import session_scope
from ..logging_setup import get_logger
from .models import UniverseSnapshot
from .nasdaq_directory import (
    NasdaqDirectoryError,
    download_nasdaq_directory,
    parse_listings,
)

log = get_logger(__name__)

EXCHANGE_SCOPE = "NYSE+NASDAQ+AMEX+NYSE_ARCA"


@dataclass
class UniverseResult:
    """The outcome of a single Stage 1 rebuild."""

    tickers: list[dict[str, Any]]
    stats: dict[str, Any]
    rebuilt_at: datetime
    exchange_scope: str = EXCHANGE_SCOPE

    def as_dict(self) -> dict[str, Any]:
        return {
            "rebuilt_at": self.rebuilt_at.isoformat(),
            "exchange_scope": self.exchange_scope,
            "n_tickers": len(self.tickers),
            "stats": self.stats,
        }


@dataclass
class _FilterReport:
    """Counts + surviving rows after each filter stage."""

    label: str
    count: int
    survivors: list[dict[str, Any]]
    extra: dict[str, Any] = field(default_factory=dict)


def _safe_get(d: dict[str, Any], *keys: str) -> Any:
    """Get the first non-None value from a dict for any of the given keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _enrich_one(symbol: str) -> dict[str, Any] | None:
    """Fetch a single ticker's metadata via yfinance. Returns None on failure."""
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001 - yfinance raises broad errors
        log.warning("universe.enrich_failed", ticker=symbol, error=str(exc))
        return None
    if not info:
        return None
    long_name = _safe_get(info, "longName", "shortName")
    symbol_resolved = _safe_get(info, "symbol", "underlyingSymbol")
    if not long_name or not symbol_resolved:
        # Treat as "no useful metadata"; caller will drop it.
        return None
    return {
        "symbol": str(symbol_resolved).upper(),
        "name": str(long_name),
        "sector": _safe_get(info, "sector"),
        "industry": _safe_get(info, "industry"),
        "currency": _safe_get(info, "financialCurrency", "currency"),
        "exchange": _safe_get(info, "exchange", "fullExchangeName"),
        "market_cap": _safe_get(info, "marketCap"),
        "avg_volume_10d": _safe_get(info, "averageVolume10days", "averageVolume"),
        "last_close": _safe_get(info, "currentPrice", "regularMarketPrice", "previousClose"),
        "history_ms": _safe_get(info, "firstTradeDateMilliseconds"),
    }


def enrich_tickers(
    symbols: list[str],
    *,
    batch_size: int | None = None,
    batch_delay_seconds: float | None = None,
    sleep_fn: Any = time.sleep,
) -> list[dict[str, Any]]:
    """Fetch ``yf.Ticker(symbol).info`` for a list of symbols in batches.

    Returns one dict per successfully enriched ticker (failures filtered out).
    The ``sleep_fn`` override exists so tests can avoid blocking sleeps.
    """
    settings = get_settings()
    batch_size = batch_size or settings.universe_batch_size
    batch_delay_seconds = (
        batch_delay_seconds if batch_delay_seconds is not None
        else settings.universe_batch_delay_seconds
    )

    enriched: list[dict[str, Any]] = []
    total = len(symbols)
    for start in range(0, total, batch_size):
        batch = symbols[start : start + batch_size]
        log.info(
            "universe.enrich.batch",
            start=start,
            end=min(start + batch_size, total),
            total=total,
        )
        for sym in batch:
            sym = sym.strip().upper()
            if not sym:
                continue
            meta = _enrich_one(sym)
            if meta is not None:
                enriched.append(meta)
        # Sleep between batches (skip after the last).
        if start + batch_size < total and batch_delay_seconds > 0:
            sleep_fn(batch_delay_seconds)
    return enriched


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _normalize_enriched(enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cast yfinance numbers to floats + coerce sector to a non-empty string."""
    normalized: list[dict[str, Any]] = []
    for row in enriched:
        sector = row.get("sector")
        if sector is not None:
            sector = str(sector).strip() or "Unknown"
        else:
            sector = "Unknown"
        history_days = None
        if row.get("history_ms") is not None:
            try:
                history_days = int(row["history_ms"]) // 86_400_000
            except (TypeError, ValueError):
                history_days = None
        normalized.append(
            {
                "ticker": row["symbol"].upper(),
                "name": row.get("name") or row["symbol"],
                "sector": sector,
                "industry": row.get("industry"),
                "exchange": row.get("exchange"),
                "currency": row.get("currency"),
                "market_cap": _coerce_float(row.get("market_cap")),
                "avg_volume_10d": _coerce_float(row.get("avg_volume_10d")),
                "last_close": _coerce_float(row.get("last_close")),
                "history_days": history_days,
            }
        )
    return normalized


def apply_liquidity_filter(
    rows: list[dict[str, Any]], *, min_avg_volume_usd: float
) -> _FilterReport:
    survivors = []
    for r in rows:
        vol = r.get("avg_volume_10d") or 0.0
        price = r.get("last_close") or 0.0
        if vol * price >= min_avg_volume_usd:
            survivors.append(r)
    return _FilterReport(
        label="liquidity",
        count=len(survivors),
        survivors=survivors,
        extra={"min_avg_volume_usd": min_avg_volume_usd},
    )


def apply_price_filter(
    rows: list[dict[str, Any]], *, min_share_price: float
) -> _FilterReport:
    survivors = [r for r in rows if (r.get("last_close") or 0.0) >= min_share_price]
    return _FilterReport(
        label="price",
        count=len(survivors),
        survivors=survivors,
        extra={"min_share_price": min_share_price},
    )


def apply_market_cap_filter(
    rows: list[dict[str, Any]], *, min_market_cap: float
) -> _FilterReport:
    survivors = [r for r in rows if (r.get("market_cap") or 0.0) >= min_market_cap]
    return _FilterReport(
        label="market_cap",
        count=len(survivors),
        survivors=survivors,
        extra={"min_market_cap": min_market_cap},
    )


def apply_sector_cap(
    rows: list[dict[str, Any]],
    *,
    max_pct: float,
    seed: int = 42,
) -> _FilterReport:
    """Cap any single sector to ``max_pct`` of the eligible universe.

    Sectors exceeding the cap are randomly downsampled using a fixed seed so
    results are reproducible. Capped-out tickers are recorded in ``extra``
    for audit purposes.
    """
    if not rows:
        return _FilterReport(
            label="sector_cap",
            count=0,
            survivors=[],
            extra={"max_pct": max_pct, "capped_out": []},
        )

    total = len(rows)
    cap = int(total * max_pct)
    by_sector: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_sector.setdefault(r["sector"] or "Unknown", []).append(r)

    rng = random.Random(seed)
    kept: list[dict[str, Any]] = []
    capped_out: list[dict[str, str]] = []
    for sector, members in by_sector.items():
        if cap <= 0 or len(members) <= cap:
            kept.extend(members)
            continue
        # Deterministic shuffle, then trim to cap.
        rng.shuffle(members)
        kept.extend(members[:cap])
        for dropped in members[cap:]:
            capped_out.append({"ticker": dropped["ticker"], "sector": sector})

    return _FilterReport(
        label="sector_cap",
        count=len(kept),
        survivors=kept,
        extra={"max_pct": max_pct, "main_cap": cap, "capped_out": capped_out},
    )


def _build_stats(
    *,
    raw: int,
    after_etf: int,
    after_metadata: int,
    liquidity: _FilterReport,
    price: _FilterReport,
    market_cap: _FilterReport,
    sector_cap: _FilterReport,
    started_at: datetime,
) -> dict[str, Any]:
    duration = (datetime.now(timezone.utc) - started_at).total_seconds()
    final = sector_cap.survivors

    def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            k = r.get(key) or "Unknown"
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    return {
        "raw_count": raw,
        "after_etf_filter": after_etf,
        "after_metadata": after_metadata,
        "after_liquidity": liquidity.count,
        "after_price": price.count,
        "after_mcap": market_cap.count,
        "after_sector_cap": sector_cap.count,
        "final_count": len(final),
        "sector_breakdown": _breakdown(final, "sector"),
        "exchange_breakdown": _breakdown(final, "exchange"),
        "duration_seconds": round(duration, 2),
        "filters": {
            "liquidity": liquidity.extra,
            "price": price.extra,
            "market_cap": market_cap.extra,
            "sector_cap": sector_cap.extra,
        },
    }


def build_universe(
    *,
    listings: list[dict[str, str]] | None = None,
    enriched: list[dict[str, Any]] | None = None,
    min_avg_volume_usd: float | None = None,
    min_share_price: float | None = None,
    min_market_cap: float | None = None,
    min_history_days: int | None = None,
    max_sector_pct: float | None = None,
    batch_size: int | None = None,
    batch_delay_seconds: float | None = None,
    nasdaq_url: str | None = None,
    other_url: str | None = None,
    sleep_fn: Any = time.sleep,
) -> UniverseResult:
    """Run the full Stage 1 pipeline and return the result.

    Each ``None`` falls through to the value defined in ``Settings``. The
    ``listings`` and ``enriched`` hooks let tests bypass the live network
    fetch + yfinance call without touching the rest of the pipeline.
    """
    settings = get_settings()
    min_avg_volume_usd = min_avg_volume_usd if min_avg_volume_usd is not None else settings.universe_min_avg_volume_usd
    min_share_price = min_share_price if min_share_price is not None else settings.universe_min_share_price
    min_market_cap = min_market_cap if min_market_cap is not None else settings.universe_min_market_cap
    min_history_days = min_history_days if min_history_days is not None else settings.universe_min_history_days
    max_sector_pct = max_sector_pct if max_sector_pct is not None else settings.universe_max_sector_pct

    started_at = datetime.now(timezone.utc)
    log.info("universe.build.start", started_at=started_at.isoformat())

    if listings is None:
        nasdaq_df, other_df = download_nasdaq_directory(
            nasdaq_url=nasdaq_url or settings.universe_nasdaq_listed_url,
            other_url=other_url or settings.universe_other_listed_url,
        )
        listings = parse_listings(nasdaq_df, other_df)

    raw_count = len(listings)

    if enriched is None:
        enriched = enrich_tickers(
            [r["symbol"] for r in listings],
            batch_size=batch_size,
            batch_delay_seconds=batch_delay_seconds,
            sleep_fn=sleep_fn,
        )

    # Merge exchange + name from the raw listings onto the enriched rows so
    # the breakdown/sector-cap logic sees canonical values ("NYSE" etc).
    listing_by_sym = {r["symbol"].upper(): r for r in listings}
    merged: list[dict[str, Any]] = []
    for e in _normalize_enriched(enriched):
        listing = listing_by_sym.get(e["ticker"])
        if listing is None:
            continue
        merged.append({**e, "exchange": listing["exchange"], "name": listing["name"]})

    # Cheap filters, ordered by selectivity.
    liquidity = apply_liquidity_filter(merged, min_avg_volume_usd=min_avg_volume_usd)
    price = apply_price_filter(liquidity.survivors, min_share_price=min_share_price)
    mcap = apply_market_cap_filter(price.survivors, min_market_cap=min_market_cap)
    sector_cap_report = apply_sector_cap(mcap.survivors, max_pct=max_sector_pct)

    stats = _build_stats(
        raw=raw_count,
        after_etf=raw_count,           # parse_listings already excludes ETFs/tests
        after_metadata=len(merged),
        liquidity=liquidity,
        price=price,
        market_cap=mcap,
        sector_cap=sector_cap_report,
        started_at=started_at,
    )
    stats["min_history_days"] = min_history_days

    final_rows = sector_cap_report.survivors
    cleaned = [
        {
            "ticker": r["ticker"],
            "exchange": r["exchange"],
            "name": r["name"],
            "sector": r["sector"],
            "market_cap": r.get("market_cap"),
            "avg_volume_10d": r.get("avg_volume_10d"),
        }
        for r in final_rows
    ]

    log.info(
        "universe.build.done",
        raw=raw_count,
        eligible=len(cleaned),
        duration_seconds=stats["duration_seconds"],
    )

    return UniverseResult(
        tickers=cleaned,
        stats=stats,
        rebuilt_at=datetime.now(timezone.utc),
    )


def save_universe(result: UniverseResult) -> int:
    """Persist ``result`` to SQLite (overwrites prior snapshots).

    Returns the id of the freshly inserted snapshot.
    """
    with session_scope() as s:
        prior = s.execute(select(UniverseSnapshot)).scalars().all()
        for row in prior:
            s.delete(row)
        snap = UniverseSnapshot(
            rebuilt_at=result.rebuilt_at,
            exchange_scope=result.exchange_scope,
            tickers=list(result.tickers),
            stats=dict(result.stats),
        )
        s.add(snap)
        s.flush()
        return int(snap.id)  # type: ignore[arg-type]


def get_current_universe() -> UniverseSnapshot | None:
    """Return the latest persisted snapshot, or ``None`` if absent."""
    with session_scope() as s:
        return s.execute(
            select(UniverseSnapshot).order_by(UniverseSnapshot.rebuilt_at.desc())
        ).scalars().first()


def get_universe_stats() -> dict[str, Any] | None:
    """Return the latest snapshot's stats dict, or ``None`` if absent."""
    snap = get_current_universe()
    return None if snap is None else dict(snap.stats)