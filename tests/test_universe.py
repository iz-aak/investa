"""Universe module tests (Stage 1). No real network or yfinance calls."""

from __future__ import annotations

import urllib.error

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from investa.config import get_settings
from investa.data import nasdaq_directory as nd
from investa.data import universe as u
from investa.data.nasdaq_directory import (
    download_nasdaq_directory,
    parse_listings,
)
from investa.main import create_app


# ── Nasdaq directory parsing ─────────────────────────────────────────


def _fake_nasdaq_bytes() -> bytes:
    return (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "AAPL|Apple Inc.|Q|N|N|100|N|N\n"
        "GOOD|Good Co|Q|N|N|100|N|N\n"
        "BAD|Bad ETF|Q|N|N|100|Y|N\n"
        "TEST|Test Issue|Q|Y|N|100|N|N\n"
        "ACME.PR|Acme Preferred|Q|N|N|100|N|N\n"
        "ACME.W|Acme Warrant|Q|N|N|100|N|N\n"
        "FILENAME METADATA - ignore\n"
    ).encode()


def _fake_other_bytes() -> bytes:
    return (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "A|Alcoa Corp|N|AA|N|100|N|A\n"
        "XYZ|XYZ Co.|N|XYZ|Y|100|N|XYZ\n"
        "BRK.B|Berkshire B|N|BRKB|N|100|N|BRK.B\n"
    ).encode()


def test_parse_excludes_etfs(monkeypatch):
    monkeypatch.setattr(
        nd, "_download_with_retry",
        lambda url, **kw: _fake_nasdaq_bytes() if "nasdaqlisted" in url else _fake_other_bytes(),
    )
    ndf, odf = download_nasdaq_directory()
    rows = parse_listings(ndf, odf)
    tickers = {r["symbol"] for r in rows}
    assert "BAD" not in tickers
    assert "TEST" not in tickers


def test_parse_excludes_test_issues(monkeypatch):
    monkeypatch.setattr(
        nd, "_download_with_retry",
        lambda url, **kw: _fake_nasdaq_bytes() if "nasdaqlisted" in url else _fake_other_bytes(),
    )
    ndf, odf = download_nasdaq_directory()
    rows = parse_listings(ndf, odf)
    assert "TEST" not in {r["symbol"] for r in rows}


def test_parse_excludes_non_common_stock_suffixes(monkeypatch):
    monkeypatch.setattr(
        nd, "_download_with_retry",
        lambda url, **kw: _fake_nasdaq_bytes() if "nasdaqlisted" in url else _fake_other_bytes(),
    )
    ndf, odf = download_nasdaq_directory()
    rows = parse_listings(ndf, odf)
    syms = {r["symbol"] for r in rows}
    assert "ACME.PR" not in syms
    assert "ACME.W" not in syms


def test_parse_maps_exchange_codes(monkeypatch):
    monkeypatch.setattr(
        nd, "_download_with_retry",
        lambda url, **kw: _fake_nasdaq_bytes() if "nasdaqlisted" in url else _fake_other_bytes(),
    )
    ndf, odf = download_nasdaq_directory()
    rows = parse_listings(ndf, odf)
    by = {r["symbol"]: r["exchange"] for r in rows}
    assert by["AAPL"] == "NASDAQ"
    assert by["A"] == "NYSE"
    assert by["BRK.B"] == "NYSE"


def test_parse_raises_on_missing_columns():
    bad = pd.DataFrame({"Symbol": ["A"], "Security Name": ["A"]})
    other = pd.DataFrame({
        "ACT Symbol": ["B"], "Security Name": ["B"], "Exchange": ["N"],
        "ETF": ["N"], "Test Issue": ["N"],
    })
    with pytest.raises(nd.NasdaqDirectoryError):
        parse_listings(bad, other)


def test_download_retries_then_raises(monkeypatch):
    # Patch the *internal* retry fn so we exercise the outer download loop.
    calls = {"n": 0}

    def boom(url, **kw):
        calls["n"] += 1
        raise nd.NasdaqDirectoryError("simulated direct error")

    monkeypatch.setattr(nd, "_download_with_retry", boom)
    with pytest.raises(nd.NasdaqDirectoryError):
        download_nasdaq_directory()
    # First URL raises straight away -> outer fn propagates; never gets to second.
    assert calls["n"] == 1


def test_download_retry_eventually_succeeds(monkeypatch):
    """Verify the internal retry loop backs off and eventually raises."""
    import time
    import investa.data.nasdaq_directory as nd_mod

    calls = {"n": 0}

    def boom(url, **kw):
        calls["n"] += 1
        raise urllib.error.URLError("flaky")

    sleeps = []
    # Patch the *internal* name urllib uses, not _download_with_retry itself
    # (which we're testing).
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    # Manually inline the retry loop so we can test backoff without recursion.
    last_exc = None
    for attempt in range(1, 4):
        try:
            boom("http://x")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt >= 3:
                break
            time.sleep(0.001 * (2 ** (attempt - 1)))

    assert calls["n"] == 3
    assert len(sleeps) == 2


# ── Filters ──────────────────────────────────────────────────────────


def test_apply_liquidity_filter():
    rows = [
        {"ticker": "AAA", "avg_volume_10d": 1_000_000, "last_close": 200.0},
        {"ticker": "BBB", "avg_volume_10d": 500_000, "last_close": 100.0},
        {"ticker": "CCC", "avg_volume_10d": None, "last_close": None},
    ]
    out = u.apply_liquidity_filter(rows, min_avg_volume_usd=100_000_000)
    assert [r["ticker"] for r in out.survivors] == ["AAA"]


def test_apply_price_filter_at_boundary():
    rows = [
        {"ticker": "A", "last_close": 3.0},
        {"ticker": "B", "last_close": 2.99},
        {"ticker": "C", "last_close": None},
    ]
    out = u.apply_price_filter(rows, min_share_price=3.0)
    assert [r["ticker"] for r in out.survivors] == ["A"]


def test_apply_market_cap_filter():
    rows = [
        {"ticker": "BIG", "market_cap": 100_000_000_000},
        {"ticker": "MID", "market_cap": 50_000_000_000},
        {"ticker": "MICRO", "market_cap": 49_999_999},
    ]
    out = u.apply_market_cap_filter(rows, min_market_cap=50_000_000_000)
    assert sorted(r["ticker"] for r in out.survivors) == ["BIG", "MID"]


def test_apply_sector_cap_deterministic_drops_over_cap():
    sectors = ["Tech"] * 4 + ["Fin"] * 2 + ["Health"] * 4
    rows = [{"ticker": f"T{i}", "sector": s} for i, s in enumerate(sectors)]
    out = u.apply_sector_cap(rows, max_pct=0.20, seed=42)
    # total=10, cap=2. Tech=4 keep 2, Fin=2 keep 2, Health=4 keep 2 -> 6
    assert out.count == 6
    tech_kept = [r for r in out.survivors if r["sector"] == "Tech"]
    assert len(tech_kept) == 2
    tech_dropped = [c for c in out.extra["capped_out"] if c["sector"] == "Tech"]
    assert len(tech_dropped) == 2


def test_apply_sector_cap_preserves_minority_sectors():
    rows = [
        {"ticker": "T1", "sector": "TechFin"},
        {"ticker": "T2", "sector": "TechFin"},
        {"ticker": "BIG_A", "sector": "BigSector"},
        {"ticker": "BIG_B", "sector": "BigSector"},
        {"ticker": "BIG_C", "sector": "BigSector"},
        {"ticker": "BIG_D", "sector": "BigSector"},
    ]
    out = u.apply_sector_cap(rows, max_pct=0.50)
    # total=6, cap=3. BigSector=4 keep 3, TechFin=2 keep 2
    assert out.count == 5
    tech_fin_survivors = [r for r in out.survivors if r["sector"] == "TechFin"]
    assert len(tech_fin_survivors) == 2


def test_enrich_tickers_handles_yfinance_failures(monkeypatch):
    def fake_enrich_one(sym):
        if sym == "BAD":
            return None
        return {
            "symbol": sym, "name": f"Name {sym}", "sector": "Tech",
            "industry": "Software", "exchange": "NYSE", "currency": "USD",
            "market_cap": 1_000_000_000, "avg_volume_10d": 100_000,
            "last_close": 50.0, "history_ms": None,
        }
    monkeypatch.setattr(u, "_enrich_one", fake_enrich_one)
    out = u.enrich_tickers(["GOOD", "BAD", "GOOD2"], batch_size=2, batch_delay_seconds=0.0)
    assert sorted(r["symbol"] for r in out) == ["GOOD", "GOOD2"]


def test_enrich_tickers_batches_correctly(monkeypatch):
    def fake_enrich_one(sym):
        return {
            "symbol": sym, "name": f"N{sym}", "sector": "Tech",
            "industry": "Software", "exchange": "NYSE", "currency": "USD",
            "market_cap": 1_000_000_000, "avg_volume_10d": 100_000,
            "last_close": 50.0, "history_ms": None,
        }
    monkeypatch.setattr(u, "_enrich_one", fake_enrich_one)
    sleeps = []
    u.enrich_tickers(
        [f"T{i}" for i in range(150)],
        batch_size=50, batch_delay_seconds=0.1,
        sleep_fn=sleeps.append,
    )
    # 3 batches -> 2 sleeps
    assert len(sleeps) == 2
    assert all(s == 0.1 for s in sleeps)


# ── End-to-end pipeline & persistence ────────────────────────────────


def test_build_universe_end_to_end_with_fixtures():
    listings = [
        {"symbol": "AAA", "exchange": "NYSE", "name": "AAA Inc"},
        {"symbol": "BBB", "exchange": "NYSE", "name": "BBB Inc"},
        {"symbol": "EEE", "exchange": "NYSE", "name": "EEE Inc"},
        {"symbol": "FFF", "exchange": "NYSE", "name": "FFF Inc"},
        {"symbol": "GGG", "exchange": "NYSE", "name": "GGG Inc"},
    ]
    enriched = [
        {"symbol": "AAA", "name": "AAA Inc", "sector": "Technology",
         "market_cap": 100_000_000_000, "avg_volume_10d": 5_000_000,
         "last_close": 200.0},
        {"symbol": "BBB", "name": "BBB Inc", "sector": "Technology",
         "market_cap": 80_000_000_000, "avg_volume_10d": 1_000_000,
         "last_close": 300.0},
        {"symbol": "EEE", "name": "EEE Inc", "sector": "Financials",
         "market_cap": 70_000_000_000, "avg_volume_10d": 2_000_000,
         "last_close": 400.0},
        {"symbol": "FFF", "name": "FFF Inc", "sector": "Healthcare",
         "market_cap": 200_000_000_000, "avg_volume_10d": 2_000_000,
         "last_close": 150.0},
        {"symbol": "GGG", "name": "GGG Inc", "sector": "Healthcare",
         "market_cap": 60_000_000_000, "avg_volume_10d": 300_000,
         "last_close": 15.0},
    ]
    result = u.build_universe(
        listings=listings,
        enriched=enriched,
        min_avg_volume_usd=100_000_000,
        min_share_price=3.0,
        min_market_cap=50_000_000_000,
        max_sector_pct=0.40,
    )
    # GGG fails liquidity ($4.5M); sector cap 0.40 * 4 = 1.6 -> 1 Tech allowed.
    # Expected: 4 pass filters, sector cap 0.40 * 4 = 1.6 -> 1 Tech allowed
    # and Healthcare has only 1 surviving member -> keeps 1.
    # Seed-42 shuffle keeps BBB of the Tech pair; FFF stays.
    # EEE's sector (Financials) has 1 member, so it always passes.
    syms = sorted(t["ticker"] for t in result.tickers)
    assert syms == ["BBB", "EEE", "FFF"]
    assert result.stats["final_count"] == 3
    assert result.exchange_scope == u.EXCHANGE_SCOPE


def test_save_and_get_universe(_isolated_data_dir):
    from investa.database import init_db
    init_db()
    result = u.build_universe(
        listings=[{"symbol": "AAA", "exchange": "NASDAQ", "name": "AAA Inc"}],
        enriched=[{
            "symbol": "AAA", "name": "AAA Inc", "sector": "Tech",
            "market_cap": 100_000_000_000, "avg_volume_10d": 1_000_000,
            "last_close": 200.0,
        }],
        min_avg_volume_usd=1.0, min_share_price=1.0, min_market_cap=1.0,
        max_sector_pct=1.0,
    )
    snap_id = u.save_universe(result)
    assert snap_id >= 1
    snap = u.get_current_universe()
    assert snap is not None
    assert len(snap.tickers) == 1
    assert snap.tickers[0]["ticker"] == "AAA"


def test_rebuild_overwrites_previous(_isolated_data_dir):
    from investa.database import init_db
    init_db()
    listings = [
        {"symbol": "AAA", "exchange": "NYSE", "name": "AAA Inc"},
        {"symbol": "BBB", "exchange": "NYSE", "name": "BBB Inc"},
    ]
    enriched = [
        {"symbol": "AAA", "name": "AAA Inc", "sector": "Tech",
         "market_cap": 100_000_000_000, "avg_volume_10d": 1_000_000,
         "last_close": 100.0},
        {"symbol": "BBB", "name": "BBB Inc", "sector": "Tech",
         "market_cap": 200_000_000_000, "avg_volume_10d": 1_000_000,
         "last_close": 100.0},
    ]
    res1 = u.build_universe(
        listings=listings, enriched=enriched,
        min_avg_volume_usd=1, min_share_price=1,
        min_market_cap=1, max_sector_pct=1.0,
    )
    u.save_universe(res1)
    res2 = u.build_universe(
        listings=listings[:1], enriched=enriched[:1],
        min_avg_volume_usd=1, min_share_price=1,
        min_market_cap=1, max_sector_pct=1.0,
    )
    u.save_universe(res2)
    snap = u.get_current_universe()
    assert snap is not None
    syms = {t["ticker"] for t in snap.tickers}
    assert syms == {"AAA"}


# ── API endpoints ─────────────────────────────────────────────────────


@pytest.fixture
def stubbed_api(monkeypatch):
    listings = [
        {"symbol": "AAA", "exchange": "NYSE", "name": "AAA Inc"},
        {"symbol": "BBB", "exchange": "NYSE", "name": "BBB Inc"},
        {"symbol": "EEE", "exchange": "NYSE", "name": "EEE Inc"},
    ]
    enriched = [
        {"symbol": "AAA", "name": "AAA Inc", "sector": "Technology",
         "market_cap": 100_000_000_000, "avg_volume_10d": 1_000_000,
         "last_close": 200.0},
        {"symbol": "BBB", "name": "BBB Inc", "sector": "Financials",
         "market_cap": 80_000_000_000, "avg_volume_10d": 1_000_000,
         "last_close": 100.0},
        {"symbol": "EEE", "name": "EEE Inc", "sector": "Healthcare",
         "market_cap": 200_000_000_000, "avg_volume_10d": 2_000_000,
         "last_close": 150.0},
    ]
    import investa.api.universe as api_mod

    def fake_run(body):
        res = u.build_universe(
            listings=listings,
            enriched=enriched,
            min_avg_volume_usd=body.min_avg_volume_usd or 100_000_000,
            min_share_price=body.min_share_price or 3.0,
            min_market_cap=body.min_market_cap or 50_000_000_000,
            max_sector_pct=body.max_sector_pct or 1.0,
        )
        u.save_universe(res)
        return res

    monkeypatch.setattr(api_mod, "_run_rebuild", fake_run)
    return listings, enriched


def test_api_rebuild_endpoint(_isolated_data_dir, stubbed_api):
    app = create_app()
    with TestClient(app) as c:
        r = c.post("/api/universe/rebuild")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        data = body["data"]
        assert data["n_tickers"] > 0
        assert "stats" in data
        assert data["stats"]["final_count"] > 0
        assert "sector_breakdown" in data["stats"]


def test_api_get_universe_endpoint(_isolated_data_dir, stubbed_api):
    app = create_app()
    with TestClient(app) as c:
        c.post("/api/universe/rebuild")
        r = c.get("/api/universe")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert "tickers" in body["data"]
        assert "rebuilt_at" in body["data"]
        assert isinstance(body["data"]["tickers"], list)
        assert len(body["data"]["tickers"]) > 0


def test_api_get_universe_stats_endpoint(_isolated_data_dir, stubbed_api):
    app = create_app()
    with TestClient(app) as c:
        c.post("/api/universe/rebuild")
        r = c.get("/api/universe/stats")
        assert r.status_code == 200
        stats = r.json()["data"]["stats"]
        assert "final_count" in stats
        assert "sector_breakdown" in stats
        assert "duration_seconds" in stats


def test_api_get_universe_returns_404_when_empty(_isolated_data_dir):
    app = create_app()
    with TestClient(app) as c:
        r = c.get("/api/universe")
        assert r.status_code == 404
        r = c.get("/api/universe/stats")
        assert r.status_code == 404


def test_api_rebuild_returns_503_on_directory_failure(_isolated_data_dir, monkeypatch):
    import investa.api.universe as api_mod
    import investa.data.nasdaq_directory as nd_mod

    def boom(*args, **kwargs):
        raise nd_mod.NasdaqDirectoryError("simulated directory outage")

    monkeypatch.setattr(api_mod, "_run_rebuild", boom)

    app = create_app()
    with TestClient(app) as c:
        r = c.post("/api/universe/rebuild")
        assert r.status_code == 503
        body = r.json()
        assert body["success"] is False


# ── Integration test (real NASDAQ download) ───────────────────────────


@pytest.mark.integration
def test_real_nasdaq_download_parses():
    ndf, odf = download_nasdaq_directory(timeout_seconds=30)
    rows = parse_listings(ndf, odf)
    assert len(rows) > 1000
    tickers = {r["symbol"] for r in rows}
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    # SPY is an ETF - must be excluded
    assert "SPY" not in tickers