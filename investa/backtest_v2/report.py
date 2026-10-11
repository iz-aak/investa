"""V2 reversal report: aggregations."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from ..pipeline.scorer import SCORE_BANDS
from .reversal import (
    VERDICT_CONFIRMED, VERDICT_NONE, VERDICT_PARTIAL, VERDICT_UNVERIFIABLE,
)

VERSION = "v2-1.0"


def _mean_n_ci(values):
    vs = [v for v in values if v is not None and not (isinstance(v, float) and not math.isfinite(v))]
    n = len(vs)
    if n == 0:
        return None
    m = statistics.mean(vs)
    if n < 2:
        return {"mean": round(m, 6), "std": 0.0, "n": n, "se": 0.0,
                "ci_low": round(m, 6), "ci_high": round(m, 6)}
    s = statistics.stdev(vs)
    se = s / math.sqrt(n)
    return {"mean": round(m, 6), "std": round(s, 6), "n": n, "se": round(se, 6),
            "ci_low": round(m - 1.96 * se, 6), "ci_high": round(m + 1.96 * se, 6)}

def _per_band_reversal_rate(rows, horizon):
    vcol = f"verdict_{horizon}"
    out = []
    for band in SCORE_BANDS:
        sub = [r for r in rows if r.band == band.label]
        sub_verifiable = [r for r in sub if getattr(r, vcol, None) != VERDICT_UNVERIFIABLE]
        nv = len(sub_verifiable)
        conf_rate = (sum(1 for r in sub_verifiable if getattr(r, vcol, None) == VERDICT_CONFIRMED) / nv) if nv else None
        out.append({"band": band.label, "emoji": band.emoji,
                    "n_events": len(sub),
                    "n_verifiable": nv,
                    "confirmed": sum(1 for r in sub_verifiable
                                      if getattr(r, vcol, None) == VERDICT_CONFIRMED),
                    "confirmation_rate": (round(conf_rate, 4) if conf_rate is not None else None)})

    return out

def _per_signal_reversal_contribution(rows, horizon):
    signal_keys = set()
    on_count = defaultdict(int)
    off_count = defaultdict(int)
    on_clean = defaultdict(list)
    off_clean = defaultdict(list)
    fld = f"fwd_return_{horizon}"
    for r in rows:
        v = getattr(r, fld, None)
        valid = v is not None and not (isinstance(v, float) and not math.isfinite(v))
        for k, fired in r.signal_flags.items():
            signal_keys.add(k)
            if fired:
                on_count[k] += 1
                if valid:
                    on_clean[k].append(v)
            else:
                off_count[k] += 1
                if valid:
                    off_clean[k].append(v)
    out = {}
    for k in sorted(signal_keys):
        total = on_count[k] + off_count[k]
        fire_rate = (on_count[k] / total) if total else 0.0
        on_n, off_n = len(on_clean[k]), len(off_clean[k])
        if on_n == 0 or off_n == 0:
            out[k] = {"fire_rate": round(fire_rate, 4),
                      "mean_on": None, "mean_off": None,
                      "delta": None, "t_stat": None}
            continue
        mo = statistics.mean(on_clean[k])
        so = statistics.mean(off_clean[k])
        std_o = statistics.stdev(on_clean[k]) if on_n >= 2 else 0.0
        std_f = statistics.stdev(off_clean[k]) if off_n >= 2 else 0.0
        den = math.sqrt((std_o ** 2) / on_n + (std_f ** 2) / off_n)
        t = (mo - so) / den if den > 0 else 0.0
        out[k] = {"fire_rate": round(fire_rate, 4),
                  "mean_on": round(mo, 6),
                  "mean_off": round(so, 6),
                  "delta": round(mo - so, 6),
                  "t_stat": round(t, 4)}
    return out


def _per_tier_fire_rates(rows, horizon):
    vcol = f"verdict_{horizon}"
    dcol = f"days_to_first_fire_{horizon}"
    cols = {f"tier_golden_cross_{horizon}": "golden_cross",
            f"tier_ema50_reclaim_{horizon}": "ema50_reclaim",
            f"tier_swing_{horizon}": "swing"}
    verifiables = [r for r in rows if getattr(r, vcol, None) != VERDICT_UNVERIFIABLE]
    nv = len(verifiables)
    out = {}
    for col, label in cols.items():
        fires = sum(1 for r in verifiables if getattr(r, col, False))
        dff = [getattr(r, dcol, None) for r in verifiables
               if getattr(r, col, False) and getattr(r, dcol, None) is not None]
        out[label] = {"fires": fires,
                      "verifiable_n": nv,
                      "fire_rate": (round(fires / nv, 4) if nv else None),
                      "median_days_to_fire": (int(statistics.median(dff)) if dff else None)}
    return out


def _per_ticker_rollup(rows, horizon):
    vcol = f"verdict_{horizon}"
    dcol = f"days_to_first_fire_{horizon}"
    by_ticker = defaultdict(list)
    for r in rows:
        if getattr(r, vcol, None) == VERDICT_UNVERIFIABLE:
            continue
        by_ticker[r.ticker].append(r)
    out = []
    for ticker, items in by_ticker.items():
        rates = [r.score for r in items]
        dff = [getattr(r, dcol, None) for r in items
               if getattr(r, vcol, None) == VERDICT_CONFIRMED and getattr(r, dcol, None) is not None]
        confirmed = sum(1 for r in items if getattr(r, vcol, None) == VERDICT_CONFIRMED)
        out.append({"ticker": ticker,
                    "n_verifiable": len(items),
                    "confirmed": confirmed,
                    "confirmation_rate": (round(confirmed / len(items), 4) if items else None),
                    "avg_score": (round(statistics.mean(rates), 4) if rates else None),
                    "median_days_to_fire": (int(statistics.median(dff)) if dff else None)})
    out.sort(key=lambda d: (-d["confirmation_rate"] or 0, -d["n_verifiable"]))
    return out

def _cross_check_returns_by_verdict(rows, primary_horizon):
    vcol = f"verdict_{primary_horizon}"
    out = {}
    for verdict in (VERDICT_CONFIRMED, VERDICT_PARTIAL, VERDICT_NONE):
        sub = [r for r in rows if getattr(r, vcol, None) == verdict]
        out[verdict] = {h: _mean_n_ci([getattr(r, f"fwd_return_{h}") for r in sub])
                        for h in ("3m", "6m", "9m")}
    return out


def _median_days_for_verified(rows, horizon):
    vcol = f"verdict_{horizon}"
    dcol = f"days_to_first_fire_{horizon}"
    vals = []
    for r in rows:
        v = getattr(r, vcol, None)
        d = getattr(r, dcol, None)
        if v == VERDICT_CONFIRMED and d is not None:
            vals.append(d)
    return int(statistics.median(vals)) if vals else None


def _reversal_rate(rows, horizon):
    """Aggregate verdict counts + confirmation rate for a given horizon.

    `horizon` is the suffix portion of the attribute name (e.g. ``"3m"``,
    ``"6m"``, ``"9m"``). We look up ``verdict_<horizon>`` on each row.
    """
    counts = {VERDICT_CONFIRMED: 0, VERDICT_PARTIAL: 0, VERDICT_NONE: 0,
              VERDICT_UNVERIFIABLE: 0}
    attr = f"verdict_{horizon}"
    for r in rows:
        v = getattr(r, attr, None)
        if v in counts:
            counts[v] += 1
    nv = counts[VERDICT_CONFIRMED] + counts[VERDICT_PARTIAL] + counts[VERDICT_NONE]
    return {
        "confirmed": counts[VERDICT_CONFIRMED],
        "partial": counts[VERDICT_PARTIAL],
        "no_reversal": counts[VERDICT_NONE],
        "unverifiable": counts[VERDICT_UNVERIFIABLE],
        "verifiable_n": nv,
        "confirmation_rate": (round(counts[VERDICT_CONFIRMED] / nv, 4) if nv else None),
    }


def build_v2_report(rows, *, horizons=("3m", "6m", "9m"), start_date=None, end_date=None,
                     metadata=None) -> dict[str, Any]:
    primary = "6m"
    section_a = {h: _reversal_rate(rows, h) for h in horizons}
    section_b = {h: {
        "verifiable_n": section_a[h]["verifiable_n"],
        "confirmation_rate": section_a[h]["confirmation_rate"],
        "median_days_to_first_fire": _median_days_for_verified(rows, h),
    } for h in horizons}
    section_c = {primary: _per_band_reversal_rate(rows, primary),
                 "3m": _per_band_reversal_rate(rows, "3m"),
                 "9m": _per_band_reversal_rate(rows, "9m")}
    section_d = _per_signal_reversal_contribution(rows, primary)
    section_e = {h: _per_tier_fire_rates(rows, h) for h in horizons}
    section_f = _per_ticker_rollup(rows, primary)
    section_g = _cross_check_returns_by_verdict(rows, primary)
    return {
        "n_events": len(rows),
        "primary_horizon": primary,
        "horizons": list(horizons),
        "version": VERSION,
        "section_a_verdict_distribution_by_horizon": section_a,
        "section_b_reversal_rate_by_horizon": section_b,
        "section_c_per_band_at_3m_6m_9m": section_c,
        "section_d_per_signal_contribution_at_6m": section_d,
        "section_e_per_tier_fire_rates_by_horizon": section_e,
        "section_f_per_ticker_rollup_at_6m": section_f,
        "section_g_cross_check_returns_by_verdict_at_6m": section_g,
        "metadata": {
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
            "primary_horizon": primary,
            "horizons_days": {"3m": 63, "6m": 126, "9m": 189},
            "sustain_days": 5,
            **(metadata or {}),
        },
    }

    return out
