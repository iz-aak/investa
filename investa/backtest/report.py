"""Backtest report aggregation.

Consumes (event, forward-returns, trade-outcome) triples and produces a
deterministic, JSON-serializable report with baseline, per-band stats, alpha
vs baseline, per-signal contribution, and documented metadata.

Statistical helpers are deliberately simple (mean / median / std / SE / 95% CI
on the mean). No p-values are computed; CIs and t-stats are exposed for
downstream interpretation.
"""

from __future__ import annotations

import math
import statistics as stats
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..pipeline.scorer import SCORE_BANDS
from .replay import SignalEvent

DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
PRIMARY_HORIZON = 10


@dataclass
class BandStats:
    label: str
    emoji: str
    n: int
    mean_return_5d: dict[str, Any] | None
    mean_return_10d: dict[str, Any] | None
    mean_return_20d: dict[str, Any] | None
    median_return_10d: float | None
    hit_rate_10d: float | None
    mean_max_drawdown_20d: float | None
    mean_realized_rr: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "band": self.label, "emoji": self.emoji, "n": self.n,
            "mean_return_5d": self.mean_return_5d,
            "mean_return_10d": self.mean_return_10d,
            "mean_return_20d": self.mean_return_20d,
            "median_return_10d": (
                round(self.median_return_10d, 6)
                if self.median_return_10d is not None else None),
            "hit_rate_10d": self.hit_rate_10d,
            "mean_max_drawdown_20d": (
                round(self.mean_max_drawdown_20d, 6)
                if self.mean_max_drawdown_20d is not None else None),
            "mean_realized_rr": (
                round(self.mean_realized_rr, 6)
                if self.mean_realized_rr is not None else None),
        }


@dataclass
class _Joined:
    event: SignalEvent
    returns: dict[int, float | None]
    max_drawdown: float | None
    max_runup: float | None
    trade_outcome_label: str | None
    realized_rr: float | None


def join(events: list[SignalEvent], forward_per_event: list[dict[str, Any]],
         trades: list[Any]) -> list[_Joined]:
    out: list[_Joined] = []
    for ev, fwd, tr in zip(events, forward_per_event, trades):
        out.append(_Joined(
            event=ev,
            returns=fwd["returns"],
            max_drawdown=fwd.get("max_drawdown"),
            max_runup=fwd.get("max_runup"),
            trade_outcome_label=tr.label,
            realized_rr=tr.realized_rr,
        ))
    return out


def build_report(
    joined: list[_Joined], *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    start_date: Any = None, end_date: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = _stats_for_subset(joined, band_label="All", emoji="•", horizons=horizons)
    per_band_list: list[BandStats] = [_stats_for_subset(
        [j for j in joined if j.event.band == band.label],
        band_label=band.label, emoji=band.emoji, horizons=horizons,
    ) for band in SCORE_BANDS]
    alpha = _alpha_vs_baseline(per_band_list, baseline, horizons=horizons)
    per_signal = _per_signal_contribution(joined)

    return {
        "n_events": len(joined),
        "horizons": list(horizons),
        "overall_baseline": baseline.as_dict(),
        "per_band": [b.as_dict() for b in per_band_list],
        "alpha_vs_baseline": alpha,
        "per_signal_contribution": per_signal,
        "metadata": {
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
            "primary_horizon": PRIMARY_HORIZON,
            **(metadata or {}),
        },
    }


def _stats_for_subset(subset: Iterable[_Joined], *, band_label: str,
                      emoji: str, horizons: tuple[int, ...]) -> BandStats:
    sub = list(subset)
    n = len(sub)
    r5, r10, r20 = _slice_returns(sub, horizons)
    r10_clean = [v for v in r10 if v is not None and not math.isnan(v)]
    dds = [j.max_drawdown for j in sub
           if j.max_drawdown is not None and not math.isnan(j.max_drawdown)]
    rrs = [j.realized_rr for j in sub
           if j.realized_rr is not None and not math.isnan(j.realized_rr)]
    return BandStats(
        label=band_label, emoji=emoji, n=n,
        mean_return_5d=_mean_ci(r5),
        mean_return_10d=_mean_ci(r10),
        mean_return_20d=_mean_ci(r20),
        median_return_10d=float(stats.median(r10_clean)) if r10_clean else None,
        hit_rate_10d=(round(sum(1 for v in r10_clean if v > 0) / len(r10_clean), 4)
                      if r10_clean else None),
        mean_max_drawdown_20d=float(stats.mean(dds)) if dds else None,
        mean_realized_rr=float(stats.mean(rrs)) if rrs else None,
    )


def _alpha_vs_baseline(bands: list[BandStats], baseline: BandStats,
                       *, horizons: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for band in bands:
        if band.label == "All":
            continue
        out[band.label] = {
            "n": band.n,
            "return_5d_alpha": _alpha_dict(
                band.mean_return_5d, baseline.mean_return_5d),
            "return_10d_alpha": _alpha_dict(
                band.mean_return_10d, baseline.mean_return_10d),
            "return_20d_alpha": _alpha_dict(
                band.mean_return_20d, baseline.mean_return_20d),
        }
    return out


def _alpha_dict(band: dict[str, Any] | None, baseline: dict[str, Any] | None
                ) -> dict[str, Any] | None:
    if band is None or baseline is None:
        return None
    alpha = band["mean"] - baseline["mean"]
    band_no_overlap = (band["ci_low"] > baseline["ci_high"]) or (
        baseline["ci_low"] > band["ci_high"])
    return {
        "alpha_return": round(alpha, 6),
        "band_ci_low": band["ci_low"],
        "band_ci_high": band["ci_high"],
        "baseline_ci_low": baseline["ci_low"],
        "baseline_ci_high": baseline["ci_high"],
        "ci_does_not_overlap": bool(band_no_overlap),
    }


def _per_signal_contribution(joined: list[_Joined]) -> dict[str, Any]:
    """Delta in mean PRIMARY_HORIZON return between 'fired' and 'did not fire'."""
    signal_keys: set[str] = set()
    for j in joined:
        signal_keys.update(j.event.signal_flags.keys())

    out: dict[str, Any] = {}
    total = max(len(joined), 1)
    for key in sorted(signal_keys):
        on_clean: list[float] = []
        off_clean: list[float] = []
        on_count = 0
        off_count = 0
        for j in joined:
            r = j.returns.get(PRIMARY_HORIZON)
            fired = bool(j.event.signal_flags.get(key, False))
            if fired:
                on_count += 1
                if r is not None and not math.isnan(r):
                    on_clean.append(r)
            else:
                off_count += 1
                if r is not None and not math.isnan(r):
                    off_clean.append(r)

        if not on_clean or not off_clean:
            out[key] = {
                "n_on": len(on_clean), "n_off": len(off_clean),
                "mean_on": None, "mean_off": None,
                "delta": None, "t_stat": None,
                "fire_rate": round(on_count / total, 4),
            }
            continue

        m_on, s_on, n_on = _mean_std_n(on_clean)
        m_off, s_off, n_off = _mean_std_n(off_clean)
        den = math.sqrt((s_on ** 2) / n_on + (s_off ** 2) / n_off)
        t = (m_on - m_off) / den if den > 0 else 0.0

        out[key] = {
            "n_on": n_on, "n_off": n_off,
            "mean_on": round(m_on, 6),
            "mean_off": round(m_off, 6),
            "delta": round(m_on - m_off, 6),
            "t_stat": round(t, 4),
            "fire_rate": round(on_count / len(joined), 4),
        }
    return out


def _slice_returns(joined: Iterable[_Joined], horizons: tuple[int, ...]
                   ) -> tuple[list[float | None], list[float | None], list[float | None]]:
    r5: list[float | None] = []
    r10: list[float | None] = []
    r20: list[float | None] = []
    get = lambda j, h: j.returns.get(h) if h else None
    for j in joined:
        r5.append(get(j, horizons[0]) if len(horizons) >= 1 else None)
        r10.append(get(j, horizons[1]) if len(horizons) >= 2 else None)
        r20.append(get(j, horizons[2]) if len(horizons) >= 3 else None)
    return r5, r10, r20


def _mean_ci(values: list[float | None]) -> dict[str, Any] | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    n = len(clean)
    m = float(stats.mean(clean))
    if n < 2:
        return {"mean": round(m, 6), "std": 0.0, "n": n, "se": 0.0,
                "ci_low": round(m, 6), "ci_high": round(m, 6)}
    s = float(stats.stdev(clean))
    se = s / math.sqrt(n)
    return {
        "mean": round(m, 6),
        "std": round(s, 6),
        "n": n,
        "se": round(se, 6),
        "ci_low": round(m - 1.96 * se, 6),
        "ci_high": round(m + 1.96 * se, 6),
    }


def _mean_std_n(values: list[float]) -> tuple[float, float, int]:
    n = len(values)
    m = float(stats.mean(values))
    s = float(stats.stdev(values)) if n >= 2 else 0.0
    return m, s, n
