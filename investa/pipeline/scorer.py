"""Weighted composite scoring engine.

Aggregates individual :class:`~investa.pipeline.signals.SignalReading`
strengths into a single 0-100 Reversal Conviction Score. Weights match the
project plan exactly and sum to 1.0.

The scorer is **implemented in full** — missing signals contribute 0. Adding
a new detector later requires zero changes here as long as it uses one of the
pre-registered keys in :data:`SIGNAL_WEIGHTS`.

The weekly-confirmation signal is handled specially: it is *not* in the
weighted set; instead it contributes a +0..5 pt bonus on top of the composite
(capped at 100). Its :class:`SignalReading`'s ``value`` field carries the
bonus points directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .signals import SignalReading

SIGNAL_WEIGHTS: dict[str, float] = {
    "rsi": 0.25,
    "macd": 0.25,
    "bollinger": 0.20,
    "volume": 0.15,
    "obv": 0.10,
    "stochastic": 0.05,
}

# The weekly signal is a bonus, not a weighted contributor.
WEEKLY_BONUS_KEY = "weekly"
WEEKLY_BONUS_MAX = 5.0

# Sanity check: weights must sum to 1.0 (± tiny float error).
assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9, "SIGNAL_WEIGHTS must sum to 1.0"


@dataclass(frozen=True)
class ScoreBand:
    label: str
    emoji: str
    lo: float
    hi: float

    def contains(self, score: float) -> bool:
        return self.lo <= score <= self.hi


SCORE_BANDS: tuple[ScoreBand, ...] = (
    ScoreBand("Weak", "🔴", 0.0, 39.999),
    ScoreBand("Moderate", "🟡", 40.0, 59.999),
    ScoreBand("Strong", "🟠", 60.0, 74.999),
    ScoreBand("Very Strong", "🟢", 75.0, 89.999),
    ScoreBand("Conviction", "⚡", 90.0, 100.0),
)


def compute_score(signals: dict[str, SignalReading]) -> float:
    """Return the weighted 0-100 composite score, plus the weekly bonus.

    Weighted signals absent from ``signals`` or with strength <= 0 contribute
    nothing. The weekly-confirmation reading (key ``"weekly"``) adds its
    ``value`` (0..5 pts) on top of the composite. Final score is clamped to
    [0, 100].
    """
    total = 0.0
    for key, weight in SIGNAL_WEIGHTS.items():
        reading = signals.get(key)
        if reading is None:
            continue
        total += max(0.0, min(1.0, reading.strength)) * weight * 100.0

    weekly = signals.get(WEEKLY_BONUS_KEY)
    if weekly is not None and weekly.value is not None:
        total += max(0.0, min(WEEKLY_BONUS_MAX, float(weekly.value)))

    return round(max(0.0, min(100.0, total)), 4)


def band_for(score: float) -> ScoreBand:
    for band in SCORE_BANDS:
        if band.contains(score):
            return band
    return SCORE_BANDS[0]
