"""Exam question: from two trend values to a seasonally adjusted forecast.

The trend of a time series is linear. Two trend values in the same year are given. Part (a) finds the rate of change
per period, part (b) projects the trend to a period in a later year (counting periods carefully across the year
boundary), and part (c) turns that trend value into a forecast with the stated seasonal index for that period.
Parts (b) and (c) carry `follow_through`.

Built from the rate of change (a whole number of tenths) and the first trend value (tenths), so every trend value is on
the tenths grid and part (a) is exact. The forecast is the product of the trend value and a two-place index, rounded
once to 1 decimal place, and a draw is kept only when it sits clear of a rounding boundary.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator
from .exd_x_r2k_support import retry_build, POSITIVE, clear_of_boundary, number_part, rd, tx

# measure, period word, plural, periods per year, ordinal-name function style
_CONTEXTS = (
    ("a cafe's coffee sales (thousands of cups)", "quarter", "quarters", 4),
    ("a toy shop's sales (thousands of dollars)", "quarter", "quarters", 4),
    ("a ski resort's visitor numbers (hundreds)", "quarter", "quarters", 4),
    ("a gym's new memberships", "month", "months", 12),
    ("a surf school's bookings", "month", "months", 12),
    ("a garden centre's takings (thousands of dollars)", "month", "months", 12),
)


def _build_k_trend_forecast(rng: random.Random) -> dict | None:
    """Rate of change from two trend values, the trend at a later period, and the seasonal forecast."""
    measure, period, plural, per_year = rng.choice(_CONTEXTS)
    for _ in attempts(3000, "a trend with a clean seasonal forecast"):
        q1 = rng.randint(1, per_year - 2)
        q2 = rng.randint(q1 + 1, per_year)
        qt = rng.randint(1, per_year)
        year = rng.choice((3, 4))
        m = Fraction(rng.choice((-1, 1, 1, 1)) * rng.randint(3, 80), 10)
        y1 = Fraction(rng.randint(150, 1500), 10) if per_year == 4 else Fraction(rng.randint(300, 2500), 10)
        y2 = y1 + m * (q2 - q1)
        t1 = per_year + q1  # t = 1 is the first period of Year 1
        tt = per_year * (year - 1) + qt
        trend_t = y1 + m * (tt - t1)
        if trend_t < 15 or y2 < 10 or y1 < 10:
            continue
        s = Fraction(rng.randint(70, 140), 100)
        if s == 1:
            continue
        forecast = trend_t * s
        if not clear_of_boundary(forecast, 1, Fraction(1, 5)):
            continue
        break
    check(y2 - y1 == m * (q2 - q1), "trend: second value is off the line")

    def label(q: int) -> str:
        return f"{period} {q}"

    stem = (
        f"The trend in {measure} is linear, with t = 1 the first {period} of Year 1 and {per_year} {plural} in each "
        f"year. The trend value for {label(q1)} of Year 2 is {tx(y1, 1)} and the trend value for {label(q2)} of Year 2 "
        f"is {tx(y2, 1)}. The seasonal index for {label(qt)} is {tx(s, 2)}."
    )
    pa = number_part(
        f"Find the rate of change of the trend per {period}, correct to 1 decimal place.",
        m,
        1,
        1,
        [f"({tx(y2, 1)} - {tx(y1, 1)}) ÷ ({q2} - {q1}) = {tx(m, 1)} per {period}."],
        [
            ((y1 - y2) / (q2 - q1) if y1 != y2 else None, "reversed-difference"),
            (Fraction(q2 - q1) / (y2 - y1) if y2 != y1 else None, "inverted-rate-of-change"),
            (y2 - y1, "change-not-rate"),
            ((y2 - y1) / (q2 - q1 + 1), "counted-both-endpoints"),
        ],
    )
    pb = number_part(
        f"Use your rate of change to find the trend value for {label(qt)} of Year {year}, correct to 1 decimal place.",
        trend_t,
        1,
        2,
        [
            f"t for {label(q1)} of Year 2 is {t1}; t for {label(qt)} of Year {year} is {tt}.",
            f"Trend = {tx(y1, 1)} {'+' if m >= 0 else '-'} {tx(abs(m), 1)} × ({tt} - {t1}) = {tx(trend_t, 1)}.",
        ],
        [
            (y1 + m * (tt - t1 - per_year), "miscounted-periods-between"),
            (y1 + m * (tt - t1 + 1), "off-by-one-period"),
            (y1 + m * tt, "measured-from-t-zero-not-known-point"),
            (y2 + m * (tt - t1), "wrong-anchor-point"),
        ],
        follow_through=f"{y1} + part_a*({tt} - {t1})",
        sensible_range=POSITIVE,
    )
    pc = number_part(
        f"Hence use the seasonal index to forecast the value for {label(qt)} of Year {year}, correct to 1 decimal place.",
        forecast,
        1,
        2,
        [f"Forecast = trend × index = {tx(trend_t, 1)} × {tx(s, 2)} = {tx(forecast, 1)}."],
        [
            (trend_t / s, "divided-by-seasonal-index"),
            (trend_t * (s - 1) if s > 1 else None, "gave-adjustment-not-forecast"),
            (trend_t, "ignored-seasonal-index"),
        ],
        follow_through=f"part_b*{s.numerator}/{s.denominator}",
        sensible_range=POSITIVE,
    )
    if pa is None or pb is None or pc is None:
        return None
    return dict(text=stem, parts=[pa, pb, pc])


@generator("exd_x_r2k_trend_forecast", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2k_trend_forecast(rng: random.Random, tier: str = "exam") -> dict:
    """Rate of change from two trend values, the trend at a later period, and the seasonal forecast."""
    check(tier == "exam", "exd_x_r2k_trend_forecast: only the exam tier exists")
    return retry_build(rng, _build_k_trend_forecast, "exd_x_r2k_trend_forecast")
