"""Exam tier for moving averages: two problems that need more than "add the window and divide".

  missing  -- a series with one unknown observation, and the moving average over a window that contains it;
              the learner works backwards from the average to the missing value.
  centred  -- a centred moving average over an EVEN window (four quarters), which has no middle observation:
              the two end values each count half, i.e. (x1/2 + x2 + x3 + x4 + x5/2) / 4.

Both are built backwards. For `missing`, the unknown and the window average are chosen so that the average is
a multiple of a quarter and the unknown is a positive whole number. For `centred`, the five values are drawn
so the weighted sum is even, making the answer an exact multiple of 0.25 (shown to two decimal places).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator
from .exd_x_g5_common import number_fields

_ORDINAL = {1: "1st", 2: "2nd", 3: "3rd"}
_MEASURES = (
    ("weekly sales (units)", "week"),
    ("daily visitors", "day"),
    ("monthly rainfall (mm)", "month"),
    ("quarterly profit ($ thousand)", "quarter"),
)


def _ord(n: int) -> str:
    return _ORDINAL.get(n, f"{n}th")


def _plain(x: Fraction) -> str:
    return f"{float(x):g}"


def _missing(rng: random.Random) -> dict:
    length = rng.randint(7, 9)
    window = rng.choice((3, 4, 5))
    step = Fraction(1, 4) if window == 4 else Fraction(1)
    start = rng.randint(1, length - window + 1)  # 1-based first index of the window
    window_idx = list(range(start, start + window))
    hidden = rng.choice(window_idx)
    while True:
        values = [rng.randint(18, 60) for _ in range(length)]
        others = sum(values[i - 1] for i in window_idx if i != hidden)
        average = step * rng.randint(int(20 / step), int(55 / step))
        unknown = average * window - others
        if unknown.denominator == 1 and 10 <= unknown <= 80 and abs(unknown - average) > 1:
            break
    values[hidden - 1] = int(unknown)
    measure, unit = rng.choice(_MEASURES)
    shown = ["?" if i + 1 == hidden else str(v) for i, v in enumerate(values)]
    text = (
        f"The {measure} for {length} consecutive {unit}s are {', '.join(shown)} (the {_ord(hidden)} value is "
        f"missing). The {window}-point moving average of {unit}s {window_idx[0]} to {window_idx[-1]} is "
        f"{_plain(average)}. Find the missing value."
    )
    check(Fraction(sum(values[i - 1] for i in window_idx), window) == average, "window average does not reproduce")
    wrong = [
        average,  # the average itself taken as the missing value
        average * window,  # the window sum, with the known values not subtracted
        Fraction(others, window - 1),  # the missing value assumed equal to the mean of the known ones
        average * (window - 1) - others,  # one fewer than the window length used as the divisor
        average * (window + 1) - others,  # one more than the window length used as the divisor
    ]
    working = [
        f"Window sum = {window} x {_plain(average)} = {_plain(average * window)}.",
        f"Known values in the window add to {others}.",
        f"Missing value = {_plain(average * window)} - {others} = {values[hidden - 1]}.",
    ]
    return number_fields(text, Fraction(values[hidden - 1]), 0, wrong, working)


def _centred(rng: random.Random) -> dict:
    while True:
        values = [rng.randint(12, 90) for _ in range(5)]
        if (values[0] + 2 * (values[1] + values[2] + values[3]) + values[4]) % 2:
            continue
        x = [Fraction(v) for v in values]
        key = Fraction(values[0] + 2 * (values[1] + values[2] + values[3]) + values[4], 8)
        slips = [sum(x) / 5, sum(x) / 4, (x[1] + x[2] + x[3] + x[4]) / 4, (x[1] + x[2] + x[3]) / 3]
        if all(abs(v - key) > 1 for v in slips) and len({round(float(v), 1) for v in slips}) == 4:
            break
    measure, unit = rng.choice(_MEASURES)
    answer = Fraction(values[0] + 2 * (values[1] + values[2] + values[3]) + values[4], 8)
    text = (
        f"The {measure} for five consecutive {unit}s are {', '.join(map(str, values))}. A 4-point moving average "
        f"is centred on the middle {unit} by giving the first and last of the five values half weight and the "
        f"three inner values full weight, then dividing by 4. Find the centred moving average for the middle "
        f"{unit}, to 2 decimal places."
    )
    x = [Fraction(v) for v in values]
    wrong = [
        sum(x) / 5,  # plain mean of all five values
        sum(x) / 4,  # all five values summed, divided by the window length
        (x[1] + x[2] + x[3] + x[4]) / 4,  # an uncentred 4-point average
        (x[1] + x[2] + x[3]) / 3,  # the inner three only
    ]
    working = [
        f"Weighted sum = 0.5 x {values[0]} + {values[1]} + {values[2]} + {values[3]} + 0.5 x {values[4]} "
        f"= {_plain(Fraction(values[0] + values[4], 2) + values[1] + values[2] + values[3])}.",
        f"Divide by 4: {float(answer):.2f}.",
    ]
    return number_fields(text, answer, 2, wrong, working)


_VARIANTS = {"missing": _missing, "centred": _centred}


@generator("exd_x_g5_moving_average", level=11, presets={"exam": dict(tier="exam")})
def exd_x_g5_moving_average(rng: random.Random, tier: str = "exam") -> dict:
    """A missing observation from a window average, or a centred even-window moving average."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _VARIANTS[rng.choice(tuple(_VARIANTS))](rng)
