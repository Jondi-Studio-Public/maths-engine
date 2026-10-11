"""Exam tier for "determine a linear equation from a table".

The practice tiers give a table of ``x`` and ``y`` and ask for ``y = mx + b`` directly, or dress the same
numbers in a flat-fee story. Here the table records time in a SMALL unit (minutes, hours, days or months)
but the equation must be written in terms of a number of the BIGGER unit (hours, days, weeks or years). The
learner has to find the gradient and intercept from the table and then re-express the relationship in the new
variable: the gradient is multiplied by the number of small units in one big unit, and the intercept does not
change. That is a composition of two ideas (a linear model from data, then a change of variable), not a bigger
table.

Built backwards: draw the per-small-unit gradient ``m`` and the intercept ``b``, three distinct positive times
in the small unit, and compute the outputs. The key is ``y = (f*m)x + b`` with ``f`` the unit factor. It is
re-derived from the printed table alone with exact fractions.

Distractors, each a named misconception:

- never converted the time unit (``y = mx + b``);
- converted the gradient but used the first table output as the intercept;
- converted the intercept as well as the gradient (``y = f*m x + f*b``);
- divided by the unit factor instead of multiplying (``y = (m/f)x + b``).
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

from ..core import check, generator, join_terms, pick_distractors

# (small unit, big unit, small units per big unit)
_UNITS = (("minute", "hour", 60), ("hour", "day", 24), ("day", "week", 7), ("month", "year", 12))

# (description, quantity name for the table row, unit label)
_QUANTITIES = (
    ("The volume of water in a tank", "Volume", "litres"),
    ("The depth of water in a dam", "Depth", "cm"),
    ("The mass of fuel in a store", "Mass", "kg"),
)


def _eq(slope: Fraction, intercept: int) -> str:
    """``y = mx + b`` the way a student writes it; a fractional gradient is written ``px/q``."""
    if slope.denominator == 1:
        return "y = " + join_terms([(int(slope), "x"), (intercept, "")])
    sign = "-" if slope < 0 else ""
    head = f"{sign}{abs(slope.numerator)}x/{slope.denominator}"
    if slope.numerator == 1:
        head = f"{sign}x/{slope.denominator}"
    tail = "" if intercept == 0 else (f" + {intercept}" if intercept > 0 else f" - {-intercept}")
    return "y = " + head + tail


@generator(
    "exd_x_a4_table_unit_change",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a4_table_unit_change(rng: random.Random, tier: str = "exam") -> dict:
    """Linear equation from a table, in a rescaled variable. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    small, big, factor = rng.choice(_UNITS)
    thing, row, unit = rng.choice(_QUANTITIES)
    while True:
        m = rng.choice([-1, 1]) * rng.randint(1, 5)
        b = rng.randint(30, 160)
        ts = sorted(rng.sample(range(2, 31), 3))
        ys = [m * t + b for t in ts]
        if min(ys) >= 5:
            break
    table = f"Time ({small}s): {', '.join(map(str, ts))}. {row} ({unit}): {', '.join(map(str, ys))}."
    text = (
        f"{thing} was recorded over time and changes at a constant rate. {table} "
        f"Write the equation for the {row.lower()} y in terms of x, the number of {big}s "
        f"(y = mx + b)."
    )

    # Independent route: parse the printed table, fit exactly, then rescale.
    t_row = [int(v) for v in re.search(r"Time \(\w+\): ([\d, ]+)\.", text).group(1).split(", ")]
    y_row = [int(v) for v in re.search(r"\(%s\): ([\d, ]+)\." % re.escape(unit), text).group(1).split(", ")]
    slope_small = Fraction(y_row[1] - y_row[0], t_row[1] - t_row[0])
    check(
        all(Fraction(y_row[i + 1] - y_row[i], t_row[i + 1] - t_row[i]) == slope_small for i in range(2)),
        "table is not linear",
    )
    intercept = y_row[0] - slope_small * t_row[0]
    key_slope = slope_small * factor
    check(key_slope == m * factor and intercept == b, "re-derived equation disagrees with the construction")

    answer_text = _eq(key_slope, b)
    candidates = [
        _eq(Fraction(m), b),  # time unit never converted
        _eq(key_slope, ys[0]),  # first output used as the intercept
        _eq(key_slope, b * factor),  # intercept rescaled as well
        _eq(Fraction(m, factor), b),  # divided by the unit factor instead of multiplying
    ]
    return dict(
        text=text,
        latex=text,
        answer={"m": int(key_slope), "b": b},
        answer_text=answer_text,
        working=[
            f"Gradient per {small} = ({y_row[1]} - {y_row[0]}) / ({t_row[1]} - {t_row[0]}) = {m}.",
            f"Intercept = {y_row[0]} - ({m})({t_row[0]}) = {b}.",
            f"One {big} is {factor} {small}s, so the gradient per {big} is {m} x {factor} = {m * factor}.",
            answer_text + ".",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )
