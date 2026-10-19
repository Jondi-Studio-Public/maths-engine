"""Generates `exd_x_w5b_e_limit_table`: a multi-part exam question on tables of (a^h - 1)/h.

Two bases p < e < q are tabulated at h = 0.1, 0.01 and 0.001. The chain a senior exam sets:

(a) the limit for p, from its table                         number, 2 places, 1 mark
(b) the limit for q, from its table                         number, 2 places, 1 mark
(c) the base for which the limit is 1, by linear
    interpolation on the two limits                         number, 2 places, 2 marks, follow-through (a), (b)
(d) where the tangent to y = q^x at (0, 1) meets the
    x-axis                                                  number, 2 places, 2 marks, follow-through (b)

The limit of (a^h - 1)/h is ln a, so the keys for (a) and (b) are ln p and ln q to two places; both are rebuilt from the
last table row and must round alike, clear of a rounding boundary. (c) and (d) are computed from the two-place values a
learner would have written in (a) and (b), which is what the follow-through formulas do. The tangent at (0, 1) is
y = 1 + L x, so it meets the x-axis where x = -1/L.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import draw_until, generator, phrase, round_half_up
from ._exd_distinct_parts import distinct_part_answers
from ._w5b_e_limit_kit import ROWS_ALL, away_from_e, base_text, draw_bracket, limit, quotient, table_text, target_text
from .exd_x_r3b_common import number_part

_STEMS = (
    "A student investigates the gradient of y = a^x at the point (0, 1), which is the limit of (a^h - 1)/h as h "
    "tends to 0. A calculator gives the quotient for two bases: {table}.",
    "The quotient (a^h - 1)/h is evaluated for two bases, each at three small values of h: {table}. "
    "The limit of this quotient as h tends to 0 is the gradient of y = a^x at (0, 1).",
    "To find the exponential function with a stated gradient at its y-intercept, (a^h - 1)/h was tabulated for two trial "
    "bases: {table}.",
)


def _clear(x: float) -> bool:
    scaled = x * 100
    return abs(scaled - math.floor(scaled) - 0.5) > 0.15


def _fraction_text(x: Fraction) -> str:
    return f"({x.numerator}/{x.denominator})"


def _build(rng: random.Random) -> dict:
    def accept(trio):
        target, p, q = trio
        if not (_clear(limit(p)) and _clear(limit(q))):
            return False
        a = round_half_up(quotient(p, "0.001"), 2)
        b = round_half_up(quotient(q, "0.001"), 2)
        if a != round_half_up(Fraction(limit(p)), 2) or b != round_half_up(Fraction(limit(q)), 2):
            return False
        if not (a < target < b) or abs(float(-1 / b)) > 4:
            return False
        x = p + (q - p) * (target - a) / (b - a)
        return p < x < q and _clear(float(x)) and _clear(float(-1 / b)) and abs(float(x) - 2.72) > 0.15

    target, p, q = draw_until(
        rng, draw_bracket, accept, tries=800, what="two bases bracketing e^L with clean two-place limits"
    )
    table, grid = table_text([p, q], ROWS_ALL, 2)
    key_a = round_half_up(Fraction(limit(p)), 2)
    key_b = round_half_up(Fraction(limit(q)), 2)
    x = round_half_up(p + (q - p) * (target - key_a) / (key_b - key_a), 2)
    t = round_half_up(-1 / key_b, 2)
    stem = phrase(rng, _STEMS, table=table)
    tt = target_text(target)
    ps, qs = base_text(p, 2), base_text(q, 2)

    part_a = number_part(
        f"Estimate the limit of (a^h - 1)/h as h tends to 0 for a = {ps}, to 2 decimal places.",
        key_a,
        2,
        1,
        [f"The values are settling near {float(grid[p]['0.001']):.4f}, which is {float(key_a):.2f} to 2 places."],
        [
            ("reads-coarsest-row-as-the-limit", round_half_up(grid[p]["0.1"], 2)),
            ("picks-the-neighbouring-base-above", round_half_up(grid[q]["0.001"], 2)),
            ("gives-the-base-instead-of-the-limit", p),
        ],
        sensible_range=(0, 3),
    )
    part_b = number_part(
        f"Estimate the limit of (a^h - 1)/h as h tends to 0 for a = {qs}, to 2 decimal places.",
        key_b,
        2,
        1,
        [f"The values are settling near {float(grid[q]['0.001']):.4f}, which is {float(key_b):.2f} to 2 places."],
        [
            ("reads-coarsest-row-as-the-limit", round_half_up(grid[q]["0.1"], 2)),
            ("picks-the-neighbouring-base-below", round_half_up(grid[p]["0.001"], 2)),
            ("gives-the-base-instead-of-the-limit", q),
        ],
        sensible_range=(0, 3),
    )
    ratio = (target - key_a) / (key_b - key_a)
    ft_c = f"{_fraction_text(p)} + ({_fraction_text(q)} - {_fraction_text(p)})*({_fraction_text(target)} - part_a)/(part_b - part_a)"
    part_c = number_part(
        f"Use your two limits and linear interpolation to estimate the value of a for which the limit of "
        f"(a^h - 1)/h is {tt}, to 2 decimal places.",
        x,
        2,
        2,
        [
            f"The limit has to rise {float(target - key_a):.2f} of the {float(key_b - key_a):.2f} between the two bases.",
            f"a = {ps} + {float(q - p):.3f} x {float(ratio):.4f}, which is about {float(x):.2f}.",
        ],
        away_from_e(
            [
                ("midpoint-of-the-two-bases-used", (p + q) / 2),
                ("interpolation-ratio-inverted", p + (q - p) * (key_b - target) / (key_b - key_a)),
                ("nearer-base-taken-without-interpolating", p if (target - key_a) < (key_b - target) else q),
                ("interpolation-fraction-not-scaled-to-the-bases", ratio),
            ]
        ),
        follow_through=ft_c,
        sensible_range=(1, 10),
    )
    part_d = number_part(
        f"The tangent to y = {qs}^x at the point (0, 1) has gradient equal to your limit for a = {qs}. "
        "Find the x-coordinate where this tangent meets the x-axis, to 2 decimal places.",
        t,
        2,
        2,
        [
            f"The tangent is y = 1 + {float(key_b):.2f}x.",
            f"It meets the x-axis where 0 = 1 + {float(key_b):.2f}x, so x = -1/{float(key_b):.2f}, about {float(t):.2f}.",
        ],
        [
            ("tangent-intercept-sign-dropped", -t),
            ("tangent-intercept-is-the-gradient", key_b),
            ("tangent-intercept-uses-the-other-base", -1 / key_a),
        ],
        follow_through="-1/part_b",
        sensible_range=(-5, 0),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_w5b_e_limit_table", level=18, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_w5b_e_limit_table(rng: random.Random, tier: str = "exam") -> dict:
    """Two limits read from tables, the base that makes the limit 1 by interpolation, and a tangent's x-intercept."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
