"""Exam tier for "estimate the areas of different shapes": a paddock estimated by pacing, then used.

The practice tiers choose the best estimate of a familiar area. Here the learner estimates a paddock's area from
a measured pace and two counts of paces, which needs the pace length applied to BOTH dimensions (a linear factor
used once is the classic slip), converts square metres to hectares, and uses the area in a stocking calculation
whose answer must be rounded down to whole animals.

Parts (b) and (c) carry ``follow_through``. Built backwards from the two side lengths in metres: the pace is
chosen so each side is a whole number of paces, and the area is a multiple of 100 square metres so the area in
hectares is exact to 2 decimal places.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, draw_until, generator
from ._exd_x_sc4_kit import num_part, txt

_PACES = (Fraction(3, 5), Fraction(3, 4), Fraction(4, 5), Fraction(9, 10))
_PLACES = (
    "A farmer estimates the area of a rectangular paddock by pacing it.",
    "A grazier estimates the area of a rectangular paddock by pacing it out.",
    "A landholder estimates the area of a rectangular paddock by pacing its sides.",
)


@generator("exd_x_sc4_est_area", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc4_est_area(rng: random.Random, tier: str = "exam") -> dict:
    """Estimate an area from paces, convert it to hectares, then use it. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")

    def draw(r):
        return r.choice(_PACES), r.randrange(60, 301, 5), r.randrange(60, 301, 5), r.randint(5, 15)

    def fine(values):
        pace, l1, l2, stock = values
        if l1 <= l2 or (l1 / pace).denominator != 1 or (l2 / pace).denominator != 1:
            return False
        area = l1 * l2
        if area % 100 != 0 or area > 80000:
            return False
        stocked = Fraction(area, 10000) * stock
        return Fraction(1, 10) <= stocked % 1 <= Fraction(9, 10)

    pace, l1, l2, stock = draw_until(rng, draw, fine, what="a paddock on whole paces and a two-decimal hectare area")
    n1, n2 = int(l1 / pace), int(l2 / pace)
    area = l1 * l2
    hectares = Fraction(area, 10000)
    sheep = math.floor(hectares * stock)
    check((hectares * 100).denominator == 1, "hectares must be exact to 2 decimal places")

    pace_text = txt(pace, 2) if (pace * 10).denominator != 1 else txt(pace, 1)
    stem = f"{rng.choice(_PLACES)} One pace is {pace_text} m. The paddock is {n1} paces long and {n2} paces wide."
    parts = [
        num_part(
            "Estimate the area of the paddock, in square metres.",
            Fraction(area),
            0,
            2,
            [
                f"Length = {n1} × {pace_text} = {l1} m, width = {n2} × {pace_text} = {l2} m.",
                f"Area = {l1} × {l2} = {area} m².",
            ],
            [
                (Fraction(n1 * n2) * pace, "linear-factor-on-area"),
                (Fraction(n1 * n2), "paces-not-converted"),
                (Fraction(2 * (l1 + l2)), "perimeter-for-area"),
            ],
        ),
        num_part(
            "Hence find the area in hectares, correct to 2 decimal places. (1 hectare = 10 000 m².)",
            hectares,
            2,
            1,
            [f"{area} ÷ 10 000 = {txt(hectares, 2)} ha."],
            [
                (Fraction(area, 1000), "conversion-power-of-ten-slip"),
                (Fraction(area, 100), "conversion-power-of-ten-slip"),
                (Fraction(area, 100000), "conversion-power-of-ten-slip"),
            ],
            follow_through="part_a/10000",
        ),
        num_part(
            f"The paddock can carry {stock} sheep per hectare. How many sheep can it carry? "
            "Only whole sheep count, so round down.",
            Fraction(sheep),
            0,
            2,
            [f"{txt(hectares, 2)} × {stock} = {float(hectares * stock):.2f}, so {sheep} sheep."],
            [
                (Fraction(sheep + 1), "rounded-up-when-capacity"),
                (Fraction(stock) / hectares, "inverted-ratio"),
                (hectares + stock, "added-instead-of-multiplied"),
            ],
            follow_through=f"floor(part_b*{stock})",
        ),
    ]
    parts[0]["marks"], parts[1]["marks"], parts[2]["marks"] = 2, 1, 2
    return dict(text=stem, parts=parts)
