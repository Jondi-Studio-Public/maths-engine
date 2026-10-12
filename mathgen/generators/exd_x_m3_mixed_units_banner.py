"""Exam-tier source for `rectangle-area-perimeter-mixed-units`: a banner with sides in metres and centimetres.

The practice tiers convert two sides to a common unit and give one perimeter or one area. Here one rectangle
is carried through four steps whose units change: a perimeter in metres, the cost of edging it, the area in
square centimetres, and then the cost of painting it by the square metre. The last step is the trap: the
area in square centimetres must be divided by 100 x 100, not by 100.

Built forwards from `(length in tenths of a metre, width in tens of centimetres, prices)`, so every answer
is exact to 2 decimal places or a whole number of square centimetres. Parts (b) and (d) carry follow-through.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, generator, phrase

from .exd_x_m3_support import finish, numeric_part

_THINGS = ["banner", "mural", "noticeboard", "shopfront sign", "stage backdrop", "canvas print"]
_STEMS = [
    "A rectangular {thing} is {L} m long and {W} cm wide.",
    "A {thing} is a rectangle. Its length is {L} m and its width is {W} cm.",
    "The school is making a rectangular {thing}, {L} m long by {W} cm wide.",
]


@generator("exd_x_m3_mixed_units_banner", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m3_mixed_units_banner(rng: random.Random, tier: str = "exam") -> dict:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a banner with distinct answers"):
        tenths = rng.randint(21, 98)  # length in tenths of a metre
        W = 10 * rng.randint(3, 19)  # width in cm
        edge_price = rng.choice([4, 5, 6, 8, 9, 12, 15])
        paint_price = rng.choice([6, 8, 9, 12, 14, 15, 18, 22])
        L = Fraction(tenths, 10)
        Lcm = 100 * L
        perimeter = 2 * (L + Fraction(W, 100))
        edge_cost = perimeter * edge_price
        area_cm2 = Lcm * W
        area_m2 = area_cm2 / 10000
        paint_cost = area_m2 * paint_price
        thing = rng.choice(_THINGS)
        stem = phrase(rng, _STEMS, thing=thing, L=f"{float(L):.1f}", W=W)
        parts = [
            numeric_part(
                f"Find the perimeter of the {thing}, in metres. Give your answer to 2 decimal places.",
                perimeter,
                2,
                2,
                [
                    f"{W} cm = {Fraction(W, 100)} m.",
                    f"Perimeter = 2 × ({float(L):.1f} + {float(Fraction(W, 100)):.1f}) m.",
                ],
                [
                    (2 * (L + W), "unit-not-converted"),
                    (2 * (L + Fraction(W, 10)), "unit-factor-wrong-power"),
                    (L + Fraction(W, 100), None),
                    (L * Fraction(W, 100), "area-for-perimeter"),
                ],
            ),
            numeric_part(
                f"Edging costs ${edge_price} per metre. Find the cost of edging all four sides, in dollars. "
                "Give your answer to 2 decimal places.",
                edge_cost,
                2,
                1,
                [f"Cost = {float(perimeter):.2f} × {edge_price}."],
                [
                    (2 * (L + W) * edge_price, "unit-not-converted"),
                    ((L + Fraction(W, 100)) * edge_price, None),
                    (L * Fraction(W, 100) * edge_price, "area-for-perimeter"),
                ],
                follow_through=f"part_a*{edge_price}",
            ),
            numeric_part(
                f"Find the area of the {thing}, in square centimetres.",
                area_cm2,
                0,
                2,
                [f"{float(L):.1f} m = {int(Lcm)} cm.", f"Area = {int(Lcm)} × {W} cm²."],
                [
                    (L * W, "unit-not-converted"),
                    (L * 10 * W, "unit-factor-wrong-power"),  # 10 cm to the metre
                    (Lcm * W / 100, "linear-factor-on-area"),  # divided by 100, as for a length
                    (2 * (Lcm + W), "perimeter-for-area"),
                ],
            ),
            numeric_part(
                f"Paint costs ${paint_price} per square metre. Find the cost of painting one face of the {thing}, "
                "in dollars. Give your answer to 2 decimal places.",
                paint_cost,
                2,
                2,
                [
                    "1 m² = 100 cm × 100 cm = 10 000 cm².",
                    f"Area = {float(area_m2):.2f} m²; cost = area × {paint_price}.",
                ],
                [
                    (area_cm2 / 100 * paint_price, "linear-factor-on-area"),
                    (area_cm2 / 1000 * paint_price, "unit-factor-wrong-power"),
                    (area_cm2 * paint_price, "unit-not-converted"),
                ],
                follow_through=f"part_c/10000*{paint_price}",
            ),
        ]
        out = finish(stem, parts)
        if out is not None:
            return out
