"""Exam-tier source for `rectangle-perimeter`: a perimeter and a relationship between the sides.

The practice tiers give the perimeter and one side and ask for the other. Here the perimeter is given along
with how much longer the length is than the width, so the learner has two unknown sides tied together and
must choose a method (halve the perimeter, take off the difference, halve again) before finding both sides
and then the area. Parts (b) and (c) carry follow-through.

Built backwards from `(width, difference)`, so the perimeter is 4 × width + 2 × difference and the width is a
whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, generator, phrase, round_half_up

from .exd_x_m3_support import finish, numeric_part

_THINGS = ["paddock", "school courtyard", "veggie patch", "skate park", "tennis court practice area", "car park"]
_STEMS = [
    "The perimeter of a rectangular {thing} is {P} m. Its length is {d} m more than its width.",
    "A rectangular {thing} has a perimeter of {P} m. The length of the {thing} is {d} m longer than the width.",
    "A rectangular {thing} is fenced with {P} m of fencing, with no gaps. It is {d} m longer than it is wide.",
]


@generator("exd_x_m3_rectangle_perimeter_unknown", level=10, presets={"exam": dict(tier="exam")})
def exd_x_m3_rectangle_perimeter_unknown(rng: random.Random, tier: str = "exam") -> dict:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a width and difference with distinct answers"):
        w = rng.randint(4, 45)
        d = rng.randint(2, 18)
        P = 4 * w + 2 * d
        length = w + d
        area = w * length
        thing = rng.choice(_THINGS)
        stem = phrase(rng, _STEMS, thing=thing, P=P, d=d)
        parts = [
            numeric_part(
                f"Find the width of the {thing}, in metres.",
                w,
                0,
                2,
                [
                    f"Two lengths and two widths: {P} = 2w + 2(w + {d}) = 4w + {2 * d}.",
                    f"4w = {P - 2 * d}, so w = {w} m.",
                ],
                [
                    # whole metres like the key, so the format never gives the answer away
                    (round_half_up(Fraction(P, 4), 0), None),
                    (round_half_up(Fraction(P - d, 4), 0), None),
                    (P / 2 - d, None),
                ],
            ),
            numeric_part(
                "Hence find the length, in metres.",
                length,
                0,
                1,
                [f"Length = {w} + {d} = {length} m."],
                [(w - d, None), (P / 2, "half-perimeter-as-side"), (w + 2 * d, None)],
                follow_through=f"part_a + {d}",
            ),
            numeric_part(
                f"Find the area of the {thing}, in square metres.",
                area,
                0,
                1,
                [f"Area = {w} × {length} = {area} m²."],
                [(P / 2, "perimeter-for-area"), (w * d, None), (length * length, None)],
                follow_through="part_a*part_b",
            ),
        ]
        out = finish(stem, parts)
        if out is not None:
            return out
