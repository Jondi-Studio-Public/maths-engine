"""Exam-tier source for `senior-calculate-perimeters-of-familiar-shapes-including-triangles-squares`.

The practice tiers add the sides of a composite shape. Here a window is a rectangle topped with a semicircle,
and the learner finds the length of the curved edge (half a circumference), the perimeter of the whole
window (the shared edge is not part of it), and then uses that perimeter to size a different shape: a plain
rectangular frame of the same width and perimeter. Parts (b) and (c) carry follow-through.

Built forwards from `(width, height)`. A draw is rejected if rounding an earlier answer first would change a
later last digit, so the exact route and the learner's route agree.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.core import attempts, generator, phrase, round_half_up

from .exd_x_m3_support import clear_of_tie, display, finish, numeric_part

PI = Fraction(math.pi)
_THINGS = ["window", "arched mirror", "picture frame", "shopfront window", "noticeboard"]
_STEMS = [
    "A {thing} is made from a rectangle {w} cm wide and {h} cm high, with a semicircle on top. The diameter of the "
    "semicircle is the top edge of the rectangle.",
    "The frame of a {thing} is a rectangle with a semicircle fitted on its top edge. The rectangle is {w} cm wide and "
    "{h} cm high, and the semicircle's diameter is the full width of the rectangle.",
]


@generator("exd_x_m3_window_perimeter_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_m3_window_perimeter_chain(rng: random.Random, tier: str = "exam") -> dict:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(3000, "a window whose roundings are unambiguous"):
        w = rng.randint(30, 140)
        h = rng.randint(40, 160)
        arc = PI * w / 2
        perimeter = arc + w + 2 * h
        # the frame: same width, same perimeter, no semicircle
        frame_h = (perimeter - 2 * w) / 2
        if frame_h <= 0 or not all(clear_of_tie(v, 1) for v in (arc, perimeter, frame_h)):
            continue
        a1, b1 = round_half_up(arc, 1), round_half_up(perimeter, 1)
        if display(a1 + w + 2 * h, 1) != display(perimeter, 1) or display((b1 - 2 * w) / 2, 1) != display(frame_h, 1):
            continue
        thing = rng.choice(_THINGS)
        stem = phrase(rng, _STEMS, thing=thing, w=w, h=h)
        parts = [
            numeric_part(
                f"Find the length of the curved edge of the {thing}, in centimetres, correct to 1 decimal place.",
                arc,
                1,
                1,
                [f"The curved edge is half a circle of diameter {w} cm.", f"Arc = (1/2) × π × {w}."],
                [
                    (PI * w, "full-circle-for-semicircle"),
                    (PI * w * w / 8, "area-formula-for-circumference"),
                    (PI * w / 4, None),
                ],
            ),
            numeric_part(
                f"Hence find the perimeter of the {thing}, in centimetres, correct to 1 decimal place.",
                perimeter,
                1,
                2,
                [
                    "The perimeter is the curved edge, the base and the two sides; the top edge of the rectangle is "
                    "inside the shape.",
                    f"P = arc + {w} + 2 × {h}.",
                ],
                [
                    (arc + 2 * w + 2 * h, None),
                    (arc + w + h, None),
                    (arc + 2 * h, None),
                    (PI * w + w + 2 * h, "full-circle-for-semicircle"),
                ],
                follow_through=f"part_a + {w} + {2 * h}",
            ),
            numeric_part(
                f"A plain rectangular frame has the same width, {w} cm, and the same perimeter as the {thing}. Find "
                "its height, in centimetres, correct to 1 decimal place.",
                frame_h,
                1,
                2,
                [f"2 × {w} + 2 × height = perimeter.", "Height = (perimeter - 2 × width) ÷ 2."],
                [
                    (perimeter / 2, "half-perimeter-as-side"),
                    (perimeter - 2 * w, None),
                    ((perimeter - w) / 2, None),
                    (perimeter / 2 - w / 2, None),
                ],
                follow_through=f"(part_b - {2 * w})/2",
            ),
        ]
        out = finish(stem, parts)
        if out is not None:
            return out
