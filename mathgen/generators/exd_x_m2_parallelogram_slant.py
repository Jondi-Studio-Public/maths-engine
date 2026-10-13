"""Exam source for `f10-parallelogram-area`: a parallelogram given by its slanted side, then area, then cost.

The perpendicular height is not given. The learner is told the base, the slanted side and how far the slanted
side leans over the base, so the height comes from a right triangle (a Pythagorean triple, so it is exact).
Then:

(a) the perpendicular height, (b) the area using it, (c) the cost of covering that area at a price per square
metre. Parts (b) and (c) carry follow-through.

Built backwards from a triple ``(d, h, s)``, a base and a price, so every answer is an exact integer.
"""

from __future__ import annotations

import random

from mathgen.core import attempts, check, generator, latex_answer
from mathgen.generators.exd_x_m2_support import keys_absent, tagged_distractors

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (6, 8, 10), (9, 12, 15), (12, 16, 20), (15, 20, 25)]
_OBJECTS = [
    "A paving slab",
    "A garden bed",
    "A section of decking",
    "A sail panel",
    "A car park",
    "A veggie patch",
    "A shade-cloth panel",
    "A concrete pad",
]


@generator("exd_x_m2_parallelogram_slant", level=7, presets={"exam": dict(tier="exam")})
def exd_x_m2_parallelogram_slant(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: height from the slanted side, then area, then cost."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a draw whose answers are not already in the text"):
        d, h, s = rng.choice(_TRIPLES)
        k = 2 if s <= 13 and rng.random() < 0.4 else 1
        d, h, s = d * k, h * k, s * k
        b = rng.randint(8, 40)
        price = rng.choice([12, 15, 18, 22, 25, 28, 35, 42, 55])
        check(d * d + h * h == s * s, "triple is not Pythagorean")
        area = b * h
        cost = area * price
        stem = (
            f"{rng.choice(_OBJECTS)} is a parallelogram with a base of {b} m and a slanted side of {s} m. "
            f"The slanted side leans over the base by a horizontal distance of {d} m. "
            f"Covering it costs ${price} per square metre."
        )
        prompts = [
            "Find the perpendicular height of the parallelogram, in metres.",
            "Hence find its area, in square metres.",
            "Find the cost of covering the whole parallelogram, in dollars.",
        ]
        a_text, b_text, c_text = str(h), str(area), str(cost)
        if keys_absent(stem, prompts, [a_text, b_text, c_text]):
            break

    a_dist = tagged_distractors(
        a_text,
        0,
        [(s, None), (s - d, None), (s + d, None), (d, None), (b - d, None)],
    )
    b_dist = tagged_distractors(
        b_text,
        0,
        [(b * s, None), (b * d, None), (b * h / 2, None), (b + h, None), (b * (s - d), None)],
    )
    c_dist = tagged_distractors(
        c_text,
        0,
        [(b * s * price, None), (area / price, None), (area + price, None), (b * h * price / 2, None)],
    )
    for dist in (a_dist, b_dist, c_dist):
        check(len(dist) >= 3, "not enough distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=h,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    "Draw a perpendicular from the top corner to the base. It makes a right triangle.",
                    f"h² = {s}² - {d}² = {s * s - d * d}, so h = {h} m.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=area,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{b}*part_a",
                working=[f"Area = base × perpendicular height = {b} × {h} = {area} m²."],
                distractors=b_dist,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=cost,
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{price}*part_b",
                working=[f"Cost = {area} × ${price} = ${cost}."],
                distractors=c_dist,
            ),
        ],
    )
