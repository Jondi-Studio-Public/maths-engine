"""Exam chain on one sector of a circle, with the angle in degrees.

(a) the area of the sector                                                      number, 2 marks
(b) the length of its arc                                                       number, 2 marks
(c) the perimeter of the sector (follow-through (b))                            number, 1 mark
(d) the area of the rest of the circle (follow-through (a))                     number, 2 marks

All answers are to 1 decimal place and use the true value of pi. The draw is kept where working from the rounded
earlier answers gives the same last digit as the exact working, so a learner who follows through is not penalised.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4g_kit import clear_of_tie, number_part, number_text, retry
from .r4g_sector_degrees_build import ANGLES, PI, area, arc

SCENES = [
    (
        "A circular lawn",
        "m",
        "A sprinkler at the centre waters a sector of the lawn with a centre angle of {theta}°.",
        "of the lawn that is not watered",
        (5, 20),
        250,
    ),
    (
        "A round pizza",
        "cm",
        "A slice cut from the centre is a sector with an angle of {theta}°.",
        "of the pizza that is left",
        (10, 18),
        120,
    ),
    (
        "A circular dance floor",
        "m",
        "A fan-shaped stage takes up a sector of the floor with a centre angle of {theta}°.",
        "of the floor that is not stage",
        (6, 14),
        200,
    ),
]


def r1(x) -> Fraction:
    return Fraction(number_text(x, 1))


@distinct_part_answers
def _build(rng):
    for _ in attempts(2000, "a sector whose follow-through values round the same way"):
        name, unit, sentence, rest, radii, top = rng.choice(SCENES)
        r = Fraction(rng.randint(*radii))
        theta = rng.choice([a for a in ANGLES if a <= top])
        a_, l_ = area(r, theta), arc(r, theta)
        whole = PI * r * r
        if not (clear_of_tie(a_, 1) and clear_of_tie(l_, 1) and clear_of_tie(whole - a_, 1)):
            continue
        if r1(r1(l_) + 2 * r) != r1(l_ + 2 * r) or r1(whole - r1(a_)) != r1(whole - a_):
            continue
        break
    peri, remaining = l_ + 2 * r, whole - a_
    check(abs(float(remaining) - (1 - theta / 360) * math.pi * float(r) ** 2) < 1e-9, "remainder re-derivation differs")
    text = f"{name} has a radius of {int(r)} {unit}. {sentence.format(theta=theta)} Use the π button on your calculator and give each answer correct to 1 decimal place."
    parts = [
        number_part(
            "Find the area of the sector.",
            a_,
            1,
            2,
            [f"Area = {theta}/360 x π x {int(r)}² = {number_text(a_, 1)} {unit}²."],
            [
                (Fraction(theta, 360) * PI * r, "r4g-radius-not-squared"),
                (arc(r, theta), "r4g-arc-formula-for-area"),
                (whole, "r4g-whole-circle-area"),
                (Fraction(theta, 360) * PI * 4 * r * r, "r4g-diameter-squared-for-radius-squared"),
            ],
        ),
        number_part(
            "Find the length of the arc of the sector.",
            l_,
            1,
            2,
            [f"Arc = {theta}/360 x 2 x π x {int(r)} = {number_text(l_, 1)} {unit}."],
            [
                (a_, "r4g-area-formula-for-arc"),
                (2 * PI * r, "r4g-whole-circumference"),
                (Fraction(theta, 360) * PI * r, "r4g-radius-for-diameter-in-circumference"),
                (Fraction(theta) * r / 180, "r4g-angle-used-as-radians"),
            ],
        ),
        number_part(
            "Use your arc length from (b) to find the perimeter of the sector.",
            peri,
            1,
            1,
            [f"Perimeter = {number_text(l_, 1)} + 2 x {int(r)} = {number_text(peri, 1)} {unit}."],
            [
                (l_, "r4g-radii-left-out-of-perimeter"),
                (l_ + r, "r4g-one-radius-in-perimeter"),
                (l_ + 4 * r, "r4g-diameter-added-twice"),
                (2 * PI * r + 2 * r, "r4g-whole-circumference"),
            ],
            follow_through=f"part_b + {2 * int(r)}",
        ),
        number_part(
            f"Use your sector area from (a) to find the area {rest}.",
            remaining,
            1,
            2,
            [
                f"Whole circle = π x {int(r)}² = {number_text(whole, 1)} {unit}²; minus the sector gives {number_text(remaining, 1)} {unit}²."
            ],
            [
                (whole + a_, "r4g-sectors-added-not-subtracted"),
                (whole, "r4g-whole-circle-area"),
                (a_, "r4g-sector-given-for-the-rest"),
                (PI * (r * r - a_), "r4g-area-subtracted-before-multiplying-by-pi"),
            ],
            follow_through=f"pi*{int(r) ** 2} - part_a",
        ),
    ]
    return dict(text=text, parts=parts)


@generator("exd_x_r4g_sector_degrees", level=11, presets={"exam": {}})
def exd_x_r4g_sector_degrees(rng):
    return retry(_build, rng)
