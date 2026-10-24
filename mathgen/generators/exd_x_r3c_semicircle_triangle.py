"""Exam source for `senior-an-angle-in-a-semicircle-is-a-right`: from a diameter and a chord to a height.

The stem gives a diameter AB (or the radius) and the chord AC, with C on the circle. The angle in a semicircle is a
right angle, so triangle ABC is right-angled at C, and:

(a) the chord BC, by Pythagoras (a whole number: the sides are an integer right triangle);
(b) the area of triangle ABC, using (a);
(c) the shortest distance from C to AB, which is twice the area over AB, using (b), correct to 1 d.p.

Parts (b) and (c) carry `follow_through`. The key is re-derived with coordinates: A = (-r, 0), B = (r, 0), and C
found from the two chord lengths; C is checked to lie on the circle.
"""

from __future__ import annotations

import math
from functools import cache
from fractions import Fraction

from mathgen.core import attempts, check, generator

from .exd_x_r3c_parts import number_part
from ._exd_distinct_parts import distinct_part_answers


#: Every integer right triangle (leg, leg, hypotenuse) with a hypotenuse from 10 to 150.
@cache
def _triples():
    return tuple((a, b, c) for c in range(10, 151) for a in range(3, c) for b in range(a, c) if a * a + b * b == c * c)


_CM_STEMS = (
    "Triangle ABC is drawn inside a circle so that AB is a diameter of length {d} cm and C lies on the circle. "
    "The chord AC is {a} cm long.",
    "A circle has centre O and radius {r} cm. The points A, B and C lie on the circle, with AOB a straight line. "
    "The chord AC is {a} cm long.",
    "AB is a diameter of a circle and C is a point on the circle, with AB = {d} cm and AC = {a} cm.",
)
_M_STEMS = (
    "A circular pond has a diameter AB of {d} m, and a post C stands on its edge, {a} m from A.",
    "A circular skating rink is {d} m across at its widest, from A to B. A light C is fixed on the edge of the "
    "rink, {a} m from A.",
)


@generator("exd_x_r3c_semicircle_triangle", level=13, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_semicircle_triangle(rng, tier="exam"):
    """Diameter and chord to the other chord, the triangle's area and the height from C."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a triangle whose height is clear of a rounding boundary"):
        x, y, d = rng.choice(_triples())
        a, b = (x, y) if rng.random() < 0.5 else (y, x)
        height = Fraction(a * b, d)
        if abs((height * 10) % 1 - Fraction(1, 2)) > Fraction(1, 100):
            break
    area = Fraction(a * b, 2)
    check(area.denominator == 1, "the area is not a whole number")
    # Independent route: coordinates with the diameter on the x-axis.
    r = d / 2
    cx = (a * a - b * b) / (2 * d)  # from AC^2 - BC^2 = 2 d cx
    cy = math.sqrt(r * r - cx * cx)
    check(abs(math.hypot(cx + r, cy) - a) < 1e-9 and abs(math.hypot(cx - r, cy) - b) < 1e-9, "C is off the circle")
    check(abs(cy - float(height)) < 1e-9, "the height disagrees with the coordinates")

    metres = d >= 100 or rng.random() < 0.3
    if metres:
        stem = rng.choice(_M_STEMS).format(d=d, a=a)
        unit, sq = "m", "m²"
    else:
        options = [s for s in _CM_STEMS if "{r}" not in s or d % 2 == 0]
        stem = rng.choice(options).format(d=d, a=a, r=d // 2)
        unit, sq = "cm", "cm²"
    part_a = number_part(
        f"Find the length of BC, in {unit}.",
        b,
        marks=2,
        sensible_range=(0, None),
        working=[f"The angle in a semicircle is 90°, so BC² = {d}² - {a}² = {d * d - a * a} and BC = {b}."],
        wrongs=[
            (d - a, "subtracted-sides-not-pythagoras"),
            (d * d - a * a, "pythagoras-without-square-root"),
            (Fraction(d, 2), "used-radius-as-chord"),
        ],
    )
    part_b = number_part(
        f"Using your answer to (a), find the area of triangle ABC, in {sq}.",
        area,
        sensible_range=(0, None),
        working=[f"The right angle is at C, so the area is ½ × {a} × {b} = {area} {sq}."],
        wrongs=[
            (a * b, "forgot-half-in-triangle-area"),
            (Fraction(a * d, 2), "used-hypotenuse-as-leg"),
            (Fraction(b * d, 2), "used-hypotenuse-as-leg"),
        ],
        follow_through=f"{a}*part_a/2",
    )
    part_c = number_part(
        f"Using your answer to (b), find the shortest distance from C to the diameter AB, in {unit}, correct to 1 decimal place.",
        height,
        dp=1,
        marks=2,
        sensible_range=(0, None),
        working=[f"Area = ½ × AB × height, so height = 2 × {area} ÷ {d} = {float(height):.1f} {unit} (1 d.p.)."],
        wrongs=[
            (area / d, "forgot-doubling-in-altitude"),
            (2 * area / a, "used-leg-as-base"),
            (2 * area / b, "used-leg-as-base"),
            (Fraction(a * b, 2 * d), "forgot-doubling-in-altitude"),
        ],
        follow_through=f"2*part_b/{d}",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])
