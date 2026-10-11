"""Generates `exd_x_bearings_cosine` exam questions: the distance between start and finish of a two-leg
journey whose legs are NOT at right angles.

The learner reads two compass bearings, works out the angle at the turning point (it is 120 or 60 degrees,
never 90), and applies the cosine rule. The side lengths come from integer triangles with a 60 or 120 degree
angle, so the straight-line distance is a whole number of kilometres.

Built backwards from a triangle (d1, d2, d3) with d3^2 = d1^2 + d2^2 + d1*d2 (turn of 60 degrees at B, so the
interior angle is 120) or d3^2 = d1^2 + d2^2 - d1*d2 (turn of 120 degrees, interior angle 60), scaled by a
small whole number. The bearings are drawn afterwards; the turn can be to the left or the right.
"""

from __future__ import annotations

import math
import random

from ..core import check, check_distinct, generator, pick_distractors

# (d1, d2, d3) with d3^2 = d1^2 + d2^2 + d1*d2: interior angle at the turning point is 120 degrees.
_OBTUSE = [(3, 5, 7), (5, 3, 7), (7, 8, 13), (8, 7, 13), (5, 16, 19), (16, 5, 19)]
# (d1, d2, d3) with d3^2 = d1^2 + d2^2 - d1*d2: interior angle is 60 degrees.
_ACUTE = [(5, 8, 7), (8, 5, 7), (3, 8, 7), (8, 3, 7), (7, 15, 13), (15, 7, 13), (8, 15, 13), (15, 8, 13)]
_TRAVELLERS = ["A yacht sails", "A hiker walks", "A ship steams", "A drone flies"]


def _bearing_text(b: int) -> str:
    return f"{b:03d}°"


def _build(rng: random.Random) -> dict:
    obtuse = rng.random() < 0.5
    d1, d2, d3 = rng.choice(_OBTUSE if obtuse else _ACUTE)
    k = rng.randint(1, 4)
    d1, d2, d3 = d1 * k, d2 * k, d3 * k
    sign = rng.choice([-1, 1])
    turn = 60 if obtuse else 120
    b1 = rng.randrange(0, 360, 5)
    b2 = (b1 + sign * turn) % 360
    interior = 180 - turn

    # Exact integer form of the cosine rule: cos(120) = -1/2 and cos(60) = 1/2.
    check(d3 * d3 == d1 * d1 + d2 * d2 + (d1 * d2 if obtuse else -d1 * d2), "triangle mismatch")
    # Second route: coordinates, east and north components of each leg.
    bx = d1 * math.sin(math.radians(b1))
    by = d1 * math.cos(math.radians(b1))
    cx = bx + d2 * math.sin(math.radians(b2))
    cy = by + d2 * math.cos(math.radians(b2))
    check(abs(math.hypot(cx, cy) - d3) < 1e-9, "coordinate route disagrees")

    wrong_angle = math.sqrt(d1 * d1 + d2 * d2 - 2 * d1 * d2 * math.cos(math.radians(turn)))
    candidates = {
        "applies-pythagoras": round(math.hypot(d1, d2)),
        "adds-the-legs": d1 + d2,
        "uses-the-turn-as-the-interior-angle": round(wrong_angle),
        "subtracts-the-legs": abs(d1 - d2),
    }
    check_distinct(d3, *candidates.values())

    who = rng.choice(_TRAVELLERS)
    text = (
        f"{who} {d1} km on a bearing of {_bearing_text(b1)} from A to B, then {d2} km on a bearing of "
        f"{_bearing_text(b2)} from B to C. How far is C from A, in whole kilometres?"
    )
    answer_text = str(d3)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=d3,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The path turns through {turn} degrees at B, so angle ABC = 180 - {turn} = {interior} degrees.",
            f"AC^2 = {d1}^2 + {d2}^2 - 2 x {d1} x {d2} x cos({interior}) = {d3 * d3}, so AC = {d3} km.",
        ],
        distractors=pick_distractors(answer_text, [str(v) for v in candidates.values()]),
    )


@generator(
    "exd_x_bearings_cosine",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_bearings_cosine(rng: random.Random, tier: str = "exam") -> dict:
    """Straight-line distance over a non-right-angled two-leg bearing journey. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
