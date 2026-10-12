"""Exam-tier source for a primary angle skill: a connected problem in an everyday setting.

``exd_x_pr2_angle_point_unknown`` (``f10-angle-relationships``): four angles meet at a point. Part (a)
finds the fourth angle from the other three, part (b) uses it to solve for x when that angle is split
into two algebraic parts, and part (c) gives the size of one of the two parts.

Built backwards from x, so each key is exact; parts (b) and (c) carry ``follow_through`` so a slip
early on does not cost the later marks.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)
from .exd_x_pr2_time_chains import _int_part, _key_not_in_prompt, _options, _whole


def _whole_only(slips):
    """Keep the slips whose value is a whole number: an angle or a value of x shown in these options is whole."""
    return [(v, tag) for v, tag in slips if Fraction(v).denominator == 1]


# ---------------------------------------------------------------------------------------------------
# f10-angle-relationships: angles at a point, then an unknown, then one angle
# ---------------------------------------------------------------------------------------------------

_POINT_STEMS = [
    "Four angles meet at a point and together make a full turn. Three of them measure {A}°, {B}° and {C}°. The "
    "fourth angle is split by a straight line from the point into two angles of ({e1})° and ({e2})°.",
    "A round pizza is cut from its centre into four slices. Three slices have angles of {A}°, {B}° and {C}° at the "
    "centre. The fourth slice is cut into two thinner slices with angles of ({e1})° and ({e2})°.",
    "A garden sprinkler waters four sectors around it, which together make a full circle. Three sectors have angles "
    "of {A}°, {B}° and {C}°. The fourth sector is divided by a path into two parts with angles of ({e1})° and ({e2})°.",
]


def _expr(coeff: int, const: int) -> str:
    return f"{'' if coeff == 1 else coeff}x + {const}"


@generator("exd_x_pr2_angle_point_unknown", level=7, presets={"exam": dict(tier="exam")})
@_key_not_in_prompt
def exd_x_pr2_angle_point_unknown(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the fourth angle at a point, then x, then the size of one of its two parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "four angles at a point that split into two algebraic parts"):
        p, q = rng.randint(1, 4), rng.randint(1, 4)
        k, m = rng.randrange(5, 26, 5), rng.randrange(5, 26, 5)
        x = rng.randint(4, 30)
        fourth = (p + q) * x + k + m
        if not 50 <= fourth <= 170 or (p, k) == (q, m):
            continue
        a_angle, b_angle = rng.randrange(45, 141, 5), rng.randrange(45, 141, 5)
        c_angle = 360 - fourth - a_angle - b_angle
        if not (40 <= c_angle <= 150 and len({a_angle, b_angle, c_angle}) == 3):
            continue
        b_slips = _whole_only(
            [
                (fourth - k - m, "forgot-to-divide-by-coefficient"),
                (Fraction(fourth, p + q), "ignored-constant-term"),
                (Fraction(fourth + k + m, p + q), "sign-error-moving-term"),
                (Fraction(fourth - k - m, p), "used-one-coefficient"),
            ]
        )
        if len({v for v, _ in b_slips}) >= 2:
            break
    first = p * x + k  # the part asked for in (c)
    other = q * x + m
    check(a_angle + b_angle + c_angle + first + other == 360, "angles do not make a full turn")
    check(first + other == fourth and (fourth - k - m) % (p + q) == 0, "split of the fourth angle is inconsistent")

    e1, e2 = _expr(p, k), _expr(q, m)
    stem = rng.choice(_POINT_STEMS).format(A=a_angle, B=b_angle, C=c_angle, e1=e1, e2=e2)
    drop = [360 - (a_angle + b_angle), 360 - (a_angle + c_angle), 360 - (b_angle + c_angle)]
    a_slips = [(360 - fourth, "gave-sum-not-remainder")] + [(v, "drops-one-angle") for v in drop]
    c_slips = [
        (x, "gave-x-not-angle"),
        (other, "read-wrong-angle"),
        (p * x, "dropped-constant-term"),
        (fourth, "gave-whole-not-part"),
    ]
    return dict(
        text=stem,
        latex=f"plain:{stem}",
        parts=[
            _int_part(
                "What is the size of the fourth angle, in degrees?",
                fourth,
                marks=2,
                distractors=_options(Fraction(fourth), a_slips, _whole),
                working=[f"{a_angle} + {b_angle} + {c_angle} = {360 - fourth}; 360 - {360 - fourth} = {fourth}."],
            ),
            _int_part(
                "Using your answer to (a), find the value of x.",
                x,
                marks=2,
                follow_through=f"(part_a - {k + m})/{p + q}",
                distractors=_options(Fraction(x), b_slips, _whole),
                working=[f"({e1}) + ({e2}) = {fourth}, so {p + q}x = {fourth - k - m} and x = {x}."],
            ),
            _int_part(
                f"Using your answer to (b), find the size of the angle ({e1})°, in degrees.",
                first,
                marks=1,
                follow_through=f"{p}*part_b + {k}",
                distractors=_options(Fraction(first), _whole_only(c_slips), _whole),
                working=[f"{p} × {x} + {k} = {first} degrees."],
            ),
        ],
    )
