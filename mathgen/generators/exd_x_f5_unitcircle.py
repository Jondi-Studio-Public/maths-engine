"""Exam tier for the unit-circle definition of tan: only one coordinate is given.

The practice tiers show the whole terminal point. Here one coordinate (or one of sin / cos) is given together with
the quadrant (or the sign of the other function), so the learner must recover the missing coordinate from
x^2 + y^2 = 1, choose its sign from the quadrant, and only then form tan = sin / cos.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, pick_distractors

_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (12, 35, 37), (9, 40, 41))
_QUADRANT_SIGNS = {1: (1, 1), 2: (-1, 1), 3: (-1, -1), 4: (1, -1)}
_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}


def _frac(f: Fraction) -> str:
    return str(f)


@generator("exd_x_f5_unitcircle", level=14, presets={"exam": dict(tier="exam")})
def exd_x_f5_unitcircle(rng: random.Random, tier: str = "exam") -> dict:
    """tan(theta) from one given coordinate / ratio plus the quadrant information."""
    if tier != "exam":
        raise ValueError(f"unknown tier: {tier}")
    leg1, leg2, hyp = rng.choice(_TRIPLES)
    legs = (leg1, leg2) if rng.random() < 0.5 else (leg2, leg1)
    quadrant = rng.choice([1, 2, 3, 4])
    sx, sy = _QUADRANT_SIGNS[quadrant]
    cos_v = Fraction(sx * legs[0], hyp)
    sin_v = Fraction(sy * legs[1], hyp)
    check(cos_v**2 + sin_v**2 == 1, "point is not on the unit circle")
    tan_v = sin_v / cos_v

    given_is = rng.choice(["x", "y"])
    given = cos_v if given_is == "x" else sin_v
    shape = rng.choice(["point", "ratio"])
    if shape == "point":
        coords = f"({_frac(given)}, y)" if given_is == "x" else f"(x, {_frac(given)})"
        text = (
            f"The terminal side of angle theta meets the unit circle at the point P{coords}, "
            f"and P lies in quadrant {_ROMAN[quadrant]}. Find tan(theta)."
        )
    else:
        if given_is == "x":
            text = f"Given that cos(theta) = {_frac(cos_v)} and sin(theta) {'>' if sy > 0 else '<'} 0, find tan(theta)."
        else:
            text = f"Given that sin(theta) = {_frac(sin_v)} and cos(theta) {'>' if sx > 0 else '<'} 0, find tan(theta)."
    answer_text = _frac(tan_v)
    # Wrong value for the missing coordinate: 1 - given^2, i.e. its square (the root is never taken).
    missing_wrong = 1 - given**2
    wrong_missing = (missing_wrong / given) if given_is == "x" else (given / missing_wrong)
    candidates = [
        _frac(1 / tan_v),  # ratio inverted (cos / sin)
        _frac(-tan_v),  # sign of the missing coordinate taken from the wrong quadrant
        _frac(wrong_missing),  # uses the square of the missing coordinate
        _frac(given),  # reports the given coordinate itself
    ]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "fewer than 3 distractors")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"cos(theta) = {_frac(cos_v)} and sin(theta) = {_frac(sin_v)}, using x^2 + y^2 = 1 and the quadrant sign.",
            f"tan(theta) = sin(theta) / cos(theta) = {answer_text}.",
        ],
        distractors=distractors,
    )
