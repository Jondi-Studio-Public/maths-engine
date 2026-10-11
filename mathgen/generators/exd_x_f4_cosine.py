"""Exam-tier cosine-rule questions.

In the practice tiers the triangle and its included angle are stated outright.
Here the learner is given a navigation or shape setting and must first work out
the included angle (and see that the cosine rule applies):

- ``bearings``: two points are given as a distance and a bearing from the same
  origin; find how far apart they are (the angle is a difference of bearings).
- ``turn``: a path of two straight legs with a change of course at the corner;
  find the distance from the start (the included angle is 180 degrees minus the
  turn).
- ``diagonal``: a parallelogram with two side lengths and one angle; find the
  longer or the shorter diagonal (the learner chooses between the angle and its
  supplement).

Answers are lengths to one decimal place.
"""

from __future__ import annotations

import math

from ..core import (
    check,
    generator,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)
from fractions import Fraction

_DP = 1

_BEARINGS = (
    "From a lighthouse, ship A is {r1} km away on a bearing of {b1:03d} degrees and ship B is {r2} km away on a bearing of {b2:03d} degrees. Find the distance between the two ships, in km, correct to 1 decimal place.",
    "A walker at a camp sees a peak {r1} km away on a bearing of {b1:03d} degrees and a lake {r2} km away on a bearing of {b2:03d} degrees. How far is the peak from the lake? Give the answer in km, to 1 decimal place.",
)
_TURN = (
    "A boat sails {r1} km on a course of {b1:03d} degrees, then changes course to {b2:03d} degrees and sails a further {r2} km. How far is it from its starting point, in km, correct to 1 decimal place?",
    "A cyclist rides {r1} km on a bearing of {b1:03d} degrees, then turns onto a bearing of {b2:03d} degrees and rides {r2} km more. Find the straight-line distance back to the start, in km, to 1 decimal place.",
)
_DIAGONAL = (
    "A parallelogram has sides of {a} cm and {b} cm, and one of its angles is {t} degrees. Find the length of its {which} diagonal, in cm, correct to 1 decimal place.",
    "Two adjacent sides of a parallelogram are {a} cm and {b} cm long and the angle between them is {t} degrees. Calculate the {which} diagonal, in cm, to 1 decimal place.",
)


def _c(a: float, b: float, deg: float, sign: int = -1) -> float:
    """sqrt(a^2 + b^2 + sign * 2ab cos(deg))."""
    return math.sqrt(a * a + b * b + sign * 2 * a * b * math.cos(math.radians(deg)))


def _bearings(rng):
    r1, r2 = rng.randint(5, 30), rng.randint(5, 30)
    b1 = rng.randint(0, 359)
    inc = rng.choice([x for x in range(25, 156) if x != 90])
    b2 = (b1 + inc * rng.choice([1, -1])) % 360
    diff = abs(b1 - b2)
    angle = diff if diff <= 180 else 360 - diff
    check(angle == inc)
    cands = [
        _c(r1, r2, angle, +1),  # sign error on the cosine term
        _c(r1, r2, b2),  # one bearing used as the included angle
        _c(r1, r2, 180 - angle),  # the supplement of the angle
        math.sqrt(r1 * r1 + r2 * r2),  # treats the angle as a right angle
    ]
    return phrase(rng, _BEARINGS, r1=r1, r2=r2, b1=b1, b2=b2), _c(r1, r2, angle), cands


def _turn(rng):
    r1, r2 = rng.randint(5, 30), rng.randint(5, 30)
    b1 = rng.randint(0, 359)
    turn = rng.choice([x for x in range(20, 151) if x != 90])
    b2 = (b1 + turn * rng.choice([1, -1])) % 360
    angle = 180 - turn
    cands = [
        _c(r1, r2, turn),  # the turn taken as the included angle
        _c(r1, r2, angle, +1),
        r1 + r2,  # follows the path rather than measuring straight back
        math.sqrt(r1 * r1 + r2 * r2),
    ]
    return phrase(rng, _TURN, r1=r1, r2=r2, b1=b1, b2=b2), _c(r1, r2, angle), cands


def _diagonal(rng):
    a, b = rng.randint(6, 25), rng.randint(6, 25)
    t = rng.choice([x for x in range(30, 151) if x != 90])
    which = rng.choice(["longer", "shorter"])
    # the diagonal opposite the angle t uses t; the other uses 180 - t
    opposite = _c(a, b, t)
    other = _c(a, b, 180 - t)
    key = max(opposite, other) if which == "longer" else min(opposite, other)
    wrong = min(opposite, other) if which == "longer" else max(opposite, other)
    cands = [
        wrong,  # the other diagonal
        math.sqrt(a * a + b * b - 2 * a * b * math.cos(t)),  # calculator left in radian mode
        a + b,  # adds the sides
        math.sqrt(a * a + b * b),
    ]
    return phrase(rng, _DIAGONAL, a=a, b=b, t=t, which=which), key, cands


_SETTINGS = (_bearings, _turn, _diagonal)
_WORKING = {
    _bearings: "The included angle is the difference of the two bearings (use 360 minus it if it exceeds 180); then the cosine rule.",
    _turn: "The included angle at the corner is 180 degrees minus the change of course; then the cosine rule.",
    _diagonal: "The diagonal opposite the given angle uses that angle; the other diagonal uses 180 degrees minus it. The larger included angle gives the longer diagonal.",
}


@generator("exd_x_f4_cosine", level=11, presets={"exam": dict(tier="exam")})
def exd_x_f4_cosine(rng, tier: str = "exam"):
    for _ in range(300):
        setting = rng.choice(_SETTINGS)
        text, exact, cands = setting(rng)
        if not rounds_unambiguously(exact, _DP, margin=0.1):
            continue
        key = round_half_up(Fraction(exact), _DP)
        answer_text = stated_precision_string(key, _DP)
        shown = [stated_precision_string(round_half_up(Fraction(c), _DP), _DP) for c in cands]
        distractors = pick_distractors(answer_text, shown, tolerance=stated_precision_tolerance(_DP) * 2)
        if len(distractors) < 3:
            continue
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(key),
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[_WORKING[setting], f"Distance = {answer_text} (rounded once, at the end)."],
            distractors=distractors,
        )
    check(False, "exd_x_f4_cosine: no unambiguous draw in 300 attempts")
