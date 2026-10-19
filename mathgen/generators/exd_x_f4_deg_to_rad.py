"""Exam-tier questions for converting an angle in degrees to radians (decimal answer).

The practice tiers give the angle in degrees and ask only for the conversion.
Here the angle has to be found first, from a geometric setting, and only then
converted. Three settings, drawn with equal chance:

- ``polygon``: the interior angle of a regular n-sided polygon.
- ``clock``: the smaller angle between the hands of an analogue clock.
- ``exterior``: the exterior angle of a triangle at the third vertex, from the
  two other interior angles.

The key is computed once from an exact rational angle in degrees and the exact
rational value of the double-precision pi, rounded half-up to three decimals.
"""

from __future__ import annotations

import math
from fractions import Fraction

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

PI = Fraction(math.pi)
_DP = 3

_POLYGON = (
    "Find the size of each interior angle of a regular polygon with {n} sides, in radians, correct to 3 decimal places.",
    "A regular {n}-sided polygon is drawn. Give the size of one interior angle in radians, to 3 decimal places.",
    "Each interior angle of a regular polygon with {n} sides is measured in radians. Find it, correct to 3 decimal places.",
)
_CLOCK = (
    "At {h}:{m:02d} the hands of an analogue clock make two angles with each other. Find the smaller of the two in radians, to 3 decimal places.",
    "What is the smaller angle between the hour hand and the minute hand of a clock showing {h}:{m:02d}? Give your answer in radians, correct to 3 decimal places.",
    "A clock shows {h}:{m:02d}. Work out, in radians correct to 3 decimal places, the smaller angle between its two hands.",
)
_EXTERIOR = (
    "Two interior angles of a triangle are {a} degrees and {b} degrees. Find the exterior angle at the third vertex in radians, correct to 3 decimal places.",
    "In a triangle, two of the interior angles measure {a} degrees and {b} degrees. Give the exterior angle at the remaining vertex in radians, to 3 decimal places.",
    "A triangle has interior angles of {a} degrees and {b} degrees at two of its vertices. Find, in radians to 3 decimal places, the exterior angle at its third vertex.",
)


def _rad(degrees: Fraction) -> Fraction:
    return degrees * PI / 180


def _show(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, _DP), _DP)


def _polygon(rng):
    n = rng.randint(7, 24)
    interior = Fraction(180 * (n - 2), n)
    cands = [
        _rad(Fraction(360, n)),  # the exterior angle
        PI * (n - 2),  # the whole angle sum, not divided among the vertices
        interior,  # the degree value left unconverted
        _rad(Fraction(180 * (n - 1), n)),  # (n - 1) in place of (n - 2)
    ]
    return phrase(rng, _POLYGON, n=n), interior, cands, f"Interior angle = 180 × ({n} - 2) / {n} degrees."


def _clock(rng):
    h = rng.randint(1, 11)
    m = rng.choice([5, 10, 15, 20, 25, 35, 40, 45, 50, 55])
    hour_hand = Fraction(30 * h) + Fraction(m, 2)
    minute_hand = Fraction(6 * m)
    gap = abs(hour_hand - minute_hand)
    smaller = min(gap, 360 - gap)
    check(0 < smaller < 180)
    cands = [
        _rad(360 - smaller) if smaller != 180 else _rad(gap),  # the larger angle
        _rad(abs(Fraction(30 * h) - minute_hand)),  # the hour hand taken as fixed at the hour mark
        _rad(Fraction(30 * h) + minute_hand),
        smaller,  # degrees left unconverted
    ]
    return phrase(rng, _CLOCK, h=h, m=m), smaller, cands, "Hour hand: 30h + m/2 degrees; minute hand: 6m degrees."


def _exterior(rng):
    a = rng.randint(28, 85)
    b = rng.randint(28, 85)
    third = 180 - a - b
    check(third >= 10)
    ext = Fraction(a + b)
    cands = [
        _rad(Fraction(third)),  # the interior angle at that vertex
        Fraction(a + b),  # degrees left unconverted
        _rad(Fraction(max(a, b))),  # takes the larger given angle
        _rad(Fraction(360 - a - b)),
    ]
    return phrase(rng, _EXTERIOR, a=a, b=b), ext, cands, f"Exterior angle = {a} + {b} degrees."


_SETTINGS = (_polygon, _clock, _exterior)


@generator("exd_x_f4_deg_to_rad", level=11, presets={"exam": dict(tier="exam")})
def exd_x_f4_deg_to_rad(rng, tier: str = "exam"):
    for _ in range(200):
        text, degrees, cands, step = rng.choice(_SETTINGS)(rng)
        exact = _rad(degrees)
        if not rounds_unambiguously(float(exact), _DP, margin=0.05):
            continue
        key = round_half_up(exact, _DP)
        # Every distractor is shown at the same stated precision; those closer to the key than the
        # grader's own tolerance are dropped, never offered.
        shown = [_show(c) for c in cands]
        distractors = pick_distractors(
            stated_precision_string(key, _DP), shown, tolerance=stated_precision_tolerance(_DP) * 2
        )
        if len(distractors) < 3:
            continue
        answer_text = stated_precision_string(key, _DP)
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(key),
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[step, f"Convert: degrees × π / 180 = {answer_text} rad (rounded once, at the end)."],
            distractors=distractors,
        )
    check(False, "exd_x_f4_deg_to_rad: no unambiguous draw in 200 attempts")
