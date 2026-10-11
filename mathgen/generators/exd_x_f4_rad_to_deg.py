"""Exam-tier questions for converting an angle in radians to degrees (decimal answer).

The practice tiers give the angle in radians and ask only for the conversion.
Here the radian value has to be found or reduced first. Three settings, drawn
with equal chance:

- ``triangle``: two angles of a triangle in radians; find the third in degrees.
- ``arc``: an arc length and a radius; find the angle at the centre in degrees.
- ``spin``: an angular speed and a time; find the angle, in degrees, of the
  final position within one turn.

Each key is computed from exact rationals and the exact rational value of the
double-precision pi, rounded half-up to one decimal place.
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
_DP = 1

_TRIANGLE = (
    "Two angles of a triangle measure {a} radians and {b} radians. Find the third angle in degrees, correct to 1 decimal place.",
    "A triangle has angles of {a} rad and {b} rad. Work out the size of its third angle in degrees, to 1 decimal place.",
    "The angles at two vertices of a triangle are {a} radians and {b} radians. What is the angle at the third vertex, in degrees correct to 1 decimal place?",
)
_ARC = (
    "An arc of length {arc} cm is cut from a circle of radius {r} cm. Find the angle the arc subtends at the centre, in degrees, correct to 1 decimal place.",
    "On a circle of radius {r} cm, an arc is {arc} cm long. Give the angle at the centre in degrees, to 1 decimal place.",
    "A circle has radius {r} cm and an arc of it has length {arc} cm. Calculate the angle at the centre in degrees, correct to 1 decimal place.",
)
_SPIN = (
    "A wheel turns at {w} radians per second for {t} seconds from a marked starting position. Find the angle, in degrees, of the mark from its start after the last complete turn is removed. Give the answer between 0 and 360, correct to 1 decimal place.",
    "A turntable rotates at {w} rad/s for {t} seconds. Ignoring whole turns, how many degrees past its starting position does it finish? Give the answer between 0 and 360, correct to 1 decimal place.",
)


def _deg(radians: Fraction) -> Fraction:
    return radians * 180 / PI


def _show(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, _DP), _DP)


def _dec2(value: int) -> str:
    return f"{value // 100}.{value % 100:02d}"


def _triangle(rng):
    a = rng.randint(40, 120)
    b = rng.randint(40, 120)
    third = PI - Fraction(a + b, 100)
    check(third > Fraction(1, 10))
    cands = [
        _deg(Fraction(a + b, 100)),  # the sum of the two given angles, not the third
        180 - Fraction(a + b, 100),  # subtracts radians from 180 degrees
        third,  # the radian value reported as if it were degrees
        _deg(Fraction(max(a, b), 100)),
    ]
    return phrase(rng, _TRIANGLE, a=_dec2(a), b=_dec2(b)), _deg(third), cands, "Third angle = pi - a - b radians."


def _arc(rng):
    r = rng.randint(4, 15)
    arc = Fraction(rng.randint(30, 200), 10)
    theta = arc / r
    check(0 < theta < 2 * PI)
    cands = [
        theta,  # the radian value reported as if it were degrees
        arc * r * 180 / PI,  # multiplies where it should divide
        arc * 180 / PI,  # forgets to divide by the radius
        _deg(theta) / 2,
    ]
    return phrase(rng, _ARC, r=r, arc=str(float(arc))), _deg(theta), cands, "Angle = arc / radius radians."


def _spin(rng):
    for _ in range(100):
        w = Fraction(rng.randint(15, 95), 10)
        t = rng.randint(3, 9)
        total = _deg(w * t)
        turns = math.floor(total / 360)
        reduced = total - 360 * turns
        if turns >= 1 and reduced > 0:
            break
    else:
        check(False, "exd_x_f4_rad_to_deg: no spin draw with a whole turn to remove")
    cands = [
        total,  # whole turns left in
        _deg(w) * t - 180 * turns,  # removes half-turns instead of whole turns
        w * t,  # the radian value reported as if it were degrees
        360 - reduced,  # measured the wrong way round
    ]
    return phrase(rng, _SPIN, w=str(float(w)), t=t), reduced, cands, "Total angle = w x t radians; remove whole turns."


_SETTINGS = (_triangle, _arc, _spin)


@generator("exd_x_f4_rad_to_deg", level=11, presets={"exam": dict(tier="exam")})
def exd_x_f4_rad_to_deg(rng, tier: str = "exam"):
    for _ in range(300):
        text, exact, cands, step = rng.choice(_SETTINGS)(rng)
        if not rounds_unambiguously(float(exact), _DP, margin=0.05):
            continue
        key = round_half_up(exact, _DP)
        answer_text = stated_precision_string(key, _DP)
        shown = [_show(c) for c in cands]
        distractors = pick_distractors(answer_text, shown, tolerance=stated_precision_tolerance(_DP) * 2)
        if len(distractors) < 3:
            continue
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(key),
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[step, f"Convert: radians x 180 / pi = {answer_text} degrees (rounded once, at the end)."],
            distractors=distractors,
        )
    check(False, "exd_x_f4_rad_to_deg: no unambiguous draw in 300 attempts")
