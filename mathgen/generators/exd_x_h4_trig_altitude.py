"""Exam-tier source for "apply the cosine and sine ratios to find lengths".

The practice tiers apply one or two ratios inside a single right-angled
triangle. An exam question has the learner create a second right-angled
triangle for themselves: in triangle ABC (right angle at B) the perpendicular
BD is dropped to the hypotenuse AC, and the asked length is one of the three
segments AD, DC or BD. Each is a product of two trigonometric ratios, found by
applying sine or cosine in one triangle and then again in the smaller triangle
it creates:

    AD = AC cos^2(A),   DC = AC sin^2(A),   BD = AC sin(A) cos(A).

The hypotenuse AC and the angle A are drawn first; the key is computed from
them and rounded half-up to one decimal place. Draws whose key lies within 0.01
of a rounding boundary, or whose distractors fall within 0.2 of the key, are
redrawn so a typed answer has one correct reading at the stated precision.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from sympy import Integer, cos, pi, sin

from ..core import (
    check,
    generator,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_ANGLES = (20, 25, 30, 35, 40, 50, 55, 60, 65, 70)
_ASKED = ("AD", "DC", "BD")


def _values(h: int, alpha: int) -> dict[str, float]:
    a = math.radians(alpha)
    return {
        "AB": h * math.cos(a),
        "BC": h * math.sin(a),
        "AD": h * math.cos(a) ** 2,
        "DC": h * math.sin(a) ** 2,
        "BD": h * math.sin(a) * math.cos(a),
    }


def _margin(x: float) -> float:
    scaled = x * 10
    return abs(scaled - math.floor(scaled) - 0.5) / 10


@generator(
    "exd_x_h4_trig_altitude",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_h4_trig_altitude(rng: random.Random, tier: str = "exam") -> dict:
    """Altitude to the hypotenuse of a right-angled triangle, found by trigonometry."""
    check(tier == "exam", f"unknown tier {tier!r}")
    tol = float(stated_precision_tolerance(1))
    for _ in range(500):
        h = rng.randint(12, 60)
        alpha = rng.choice(_ANGLES)
        asked = rng.choice(_ASKED)
        v = _values(h, alpha)
        key_exact = v[asked]
        others = {
            "AD": ("AB", "BD", "DC"),
            "DC": ("BC", "BD", "AD"),
            "BD": ("AB", "BC", "AD"),
        }[asked]
        if _margin(key_exact) < 0.01:
            continue
        if any(abs(v[o] - key_exact) < 4 * tol + 0.05 for o in others):
            continue
        break
    else:  # pragma: no cover
        raise AssertionError("no clean draw")

    # Independent route: exact sympy evaluation of the same length.
    rad = Integer(alpha) * pi / 180
    c, s = cos(rad), sin(rad)
    exact = {"AD": h * c * c, "DC": h * s * s, "BD": h * s * c}[asked]
    check(abs(float(exact.evalf(30)) - key_exact) < 1e-9, "trig_altitude: sympy value disagrees")

    key = round_half_up(Fraction(key_exact).limit_denominator(10**9), 1)
    answer_text = stated_precision_string(key, 1)
    answer = numeric_answer(key)
    cands = [stated_precision_string(round_half_up(Fraction(v[o]).limit_denominator(10**9), 1), 1) for o in others]
    distractors = pick_distractors(answer_text, cands, tolerance=stated_precision_tolerance(1))
    check(len(distractors) >= 2, "too few distractors")

    text = phrase(
        rng,
        (
            "In triangle ABC the angle at B is a right angle, AC = {h} cm and angle BAC = {a} degrees. "
            "D is the point on AC such that BD is perpendicular to AC. Find the length of {asked}, in cm, "
            "correct to 1 decimal place.",
            "Triangle ABC has a right angle at B, with hypotenuse AC = {h} cm and angle BAC = {a} degrees. "
            "The perpendicular from B meets AC at D. Calculate {asked} in cm, to 1 decimal place.",
        ),
        h=h,
        a=alpha,
        asked=asked,
    )
    working = {
        "AD": [
            f"AB = {h} cos({alpha}) in right-angled triangle ABC.",
            f"In right-angled triangle ABD, AD = AB cos({alpha}) = {h} cos^2({alpha}).",
        ],
        "DC": [
            f"BC = {h} sin({alpha}) in right-angled triangle ABC.",
            f"Angle DBC = {alpha} degrees, so DC = BC sin({alpha}) = {h} sin^2({alpha}).",
        ],
        "BD": [
            f"AB = {h} cos({alpha}) in right-angled triangle ABC.",
            f"In right-angled triangle ABD, BD = AB sin({alpha}) = {h} sin({alpha}) cos({alpha}).",
        ],
    }[asked] + [f"{asked} = {answer_text} cm."]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
