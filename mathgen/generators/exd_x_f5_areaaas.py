"""Exam tier for the area of a triangle from the sine formula: two angles and one side.

The practice tiers give two sides and the included angle (or the area). Here only one side is given, with two
angles, and the side is opposite any one of the three vertices. The learner must find the third angle, use the
sine rule to get a second side, and then identify an included angle for ``1/2 a b sin C``. Equivalently
``area = s^2 sin Q sin R / (2 sin P)`` for the given side ``s`` opposite angle ``P``.
"""

from __future__ import annotations

import math
import random

from ..core import GRADER_TOLERANCE, check, generator, pick_distractors

_LETTERS = "ABC"


def _clear(value: float, eps: float = 0.03) -> bool:
    return abs((value * 10) % 1 - 0.5) > eps


@generator("exd_x_f5_areaaas", level=12, presets={"exam": dict(tier="exam")})
def exd_x_f5_areaaas(rng: random.Random, tier: str = "exam") -> dict:
    """Area of a triangle from two angles and one side."""
    if tier != "exam":
        raise ValueError(f"unknown tier: {tier}")
    s = lambda deg: math.sin(math.radians(deg))  # noqa: E731
    while True:
        angles = [rng.randint(30, 100) for _ in range(2)]
        third = 180 - sum(angles)
        if not 25 <= third <= 120:
            continue
        angles.append(third)
        given = rng.randrange(3)  # index of the vertex opposite the given side
        side = rng.randint(6, 25)
        p, q, r = angles[given], angles[(given + 1) % 3], angles[(given + 2) % 3]
        area_exact = side * side * s(q) * s(r) / (2 * s(p))
        if not _clear(area_exact):
            continue
        area = round(area_exact, 1)
        other_exact = side * s(p) / s(q)  # sine rule with the ratio flipped
        cands_exact = [
            2 * area_exact,  # forgets the one half
            0.5 * side * side * s(r),  # uses the given side twice
            0.5 * side * other_exact * s(r),  # flipped sine-rule ratio
            0.5 * side * side * s(p),  # given side twice with its own opposite angle
        ]
        distractors = pick_distractors(
            f"{area:.1f}", [f"{c:.1f}" for c in cands_exact], n=3, tolerance=GRADER_TOLERANCE
        )
        if len(distractors) == 3:
            break
    # Independent route: Heron's formula on the three sides found from the sine rule.
    sides = [side * s(angle) / s(p) for angle in (p, q, r)]
    sp = sum(sides) / 2
    heron = math.sqrt(sp * (sp - sides[0]) * (sp - sides[1]) * (sp - sides[2]))
    check(abs(heron - area_exact) < 1e-9, "Heron's formula disagrees with the sine-rule area")
    known = [_LETTERS[(given + 1) % 3], _LETTERS[(given + 2) % 3]]
    k1, k2 = sorted(known)
    text = (
        f"In triangle ABC, angle {k1} = {angles[_LETTERS.index(k1)]}° and angle {k2} = {angles[_LETTERS.index(k2)]}°. "
        f"The side {_LETTERS[given].lower()}, opposite angle {_LETTERS[given]}, is {side} cm long. "
        f"Find the area of the triangle in cm² (1 d.p.)."
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=area,
        answer_text=f"{area:.1f}",
        answer_latex=f"{area:.1f}",
        working=[
            f"Angle {_LETTERS[given]} = 180 - {angles[_LETTERS.index(k1)]} - {angles[_LETTERS.index(k2)]} = {p}°.",
            f"Sine rule for a second side, then area = 1/2 ab sin C = {area:.1f}.",
        ],
        distractors=distractors,
    )
