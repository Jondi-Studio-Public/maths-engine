"""Exam-tier source for "the alternate segment theorem".

The practice tiers give the angle between a tangent and a chord and read off
the angle in the alternate segment. An exam question starts from the point
OUTSIDE the circle: two tangents PA and PB are drawn from P, and Q is a point on
the major arc AB. The learner must combine three ideas:

1. tangents from an external point are equal, so triangle PAB is isosceles and
   angle PAB = (180 - angle APB) / 2;
2. angle PAB is the tangent-chord angle at A, so by the alternate segment
   theorem angle AQB equals it;
3. therefore angle APB + 2 x angle AQB = 180.

Three shapes: find angle AQB from angle APB, find angle APB from angle AQB, and
a version where both are linear expressions in x that must first be solved.
"""

from __future__ import annotations

import random

from sympy import Integer, Symbol, solve

from ..core import check, generator, join_terms, phrase, pick_distractors


def _lin(a: int, b: int) -> str:
    return join_terms([(a, "x"), (b, "")])


def _pack(text, key, wrong, working):
    answer_text = str(key)
    distractors = pick_distractors(answer_text, [str(w) for w in wrong if 0 < w < 360])
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _from_apb(rng):
    p = 2 * rng.randint(10, 70)
    q = 90 - p // 2
    text = phrase(
        rng,
        (
            "Tangents PA and PB are drawn from an external point P to a circle, touching it at A and B. "
            "Angle APB = {p} degrees. Q is a point on the major arc AB. Find angle AQB, in degrees.",
            "From a point P outside a circle, two tangents touch the circle at A and B, and angle APB is {p} "
            "degrees. The point Q lies on the major arc AB. Calculate angle AQB, in degrees.",
        ),
        p=p,
    )
    working = [
        "Tangents from P are equal, so triangle PAB is isosceles with PA = PB.",
        f"Angle PAB = (180 - {p}) / 2 = {q} degrees.",
        f"By the alternate segment theorem, angle AQB = angle PAB = {q} degrees.",
    ]
    return _pack(text, q, [p // 2, 180 - p, p, 180 - q], working)


def _from_aqb(rng):
    q = rng.randint(20, 80)
    p = 180 - 2 * q
    text = phrase(
        rng,
        (
            "Tangents PA and PB are drawn from an external point P to a circle, touching it at A and B. "
            "Q is a point on the major arc AB and angle AQB = {q} degrees. Find angle APB, in degrees.",
            "The tangents from a point P touch a circle at A and B. A point Q on the major arc AB makes angle AQB "
            "equal to {q} degrees. Calculate angle APB, in degrees.",
        ),
        q=q,
    )
    working = [
        f"By the alternate segment theorem, angle PAB = angle AQB = {q} degrees.",
        "Tangents from P are equal, so angle PBA = angle PAB as well.",
        f"Angle APB = 180 - 2 x {q} = {p} degrees.",
    ]
    return _pack(text, p, [2 * q, 180 - q, 90 - q, q, 90 + q], working)


def _algebraic(rng):
    for _ in range(500):
        x = rng.randint(3, 20)
        c = rng.randint(1, 4)
        d = rng.randint(-10, 25)
        q = c * x + d
        p = 180 - 2 * q
        a = rng.randint(1, 6)
        b = p - a * x
        if 15 <= q <= 80 and abs(b) <= 90:
            break
    xs = Symbol("x")
    sol = solve((a * xs + b) + 2 * (c * xs + d) - 180, xs)
    check(sol == [Integer(x)], "tangent_pair: x disagrees")
    text = phrase(
        rng,
        (
            "Tangents PA and PB are drawn from an external point P to a circle, touching it at A and B. "
            "Q is a point on the major arc AB. Angle APB = ({u}) degrees and angle AQB = ({v}) degrees. "
            "Find angle APB, in degrees.",
            "From an external point P, tangents touch a circle at A and B, and Q lies on the major arc AB. "
            "Angle APB is ({u}) degrees and angle AQB is ({v}) degrees. Calculate angle APB, in degrees.",
        ),
        u=_lin(a, b),
        v=_lin(c, d),
    )
    working = [
        "Triangle PAB is isosceles (equal tangents) and angle PAB = angle AQB (alternate segment).",
        f"So angle APB + 2 x angle AQB = 180: ({_lin(a, b)}) + 2({_lin(c, d)}) = 180.",
        f"x = {x}, so angle APB = {p} degrees.",
    ]
    return _pack(text, p, [q, 180 - q, a * x, x], working)


@generator(
    "exd_x_h4_tangent_pair_alternate_segment",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_h4_tangent_pair_alternate_segment(rng: random.Random, tier: str = "exam") -> dict:
    """Two tangents from an external point plus the alternate segment theorem."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return rng.choice((_from_apb, _from_aqb, _algebraic))(rng)
