"""Exam-tier source for "an angle in a semicircle is a right angle".

The practice tiers only ask for an angle. An exam question uses the theorem as a
step in a longer length argument: once the angle at the circumference is known
to be a right angle, Pythagoras or the altitude (geometric mean) relation does
the real work.

Three question shapes, each built backwards from integer lengths so the key is
exact:

* ``altitude``: AB is a diameter, C is on the circle and H is the foot of the
  perpendicular from C to AB. AH and HB are given; find CH. Because angle ACB is
  a right angle, CH^2 = AH * HB.
* ``area``: the radius and one chord AC are given; find the area of triangle
  ABC. The other leg comes from Pythagoras in the right-angled triangle ABC.
* ``radius``: the two chords AC and BC are given; find the radius of the circle.
  The hypotenuse AB is a diameter.
"""

from __future__ import annotations

import random

from sympy import Integer, sqrt

from ..core import check, check_equivalent, generator, phrase, pick_distractors

_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41))


def _even_hyp_triple(rng: random.Random) -> tuple[int, int, int]:
    """A right-triangle triple (legs a, b, hypotenuse c) scaled so c is even."""
    a, b, c = rng.choice(_TRIPLES)
    k = rng.randint(1, 4)
    if c % 2 == 1 and k % 2 == 1:
        k += 1
    a, b, c = a * k, b * k, c * k
    if rng.random() < 0.5:
        a, b = b, a
    return a, b, c


def _altitude(rng: random.Random) -> dict:
    while True:
        m = rng.randint(1, 4)
        s, t = rng.randint(1, 4), rng.randint(1, 4)
        if s != t:
            break
    p, q = m * s * s, m * t * t
    h = m * s * t
    check_equivalent(sqrt(Integer(p * q)), h, "altitude: geometric mean disagrees")
    text = phrase(
        rng,
        (
            "AB is a diameter of a circle and C is a point on the circle. The perpendicular from C meets AB at H, "
            "where AH = {p} cm and HB = {q} cm. Find the length CH, in cm.",
            "A circle has diameter AB, and C lies on the circle. The foot of the perpendicular from C to AB is H, "
            "with AH = {p} cm and HB = {q} cm. Calculate CH, in cm.",
        ),
        p=p,
        q=q,
    )
    wrong = [p * q, p + q, abs(p - q), (p + q) // 2 if (p + q) % 2 == 0 else p + q + 1]
    working = [
        "Angle ACB is in a semicircle, so it is a right angle.",
        "In a right triangle the altitude to the hypotenuse satisfies CH^2 = AH x HB.",
        f"CH^2 = {p} x {q} = {p * q}, so CH = {h}.",
    ]
    return _pack(text, h, wrong, working)


def _area(rng: random.Random) -> dict:
    a, b, c = _even_hyp_triple(rng)
    r = c // 2
    area = a * b // 2
    check(a * b % 2 == 0, "area: half-product not an integer")
    check_equivalent(sqrt(Integer(4 * r * r - a * a)), b, "area: second leg disagrees")
    text = phrase(
        rng,
        (
            "Triangle ABC is drawn inside a circle of radius {r} cm so that AB is a diameter and C lies on the "
            "circle. The chord AC is {a} cm long. Find the area of triangle ABC, in cm^2.",
            "A, B and C lie on a circle of radius {r} cm, and AB passes through the centre. Given AC = {a} cm, "
            "calculate the area of triangle ABC, in cm^2.",
        ),
        r=r,
        a=a,
    )
    wrong = [a * c // 2, a * b, b, a * r]
    working = [
        "AB is a diameter, so angle ACB is a right angle (angle in a semicircle).",
        f"BC^2 = AB^2 - AC^2 = {c}^2 - {a}^2 = {b * b}, so BC = {b} cm.",
        f"Area = (1/2) x {a} x {b} = {area} cm^2.",
    ]
    return _pack(text, area, wrong, working)


def _radius(rng: random.Random) -> dict:
    a, b, c = _even_hyp_triple(rng)
    r = c // 2
    check_equivalent(sqrt(Integer(a * a + b * b)), c, "radius: hypotenuse disagrees")
    text = phrase(
        rng,
        (
            "Points A, B and C lie on a circle and AB is a diameter. The chords are AC = {a} cm and BC = {b} cm. "
            "Find the radius of the circle, in cm.",
            "Triangle ABC has all three vertices on a circle, with AB passing through the centre. If AC = {a} cm "
            "and BC = {b} cm, what is the radius of the circle, in cm?",
        ),
        a=a,
        b=b,
    )
    wrong = [c, a + b, (a + b) // 2, abs(a - b)]
    working = [
        "AB is a diameter, so angle ACB is a right angle.",
        f"AB^2 = {a}^2 + {b}^2 = {c * c}, so AB = {c} cm.",
        f"The radius is half the diameter: {r} cm.",
    ]
    return _pack(text, r, wrong, working)


def _pack(text: str, key: int, wrong: list[int], working: list[str]) -> dict:
    answer_text = str(key)
    distractors = pick_distractors(answer_text, [str(w) for w in wrong if w > 0])
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


@generator(
    "exd_x_h4_semicircle_composed",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_h4_semicircle_composed(rng: random.Random, tier: str = "exam") -> dict:
    """Composed length question that needs the angle-in-a-semicircle theorem."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return rng.choice((_altitude, _area, _radius))(rng)
