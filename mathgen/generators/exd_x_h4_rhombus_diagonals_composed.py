"""Exam-tier source for "the diagonals of a parallelogram meet at right angles if and only if it is a rhombus".

The practice tiers find a side or a diagonal of a rhombus that is already named.
An exam question makes the learner use the theorem as a step: the diagonals of
any parallelogram bisect each other, so the half-diagonals and the half-side
form a triangle whose shape decides whether the diagonals are perpendicular.

Three shapes, each built backwards from the half-diagonals p and q and the side
s with p^2 + q^2 = s^2 (a Pythagorean triple, scaled):

* ``area``: a rhombus is described by its side s and one diagonal 2p; find the
  area, which needs the other diagonal 2q first.
* ``perimeter``: a rhombus is described by its area 2pq and one diagonal 2p; find
  the perimeter 4s, which needs the other diagonal and then the side.
* ``recognise``: a parallelogram ABCD is only said to have AB = s, AC = 2p and
  BD = 2q. Because p^2 + q^2 = s^2, the angle between the diagonals is a right
  angle, so ABCD is a rhombus and the perimeter is 4s. The learner has to see
  this; nothing in the wording says "rhombus".
"""

from __future__ import annotations

import random

from sympy import Integer, sqrt

from ..core import check, check_equivalent, generator, phrase, pick_distractors

_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29))


def _triple(rng):
    a, b, c = rng.choice(_TRIPLES)
    k = rng.randint(1, 3)
    p, q, s = a * k, b * k, c * k
    if rng.random() < 0.5:
        p, q = q, p
    check(p * p + q * q == s * s, "rhombus: not a triple")
    return p, q, s


def _pack(text, key, wrong, working):
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


def _area(rng):
    p, q, s = _triple(rng)
    check_equivalent(sqrt(Integer(s * s - p * p)), q, "rhombus area: half-diagonal disagrees")
    key = 2 * p * q
    text = phrase(
        rng,
        (
            "A rhombus has sides of length {s} cm and one diagonal of length {d} cm. Find the area of the "
            "rhombus, in cm^2.",
            "Each side of a rhombus is {s} cm and one of its diagonals is {d} cm long. Calculate the area of the "
            "rhombus, in cm^2.",
        ),
        s=s,
        d=2 * p,
    )
    working = [
        "The diagonals of a rhombus bisect each other at right angles.",
        f"The other half-diagonal is sqrt({s}^2 - {p}^2) = {q} cm, so the diagonals are {2 * p} cm and {2 * q} cm.",
        f"Area = (1/2) x {2 * p} x {2 * q} = {key} cm^2.",
    ]
    return _pack(text, key, [2 * p * s, p * q, p * s, 2 * q * s], working)


def _perimeter(rng):
    p, q, s = _triple(rng)
    area = 2 * p * q
    key = 4 * s
    text = phrase(
        rng,
        (
            "A rhombus has an area of {area} cm^2 and one diagonal of length {d} cm. Find the perimeter of the "
            "rhombus, in cm.",
            "The area of a rhombus is {area} cm^2 and one of its diagonals is {d} cm long. Calculate the "
            "perimeter of the rhombus, in cm.",
        ),
        area=area,
        d=2 * p,
    )
    working = [
        f"Area = (1/2) x d1 x d2, so the other diagonal is 2 x {area} / {2 * p} = {2 * q} cm.",
        f"The diagonals meet at right angles, so each side is sqrt({p}^2 + {q}^2) = {s} cm.",
        f"Perimeter = 4 x {s} = {key} cm.",
    ]
    return _pack(text, key, [s, 2 * (p + q), 4 * q, 4 * p, 2 * s], working)


def _recognise(rng):
    p, q, s = _triple(rng)
    key = 4 * s
    text = phrase(
        rng,
        (
            "In parallelogram ABCD, AB = {s} cm, the diagonal AC = {d1} cm and the diagonal BD = {d2} cm. "
            "Find the perimeter of ABCD, in cm.",
            "ABCD is a parallelogram with AB = {s} cm. Its diagonals measure AC = {d1} cm and BD = {d2} cm. "
            "Calculate the perimeter of ABCD, in cm.",
        ),
        s=s,
        d1=2 * p,
        d2=2 * q,
    )
    working = [
        "The diagonals of a parallelogram bisect each other at M, so AM = " + f"{p} cm and BM = {q} cm.",
        f"AM^2 + BM^2 = {p * p} + {q * q} = {s * s} = AB^2, so angle AMB is a right angle.",
        "Perpendicular diagonals mean ABCD is a rhombus, so all four sides are equal.",
        f"Perimeter = 4 x {s} = {key} cm.",
    ]
    return _pack(text, key, [2 * (s + p), 2 * (s + q), 2 * (2 * p + 2 * q), 2 * s + 2 * p + 2 * q], working)


@generator(
    "exd_x_h4_rhombus_diagonals_composed",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_h4_rhombus_diagonals_composed(rng: random.Random, tier: str = "exam") -> dict:
    """Perpendicular-diagonals theorem used as a step in an area or perimeter question."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return rng.choice((_area, _perimeter, _recognise))(rng)
