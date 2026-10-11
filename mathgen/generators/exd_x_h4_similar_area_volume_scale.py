"""Exam-tier source for "use the scale factor for two similar figures".

The practice tiers scale a length by a linear scale factor that is given or read
off from a pair of lengths. An exam question gives AREAS or VOLUMES instead, so
the learner has to recover the linear scale factor first (the square root of the
area ratio or the cube root of the volume ratio), or go the other way and turn a
linear ratio into an area.

Three shapes, all built backwards from the linear ratio p : q (coprime, p != q):

* ``area_to_length``: the areas of two similar figures and a length of the first
  are given; find the corresponding length of the second.
* ``volume_to_length``: the volumes of two similar solids and a length of the
  first are given; find the corresponding length of the second.
* ``length_to_area``: two corresponding lengths and the area of the first figure
  are given; find the area of the second.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from sympy import Rational, cbrt, sqrt

from ..core import VerificationError, check, check_equivalent, generator, phrase, pick_distractors


def _ratio(rng, hi):
    while True:
        p, q = rng.randint(1, hi), rng.randint(1, hi)
        if p != q and math.gcd(p, q) == 1:
            return p, q


def _pack(text, key, wrong, working):
    answer_text = str(key)
    keep = []
    for w in wrong:
        w = Fraction(w)
        if w.denominator == 1 and w > 0:
            keep.append(str(int(w)))
    distractors = pick_distractors(answer_text, keep)
    if len(distractors) < 2:
        return None  # too few distinct wrong answers for this draw; the caller redraws
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _area_to_length(rng):
    p, q = _ratio(rng, 7)
    n = rng.randint(2, 9)
    a1, a2 = n * q * q, n * p * p
    length = q * q * rng.randint(1, 3)
    key = length * p // q
    check_equivalent(sqrt(Rational(a2, a1)), Rational(p, q), "area_to_length: ratio disagrees")
    text = phrase(
        rng,
        (
            "Two similar figures have areas {a1} cm^2 and {a2} cm^2. A side of the first figure, the "
            "one of area {a1} cm^2, is {length} cm long. Find the length of the corresponding side of the figure "
            "of area {a2} cm^2, in cm.",
            "Figure X has area {a1} cm^2 and figure Y has area {a2} cm^2, and the figures are similar. A side of "
            "X measures {length} cm. What is the length of the corresponding side of Y, in cm?",
        ),
        a1=a1,
        a2=a2,
        length=length,
    )
    working = [
        f"Area ratio = {a2} : {a1} = {p * p} : {q * q}.",
        f"The ratio of lengths is the square root of the ratio of areas: {p} : {q}.",
        f"Length of the corresponding side = {length} x {p} / {q} = {key} cm.",
    ]
    wrong = [Fraction(length * p * p, q * q), length * p, Fraction(length * q, p), length + a2 - a1]
    return _pack(text, key, wrong, working)


def _volume_to_length(rng):
    p, q = _ratio(rng, 5)
    n = rng.randint(2, 6)
    v1, v2 = n * q**3, n * p**3
    length = q * q * rng.randint(1, 3)
    key = length * p // q
    check_equivalent(cbrt(Rational(v2, v1)), Rational(p, q), "volume_to_length: ratio disagrees")
    text = phrase(
        rng,
        (
            "Two similar solids have volumes {v1} cm^3 and {v2} cm^3. An edge of the solid of volume {v1} cm^3 is "
            "{length} cm long. Find the length of the corresponding edge of the solid of volume {v2} cm^3, in cm.",
            "Solid X has volume {v1} cm^3 and solid Y has volume {v2} cm^3, and the solids are similar. An edge of "
            "X measures {length} cm. What is the length of the corresponding edge of Y, in cm?",
        ),
        v1=v1,
        v2=v2,
        length=length,
    )
    working = [
        f"Volume ratio = {v2} : {v1} = {p**3} : {q**3}.",
        f"The ratio of lengths is the cube root of the ratio of volumes: {p} : {q}.",
        f"Length of the corresponding edge = {length} x {p} / {q} = {key} cm.",
    ]
    wrong = [
        Fraction(length * p * p, q * q),
        length * p,
        Fraction(length * p**3, q**3),
        length + v2 - v1,
        Fraction(length * q, p),
    ]
    return _pack(text, key, wrong, working)


def _length_to_area(rng):
    p, q = _ratio(rng, 7)
    n = rng.randint(2, 9)
    a1, a2 = n * q * q, n * p * p
    mult = rng.randint(1, 4)
    l1, l2 = q * mult, p * mult
    check(Fraction(l2, l1) ** 2 * a1 == a2, "length_to_area: key disagrees")
    text = phrase(
        rng,
        (
            "Two similar figures have corresponding sides of {l1} cm and {l2} cm. The figure with the {l1} cm "
            "side has an area of {a1} cm^2. Find the area of the other figure, in cm^2.",
            "A figure with area {a1} cm^2 has a side of {l1} cm. A similar figure has the corresponding side "
            "{l2} cm long. Calculate the area of the similar figure, in cm^2.",
        ),
        l1=l1,
        l2=l2,
        a1=a1,
    )
    working = [
        f"The linear scale factor is {l2} / {l1} = {p} / {q}.",
        f"Areas scale by the square of the linear factor: {p * p} / {q * q}.",
        f"Area = {a1} x {p * p} / {q * q} = {a2} cm^2.",
    ]
    wrong = [Fraction(a1 * p, q), a1 * p, a1 + (l2 - l1), Fraction(a1 * p**3, q**3)]
    return _pack(text, a2, wrong, working)


@generator(
    "exd_x_h4_similar_area_volume_scale",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_h4_similar_area_volume_scale(rng: random.Random, tier: str = "exam") -> dict:
    """Similar figures where areas or volumes, not lengths, carry the scale factor."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(50):
        question = rng.choice((_area_to_length, _volume_to_length, _length_to_area))(rng)
        if question is not None:
            return question
    raise VerificationError("no draw with enough distractors")  # pragma: no cover
