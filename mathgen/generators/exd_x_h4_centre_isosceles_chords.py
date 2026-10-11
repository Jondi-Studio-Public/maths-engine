"""Exam-tier source for "the angle at the centre subtended by an arc is twice the angle at the circumference".

The practice tiers give an angle at the centre or the circumference directly.
An exam question gives neither: it gives base angles of the isosceles triangles
that the radii create, and the learner has to build the angle at the
circumference from them before the theorem can be used.

A, B and C lie on a circle with centre O, and O is inside angle ABC. Because
OA = OB = OC, triangles OAB and OBC are isosceles, so angle OBA = angle OAB = a
and angle OBC = angle OCB = c. Then angle ABC = a + c, and the angle at the
centre on the same arc AC is twice that.

Three shapes:

* ``acute``: a + c < 90, B on the major arc AC; find angle AOC = 2(a + c).
* ``obtuse``: a + c > 90, B on the minor arc AC; the angle at the centre on the
  arc AC is reflex, so the (non-reflex) angle AOC is 360 - 2(a + c).
* ``unknown_base``: angle AOC (non-reflex) and a are given, B on the major arc;
  find angle OCB = angle AOC / 2 - a.
"""

from __future__ import annotations

import random

from ..core import check, generator, phrase, pick_distractors


def _pack(text, key, wrong, working):
    answer_text = str(key)
    distractors = pick_distractors(answer_text, [str(w) for w in wrong if 0 < w < 400])
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


def _acute(rng):
    while True:
        a, c = rng.randint(12, 60), rng.randint(12, 60)
        if a + c < 86:
            break
    s = a + c
    key = 2 * s
    text = phrase(
        rng,
        (
            "A, B and C lie on a circle with centre O. B is on the major arc AC and O lies inside angle ABC. "
            "Angle OAB = {a} degrees and angle OCB = {c} degrees. Find angle AOC, in degrees.",
            "The points A, B and C are on a circle with centre O, with B on the major arc AC and O inside angle "
            "ABC. If angle OAB is {a} degrees and angle OCB is {c} degrees, calculate angle AOC, in degrees.",
        ),
        a=a,
        c=c,
    )
    working = [
        "OA = OB, so triangle OAB is isosceles and angle OBA = angle OAB.",
        f"Similarly angle OBC = angle OCB = {c}, so angle ABC = {a} + {c} = {s} degrees.",
        f"The angle at the centre is twice the angle at the circumference: angle AOC = 2 x {s} = {key} degrees.",
    ]
    return _pack(text, key, [s, a + c + a, 180 - s, 360 - key], working)


def _obtuse(rng):
    while True:
        a, c = rng.randint(35, 80), rng.randint(35, 80)
        if 94 < a + c < 170:
            break
    s = a + c
    key = 360 - 2 * s
    text = phrase(
        rng,
        (
            "A, B and C lie on a circle with centre O. B is on the minor arc AC and O lies inside angle ABC. "
            "Angle OAB = {a} degrees and angle OCB = {c} degrees. Find the non-reflex angle AOC, in degrees.",
            "The points A, B and C are on a circle with centre O, with B on the minor arc AC and O inside angle "
            "ABC. Given angle OAB = {a} degrees and angle OCB = {c} degrees, calculate the non-reflex angle AOC, "
            "in degrees.",
        ),
        a=a,
        c=c,
    )
    working = [
        "Triangles OAB and OBC are isosceles (OA = OB = OC), so angle OBA = " + f"{a} and angle OBC = {c}.",
        f"Angle ABC = {a} + {c} = {s} degrees, which is obtuse, so B is on the minor arc.",
        f"The reflex angle AOC is 2 x {s} = {2 * s} degrees, so the non-reflex angle AOC is 360 - {2 * s} = {key}.",
    ]
    return _pack(text, key, [2 * s, s, 180 - s, 2 * s - 180], working)


def _unknown_base(rng):
    while True:
        a, c = rng.randint(12, 60), rng.randint(12, 60)
        if a + c < 86:
            break
    m = 2 * (a + c)
    text = phrase(
        rng,
        (
            "A, B and C lie on a circle with centre O. B is on the major arc AC and O lies inside angle ABC. "
            "Angle OAB = {a} degrees and the non-reflex angle AOC = {m} degrees. Find angle OCB, in degrees.",
            "The points A, B and C are on a circle with centre O, with B on the major arc AC and O inside angle "
            "ABC. Angle OAB is {a} degrees and the non-reflex angle AOC is {m} degrees. Calculate angle OCB, in "
            "degrees.",
        ),
        a=a,
        m=m,
    )
    working = [
        f"The angle at the circumference is half the angle at the centre: angle ABC = {m} / 2 = {m // 2} degrees.",
        f"Triangle OAB is isosceles, so angle OBA = angle OAB = {a} degrees.",
        f"Angle OBC = {m // 2} - {a} = {c} degrees, and triangle OBC is isosceles, so angle OCB = {c} degrees.",
    ]
    return _pack(text, c, [m // 2 + a, m // 2, m - a, 180 - m // 2 - a], working)


@generator(
    "exd_x_h4_centre_isosceles_chords",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_h4_centre_isosceles_chords(rng: random.Random, tier: str = "exam") -> dict:
    """Centre-angle theorem reached through two isosceles radius triangles."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return rng.choice((_acute, _obtuse, _unknown_base))(rng)
