"""Exam-tier source for "angles at the circumference subtended by the same arc are equal".

The practice tiers use two points on the same arc. An exam question sets the
theorem inside a figure with two crossing chords: the learner has to pick out
which pairs of angles stand on the same arc, solve for the unknown from that,
and then finish with the angle sum of a triangle or with adding adjacent angles.

Chords AC and BD of a circle meet at P inside the circle, so A, B, C, D lie in
that order round the circle. Angle BAC and angle BDC stand on arc BC, and angle
DAC and angle DBC stand on arc DC.

Two question shapes, both built backwards from the chosen integer x:

* ``vertical``: angle BAC and angle BDC are given as linear expressions in x;
  with angle ABD also given, find angle CPD. Triangle PAB gives angle APB, which
  equals angle CPD (vertically opposite).
* ``adjacent``: the same equation fixes x; angle ABD is an expression in x and
  angle DAC is a number; find angle ABC = angle ABD + angle DBC, where angle DBC
  equals angle DAC.

The four arcs of the circle must add to 360 degrees, which bounds the angles.
"""

from __future__ import annotations

import random

from sympy import Integer, Symbol, solve

from ..core import check, check_equivalent, generator, join_terms, phrase, pick_distractors


def _lin(a: int, b: int) -> str:
    return join_terms([(a, "x"), (b, "")])


def _draw(rng: random.Random):
    """x and the two expressions that agree at x (angle BAC = angle BDC)."""
    for _ in range(500):
        x = rng.randint(4, 25)
        a1, a2 = rng.sample(range(1, 6), 2)
        value = rng.randint(24, 70)
        b1, b2 = value - a1 * x, value - a2 * x
        if abs(b1) <= 60 and abs(b2) <= 60:
            return x, value, (a1, b1), (a2, b2)
    raise AssertionError("no draw")  # pragma: no cover


def _pack(text: str, key: int, wrong: list[int], working: list[str]) -> dict:
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


def _vertical(rng: random.Random) -> dict:
    for _ in range(500):
        x, bac, e1, e2 = _draw(rng)
        abd = rng.randint(20, 70)
        if bac + abd < 160:
            break
    key = 180 - bac - abd
    xs = Symbol("x")
    sol = solve(e1[0] * xs + e1[1] - (e2[0] * xs + e2[1]), xs)
    check(sol == [Integer(x)], "vertical: x disagrees")
    check_equivalent(180 - (e1[0] * x + e1[1]) - abd, key, "vertical: key disagrees")
    text = phrase(
        rng,
        (
            "Chords AC and BD of a circle meet at P inside the circle. Angle BAC = ({u}) degrees and angle BDC = "
            "({v}) degrees. Angle ABD = {t} degrees. Find angle CPD, in degrees.",
            "In a circle, chords AC and BD cross at the point P. Angle BAC is ({u}) degrees, angle BDC is ({v}) "
            "degrees and angle ABD is {t} degrees. Calculate angle CPD, in degrees.",
        ),
        u=_lin(*e1),
        v=_lin(*e2),
        t=abd,
    )
    wrong = [bac + abd, bac, abd, 180 - bac]
    working = [
        "Angles BAC and BDC stand on the same arc BC, so they are equal.",
        f"{_lin(*e1)} = {_lin(*e2)} gives x = {x}, so angle BAC = {bac} degrees.",
        f"In triangle PAB, angle APB = 180 - {bac} - {abd} = {key} degrees.",
        f"Angle CPD is vertically opposite angle APB, so it is {key} degrees.",
    ]
    return _pack(text, key, wrong, working)


def _adjacent(rng: random.Random) -> dict:
    for _ in range(500):
        x, bac, e1, e2 = _draw(rng)
        c, d = rng.randint(1, 4), rng.randint(-20, 20)
        abd = c * x + d
        dac = rng.randint(18, 60)
        if 15 <= abd <= 70 and bac + abd + dac < 165:
            break
    key = abd + dac
    xs = Symbol("x")
    sol = solve(e1[0] * xs + e1[1] - (e2[0] * xs + e2[1]), xs)
    check(sol == [Integer(x)], "adjacent: x disagrees")
    text = phrase(
        rng,
        (
            "Chords AC and BD of a circle meet at P inside the circle. Angle BAC = ({u}) degrees and angle BDC = "
            "({v}) degrees. Angle ABD = ({w}) degrees and angle DAC = {t} degrees. Find angle ABC, in degrees.",
            "In a circle, chords AC and BD cross at P. Angle BAC is ({u}) degrees and angle BDC is ({v}) degrees, "
            "while angle ABD is ({w}) degrees and angle DAC is {t} degrees. Calculate angle ABC, in degrees.",
        ),
        u=_lin(*e1),
        v=_lin(*e2),
        w=_lin(c, d),
        t=dac,
    )
    wrong = [abd + bac, abd, dac, abd + x]
    working = [
        "Angles BAC and BDC stand on the same arc BC, so they are equal.",
        f"{_lin(*e1)} = {_lin(*e2)} gives x = {x}.",
        f"Angle ABD = {_lin(c, d)} = {abd} degrees.",
        f"Angle DBC stands on arc DC, as does angle DAC, so angle DBC = {dac} degrees.",
        f"Angle ABC = angle ABD + angle DBC = {abd} + {dac} = {key} degrees.",
    ]
    return _pack(text, key, wrong, working)


@generator(
    "exd_x_h4_chords_intersecting",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_h4_chords_intersecting(rng: random.Random, tier: str = "exam") -> dict:
    """Crossing chords: same-arc angle equality plus a triangle or an angle sum."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return rng.choice((_vertical, _adjacent))(rng)
