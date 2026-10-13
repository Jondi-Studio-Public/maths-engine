"""Exam-tier source for `beam-reactions-point-load`: the reactions of a simply supported beam under two point loads.

The loads are not at the middle and not equal, so neither support carries half the total. The learner takes moments
about A to find R_B, then uses vertical equilibrium for R_A. Built backwards: the loads and positions are drawn, and
the draw is kept only when R_B comes out as a multiple of 0.5 kN.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)
from .exd_x_u6_support import keys_clear, tagged

_SETTINGS = [
    "A simply supported timber beam AB spans {L} m.",
    "A steel beam AB rests on supports at A and B, {L} m apart.",
    "A footbridge girder AB is simply supported at its ends and spans {L} m.",
    "A plank AB is laid across a gap of {L} m between supports at A and B.",
]
_LOADS = [
    "Two point loads act on it: {w1} kN at {a1} m from A and {w2} kN at {a2} m from A.",
    "It carries a point load of {w1} kN at {a1} m from A and a second point load of {w2} kN at {a2} m from A.",
    "Loads of {w1} kN and {w2} kN act downwards at distances of {a1} m and {a2} m from A respectively.",
]


def _text(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 1), 1)


def _build(rng: random.Random):
    L = rng.choice([6, 8, 10, 12])
    a1 = rng.randint(1, L - 2)
    a2 = rng.randint(a1 + 1, L - 1)
    w1, w2 = rng.choice(range(10, 70, 10)), rng.choice(range(10, 70, 10))
    r_b = Fraction(w1 * a1 + w2 * a2, L)
    r_a = w1 + w2 - r_b
    if (2 * r_b).denominator != 1 or w1 == w2 or r_a == r_b or a1 + a2 == L:
        return None
    check(w1 * (L - a1) + w2 * (L - a2) == r_a * L, "moments about B disagree with the reactions")

    stem = f"{rng.choice(_SETTINGS).format(L=L)} {rng.choice(_LOADS).format(w1=w1, w2=w2, a1=a1, a2=a2)}"
    p1 = "Find the reaction R_B at support B, in kN."
    p2 = "Using your answer to (a), find the reaction R_A at support A, in kN."
    if not keys_clear(" ".join([stem, p1, p2]), [r_a, r_b]):
        return None

    def tag(value, mid):
        value = round_half_up(value, 1)
        return tagged(_text(value), mid, value)

    total = w1 + w2
    a_d = pick_distractors(
        _text(r_b),
        [
            tag(r_a, "swapped-support-reactions"),
            tag(Fraction(total, 2), "equal-reactions-assumed"),
            tag(Fraction(w2 * a2, L), "left-out-point-load"),
            tag(Fraction(w1 * a1 + w2 * a2), "reaction-not-divided-by-span"),
        ],
        3,
    )
    b_d = pick_distractors(
        _text(r_a),
        [
            tag(r_b, "swapped-support-reactions"),
            tag(Fraction(total, 2), "equal-reactions-assumed"),
            tag(Fraction(w1 * (L - a1), L), "left-out-point-load"),
            tag(Fraction(total), None),
        ],
        3,
    )
    if min(len(a_d), len(b_d)) < 2:
        return None

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p1,
                answer_kind="number",
                answer=numeric_answer(r_b),
                answer_text=_text(r_b),
                answer_latex=latex_answer(_text(r_b)),
                marks=3,
                sensible_range=(0, None),
                working=[
                    f"Moments about A: R_B × {L} = {w1} × {a1} + {w2} × {a2} = {w1 * a1 + w2 * a2}.",
                    f"R_B = {_text(r_b)} kN.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=p2,
                answer_kind="number",
                answer=numeric_answer(r_a),
                answer_text=_text(r_a),
                answer_latex=latex_answer(_text(r_a)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{total} - part_a",
                working=[f"R_A + R_B = {w1} + {w2} = {total}.", f"R_A = {total} - {_text(r_b)} = {_text(r_a)} kN."],
                distractors=b_d,
            ),
        ],
    )


@generator("exd_x_u6_beam_two_point_loads", level=20, presets={"exam": dict(tier="exam")})
def exd_x_u6_beam_two_point_loads(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: reactions of a simply supported beam under two unequal point loads."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "loads giving reactions on the 0.5 kN grid"):
        question = _build(rng)
        if question is not None:
            return question
