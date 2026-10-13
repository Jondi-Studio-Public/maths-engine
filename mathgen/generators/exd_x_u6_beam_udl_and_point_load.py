"""Exam-tier source for `bending-moment-point-load`: the bending moment under a point load when the beam also carries
a uniformly distributed load over its whole span.

The learner

1. finds the support reaction R_A from the point load and the UDL together,
2. finds the bending moment under the point load (the UDL contributes a negative term), and
3. finds the bending moment at midspan, which lies beyond the point load, so the point load now enters that moment too.

Built backwards: the span, load, position and UDL are drawn and a draw is kept only when R_A is a multiple of
0.5 kN, so every answer sits on the 0.5 grid. The point load always acts nearer to A than midspan.
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
    "A simply supported floor beam AB spans {L} m and carries a uniformly distributed load of {w} kN/m over its whole "
    "length. A point load of {W} kN also acts on it, {a} m from A.",
    "A bridge girder AB spans {L} m between simple supports. Its own weight and deck act as a uniformly distributed "
    "load of {w} kN/m, and a vehicle adds a point load of {W} kN at {a} m from A.",
    "A simply supported beam AB, {L} m long, supports a point load of {W} kN at {a} m from end A together with a "
    "uniformly distributed load of {w} kN/m along its full length.",
]


def _text(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 1), 1)


def _build(rng: random.Random):
    L = rng.choice([4, 6, 8, 10, 12])
    w = rng.choice([2, 4, 6, 8])
    W = rng.choice(range(10, 70, 10))
    a = rng.randint(1, (L - 1) // 2)
    if 2 * a >= L:
        return None
    b = L - a
    r_a = Fraction(W * b, L) + Fraction(w * L, 2)
    if (2 * r_a).denominator != 1:
        return None
    r_b = W + w * L - r_a
    m_load = r_a * a - Fraction(w * a * a, 2)
    m_mid = r_a * Fraction(L, 2) - Fraction(w * L * L, 8) - W * (Fraction(L, 2) - a)
    check(m_mid == r_b * Fraction(L, 2) - Fraction(w * L * L, 8), "midspan moment disagrees when taken from B")
    check(m_load == r_b * b - Fraction(w * b * b, 2), "moment under the load disagrees when taken from B")
    if min(m_load, m_mid) <= 0 or m_load == m_mid:
        return None

    stem = rng.choice(_SETTINGS).format(L=L, w=w, W=W, a=a)
    p1 = "Find the reaction R_A at support A, in kN."
    p2 = "Using your answer to (a), find the bending moment under the point load, in kN·m."
    p3 = "Using your answer to (a), find the bending moment at midspan, in kN·m."
    if not keys_clear(" ".join([stem, p1, p2, p3]), [r_a, m_load, m_mid]):
        return None

    def tag(value, mid):
        value = round_half_up(value, 1)
        return tagged(_text(value), mid, value)

    a_d = pick_distractors(
        _text(r_a),
        [
            tag(Fraction(w * L, 2), "left-out-point-load"),
            tag(Fraction(W * b, L), "left-out-udl-load"),
            tag(Fraction(W, 2) + Fraction(w * L, 2), "equal-reactions-assumed"),
            tag(r_b, "swapped-support-reactions"),
        ],
        3,
    )
    b_d = pick_distractors(
        _text(m_load),
        [
            tag(r_a * a, "section-moment-ignores-udl"),
            tag(r_a * a - w * a * a, "udl-lever-arm-full-length"),
            tag(r_b * a - Fraction(w * a * a, 2), "swapped-support-reactions"),
        ],
        3,
    )
    c_d = pick_distractors(
        _text(m_mid),
        [
            tag(r_a * Fraction(L, 2) - Fraction(w * L * L, 8), "left-out-point-load"),
            tag(r_a * Fraction(L, 2) - W * (Fraction(L, 2) - a), "section-moment-ignores-udl"),
            tag(r_a * Fraction(L, 2) - Fraction(w * L * L, 2), "wrong-udl-moment-formula"),
        ],
        3,
    )
    if min(len(a_d), len(b_d), len(c_d)) < 2:
        return None

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p1,
                answer_kind="number",
                answer=numeric_answer(r_a),
                answer_text=_text(r_a),
                answer_latex=latex_answer(_text(r_a)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"R_A = W × {b}/{L} + w × {L}/2 = {_text(Fraction(W * b, L))} + {w * L // 2} = {_text(r_a)} kN."
                ],
                distractors=a_d,
            ),
            dict(
                prompt=p2,
                answer_kind="number",
                answer=numeric_answer(m_load),
                answer_text=_text(m_load),
                answer_latex=latex_answer(_text(m_load)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{a} - {w}*{a}**2/2",
                working=[
                    f"M = R_A × {a} - w × {a}²/2 = {_text(r_a)} × {a} - {_text(Fraction(w * a * a, 2))} = {_text(m_load)} kN·m."
                ],
                distractors=b_d,
            ),
            dict(
                prompt=p3,
                answer_kind="number",
                answer=numeric_answer(m_mid),
                answer_text=_text(m_mid),
                answer_latex=latex_answer(_text(m_mid)),
                marks=3,
                sensible_range=(0, None),
                follow_through=f"part_a*{L}/2 - {w}*{L}**2/8 - {W}*({L}/2 - {a})",
                working=[
                    f"M = R_A × {L}/2 - w × {L}²/8 - W × ({L}/2 - {a}).",
                    f"= {_text(r_a)} × {L // 2} - {_text(Fraction(w * L * L, 8))} - {_text(W * (Fraction(L, 2) - a))} = {_text(m_mid)} kN·m.",
                ],
                distractors=c_d,
            ),
        ],
    )


@generator("exd_x_u6_beam_udl_and_point_load", level=20, presets={"exam": dict(tier="exam")})
def exd_x_u6_beam_udl_and_point_load(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: reaction and bending moments of a simply supported beam under a UDL and a point load."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a load case on the 0.5 kN grid"):
        question = _build(rng)
        if question is not None:
            return question
