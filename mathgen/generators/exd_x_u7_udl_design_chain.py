"""Exam-tier source for `max-bending-moment-udl`: work back from a design moment to the load, then forward again.

A simply supported beam carries a uniformly distributed load. The learner is not given the load: the stem gives
the span and the maximum bending moment the beam has been designed for, and the question is a chain:

(a) the load intensity w that produces that maximum moment (the formula M = wL^2/8 used backwards);
(b) the maximum shear force at the supports, from that load;
(c) the bending moment at a stated distance from one support, from the reactions and the load on the part of
    the span up to the section.

Built backwards from a whole-number load, with an even intensity so that the moment of the midspan and every
section moment is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, draw_until, generator, latex_answer

_STEMS = [
    "A timber floor joist spans {L} m between two walls and is simply supported at both ends. It is designed for "
    "a uniformly distributed load over its whole span, giving a maximum bending moment of {M} kN·m.",
    "A simply supported footbridge girder of span {L} m carries a uniformly distributed load along its whole "
    "length. The largest bending moment in the girder is {M} kN·m.",
    "A lintel of span {L} m rests on two brick piers and carries a uniformly distributed load over its full "
    "span. Its maximum bending moment is {M} kN·m.",
    "A steel purlin of span {L} m is simply supported at its ends and carries a uniformly distributed roof "
    "load. The maximum bending moment it must resist is {M} kN·m.",
]


def _slips(correct: int, slips: list[tuple[Fraction, str]]) -> list[Distractor]:
    out, seen = [], {str(correct)}
    for value, tag in slips:
        if value <= 0 or value.denominator != 1 or str(int(value)) in seen:
            continue  # offered only when a whole number, like the key
        seen.add(str(int(value)))
        out.append(Distractor(str(int(value)), tag, int(value)))
    return out


def _draw(rng: random.Random):
    L = rng.choice([4, 6, 8, 10, 12, 14])
    w = rng.choice(range(2, 21, 2))
    x = rng.choice([v for v in range(1, L) if 2 * v != L])
    return L, w, x


def _build(rng: random.Random) -> dict | None:
    """One candidate question, or None when this draw leaves too few distinct wrong answers."""
    L, w, x = draw_until(rng, _draw, lambda t: True, what="a beam draw")
    M = Fraction(w * L * L, 8)
    check(M.denominator == 1, "design moment is not whole")
    M = int(M)
    shear = w * L // 2
    section = Fraction(w * x * (L - x), 2)  # R*x - w x^2/2 with R = wL/2
    check(section == Fraction(w * L, 2) * x - Fraction(w * x * x, 2), "section moment routes disagree")
    check(section.denominator == 1, "section moment is not whole")
    section = int(section)

    stem = rng.choice(_STEMS).format(L=L, M=M)
    cl = Fraction(L)
    a_dist = _slips(
        w,
        [
            (M / (cl * cl) * 2, "wrong-udl-moment-formula"),  # M = wL^2/2
            (M / (cl * cl) * 12, "wrong-udl-moment-formula"),  # M = wL^2/12
            (Fraction(8 * M) / cl, "forgot-to-square-span"),  # M = wL/8
        ],
    )
    b_dist = _slips(
        shear,
        [
            (Fraction(w * L), "used-total-load-as-shear"),
            (Fraction(w * L * L, 8), "confused-shear-with-moment"),
        ],
    )
    c_dist = _slips(
        section,
        [
            (Fraction(M), "gave-maximum-for-section"),
            (Fraction(w * x * x, 2), "section-moment-without-reaction"),
            (Fraction(w * L * x, 2), "section-moment-ignores-udl"),
        ],
    )
    if not all(len(dists) >= 2 for dists in (a_dist, b_dist, c_dist)):
        return None

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the intensity of the load, in kN/m.",
                answer_kind="number",
                answer=w,
                answer_text=str(w),
                answer_latex=latex_answer(str(w)),
                marks=2,
                sensible_range=(0, None),
                working=[f"M_max = wL²/8, so w = 8M/L² = 8 × {M} / {L}² = {w} kN/m."],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence find the maximum shear force in the beam, in kN.",
                answer_kind="number",
                answer=shear,
                answer_text=str(shear),
                answer_latex=latex_answer(str(shear)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a*{L}/2",
                working=[f"V_max = wL/2 = {w} × {L} / 2 = {shear} kN, at each support."],
                distractors=b_dist,
            ),
            dict(
                prompt=f"Find the bending moment at a section {x} m from one support, in kN·m.",
                answer_kind="number",
                answer=section,
                answer_text=str(section),
                answer_latex=latex_answer(str(section)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{L}/2*{x} - part_a*{x}**2/2",
                working=[
                    f"M = Rx − wx²/2 = {shear} × {x} − {w} × {x}² / 2 = {section} kN·m.",
                ],
                distractors=c_dist,
            ),
        ],
    )


@generator("exd_x_u7_udl_design_chain", level=20, presets={"exam": dict(tier="exam")})
def exd_x_u7_udl_design_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: see the module docstring."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "a draw with enough distinct wrong answers"):
        question = _build(rng)
        if question is not None:
            return question
    raise AssertionError("unreachable")
