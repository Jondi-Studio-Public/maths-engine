"""Exam-tier source for `cantilever-beam-reactions`: a loaded cantilever taken from the reaction to a section.

The beam carries a uniformly distributed load over its whole length and a point load at the free end. The
question is a chain:

(a) the vertical reaction at the fixed end (both loads count);
(b) the fixing moment (the distributed load acts at its midpoint, the point load at the tip);
(c) the bending moment at a section a stated distance from the fixed end, from the support reaction and moment
    (the distributed load still carries on between the support and the section).

Built backwards from whole-number loads and lengths, so every answer is an exact integer. Part (c) is also the
moment of the loads beyond the section, which is how the test re-derives it independently.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, draw_until, generator, latex_answer, numeric_answer, pick_distractors

_STEMS = [
    "A steel awning beam projects {L} m from a brick wall, where it is rigidly fixed. It carries a uniformly "
    "distributed load of {w} kN/m over its whole length and a point load of {P} kN at its free end.",
    "A balcony beam is fixed into a concrete wall and projects {L} m. A uniformly distributed load of {w} kN/m "
    "acts along its whole length and a {P} kN point load acts at the free end.",
    "A cantilevered jetty beam, fixed at the shore end, is {L} m long. It supports a uniformly distributed load "
    "of {w} kN/m over its full length and a {P} kN point load at the outer tip.",
    "A cantilever crane arm {L} m long is fixed at its base. It carries its own weight as a uniformly "
    "distributed load of {w} kN/m and a {P} kN hanging load at the free end.",
]


def _slips(correct: Fraction, slips: list[tuple[Fraction, str]]) -> list[Distractor]:
    out, seen = [], {str(int(correct))}
    for value, tag in slips:
        if value <= 0 or value.denominator != 1:
            continue  # a wrong answer is offered only when it is a whole number, like the key
        text = str(int(value))
        if text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, int(value)))
    return out


def _draw(rng: random.Random):
    L = rng.randint(3, 8)
    w = rng.choice([2, 4, 6, 8])
    P = rng.choice([4, 5, 6, 8, 10, 12, 15, 20])
    d = rng.randint(1, L - 1)
    return L, w, P, d


def _build(rng: random.Random) -> dict | None:
    """One candidate question, or None when this draw leaves too few distinct wrong answers."""
    L, w, P, d = draw_until(rng, _draw, lambda t: True, what="a cantilever draw")
    reaction = w * L + P
    fixing = Fraction(w * L * L, 2) + P * L
    # Moment at the section a distance d from the support, from the loads beyond the section.
    x = L - d
    section = Fraction(w * x * x, 2) + P * x
    # Independent route: the support values carried back along the beam.
    check(section == fixing - reaction * d + Fraction(w * d * d, 2), "section moment disagrees with the support route")
    check(fixing.denominator == 1 and section.denominator == 1, "an answer is not whole")
    fixing, section = int(fixing), int(section)

    stem = rng.choice(_STEMS).format(L=L, w=w, P=P)
    total_udl = w * L

    a_dist = _slips(
        Fraction(reaction),
        [
            (Fraction(total_udl), "left-out-point-load"),
            (Fraction(total_udl, 2) + P, "cantilever-treated-as-simply-supported"),
        ],
    )
    b_dist = _slips(
        Fraction(fixing),
        [
            (Fraction(w * L * L, 2), "left-out-point-load"),
            (Fraction(w * L * L + P * L), "udl-lever-arm-full-length"),
            (Fraction(w * L * L, 8) + P * L, "cantilever-treated-as-simply-supported"),
        ],
    )
    c_dist = _slips(
        Fraction(section),
        [
            (Fraction(fixing), "section-moment-is-support-moment"),
            (Fraction(fixing) - reaction * d, "section-moment-ignores-udl"),
            (Fraction(w * x * x, 2), "left-out-point-load"),
        ],
    )
    if not all(len(dists) >= 2 for dists in (a_dist, b_dist, c_dist)):
        return None

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the vertical reaction at the fixed end, in kN.",
                answer_kind="number",
                answer=reaction,
                answer_text=str(reaction),
                answer_latex=latex_answer(str(reaction)),
                marks=2,
                sensible_range=(0, None),
                working=[f"R = wL + P = {w} × {L} + {P} = {reaction} kN."],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the magnitude of the fixing moment at the wall, in kN·m.",
                answer_kind="number",
                answer=fixing,
                answer_text=str(fixing),
                answer_latex=latex_answer(str(fixing)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"M = wL²/2 + PL = {w} × {L}² / 2 + {P} × {L} = {fixing} kN·m.",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=f"Hence find the magnitude of the bending moment at a section {d} m from the wall, in kN·m.",
                answer_kind="number",
                answer=numeric_answer(Fraction(section)),
                answer_text=str(section),
                answer_latex=latex_answer(str(section)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b - part_a*{d} + {w}*{d}**2/2",
                working=[
                    f"M = M_fixed − R × {d} + w × {d}²/2 = {fixing} − {reaction} × {d} + {w} × {d}² / 2 = {section} kN·m."
                ],
                distractors=c_dist,
            ),
        ],
    )


@generator("exd_x_u7_cantilever_chain", level=20, presets={"exam": dict(tier="exam")})
def exd_x_u7_cantilever_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: see the module docstring."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "a draw with enough distinct wrong answers"):
        question = _build(rng)
        if question is not None:
            return question
    raise AssertionError("unreachable")
