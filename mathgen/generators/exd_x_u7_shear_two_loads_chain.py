"""Exam-tier source for `max-shear-force-udl`: both reactions of a beam with a distributed load and a point load.

A simply supported beam carries a uniformly distributed load over its whole span and one point load that is not at
midspan. The question is a chain:

(a) the reaction at the far support B, from moments about A (both loads count);
(b) the reaction at A, from vertical equilibrium with the total load;
(c) the maximum shear force, which is the larger of the two reactions: the learner must choose.

The point load sits closer to A in some draws and closer to B in others, so the larger reaction is sometimes at A
and sometimes at B. Built by rejection from whole-number loads so that both reactions are whole.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, draw_until, generator, latex_answer

_STEMS = [
    "A simply supported timber bridge beam spans {L} m between supports A and B. It carries a uniformly "
    "distributed load of {w} kN/m over its whole span and a point load of {P} kN placed {a} m from A.",
    "A simply supported steel girder AB has a span of {L} m. A uniformly distributed load of {w} kN/m acts over "
    "the full span, and a {P} kN point load acts {a} m from end A.",
    "A freight platform beam rests on supports A and B, {L} m apart. It carries {w} kN/m uniformly over its "
    "whole length and a {P} kN load at a point {a} m from A.",
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
    L = rng.choice([4, 5, 6, 8, 10])
    w = rng.choice([2, 3, 4, 5, 6, 8])
    P = rng.choice(range(6, 61, 2))
    a = rng.randint(1, L - 1)
    return L, w, P, a


def _usable(t) -> bool:
    L, w, P, a = t
    rb = Fraction(w * L * L, 2) + P * a
    ra = Fraction(w * L + P) - rb / L
    return rb % L == 0 and 2 * a != L and ra != rb / L


def _build(rng: random.Random) -> dict | None:
    """One candidate question, or None when this draw leaves too few distinct wrong answers."""
    L, w, P, a = draw_until(rng, _draw, _usable, what="a beam with whole-number reactions")
    total = w * L + P
    rb = (Fraction(w * L * L, 2) + P * a) / L
    ra = total - rb
    # Independent route: moments about B give R_A directly.
    check(ra == (Fraction(w * L * L, 2) + P * (L - a)) / L, "the two moment routes disagree")
    check(rb.denominator == 1 and ra.denominator == 1, "reactions are not whole")
    rb, ra = int(rb), int(ra)
    vmax = max(ra, rb)

    stem = rng.choice(_STEMS).format(L=L, w=w, P=P, a=a)
    b_arm = L - a
    # Slips: the point load left out, the moment arm measured from the wrong end (which gives R_A), equal shares.
    a_dist = _slips(
        rb,
        [
            (Fraction(w * L, 2), "left-out-point-load"),
            ((Fraction(w * L * L, 2) + P * b_arm) / L, "swapped-support-reactions"),
            (Fraction(total, 2), "equal-reactions-assumed"),
        ],
    )
    b_dist = _slips(
        ra,
        [
            (Fraction(w * L, 2), "left-out-point-load"),
            (Fraction(rb), "swapped-support-reactions"),
            (Fraction(total, 2), "equal-reactions-assumed"),
        ],
    )
    c_dist = _slips(
        vmax,
        [
            (Fraction(min(ra, rb)), "picked-smaller-reaction"),
            (Fraction(total), "used-total-load-as-shear"),
            (Fraction(w * L, 2), "left-out-point-load"),
        ],
    )
    if not all(len(dists) >= 2 for dists in (a_dist, b_dist, c_dist)):
        return None

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Take moments about A to find the reaction at B, in kN.",
                answer_kind="number",
                answer=rb,
                answer_text=str(rb),
                answer_latex=latex_answer(str(rb)),
                marks=2,
                sensible_range=(0, None),
                working=[f"R_B × {L} = {w} × {L} × {L}/2 + {P} × {a}, so R_B = {rb} kN."],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence find the reaction at A, in kN.",
                answer_kind="number",
                answer=ra,
                answer_text=str(ra),
                answer_latex=latex_answer(str(ra)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{w * L + P} - part_a",
                working=[f"R_A = {w} × {L} + {P} − R_B = {total} − {rb} = {ra} kN."],
                distractors=b_dist,
            ),
            dict(
                prompt="Find the magnitude of the maximum shear force in the beam, in kN.",
                answer_kind="number",
                answer=vmax,
                answer_text=str(vmax),
                answer_latex=latex_answer(str(vmax)),
                marks=1,
                sensible_range=(0, None),
                follow_through="Max(part_a, part_b)",
                working=[f"The shear is greatest at a support, so V_max is the larger of {ra} and {rb}: {vmax} kN."],
                distractors=c_dist,
            ),
        ],
    )


@generator("exd_x_u7_shear_two_loads_chain", level=20, presets={"exam": dict(tier="exam")})
def exd_x_u7_shear_two_loads_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: see the module docstring."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "a draw with enough distinct wrong answers"):
        question = _build(rng)
        if question is not None:
            return question
    raise AssertionError("unreachable")
