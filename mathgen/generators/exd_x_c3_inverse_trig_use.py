"""Exam tier for using inverse trigonometric functions: compound angles built from arcsine, arccosine, arctangent.

The hard tier finds one ratio from another for a single angle. Here an angle is the sum of two inverse-trig values
(or the double of one), and the learner evaluates a trig function of it exactly: find the missing ratio of each
angle from its principal-value range, then apply the addition or double-angle formula. Arguments come from
Pythagorean triples, and some are negative, so the range of arccos (which gives a positive sine) matters.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, pick_distractors, rational_text
from .exd_x_c3_support import Redraw, draw

# (opposite, adjacent, hypotenuse) triples used for the rational sine/cosine pairs.
_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]


def _ratio(rng: random.Random, allow_negative: bool) -> tuple[Fraction, Fraction]:
    """(value, partner): `value` = p/c with a random sign and `partner` = sqrt(1 - value^2) >= 0, both rational."""
    a, b, c = rng.choice(_TRIPLES)
    num, other = (a, b) if rng.random() < 0.5 else (b, a)
    sign = -1 if allow_negative and rng.random() < 0.5 else 1
    return Fraction(sign * num, c), Fraction(other, c)


def _sum_of_inverse_trig(rng: random.Random) -> dict:
    """sin(arcsin u + arccos v) = u v + sqrt(1-u^2) sqrt(1-v^2)."""
    u, cu = _ratio(rng, True)
    v, sv = _ratio(rng, True)
    answer = u * v + cu * sv
    cands = [
        u * v - cu * sv,  # subtracts in the addition formula
        u + sv,  # sin(A) + sin(B)
        cu * v - u * sv,  # uses the cosine addition pattern
        u * v + cu * -sv,  # takes the negative root for sin(arccos v)
        cu * sv,  # forgets the first product
    ]
    text = f"Evaluate sin(arcsin({rational_text(u)}) + arccos({rational_text(v)})) exactly."
    return _pack(text, answer, cands)


def _double_angle(rng: random.Random) -> dict:
    """sin(2 arccos v) = 2 v sqrt(1 - v^2), where arccos lies in [0, pi] so sin(arccos v) >= 0."""
    v, sv = _ratio(rng, True)
    answer = 2 * v * sv
    cands = [
        -answer,  # takes the negative root for sin(arccos v)
        2 * v,  # forgets the sine factor
        2 * sv,  # forgets the cosine factor
        v * v - sv * sv,  # uses cos(2A) in place of sin(2A)
        v * sv,  # forgets the 2
    ]
    text = f"Evaluate sin(2·arccos({rational_text(v)})) exactly."
    return _pack(text, answer, cands)


def _tan_of_sum(rng: random.Random) -> dict:
    """tan(arcsin u + arctan t) = (tan A + t) / (1 - t tan A), with tan A = u / sqrt(1 - u^2)."""
    u, cu = _ratio(rng, True)
    t = Fraction(rng.choice([1, 2, 3, 4]), rng.choice([1, 2, 3, 5]))
    ta = u / cu
    denominator = 1 - ta * t
    # Every wrong-answer formula below divides by one of these, so none may vanish.
    if 0 in (denominator, ta, ta + t, 1 + ta * t, 1 - u * t):
        raise Redraw
    answer = (ta + t) / denominator
    cands = [
        (ta + t) / (1 + ta * t),  # sign slip: the tan(A - B) denominator
        ta + t,  # tan(A) + tan(B)
        (ta - t) / (1 + ta * t),  # tan(A - B)
        denominator / (ta + t),  # inverted
        (u + t) / (1 - u * t),  # uses sin A where tan A belongs
    ]
    text = f"Evaluate tan(arcsin({rational_text(u)}) + arctan({rational_text(t)})) exactly."
    return _pack(text, answer, cands)


def _pack(text: str, answer: Fraction, cands) -> dict:
    check(answer.denominator < 10**4, "unexpectedly large denominator")
    answer_text = rational_text(answer)
    distractors = pick_distractors(answer_text, [rational_text(c) for c in cands])
    if len(distractors) < 3:
        raise Redraw
    return dict(
        text=text,
        latex="",
        answer=answer_text,
        answer_text=answer_text,
        distractors=distractors,
    )


_KINDS = (_sum_of_inverse_trig, _double_angle, _tan_of_sum)


@generator("exd_x_c3_inverse_trig_use", level=11, presets={"exam": dict(tier="exam")})
def exd_x_c3_inverse_trig_use(rng: random.Random, tier: str = "exam") -> dict:
    """Evaluate a trig function of a sum or double of inverse-trig values exactly."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(rng.choice(_KINDS), rng)
