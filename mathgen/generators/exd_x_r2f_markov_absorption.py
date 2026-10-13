"""Exam source for "hitting probabilities of a Markov chain": a four-part chain by first-step analysis.

A particle moves on the states 0, 1, 2, 3. States 0 and 3 are absorbing. From state 1 it moves up (to 2) with probability
p1, down (to 0) with probability q1 and stays with probability s1; from state 2 it moves up (to 3) with probability p2,
down (to 1) with probability q2 and stays with probability s2. Write r1 = p1 / (p1 + q1) and r2 = p2 / (p2 + q2), the
probabilities that a move away from state 1 (or 2) is the upward one. With h_i = P(absorbed at 3 | start in state i):

(a) r1                                        the first move away from state 1 is upward         fraction, 2 marks
(b) h2 = r2 / (1 - (1 - r2) r1)               first-step analysis from state 2                   fraction, 3 marks, from (a)
(c) h1 = r1 h2                                                                                    fraction, 1 mark,  from (a), (b)
(d) (h1 + h2) / 2                             the walk starts in state 1 or 2 with equal chance    fraction, 2 marks, from (b), (c)

Staying put never changes where the walk ends up, so only the probabilities of the other two moves matter: that is the
idea (a) isolates. r1 and r2 are drawn from different rows, so the answers do not collapse to a single parameter.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Rational

from ..core import attempts, check, generator, rational_text
from .exd_x_r2f_common import frac_part

_DENOMINATORS = (6, 8, 9, 10, 12)


def _row(rng: random.Random):
    """(up, down, stay) as numerators over a common denominator: all positive, summing to it."""
    d = rng.choice(_DENOMINATORS)
    up = rng.randint(1, d - 2)
    down = rng.randint(1, d - 1 - up)
    return up, down, d - up - down, d


def _prob(numerator: int, d: int) -> str:
    g = Fraction(numerator, d)
    return rational_text(Rational(g.numerator, g.denominator))


@generator(
    "exd_x_r2f_markov_absorption",
    level=17,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_r2f_markov_absorption(rng: random.Random, tier: str = "exam") -> dict:
    """The escape probability from state 1, then the absorption probabilities from states 2 and 1, then an average."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "two rows with distinct, non-trivial answers"):
        p1, q1, s1, d1 = _row(rng)
        p2, q2, s2, d2 = _row(rng)
        r1, r2 = Fraction(p1, p1 + q1), Fraction(p2, p2 + q2)
        h2 = r2 / (1 - (1 - r2) * r1)
        h1 = r1 * h2
        mean = (h1 + h2) / 2
        values = (r1, h2, h1, mean)
        if (
            len(set(values)) == 4
            and max(v.denominator for v in values) <= 400
            and Fraction(p1, d1) != Fraction(p1, p1 + q1)
        ):
            break

    # Independent route: solve the two first-step equations as a linear system with exact arithmetic.
    #   h1 (1 - s1) = p1 h2          (1 - s1 = (d1 - s1) / d1 in the common denominator)
    #   h2 (1 - s2) = p2 + q2 h1
    a11, a12, b1 = Fraction(d1 - s1, d1), Fraction(-p1, d1), Fraction(0)
    a21, a22, b2 = Fraction(-q2, d2), Fraction(d2 - s2, d2), Fraction(p2, d2)
    det = a11 * a22 - a12 * a21
    check((b1 * a22 - a12 * b2) / det == h1 and (a11 * b2 - b1 * a21) / det == h2, "linear system disagrees")
    check(0 < h1 < h2 < 1, "absorption probabilities out of range")

    stem = (
        "A particle moves on the states 0, 1, 2 and 3, and states 0 and 3 are absorbing. In one step from state 1 it moves "
        f"to state 2 with probability {_prob(p1, d1)}, to state 0 with probability {_prob(q1, d1)}, and otherwise stays "
        f"in state 1. In one step from state 2 it moves to state 3 with probability {_prob(p2, d2)}, to state 1 with "
        f"probability {_prob(q2, d2)}, and otherwise stays in state 2."
    )
    r2_text = rational_text(Rational(r2.numerator, r2.denominator))
    a_part = frac_part(
        "Starting in state 1, find the probability that the first move away from state 1 is to state 2.",
        Rational(r1.numerator, r1.denominator),
        2,
        [
            f"Staying put does not matter. P(up) ÷ (P(up) + P(down)) = {_prob(p1, d1)} ÷ ({_prob(p1 + q1, d1)}) = {rational_text(Rational(r1.numerator, r1.denominator))}."
        ],
        [
            ("ignores-holding-probability", Rational(p1, d1)),
            ("swaps-up-and-down", Rational((1 - r1).numerator, (1 - r1).denominator)),
            ("odds-for-probability", Rational(p1, q1)),
            ("gives-the-complement", Rational(d1 - p1, d1)),
        ],
        sensible_range=(0, 1),
    )
    r1_value = Rational(r1.numerator, r1.denominator)
    r2_value = Rational(r2.numerator, r2.denominator)
    b_part = frac_part(
        "Let h₂ be the probability that the particle is eventually absorbed in state 3, starting from state 2. Find h₂.",
        Rational(h2.numerator, h2.denominator),
        3,
        [
            f"From state 2 the next move away is up with probability {r2_text} and down with probability {rational_text(1 - r2_value)}.",
            f"h₂ = {r2_text} + {rational_text(1 - r2_value)} × h₁ and h₁ = {rational_text(r1_value)} h₂, so "
            f"h₂ = {r2_text} ÷ (1 − {rational_text(1 - r2_value)} × {rational_text(r1_value)}) = {rational_text(Rational(h2.numerator, h2.denominator))}.",
        ],
        [
            ("ignores-the-return-path", r2_value),
            ("stops-after-one-loop", r2_value + (1 - r2_value) * r1_value),
            ("wrong-loop-probability", r2_value / (1 - r2_value * r1_value)),
            ("drops-later-step-probability", r1_value * r2_value),
        ],
        follow_through=f"({r2_text})/(1 - (1 - ({r2_text}))*part_a)",
        sensible_range=(0, 1),
    )
    c_part = frac_part(
        "Hence find h₁, the probability of absorption in state 3 starting from state 1.",
        Rational(h1.numerator, h1.denominator),
        1,
        [f"h₁ = {rational_text(r1_value)} × h₂ = {rational_text(Rational(h1.numerator, h1.denominator))}."],
        [
            ("drops-later-step-probability", r1_value),
            ("confuses-start-states", Rational(h2.numerator, h2.denominator)),
            ("gives-the-complement", 1 - Rational(h1.numerator, h1.denominator)),
            ("ignores-the-return-path", r1_value * r2_value),
        ],
        follow_through="part_a*part_b",
        sensible_range=(0, 1),
    )
    d_part = frac_part(
        "The particle is equally likely to start in state 1 or in state 2. Find the probability that it is eventually absorbed in state 3.",
        Rational(mean.numerator, mean.denominator),
        2,
        [
            f"(h₁ + h₂) ÷ 2 = ({rational_text(Rational(h1.numerator, h1.denominator))} + {rational_text(Rational(h2.numerator, h2.denominator))}) ÷ 2 = {rational_text(Rational(mean.numerator, mean.denominator))}."
        ],
        [
            ("forgets-to-average", Rational((h1 + h2).numerator, (h1 + h2).denominator)),
            ("confuses-start-states", Rational(h2.numerator, h2.denominator)),
            ("confuses-start-states", Rational(h1.numerator, h1.denominator)),
            ("drops-later-step-probability", Rational((h1 * h2).numerator, (h1 * h2).denominator)),
        ],
        follow_through="(part_b + part_c)/2",
        sensible_range=(0, 1),
    )
    return dict(text=stem, parts=[a_part, b_part, c_part, d_part])
