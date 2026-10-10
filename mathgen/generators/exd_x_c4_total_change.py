"""Exam source for total change by integrating a rate: the rate has an unknown constant.

The prompt gives the rate as ``r(t) = k * t^p`` with ``k`` unknown, and the total change over a first interval
``[0, a]``. The learner must use that to recover ``k`` and then integrate over a different interval ``[a, b]``.
That is two ideas composed (inverse use of a definite integral, then a forward one), not a bigger number range.

Built backwards from the answer: ``p``, ``a``, ``b`` and a multiplier ``m`` are drawn first and everything else is
derived with exact arithmetic, so the key is exact for every seed.

    P = p + 1 = num/den (in lowest terms)
    k = m * num                      (an integer)
    N = k * a^P / P = m * den * a^P  (the stated first-interval change)
    answer = m * den * (b^P - a^P)

For ``p = 1/2`` the end points ``a`` and ``b`` are perfect squares so every power is an integer.
"""

from __future__ import annotations

import random

from sympy import Integer, Rational, Symbol, integrate, sqrt

from ..core import check, check_distinct, check_equivalent, generator, latex_prompt, pick_distractors

_T = Symbol("t", positive=True)
_K = Symbol("k", positive=True)

_CONTEXTS = [
    ("the volume of water in a reservoir", "megalitres per hour", "hour", "hours"),
    ("the number of bacteria in a culture", "thousands per day", "day", "days"),
    ("the total cost of running a plant", "dollars per month", "month", "months"),
    ("the mass of ice on a glacier", "tonnes per year", "year", "years"),
    ("the charge stored in a capacitor", "coulombs per second", "second", "seconds"),
]

_EXPONENTS = [Integer(1), Integer(2), Integer(3), Rational(1, 2)]


def _rate_text(p) -> str:
    return "k·sqrt(t)" if p == Rational(1, 2) else (f"k·t^{p}" if p != 1 else "k·t")


def _pow(base: int, expo: Rational) -> Rational:
    """base ** expo for an exact rational exponent, with base a perfect square when expo has denominator 2."""
    return Rational(base) ** expo


def _build(rng: random.Random) -> dict:
    p = rng.choice(_EXPONENTS)
    big_p = p + 1
    num, den = big_p.p, big_p.q
    m = rng.randint(1, 4)
    if p == Rational(1, 2):
        s = rng.randint(1, 3)
        u = rng.randint(s + 1, s + 3)
        a, b = s * s, u * u
    else:
        a = rng.randint(1, 4)
        b = a + rng.randint(2, 4)
    k = m * num
    n_first = m * den * _pow(a, big_p)
    answer = m * den * (_pow(b, big_p) - _pow(a, big_p))

    rate = k * _T**p
    check_equivalent(integrate(rate, (_T, 0, a)), n_first)
    check_equivalent(integrate(rate, (_T, a, b)), answer)

    full_b = n_first + answer  # change over [0, b]: the learner counted the first interval again
    candidates = [
        n_first * Rational(b - a, a),  # assumed a constant rate over the first interval
        full_b,  # integrated from 0 to b, forgetting to start at a
        n_first * (_pow(b, big_p) - _pow(a, big_p)) / big_p,  # used the stated change N as the constant k
        k * (_pow(b, big_p) - _pow(a, big_p)),  # integrated but did not divide by the new power
    ]
    for c in candidates:
        check_distinct(answer, c)
    answer_text = str(answer)
    # keep the options one shape: whole numbers only, like the key
    distractors = pick_distractors(answer_text, [str(c) for c in candidates if Rational(c).q == 1])

    what, unit, tone, tmany = rng.choice(_CONTEXTS)
    tunit = tone if a == 1 else tmany
    if rng.random() < 0.5:
        text = (
            f"The rate of change of {what} is r(t) = {_rate_text(p)} ({unit}), where k is a positive constant and t is "
            f"measured in {tmany}. The total change during the first {a if a > 1 else ''} {tunit} is {n_first}. "
            f"Find the total change between t = {a} and t = {b}."
        )
    else:
        text = (
            f"For t >= 0, {what} changes at the rate r(t) = {_rate_text(p)} ({unit}) for some constant k. "
            f"Between t = 0 and t = {a} the quantity increases by {n_first}. "
            f"By how much does it increase between t = {a} and t = {b}?"
        )
    text = text.replace("first  ", "first ")
    return {
        "text": text,
        "latex": latex_prompt(text),
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": [
            f"∫ from 0 to {a} of {_rate_text(p)} dt = {n_first}, so k = {k}",
            f"Total change = ∫ from {a} to {b} of {k}·{_rate_text(p)[2:]} dt = {answer_text}",
        ],
    }


@generator("exd_x_c4_total_change", level=17, presets={"exam": dict(tier="exam")})
def exd_x_c4_total_change(rng: random.Random, tier: str = "exam") -> dict:
    """Total change over [a, b] when the rate's constant must first be recovered from the change over [0, a]."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
