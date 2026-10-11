"""Exam-tier binomial questions: an event built from several point probabilities.

The practice tier asks for one point probability ``P(X = x)`` from ``X ~ Bin(n, p)``. The exam tier asks for an
event that is not a single value, so the learner has to decide which point probabilities to combine (or whether
to use the complement):

* ``at_least``: ``P(X >= m)``, best found as ``1 - P(X <= m - 1)`` when ``m`` is small.
* ``between``: ``P(a <= X <= b)``, a sum of several point probabilities.
* ``given``: ``P(X = x | X >= 1)``, a conditional probability whose denominator is itself a complement.

Probabilities are held as exact fractions and rounded half-up to 4 decimal places only at the end; a draw is
rejected when the exact value sits close to a rounding boundary. The exact value is re-derived by adding the
probability of every success/failure sequence that satisfies the event, an independent route that uses no
binomial coefficient.
"""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import product
from math import comb

from ..core import (
    check,
    generator,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_P_CHOICES = (
    Fraction(1, 5),
    Fraction(1, 4),
    Fraction(3, 10),
    Fraction(2, 5),
    Fraction(1, 2),
    Fraction(3, 5),
    Fraction(3, 4),
)

_CONTEXTS = (
    (
        "a multiple-choice quiz has {n} questions and each is answered by guessing, with probability {p} of being correct",
        "correct answers",
    ),
    ("a machine makes {n} items in a batch and each item is independently faulty with probability {p}", "faulty items"),
    ("a player takes {n} shots and scores each independently with probability {p}", "scored shots"),
    ("{n} seeds are planted and each independently germinates with probability {p}", "seeds that germinate"),
)


def _pmf(n: int, p: Fraction, k: int) -> Fraction:
    return Fraction(comb(n, k)) * p**k * (1 - p) ** (n - k)


def _prob_by_sequences(n: int, p: Fraction, event) -> Fraction:
    """Add up the probability of every success/failure sequence of length n whose success count satisfies `event`."""
    total = Fraction(0)
    for outcome in product((0, 1), repeat=n):
        successes = sum(outcome)
        if event(successes):
            weight = Fraction(1)
            for o in outcome:
                weight *= p if o else 1 - p
            total += weight
    return total


def _dp4(value: Fraction) -> Fraction:
    return round_half_up(value, 4)


def _show(value: Fraction) -> str:
    return stated_precision_string(_dp4(value), 4)


def _decimal(p: Fraction) -> str:
    return f"{float(p):g}"


def _draw(rng: random.Random) -> tuple[int, Fraction, str, str]:
    n = rng.randint(5, 9)
    p = rng.choice(_P_CHOICES)
    setting, word = rng.choice(_CONTEXTS)
    return n, p, setting.format(n=n, p=_decimal(p)), word


def _build(rng: random.Random) -> dict:
    while True:
        n, p, setting, word = _draw(rng)
        kind = rng.choice(("at_least", "between", "given"))
        if kind == "at_least":
            m = rng.randint(2, n - 1)
            event = lambda s, m=m: s >= m  # noqa: E731
            exact = sum(_pmf(n, p, k) for k in range(m, n + 1))
            ask = f"the number of {word} is at least {m}"
            cands = [
                _pmf(n, p, m),  # only the boundary value
                sum(_pmf(n, p, k) for k in range(m + 1, n + 1)),  # P(X > m), boundary left out
                sum(_pmf(n, p, k) for k in range(0, m)),  # the opposite tail
            ]
            how = f"P(X >= {m}) = 1 - P(X <= {m - 1}) = {_show(exact)}"
        elif kind == "between":
            a = rng.randint(1, n - 3)
            b = rng.randint(a + 2, n - 1)
            event = lambda s, a=a, b=b: a <= s <= b  # noqa: E731
            exact = sum(_pmf(n, p, k) for k in range(a, b + 1))
            ask = f"the number of {word} is between {a} and {b}, inclusive"
            cands = [
                _pmf(n, p, a) + _pmf(n, p, b),  # the two end values only
                sum(_pmf(n, p, k) for k in range(a + 1, b)),  # both ends left out
                sum(_pmf(n, p, k) for k in range(0, b + 1)),  # no lower bound
            ]
            how = f"P({a} <= X <= {b}) = sum of the point probabilities from {a} to {b} = {_show(exact)}"
        else:
            x = rng.randint(1, n - 1)
            event = None
            exact = _pmf(n, p, x) / (1 - _pmf(n, p, 0))
            ask = None
            cands = [
                _pmf(n, p, x),  # the condition ignored
                1 - _pmf(n, p, 0),  # the probability of the condition itself
                _pmf(n, p, x) / _pmf(n, p, 0),  # divides by the wrong probability
            ]
            how = f"P(X = {x} | X >= 1) = P(X = {x}) / (1 - P(X = 0)) = {_show(exact)}"
        scaled = float(exact * 10000)
        if abs(scaled - int(scaled) - 0.5) < 0.02:
            continue  # too close to a rounding boundary
        answer_text = _show(exact)
        wrong = [_show(c) for c in cands if 0 < _dp4(c) < 1]
        distractors = pick_distractors(answer_text, wrong, tolerance=stated_precision_tolerance(4) * 4)
        if distractors:
            break

    if kind == "given":
        check(
            _prob_by_sequences(n, p, lambda s: s == x) / _prob_by_sequences(n, p, lambda s: s >= 1) == exact,
            "conditional probability disagrees with the sequence sum",
        )
        text = phrase(
            rng,
            (
                "Suppose {setting}. Given that the number of {word} is at least 1, find the probability that "
                "the number of {word} is exactly {x}. Give your answer to 4 decimal places.",
                "{setting_cap}. Find the probability that the number of {word} is exactly {x}, given that "
                "it is at least 1. Give your answer correct to 4 decimal places.",
            ),
            setting=setting,
            setting_cap=setting[0].upper() + setting[1:],
            word=word,
            x=x,
        )
    else:
        check(_prob_by_sequences(n, p, event) == exact, "probability disagrees with the sequence sum")
        text = phrase(
            rng,
            (
                "Suppose {setting}. Find the probability that {ask}. Give your answer to 4 decimal places.",
                "{setting_cap}. What is the probability that {ask}? Give your answer correct to 4 decimal places.",
            ),
            setting=setting,
            setting_cap=setting[0].upper() + setting[1:],
            ask=ask,
        )
    return dict(
        text=text,
        latex=text,
        answer=float(_dp4(exact)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"With X ~ Bin({n}, {_decimal(p)}): {how}."],
        distractors=distractors,
    )


@generator(
    "exd_x_h2_binomial_point_probability",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_h2_binomial_point_probability(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier binomial probability: a range, a tail or a conditional event, to 4 decimal places."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
