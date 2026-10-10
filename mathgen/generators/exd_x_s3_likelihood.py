"""Exam-tier source for `probability_likelihood_compare`: event A is a two-draw compound event whose probability
the learner has to work out, and event B is given as a probability. The learner then compares them.

Event A is one of five two-draw events from a bag of red and blue counters, with or without replacement:

* both red (with replacement)
* both red (without replacement)
* both the same colour (without replacement)
* at least one red (with replacement)
* exactly one red (without replacement)

Each needs a different method (independent product, conditional product, complement, or a sum over orderings), so
the learner must choose it. The key is computed with exact fractions from the formula, then re-derived by
enumerating every ordered pair of labelled counters. Event B is either an equivalent fraction (so "equally likely"
occurs), or a fraction close to P(A) on either side, so neither a quick estimate nor a common-denominator slip is
safe.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

from ..core import check, generator, latex_answer, pick_distractors
from .p4d_outcomes_likelihood_build import OPTION_A_MORE, OPTION_B_MORE, OPTION_EQUAL, OPTIONS_B

# (event wording, with replacement?, predicate on the two colours)
_EVENTS = (
    ("both counters are red", True, lambda c1, c2: c1 == "R" and c2 == "R"),
    ("both counters are red", False, lambda c1, c2: c1 == "R" and c2 == "R"),
    ("the two counters are the same colour", False, lambda c1, c2: c1 == c2),
    ("at least one counter is red", True, lambda c1, c2: "R" in (c1, c2)),
    ("exactly one counter is red", False, lambda c1, c2: (c1 == "R") != (c2 == "R")),
)


def _formula(idx: int, r: int, b: int) -> Fraction:
    n = r + b
    if idx == 0:
        return Fraction(r, n) ** 2
    if idx == 1:
        return Fraction(r * (r - 1), n * (n - 1))
    if idx == 2:
        return Fraction(r * (r - 1) + b * (b - 1), n * (n - 1))
    if idx == 3:
        return 1 - Fraction(b, n) ** 2
    return Fraction(2 * r * b, n * (n - 1))


def _enumerate(idx: int, r: int, b: int) -> Fraction:
    """Count favourable ordered draws over all ordered draws of labelled counters."""
    _, replace, pred = _EVENTS[idx]
    counters = ["R"] * r + ["B"] * b
    labelled = list(enumerate(counters))
    pairs = itertools.product(labelled, repeat=2) if replace else itertools.permutations(labelled, 2)
    total = good = 0
    for (_, c1), (_, c2) in pairs:
        total += 1
        good += bool(pred(c1, c2))
    return Fraction(good, total)


@generator(
    "exd_x_s3_likelihood",
    level=11,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s3_likelihood(rng: random.Random, tier: str = "exam"):
    """Compare a two-draw compound event with an event of stated probability."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    idx = rng.randrange(len(_EVENTS))
    r, b = rng.randint(2, 7), rng.randint(2, 7)
    p_a = _formula(idx, r, b)
    check(p_a == _enumerate(idx, r, b), "formula disagrees with enumeration of draws")
    check(0 < p_a < 1, "event A must be neither impossible nor certain")

    if rng.random() < 0.3 and p_a.denominator <= 20:
        m = rng.choice([2, 3])
        num_b, den_b = p_a.numerator * m, p_a.denominator * m
    else:
        den_b = rng.choice([10, 12, 15, 20, 25, 30, 36, 40, 45, 50])
        num_b = round(p_a * den_b) + rng.choice([-1, 0, 1])
        num_b = min(max(num_b, 1), den_b - 1)

    p_b = Fraction(num_b, den_b)
    key = OPTION_A_MORE if p_a > p_b else OPTION_B_MORE if p_a < p_b else OPTION_EQUAL
    # Cross-multiplication on raw integers as the independent check.
    lhs, rhs = p_a.numerator * p_b.denominator, p_b.numerator * p_a.denominator
    check((key == OPTION_A_MORE) == (lhs > rhs) and (key == OPTION_EQUAL) == (lhs == rhs), "comparison disagrees")

    wording, replace, _ = _EVENTS[idx]
    how = "with replacement" if replace else "without replacement"
    text = (
        f"A bag holds {r} red and {b} blue counters. Two counters are drawn at random, one after the other, "
        f"{how}. Event A is that {wording}. Event B has probability {num_b}/{den_b}. "
        f"Which of the following is true?"
    )
    working = [
        f"P(A) = {p_a.numerator}/{p_a.denominator}.",
        f"Compare {p_a.numerator}/{p_a.denominator} with {num_b}/{den_b} by cross-multiplication: {lhs} and {rhs}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        working=working,
        distractors=pick_distractors(key, [o for o in OPTIONS_B if o != key], n=2),
    )
