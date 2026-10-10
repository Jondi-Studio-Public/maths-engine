"""Generates `exd_x_s1_replacement` questions: the four two-draw colour outcomes when the first ball is NOT
returned.

A bag holds ``R`` balls of one colour and ``B`` of another (``T = R + B``). Two balls are drawn one after
the other without replacement, and the learner lists the probability of each of the four ordered outcomes in
a stated, shuffled order.  The draws are dependent, so the second factor changes with the first outcome.

The four probabilities are cross-checked two ways: they must sum to 1, and each must equal the count of
favourable ordered pairs of distinct balls (found by enumeration) over ``T * (T - 1)``.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

from ..core import check, generator, pairwise_distinct, rational_text

_COLOUR_PAIRS = [("red", "blue"), ("green", "yellow"), ("black", "white"), ("orange", "purple")]
_CODES = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]


def _enumerated(R: int, B: int) -> dict:
    """Count ordered pairs of distinct balls by colour: balls 0..R-1 are A, the rest B."""
    T = R + B
    colour = ["A"] * R + ["B"] * B
    counts = {pair: 0 for pair in _CODES}
    for i, j in itertools.permutations(range(T), 2):
        counts[(colour[i], colour[j])] += 1
    return {pair: Fraction(n, T * (T - 1)) for pair, n in counts.items()}


@generator(
    "exd_x_s1_replacement",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_s1_replacement(rng: random.Random, tier: str = "exam") -> dict:
    """Two draws without replacement: list the four ordered-colour probabilities."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(400):
        colour_a, colour_b = rng.choice(_COLOUR_PAIRS)
        names = {"A": colour_a, "B": colour_b}
        T = rng.randint(7, 12)
        R = rng.randint(2, T - 2)
        B = T - R
        if R == B:
            continue
        order = list(_CODES)
        rng.shuffle(order)
        if order == _CODES:
            continue
        probs = {
            ("A", "A"): Fraction(R, T) * Fraction(R - 1, T - 1),
            ("A", "B"): Fraction(R, T) * Fraction(B, T - 1),
            ("B", "A"): Fraction(B, T) * Fraction(R, T - 1),
            ("B", "B"): Fraction(B, T) * Fraction(B - 1, T - 1),
        }
        check(probs == _enumerated(R, B), "enumeration disagrees with the sequential product")
        check(sum(probs.values()) == 1, "outcome probabilities do not sum to 1")
        p, q = Fraction(R, T), Fraction(B, T)
        stay = {"A": p, "B": q}
        answer_values = [probs[pair] for pair in order]
        # the first ball is treated as returned: both draws use the original proportions
        with_return = [stay[x] * stay[y] for x, y in order]
        # the second count drops by one for a matching colour but the total is left at T
        wrong_total = [stay[x] * Fraction((R if y == "A" else B) - (1 if x == y else 0), T) for x, y in order]
        # the two colours' counts are mixed up throughout
        swapped_counts = {"A": B, "B": R}
        swapped_values = [
            Fraction(swapped_counts[x], T) * Fraction(swapped_counts[y] - (1 if x == y else 0), T - 1) for x, y in order
        ]
        lists = [answer_values, with_return, wrong_total, swapped_values]
        if not pairwise_distinct(*lists):
            continue
        if any(not (0 <= v <= 1) for lst in lists for v in lst):
            continue
        labels = [f"{names[x]}-then-{names[y]}" for x, y in order]
        text = (
            f"A bag contains {R} {colour_a} balls and {B} {colour_b} balls, {T} in total. "
            "Two balls are drawn at random one after the other, and the first ball is not returned to the bag. "
            f"List the probability of each of these four outcomes, in this order: {', '.join(labels)}."
        )
        fmt = lambda vals: ", ".join(rational_text(v) for v in vals)  # noqa: E731
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=answer_values,
            answer_text=fmt(answer_values),
            answer_latex=fmt(answer_values),
            working=[
                "The second draw is made from T-1 balls, with the count of the first colour reduced by one.",
                f"In stated order: {fmt(answer_values)}.",
            ],
            distractors=[fmt(with_return), fmt(wrong_total), fmt(swapped_values)],
        )
    raise RuntimeError("could not build an exd_x_s1_replacement question")
