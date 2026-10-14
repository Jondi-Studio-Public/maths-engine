"""Exam chain for `f10-compound-outcomes-replacement`: two draws with replacement, then the effect of not replacing.

The practice tiers list the outcomes of a two-stage event and their probabilities. The exam question builds one
connected argument on a container of two colours, A and B:

(a) the probability of one named order, first A then B                    fraction, 2 marks
(b) the probability that exactly one is colour A                          fraction, 1 mark, follow-through (a)
(c) the probability that at least one is colour A                         fraction, 2 marks, follow-through (b)
(d) how much that probability rises if the first item is not put back    fraction, 2 marks, follow-through (c)

Part (b) uses the symmetry of drawing with replacement (the two mixed orders are equally likely). Part (c) adds the
outcome with both A (which the learner finds from the counts). Part (d) needs the same event without replacement,
built from the changed counts.

## Construction

The counts ``a`` and ``b`` are different whole numbers with a total of 7 to 20. Every part is recomputed by listing
all ``n * n`` ordered draws (with replacement) and all ``n * (n - 1)`` ordered draws (without) and counting, an
independent route from the formulas. The rise in (d) is always positive, because removing replacement makes two As
less likely to be avoided and so "no A" less likely.

## Distractors

(a) the draw without replacement, the same colour twice, one denominator in place of the square.
(b) one order only, the same colour twice, the draw without replacement.
(c) the answer to (b), the probability of no A, the probability of both A.
(d) the without-replacement probability itself, the answer to (c) itself, the reduced denominator on only one factor.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_r2x_kit import frac_part

_SCENES = (
    "A bag holds {a} {ca} counters and {b} {cb} counters. Two counters are drawn at random, one after the other, and "
    "the first is put back in the bag before the second is drawn.",
    "A jar contains {a} {ca} jelly beans and {b} {cb} jelly beans. Two are taken at random, one at a time, with "
    "the first returned to the jar before the second is taken.",
    "A tin has {a} {ca} marbles and {b} {cb} marbles. A marble is picked at random, noted, and put back, and then a "
    "second marble is picked at random.",
)
_COLOURS = (("red", "blue"), ("green", "yellow"), ("black", "white"), ("orange", "purple"), ("blue", "green"))


@generator("exd_x_r2x_compound_outcomes_replacement", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_r2x_compound_outcomes_replacement(rng, tier: str = "exam"):
    """Probability of an order, of exactly one of a colour, of two the same, then the change without replacement."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "two different counts"):
        a, b = rng.randint(2, 14), rng.randint(2, 14)
        if a != b and 7 <= a + b <= 20:
            break
    n = a + b
    ca, cb = rng.choice(_COLOURS)
    items = ["A"] * a + ["B"] * b

    # Independent route: enumerate every ordered draw.
    with_rep = list(product(range(n), repeat=2))
    without = [(i, j) for i, j in with_rep if i != j]
    p_ab = Fraction(sum(1 for i, j in with_rep if items[i] == "A" and items[j] == "B"), len(with_rep))
    p_one = Fraction(sum(1 for i, j in with_rep if (items[i] == "A") != (items[j] == "A")), len(with_rep))
    p_any = Fraction(sum(1 for i, j in with_rep if "A" in (items[i], items[j])), len(with_rep))
    p_any_wo = Fraction(sum(1 for i, j in without if "A" in (items[i], items[j])), len(without))
    rise = p_any_wo - p_any
    check(p_ab == Fraction(a * b, n * n) and p_one == 2 * p_ab, "formulas disagree with counting")
    check(p_any == p_one + Fraction(a * a, n * n), "at least one disagrees with exactly one plus both")
    check(rise > 0, "the rise must be positive")

    stem = phrase(rng, _SCENES, a=a, b=b, ca=ca, cb=cb)
    wo_text = f"{p_any_wo.numerator}/{p_any_wo.denominator}"
    parts = [
        frac_part(
            f"Find the probability that the first is {ca} and the second is {cb}. Give the answer as a fraction in "
            f"simplest form.",
            p_ab,
            2,
            [
                f"P({ca} then {cb}) = {a}/{n} × {b}/{n} = {a * b}/{n * n}.",
                f"In simplest form this is {p_ab.numerator}/{p_ab.denominator}.",
            ],
            [
                (Fraction(a, n) * Fraction(b, n - 1), "r2x-comp-no-replacement-values"),
                (Fraction(a * a, n * n), "r2x-comp-same-colour-for-different"),
                (Fraction(a * b, n), "r2x-comp-denominator-not-squared"),
                (Fraction(b, n) * Fraction(a, n - 1), "r2x-comp-no-replacement-values"),
            ],
        ),
        frac_part(
            f"Using your answer to (a), find the probability that exactly one of the two is {ca}. Give the answer as a "
            f"fraction in simplest form.",
            p_one,
            1,
            [
                f"{ca} then {cb} and {cb} then {ca} are equally likely, so P(exactly one is {ca}) = 2 × {p_ab.numerator}/{p_ab.denominator}."
            ],
            [
                (p_ab, "r2x-comp-one-order-only"),
                (Fraction(a * a, n * n), "r2x-comp-same-colour-for-different"),
                (Fraction(2 * a * b, n * (n - 1)), "r2x-comp-no-replacement-values"),
                (Fraction(a, n), "r2x-comp-single-draw-probability"),
            ],
            follow="2*part_a",
        ),
        frac_part(
            f"Using your answer to (b), find the probability that at least one of the two is {ca}. Give the answer as "
            f"a fraction in simplest form.",
            p_any,
            2,
            [
                f"At least one {ca} is exactly one {ca} or both {ca}.",
                f"P(both {ca}) = {a}/{n} × {a}/{n} = {a * a}/{n * n}, so the total is {p_one.numerator}/{p_one.denominator} + {a * a}/{n * n}.",
            ],
            [
                (p_one, "r2x-comp-exactly-one-for-at-least-one"),
                (Fraction(b * b, n * n), "r2x-comp-none-for-at-least-one"),
                (Fraction(a * a, n * n), "r2x-comp-both-for-at-least-one"),
                (1 - Fraction(a * a, n * n), "r2x-comp-complement-of-both"),
            ],
            follow=f"part_b + {a * a}/{n * n}",
        ),
        frac_part(
            f"Now suppose the first is not put back. Using your answer to (c), by how much does the probability that at "
            f"least one is {ca} rise? Give the answer as a fraction in simplest form.",
            rise,
            2,
            [
                f"Without replacement, P(no {ca}) = {b}/{n} × {b - 1}/{n - 1}, so P(at least one {ca}) = {wo_text}.",
                f"The rise is {wo_text} - {p_any.numerator}/{p_any.denominator}.",
            ],
            [
                (p_any_wo, "r2x-comp-new-value-not-change"),
                (p_any, "r2x-comp-new-value-not-change"),
                (1 - Fraction(b * (b - 1), n * n) - p_any, "r2x-comp-first-denominator-not-reduced"),
                (p_any_wo - p_one, "r2x-comp-compared-with-exactly-one"),
            ],
            follow=f"{wo_text} - part_c",
        ),
    ]
    return dict(text=stem, parts=parts)
