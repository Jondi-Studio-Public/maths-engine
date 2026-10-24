"""Exam chain for ``f10-probability-tree-diagrams``: a partly filled tree for two draws without replacement.

A bag holds ``a`` counters of one colour and ``b`` of another, and two are drawn one after the other without
replacement. The tree diagram is the stimulus (the same ``tree_diagram`` figure the practice tiers draw) with two
branches blank, one in the first set of branches and one in the set that follows the second colour. Set the way a
Year 10 test sets it:

(a) the blank first-draw branch                                          fraction, 1 mark
(b) the blank second-draw branch (the same colour twice)                 fraction, 1 mark
(c) the probability that both counters are the first colour             fraction, 2 marks, follow-through (a)
(d) the probability that the two counters are the same colour           fraction, 2 marks, follow-through (b), (c)

Nothing the learner is asked for is printed on the tree, and the question never states a counter count: each blank
is one minus its siblings.

## Checks (on every question)

The tree's branches are the conditional probabilities of drawing from ``a`` and ``b`` counters, and every key is
also found by listing every ordered pair of distinct counters (brute force) and counting the pairs that satisfy the
event. The two must agree. No answer may appear in the figure's labels.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

from ..core import VerificationError, attempts, check, generator, phrase, rational_text
from . import _space_kit as kit
from ._exd_x_fc_kit import fraction_part
from .fig_tree_build import _COLOURS, _NOUNS, _build_tree

_STEMS = (
    "A bag holds {c1} and {c2} {plural}. Two {plural} are drawn at random, one after the other, and the first is "
    "not put back. The tree diagram shows the probabilities, but two of them are missing.",
    "Two {plural} are taken at random from a bag of {c1} and {c2} {plural}, one at a time, without being put back. "
    "Some of the probabilities on the tree diagram have been left out.",
    "A {noun} is drawn at random from a bag of {c1} and {c2} {plural} and kept out, then a second {noun} is drawn. "
    "Two of the probabilities on the tree diagram are missing.",
)


def _brute_force(a: int, b: int):
    """(P first is colour 1, P second is colour 2 after colour 2, P both colour 1, P same colour) by listing every
    ordered pair of distinct counters."""
    bag = ["1"] * a + ["2"] * b
    pairs = list(itertools.permutations(range(len(bag)), 2))
    first_1 = sum(1 for i, _ in pairs if bag[i] == "1")
    after_2 = [(i, j) for i, j in pairs if bag[i] == "2"]
    both_2_given_first_2 = sum(1 for i, j in after_2 if bag[j] == "2")
    return (
        Fraction(first_1, len(pairs)),
        Fraction(both_2_given_first_2, len(after_2)),
        Fraction(sum(1 for i, j in pairs if bag[i] == bag[j] == "1"), len(pairs)),
        Fraction(sum(1 for i, j in pairs if bag[i] == bag[j]), len(pairs)),
    )


def _a(word: str) -> str:
    """The indefinite article for a colour word."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _build(rng: random.Random, cols: tuple, noun: str, plural: str) -> dict:
    for _ in attempts(300, "a tree whose answers stay off the figure"):
        t = rng.randint(6, 16)
        a = rng.randint(2, t - 2)
        b = t - a
        first = [Fraction(a, t), Fraction(b, t)]
        second = [
            [Fraction(a - 1, t - 1), Fraction(b, t - 1)],
            [Fraction(a, t - 1), Fraction(b - 1, t - 1)],
        ]
        fig = _build_tree(cols, first, second, blank={(0,), (1, 1)})
        key_a, key_b = first[0], second[1][1]
        key_c = first[0] * second[0][0]
        key_d = key_c + first[1] * second[1][1]
        check(0 < key_a < 1 and 0 < key_b < 1 and 0 < key_c < 1 and 0 < key_d < 1, "a key left (0, 1)")
        if any(kit.leaks(fig, rational_text(k)) for k in (key_a, key_b, key_c, key_d)):
            continue
        break
    brute = _brute_force(a, b)
    check(brute == (key_a, key_b, key_c, key_d), "the tree and the brute-force count disagree")

    c1, c2 = cols
    stem = phrase(rng, _STEMS, c1=c1, c2=c2, noun=noun, plural=plural)
    r_t = rational_text
    part_a = fraction_part(
        f"Find the missing probability on the branch for {_a(c1)} {c1} {noun} on the first draw.",
        key_a,
        1,
        [
            f"The two first-draw branches add to 1, so the missing one is 1 - {r_t(first[1])} = {r_t(key_a)}.",
        ],
        [
            (first[1], "gives-the-sibling-branch"),
            (Fraction(a, t - 1), "second-draw-total-on-first-draw"),
            (second[0][0], "copies-a-branch-from-another-set"),
            (Fraction(a - 1, t), "first-draw-total-on-second-draw"),
        ],
    )
    part_b = fraction_part(
        f"Find the missing probability on the branch for {_a(c2)} {c2} {noun} on the second draw, after {_a(c2)} {c2} {noun} on "
        "the first draw.",
        key_b,
        1,
        [
            f"After {_a(c2)} {c2} {noun} is removed the second-draw branches add to 1: 1 - {r_t(second[1][0])} = {r_t(key_b)}.",
        ],
        [
            (second[1][0], "gives-the-sibling-branch"),
            (Fraction(b, t - 1), "forgets-the-removed-counter"),
            (first[1], "uses-with-replacement-model"),
            (Fraction(b - 1, t), "first-draw-total-on-second-draw"),
        ],
    )
    part_c = fraction_part(
        f"Using your answer to (a), find the probability that both {plural} are {c1}.",
        key_c,
        2,
        [f"Multiply along the {c1}, {c1} path: {r_t(key_a)} × {r_t(second[0][0])} = {r_t(key_c)}."],
        [
            (key_a * key_a, "assumes-with-replacement"),
            (key_a + second[0][0], "adds-along-a-path"),
            (key_a, "stops-at-the-first-branch"),
            (first[1] * second[1][1], "multiplies-the-wrong-path"),
        ],
        follow_through=f"part_a*{a - 1}/{t - 1}",
    )
    part_d = fraction_part(
        f"Using your answers to (b) and (c), find the probability that the two {plural} are the same colour.",
        key_d,
        2,
        [
            f"Same colour is {c1}, {c1} or {c2}, {c2}. The {c2}, {c2} path is {r_t(first[1])} × {r_t(key_b)} = "
            f"{r_t(first[1] * key_b)}.",
            f"{r_t(key_c)} + {r_t(first[1] * key_b)} = {r_t(key_d)}.",
        ],
        [
            (key_c, "leaves-out-a-path"),
            (key_c * first[1] * key_b, "multiplies-paths-instead-of-adding"),
            (1 - key_d, "gives-the-complement"),
            (key_c + first[1] * first[1], "assumes-with-replacement"),
        ],
        follow_through=f"part_c+part_b*{b}/{t}",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d], figure=fig)


@generator("exd_x_fc_probability_tree", level=10, presets={"exam": {"tier": "exam"}})
def exd_x_fc_probability_tree(rng: random.Random, tier: str = "exam"):
    """Two missing branches, then both counters one colour, then the two counters the same colour."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    cols = rng.choice(_COLOURS)
    cols = cols if rng.random() < 0.5 else (cols[1], cols[0])
    noun, plural = rng.choice(_NOUNS)
    for _ in attempts(60, "a tree with enough distinct wrong answers"):
        try:
            return _build(rng, cols, noun, plural)
        except VerificationError:
            continue
    raise AssertionError("unreachable")
