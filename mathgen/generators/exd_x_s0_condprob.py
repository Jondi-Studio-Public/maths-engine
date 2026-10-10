"""Exam source for `conditional-probability-counts`: conditioning on a compound event.

The practice tiers ask for P(A | B) with B a single event. Here the learner is
given only the group totals (|A|, |B| and the number with neither) and is asked
for a probability conditioned on a *compound* event -- "at least one", "exactly
one" or "not both" -- so the overlap must be recovered by inclusion-exclusion
first and the right restricted sample space chosen second.

Questions are built backwards: the four cell counts are drawn, the totals are
derived from them, and the answer is the exact ratio of two cell sums.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, phrase, pick_distractors

_SUBJECTS = [
    ("take Music", "take Drama"),
    ("play chess", "play draughts"),
    ("own a bicycle", "own a scooter"),
    ("speak French", "speak German"),
    ("subscribe to the newsletter", "follow the blog"),
]

_PATTERNS = {
    "atleast": [
        "Of {n} people, {a} {ta}, {b} {tb}, and {z} do neither. One person who does at least one of these is "
        "chosen at random. Find the probability that the person does both.",
        "In a group of {n}, {a} {ta} and {b} {tb}; {z} do neither. Given that a person does at least one of "
        "the two, find the probability that they do both.",
    ],
    "exactly": [
        "Of {n} people, {a} {ta}, {b} {tb}, and {z} do neither. A person who does exactly one of the two is "
        "chosen at random. Find the probability that this person is one who {ta}.",
        "In a group of {n}, {a} {ta} and {b} {tb}; {z} do neither. Given that a person does exactly one of "
        "the two, find the probability that they {ta}.",
    ],
    "notboth": [
        "Of {n} people, {a} {ta}, {b} {tb}, and {z} do neither. A person who does not do both is chosen at "
        "random. Find the probability that they {ta} but do not {tb}.",
        "In a group of {n}, {a} {ta} and {b} {tb}; {z} do neither. Given that a person does not do both, "
        "find the probability that they {ta} but do not {tb}.",
    ],
}


def _text(fr: Fraction) -> str:
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


@generator(
    "exd_x_s0_condprob",
    level=12,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s0_condprob(rng: random.Random, tier: str = "exam") -> dict:
    """P(event | compound event) from totals only. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        both = rng.randint(2, 9)
        only_a = rng.randint(2, 11)
        only_b = rng.randint(2, 11)
        neither = rng.randint(2, 14)
        if only_a == only_b:
            continue
        a, b = both + only_a, both + only_b
        n = both + only_a + only_b + neither
        kind = rng.choice(["atleast", "exactly", "notboth"])
        union = n - neither
        derived_both = a + b - union  # inclusion-exclusion
        check(derived_both == both, "inclusion-exclusion mismatch")
        if kind == "atleast":
            num, den = both, union
            cands = [
                Fraction(both, n),  # divides by the whole group
                Fraction(both, a + b),  # double-counts the overlap in the denominator
                Fraction(both, b),  # conditions on one event only
                Fraction(only_a + only_b, union),  # asks for the complement
            ]
        elif kind == "exactly":
            num, den = only_a, only_a + only_b
            cands = [
                Fraction(a, a + b),  # counts the overlap as part of "exactly one"
                Fraction(only_a, union),  # denominator is "at least one"
                Fraction(only_a, n),
                Fraction(only_b, only_a + only_b),  # picks the other group
            ]
        else:
            num, den = only_a, n - both
            cands = [
                Fraction(only_a, n),
                Fraction(only_a, union),  # forgets that "neither" is also "not both"
                Fraction(a, n - both),  # keeps the overlap in the numerator
                Fraction(only_b, n - both),
            ]
        answer = Fraction(num, den)
        if len({answer, *cands}) != len(cands) + 1 or answer in (0, 1):
            continue
        break
    else:
        raise RuntimeError("exd_x_s0_condprob: no collision-free draw")
    ta, tb = rng.choice(_SUBJECTS)
    text = phrase(rng, _PATTERNS[kind], n=n, a=a, b=b, z=neither, ta=ta, tb=tb)
    answer_text = _text(answer)
    return dict(
        text=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Overlap = {a} + {b} - ({n} - {neither}) = {both}; only first = {only_a}; only second = {only_b}.",
            f"Restricted group has {den} people, of whom {num} qualify: {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, [_text(c) for c in cands]),
    )
