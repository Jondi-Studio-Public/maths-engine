"""Exam chain for ``f10-venn-diagram-counts``: a two-set Venn diagram with one region missing, then probabilities.

A group is surveyed about two activities and the Venn diagram (the same ``venn`` figure the practice tiers draw)
shows every region's count except one, the number who do only the second activity, marked ``?``. The sentence gives
the size of the group. Set the way a Year 9 or 10 test sets it:

(a) the missing region, from the group total                                   number,   1 mark
(b) everyone in the second set's circle                                         number,   1 mark, follow-through (a)
(c) the probability a person chosen at random does exactly one activity         fraction, 2 marks, follow-through (a)
(d) the probability a person known to do the second also does the first         fraction, 2 marks, follow-through (b)

No answer appears on the diagram: the missing count, the circle total and both probabilities are worked out.

## Checks (on every question)

The group is rebuilt person by person (a list of which sets each person is in), and every key is found by counting
people who satisfy the event. It must equal the key built from the regions.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import VerificationError, attempts, check, generator, phrase, rational_text
from . import _space_kit as kit
from ._exd_x_fc_kit import fraction_part
from ._exd_x_sc4_kit import count_part
from .fig_venn_build import _CONTEXTS, _figure

_NOUNS = ("students", "children", "people")
# verb -> third-person singular, for "a student who plays ..."
_THIRD = {"play": "plays", "study": "studies"}
_SINGULAR = {"students": "student", "children": "child", "people": "person"}

_STEMS = (
    "{total} {noun} were asked whether they {verb} {a} and {b}. The Venn diagram shows the results, with one region "
    "marked ?.",
    "The Venn diagram shows how many of {total} {noun} {verb} {a} and {b}. One region is marked ?.",
    "Each of {total} {noun} was asked about {a} and {b}, and the Venn diagram shows the answers, apart from one "
    "region marked ?.",
)


def _build(rng: random.Random, verb: str, names: list, noun: str) -> dict:
    for _ in attempts(400, "a Venn chain whose answers stay off the figure"):
        only_a, both, only_b, none = rng.randint(5, 22), rng.randint(3, 12), rng.randint(5, 22), rng.randint(2, 10)
        counts = {"A": only_a, "B": only_b, "AB": both, "none": none}
        total = only_a + only_b + both + none
        fig = _figure(names, counts, noun, "B")
        key_a = only_b
        key_b = only_b + both
        key_c = Fraction(only_a + only_b, total)
        key_d = Fraction(both, key_b)
        if any(kit.leaks(fig, t) for t in (str(key_a), str(key_b), rational_text(key_c), rational_text(key_d))):
            continue
        if not (0 < key_c < 1 and 0 < key_d < 1):
            continue
        break
    # The same group, person by person.
    people = [{"A"}] * only_a + [{"B"}] * only_b + [{"A", "B"}] * both + [set()] * none
    check(len(people) == total, "the group is not the stated size")
    check(sum(1 for p in people if "B" in p) == key_b, "the circle total disagrees with the head count")
    check(Fraction(sum(1 for p in people if len(p) == 1), len(people)) == key_c, "exactly-one disagrees")
    in_b = [p for p in people if "B" in p]
    check(Fraction(sum(1 for p in in_b if "A" in p), len(in_b)) == key_d, "the conditional disagrees")

    a, b = names
    third = _THIRD[verb]
    one = _SINGULAR[noun]
    stem = phrase(rng, _STEMS, total=total, noun=noun, verb=verb, a=a, b=b)
    part_a = count_part(
        "Find the number marked ?.",
        key_a,
        1,
        [
            f"The regions add to the total: {only_a} + {both} + ? + {none} = {total}, so ? = {total} - "
            f"{only_a + both + none} = {key_a}."
        ],
        [
            (only_a + both + none, "adds-shown-without-total"),
            (total - only_a - both, "leaves-out-a-region"),
            (total - only_a - none, "leaves-out-a-region"),
            (total - both - none, "leaves-out-a-region"),
        ],
        lo=1,
    )
    part_b = count_part(
        f"Using your answer to (a), find how many {noun} {verb} {b}.",
        key_b,
        1,
        [f"The {b} circle holds the part outside the overlap and the overlap: {key_a} + {both} = {key_b}."],
        [
            (key_a, "overlap-left-out-of-set"),
            (only_a + both, "reads-the-other-circle"),
            (key_a + both + only_a, "counts-either-set-for-one-set"),
            (both, "gives-the-overlap-alone"),
        ],
        follow_through=f"part_a+{both}",
        lo=1,
    )
    part_c = fraction_part(
        f"One of the {noun} is chosen at random. Using your answer to (a), find the probability that this {one} "
        f"{third} exactly one of {a} and {b}.",
        key_c,
        2,
        [
            f"Exactly one means {a} only or {b} only: {only_a} + {key_a} = {only_a + key_a} out of {total}, "
            f"which is {rational_text(key_c)}."
        ],
        [
            (Fraction(only_a + key_a + both, total), "exactly-one-with-overlap"),
            (Fraction(only_a + key_a, total - none), "denominator-leaves-out-neither"),
            (Fraction(both + none, total), "gives-the-complement"),
            (Fraction(key_a, total), "counts-one-circle-only"),
        ],
        follow_through=f"(part_a+{only_a})/{total}",
    )
    part_d = fraction_part(
        f"One of the {noun} who {verb} {b} is chosen at random. Using your answer to (b), find the probability that "
        f"this {one} also {third} {a}.",
        key_d,
        2,
        [f"Only the {key_b} {noun} in the {b} circle count, and {both} of them are in the overlap: {both}/{key_b}."],
        [
            (Fraction(both, total), "gives-joint-not-conditional"),
            (Fraction(both, only_a + both), "reversed-conditional"),
            (1 - key_d, "gives-the-complement"),
            (Fraction(both, total - none), "denominator-leaves-out-neither"),
        ],
        follow_through=f"{both}/part_b",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d], figure=fig)


@generator("exd_x_fc_venn_counts", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_fc_venn_counts(rng: random.Random, tier: str = "exam"):
    """The missing region, the second circle, exactly one, and a conditional probability."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    verb, pool = rng.choice(_CONTEXTS)
    names = rng.sample(pool, 2)
    noun = rng.choice(_NOUNS)
    for _ in attempts(60, "a Venn chain with enough distinct wrong answers"):
        try:
            return _build(rng, verb, names, noun)
        except VerificationError:
            continue
    raise AssertionError("unreachable")
