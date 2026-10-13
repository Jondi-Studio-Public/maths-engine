"""Exam tier for `f10-unit-fractions-combine-whole`: combine unit-fraction serves, then share what is left.

A food (or other whole) is cut into D equal pieces, so each piece is the unit fraction 1/D. Two people take some
pieces each; the pieces left are shared equally among a group. The learner combines same-denominator fractions,
completes the whole, and then divides the remaining fraction equally.

* (a) the fraction of the whole the two people took (2 marks);
* (b) the fraction left over: 1 - (a) (1 mark, follow-through from (a));
* (c) the fraction of the whole each person in the group receives: (b) divided by the group size (2 marks,
  follow-through from (b)).

Built backwards: the pieces left over are a multiple of the group size, so each person gets a whole number of pieces.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors, rational_text

# (whole, piece singular, piece plural, first person, second person, the group sharing the rest, range of pieces)
_SETTINGS = [
    ("A lasagne", "serve", "serves", "Mum", "Dad", "the {k} children", (8, 20)),
    ("A watermelon at a Queensland picnic", "wedge", "wedges", "Callum", "Sienna", "the {k} other players", (8, 24)),
    (
        "A tray of lamingtons from the school fete",
        "piece",
        "pieces",
        "Ms Nguyen",
        "Mr Taylor",
        "the {k} students on stall duty",
        (12, 48),
    ),
    ("A long loaf of damper at a camp", "slice", "slices", "Kai", "Ruby", "the {k} other campers", (8, 30)),
    ("A rectangular chocolate slab", "square", "squares", "Mia", "Tom", "the {k} friends who arrive later", (12, 40)),
]


def _count(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


class _Retry(Exception):
    pass


def _text(f: Fraction) -> str:
    return rational_text(f)


def _attempt(rng: random.Random) -> dict:
    whole, one, noun, name1, name2, group, (low, high) = rng.choice(_SETTINGS)
    for _ in attempts(1000, "a number of pieces that the remainder shares equally"):
        pieces = rng.randint(low, high)
        k = rng.randint(2, 8)
        s1, s2 = rng.randint(1, pieces // 3), rng.randint(1, pieces // 3)
        left = pieces - s1 - s2
        if left < 2 * k or left % k:
            continue
        each_pieces = left // k
        taken = Fraction(s1 + s2, pieces)
        if taken.denominator == 1 or Fraction(each_pieces, pieces).denominator == 1:
            continue
        if Fraction(each_pieces, pieces) in (Fraction(1, k), taken) or len({s1, s2, each_pieces}) < 2:
            continue
        if {s1 + s2, left, each_pieces} & {pieces, k, s1, s2}:
            continue
        break

    group = group.format(k=k)
    who = f"{name1} and {name2}"
    stem = (
        f"{whole} is cut into {pieces} equal {noun}, so each one is 1/{pieces} of the whole. "
        f"{name1} takes {_count(s1, one, noun)} and {name2} takes {_count(s2, one, noun)}. The {noun} that are left are shared equally among {group}."
    )
    first = Fraction(s1, pieces)
    second = Fraction(s2, pieces)
    leftover = Fraction(left, pieces)
    each = Fraction(each_pieces, pieces)
    check(taken + leftover == 1 and each * k == leftover, "pieces do not add up")

    d_a = pick_distractors(
        _text(taken),
        [
            Distractor(
                _text(Fraction(s1 + s2, 2 * pieces)), "adds-numerators-and-denominators", Fraction(s1 + s2, 2 * pieces)
            ),
            Distractor(_text(first), "omitted-a-term", first),
            Distractor(_text(second), "omitted-a-term", second),
            Distractor(_text(leftover), "answers-amount-used-not-remaining", leftover),
        ],
        3,
    )
    d_b = pick_distractors(
        _text(leftover),
        [
            Distractor(_text(taken), "answers-amount-used-not-remaining", taken),
            Distractor(_text(1 - first), "omitted-a-term", 1 - first),
            Distractor(
                _text(Fraction(left, 2 * pieces)), "adds-numerators-and-denominators", Fraction(left, 2 * pieces)
            ),
        ],
        3,
    )
    d_c = pick_distractors(
        _text(each),
        [
            Distractor(_text(leftover), "forgot-to-share-remainder", leftover),
            Distractor(_text(Fraction(1, k)), "share-taken-of-whole", Fraction(1, k)),
            Distractor(_text(Fraction(1, pieces * k)), "share-taken-of-whole", Fraction(1, pieces * k)),
            Distractor(
                _text(Fraction(left, pieces + k)), "adds-numerators-and-denominators", Fraction(left, pieces + k)
            ),
        ],
        3,
    )
    if min(len(d_a), len(d_b), len(d_c)) < 3:
        raise _Retry

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"What fraction of the whole did {who} take altogether? Give your answer in simplest form.",
                answer_kind="fraction",
                answer={"p": taken.numerator, "q": taken.denominator},
                answer_text=_text(taken),
                answer_latex=latex_answer(_text(taken)),
                marks=2,
                sensible_range=(0, 1),
                working=[f"{s1}/{pieces} + {s2}/{pieces} = {s1 + s2}/{pieces} = {_text(taken)}."],
                distractors=d_a,
            ),
            dict(
                prompt="What fraction of the whole is left to be shared? Give your answer in simplest form.",
                answer_kind="fraction",
                answer={"p": leftover.numerator, "q": leftover.denominator},
                answer_text=_text(leftover),
                answer_latex=latex_answer(_text(leftover)),
                marks=1,
                sensible_range=(0, 1),
                follow_through="1 - part_a",
                working=[f"1 - {_text(taken)} = {_text(leftover)}."],
                distractors=d_b,
            ),
            dict(
                prompt=f"What fraction of the whole does each of {group} get? Give your answer in simplest form.",
                answer_kind="fraction",
                answer={"p": each.numerator, "q": each.denominator},
                answer_text=_text(each),
                answer_latex=latex_answer(_text(each)),
                marks=2,
                sensible_range=(0, 1),
                follow_through=f"part_b/{k}",
                working=[f"{_text(leftover)} ÷ {k} = {_text(each)}."],
                distractors=d_c,
            ),
        ],
    )


def _build(rng: random.Random) -> dict:
    for _ in attempts(200, "a draw with three distinct distractors per part"):
        try:
            return _attempt(rng)
        except _Retry:
            continue


@generator("exd_x_pr5_serves_shared", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr5_serves_shared(rng: random.Random, tier: str = "exam") -> dict:
    """Unit fractions combined, the whole completed, and the remaining fraction shared equally."""
    check(tier == "exam", "exd_x_pr5_serves_shared: only the exam tier exists")
    return _build(rng)
