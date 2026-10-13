"""Exam tier for `f10-equivalent-fractions-related-denominators`: fractions of a marked trail, then the gap.

A trail has a marker at every kilometre. Two walkers stop at different fractions of the trail whose denominators
both divide the trail's length. Each stopping fraction is converted to an equivalent fraction with the trail's
length as denominator, which gives the marker number; the gap between them is then written as a fraction.

* (a) the marker where the first walker stops (1 mark);
* (b) the marker where the second walker stops (1 mark);
* (c) the fraction of the trail between the two walkers, in simplest form (2 marks, follow-through from (a) and
  (b)).

Built backwards from the trail length: it is a multiple of both denominators, so each marker is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, numeric_answer, pick_distractors, rational_text

_TRAILS = [
    "The Heysen Trail section near Wilpena has a marker at every kilometre.",
    "A coastal walking track near Port Macquarie has a marker at every kilometre.",
    "The Great Ocean Walk has a marker at every kilometre.",
    "A bush walk in the Grampians has a marker at every kilometre.",
]
_LENGTHS = [12, 18, 20, 24, 30, 36, 40, 42, 48, 60]
_NAMES = [("Zoe", "Ben"), ("Amira", "Liam"), ("Tessa", "Noah"), ("Priya", "Jack"), ("Chloe", "Sam")]


class _Retry(Exception):
    pass


def _whole(pairs):
    """Distractors for the (value, misconception) pairs that are positive whole numbers."""
    return [Distractor(str(int(v)), m, v) for v, m in pairs if v.denominator == 1 and v > 0]


def _text(f: Fraction) -> str:
    return rational_text(f)


def _attempt(rng: random.Random) -> dict:
    for _ in attempts(1000, "two stopping fractions whose denominators divide the trail length"):
        length = rng.choice(_LENGTHS)
        divisors = [d for d in range(3, length) if length % d == 0 and d <= 15]
        d1, d2 = rng.choice(divisors), rng.choice(divisors)
        if d1 == d2:
            continue
        f1 = Fraction(rng.randint(1, d1 - 1), d1)
        f2 = Fraction(rng.randint(1, d2 - 1), d2)
        if f1.denominator != d1 or f2.denominator != d2 or f1 >= f2:
            continue
        m1, m2 = f1 * length, f2 * length
        gap = f2 - f1
        if m1.denominator != 1 or m2.denominator != 1 or gap.denominator == 1:
            continue
        shown = {length, f1.numerator, d1, f2.numerator, d2}
        if {int(m1), int(m2), int(m2 - m1)} & shown or len({m1, m2, m2 - m1}) < 3:
            continue
        if gap in (f1, f2):
            continue
        break

    swapped = rng.random() < 0.5  # the first walker named may be the one who goes further
    if swapped:
        f1, f2, d1, d2, m1, m2 = f2, f1, d2, d1, m2, m1
    span = abs(m2 - m1)
    place = rng.choice(_TRAILS)
    walker1, walker2 = rng.choice(_NAMES)
    stem = (
        f"{place} It is {length} kilometres long. {walker1} stops after {_text(f1)} of the trail "
        f"and {walker2} stops after {_text(f2)} of the trail."
    )
    unit1 = Fraction(length, d1)
    unit2 = Fraction(length, d2)
    n1, n2 = f1.numerator, f2.numerator
    d_a = pick_distractors(
        str(int(m1)),
        _whole(
            [
                (unit1, "ignores-numerator"),
                (Fraction(n1), "scale-factor-used-as-position"),
                (length - m1, "measured-from-wrong-end"),
                (unit1 + n1, "added-scale-factor"),
            ]
        ),
        3,
    )
    d_b = pick_distractors(
        str(int(m2)),
        _whole(
            [
                (unit2, "ignores-numerator"),
                (Fraction(n2), "scale-factor-used-as-position"),
                (length - m2, "measured-from-wrong-end"),
                (unit2 + n2, "added-scale-factor"),
            ]
        ),
        3,
    )
    between = Fraction(int(span), length)
    difference_of_parts = Fraction(n2 - n1, d2 - d1) if d2 > d1 and n2 > n1 else 1 - between
    d_c = pick_distractors(
        _text(between),
        [
            c
            for c in (
                Distractor(_text(difference_of_parts), "subtracts-numerators-and-denominators", difference_of_parts),
                Distractor(_text(1 - between), "answers-amount-used-not-remaining", 1 - between),
                Distractor(_text(Fraction(int(span), d2)), "wrong-denominator-for-gap", Fraction(int(span), d2)),
                Distractor(_text(f1), "omitted-a-term", f1),
            )
            if 0 < c.value < 1
        ],
        3,
    )
    if min(len(d_a), len(d_b), len(d_c)) < 3:
        raise _Retry
    check(between == gap, "gap disagrees with the fractions")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"At which marker (number of kilometres from the start) does {walker1} stop?",
                answer_kind="number",
                answer=numeric_answer(m1),
                answer_text=str(int(m1)),
                answer_latex=latex_answer(str(int(m1))),
                marks=1,
                sensible_range=(0, None),
                working=[f"{_text(f1)} = {int(m1)}/{length}, so {walker1} stops at marker {int(m1)}."],
                distractors=d_a,
            ),
            dict(
                prompt=f"At which marker does {walker2} stop?",
                answer_kind="number",
                answer=numeric_answer(m2),
                answer_text=str(int(m2)),
                answer_latex=latex_answer(str(int(m2))),
                marks=1,
                sensible_range=(0, None),
                working=[f"{_text(f2)} = {int(m2)}/{length}, so {walker2} stops at marker {int(m2)}."],
                distractors=d_b,
            ),
            dict(
                prompt=f"What fraction of the whole trail lies between {walker1} and {walker2}? "
                "Give your answer in simplest form.",
                answer_kind="fraction",
                answer={"p": between.numerator, "q": between.denominator},
                answer_text=_text(between),
                answer_latex=latex_answer(_text(between)),
                marks=2,
                sensible_range=(0, 1),
                follow_through=f"(part_a - part_b)/{length}" if swapped else f"(part_b - part_a)/{length}",
                working=[
                    f"{int(max(m1, m2))} - {int(min(m1, m2))} = {int(span)} km, so {int(span)}/{length} = {_text(between)}."
                ],
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


@generator("exd_x_pr5_trail_markers", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr5_trail_markers(rng: random.Random, tier: str = "exam") -> dict:
    """Two stopping fractions of a marked trail converted to markers, then the gap written as a fraction."""
    check(tier == "exam", "exd_x_pr5_trail_markers: only the exam tier exists")
    return _build(rng)
