"""Exam-tier source for `calculate-range`: compare the ranges of two data sets in context.

The practice tier subtracts the least from the greatest in one short list. Here two schools each record monthly
rainfall for six months, and the learner finds the range of each set and then compares them, so the range is used
as a measure of spread rather than computed once:

(a) the range of the first school's readings                        number, 1 mark
(b) the range of the second school's readings                       number, 1 mark
(c) how many millimetres greater the first range is than the second number, 1 mark, ``part_a - part_b``

Built backwards from the two ranges. Every middle reading lies strictly between its set's least and greatest, so
the extremes are unique and the lists are in no particular order.
"""

from __future__ import annotations

import random

from mathgen.answers import Distractor, exact_value
from mathgen.core import attempts, check, generator, latex_answer, phrase, pick_distractors

_SCHOOLS = (
    "Banksia Primary School",
    "Wattle Grove Primary School",
    "Ironbark Public School",
    "Coolibah Primary School",
    "Boronia Primary School",
    "Mallee Public School",
    "Kurrajong Primary School",
    "Gumtree Public School",
)


def _readings(rng: random.Random, lo: int, spread: int, count: int = 6) -> list[int]:
    """`count` readings with least `lo` and greatest `lo + spread`, the rest strictly between, in random order."""
    middle = rng.sample(range(lo + 1, lo + spread), count - 2)
    values = [lo, lo + spread, *middle]
    rng.shuffle(values)
    return values


def _number_part(prompt, value, working, candidates, follow_through=None):
    text = str(value)
    options = [Distractor(str(v), tag, exact_value(v)) for tag, v in candidates if v != value and v > 0]
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=1,
        working=working,
        distractors=pick_distractors(text, options, 3),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _draw(rng: random.Random) -> dict | None:
    first, second = rng.sample(_SCHOOLS, 2)
    spread_a = rng.randint(120, 380)
    spread_b = rng.randint(30, spread_a - 25)
    a = _readings(rng, rng.randint(8, 70), spread_a)
    b = _readings(rng, rng.randint(25, 120), spread_b)
    range_a, range_b = max(a) - min(a), max(b) - min(b)
    check(range_a == spread_a and range_b == spread_b, "constructed range disagrees with the data")
    difference = range_a - range_b
    check(difference > 0, "the first range is not greater")

    stem = phrase(
        rng,
        (
            "{s1} and {s2} each recorded their monthly rainfall, in millimetres, for six months. {s1}: {a}. {s2}: {b}.",
            "Monthly rainfall in millimetres over six months. {s1}: {a}. {s2}: {b}.",
            "Two schools kept a rain gauge for six months. The monthly readings, in millimetres, were {a} at {s1} "
            "and {b} at {s2}.",
        ),
        s1=first,
        s2=second,
        a=", ".join(map(str, a)),
        b=", ".join(map(str, b)),
    )

    def wrong_set(values):
        return [
            ("reports-maximum", max(values)),
            ("range-first-last", abs(values[-1] - values[0])),
            (None, max(values) - sorted(values)[1]),
            (None, max(values) + min(values)),
        ]

    part_a = _number_part(
        f"Find the range of the monthly rainfall at {first}.",
        range_a,
        [f"Range = {max(a)} − {min(a)} = {range_a} mm."],
        wrong_set(a),
    )
    part_b = _number_part(
        f"Find the range of the monthly rainfall at {second}.",
        range_b,
        [f"Range = {max(b)} − {min(b)} = {range_b} mm."],
        wrong_set(b),
    )
    part_c = _number_part(
        f"How many millimetres greater is the range at {first} than the range at {second}?",
        difference,
        [f"{range_a} − {range_b} = {difference} mm."],
        [
            (None, range_a + range_b),
            (None, abs(max(a) - max(b))),
            (None, abs(min(a) - min(b))),
            (None, range_b),
        ],
        follow_through="part_a - part_b",
    )
    if any(len(p["distractors"]) < 3 for p in (part_a, part_b, part_c)):
        return None
    return dict(text=stem, parts=[part_a, part_b, part_c])


@generator("exd_x_u5_range_compare", level=8, presets={"exam": dict(tier="exam")})
def exd_x_u5_range_compare(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the range of each of two rainfall data sets, then how much greater one range is."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "two data sets with three distinct distractors per part"):
        question = _draw(rng)
        if question is not None:
            return question
