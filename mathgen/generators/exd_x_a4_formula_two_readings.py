"""Exam tier for "develop a linear formula from a word description".

The practice tiers state the rate of change and one starting value. Here the rate is NOT stated: the learner
is told the value of a steadily changing quantity at two different times (neither of them time zero), so
they must find the rate from the two readings (the change in value over the change in time) and then
back-solve the starting value, before writing ``V = b + m t``. That is a two-stage method the learner has
to choose, not a larger range of numbers.

Built backwards: draw the rate ``m`` (a nonzero integer, either sign), the starting value ``b``, the two
times, then compute both readings. Both readings are kept positive. The key is re-derived from the two
readings alone with exact fractions, independently of the construction.

Distractors, each a named misconception:

- used the first reading as the starting value (``V = V1 + m t``);
- wrong sign on the rate (``V = b - m t``);
- swapped rate and starting value (``V = m + b t``);
- divided nothing: used the raw change in value as the rate (``V = (V2 - V1) t + c``), which differs from the
  key because the time gap is always at least 2.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Symbol

from ..core import check, check_distinct, check_equivalent, generator, join_terms, phrase, pick_distractors

_T = Symbol("t")

# (reading wording, with two phrasings sharing one slot set; formula request; output letter)
_CONTEXTS = (
    (
        (
            "The water level in a tank was {V1} cm after {T1} and {V2} cm after {T2}, changing at a constant rate.",
            "The water level in a tank was {V1} cm after {T1}. A further {G} later it was {V2} cm, "
            "having changed at a constant rate.",
        ),
        "L",
        "Write a formula for the water level L (in cm) after t hours.",
        "hour",
    ),
    (
        (
            "An account held ${V1} after {T1} and ${V2} after {T2}, changing by the same amount each time period.",
            "An account held ${V1} after {T1}. A further {G} later it held ${V2}, "
            "changing by the same amount each time period.",
        ),
        "B",
        "Write a formula for the balance B (in dollars) after t weeks.",
        "week",
    ),
    (
        (
            "The depth of snow at a ski field was {V1} cm after {T1} and {V2} cm after {T2}, "
            "changing at a steady rate.",
            "The depth of snow at a ski field was {V1} cm after {T1}. A further {G} later it was {V2} cm, "
            "changing at a steady rate.",
        ),
        "D",
        "Write a formula for the depth D (in cm) after t days.",
        "day",
    ),
)


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _ordered(slope: int, const: int) -> str:
    """Constant first when the rate is negative reads as "120 - 7t"; otherwise "7t + 3"."""
    if slope < 0:
        return join_terms([(const, ""), (slope, "t")])
    return join_terms([(slope, "t"), (const, "")])


def _expr(slope: int, const: int):
    return slope * _T + const


@generator(
    "exd_x_a4_formula_two_readings",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a4_formula_two_readings(rng: random.Random, tier: str = "exam") -> dict:
    """A steadily changing quantity read at two times; find the formula. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    while True:
        m = rng.choice([-1, 1]) * rng.randint(2, 12)
        b = rng.randint(20, 90)
        t1 = rng.randint(1, 4)
        gap = rng.randint(2, 5)
        t2 = t1 + gap
        v1, v2 = b + m * t1, b + m * t2
        if min(v1, v2) >= 3 and abs(m) != b:
            break

    readings, out, request, unit = rng.choice(_CONTEXTS)
    text = phrase(
        rng,
        readings,
        V1=v1,
        V2=v2,
        T1=_count(t1, unit),
        T2=_count(t2, unit),
        G=_count(gap, unit),
    )
    text = f"{text} {request}"

    # Independent route, from the two readings alone.
    slope = Fraction(v2 - v1, t2 - t1)
    start = v1 - slope * t1
    check(slope.denominator == 1 and start.denominator == 1, "rate and start must be whole numbers")
    check(slope == m and start == b, "re-derived rate and start disagree with the construction")
    key = _expr(m, b)
    check_equivalent(key.subs(_T, t1), v1, "formula does not reproduce the first reading")
    check_equivalent(key.subs(_T, t2), v2, "formula does not reproduce the second reading")

    answer_text = f"{out} = {_ordered(m, b)}"

    raw_change = v2 - v1
    wrong = [
        (m, v1),  # first reading used as the starting value
        (-m, b),  # sign of the rate reversed
        (raw_change, v1 - raw_change * t1),  # change in value never divided by the change in time
        (b, m),  # rate and starting value swapped
    ]
    for s, c in wrong:
        check_distinct(key, _expr(s, c), msg="distractor must differ from the key")
    candidates = [f"{out} = {_ordered(s, c)}" for s, c in wrong]

    return dict(
        text=text,
        latex=text,
        answer=str(key),
        answer_text=answer_text,
        working=[
            f"Rate = ({v2} - {v1}) / ({t2} - {t1}) = {m}.",
            f"Starting value = {v1} - ({m})({t1}) = {b}.",
            answer_text + ".",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )
