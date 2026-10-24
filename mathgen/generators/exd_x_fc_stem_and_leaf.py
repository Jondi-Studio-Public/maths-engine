"""Exam chain for ``f10-stem-and-leaf-plots``: quartiles, the interquartile range and the outlier boundary.

The plot is written as text, exactly as the practice tiers write it (rows ``stem | leaves`` and a key line), with
the leaves of each row in order. There are 11 or 15 values, so the median is a single value and each half has an odd
number of values; the stem states that the median is left out of both halves. Set the way a Year 10 test sets it:

(a) the lower quartile Q1                                                  number, 2 marks
(b) the upper quartile Q3                                                  number, 2 marks
(c) the interquartile range                                                number, 1 mark,  follow-through (a), (b)
(d) the boundary above which a value is an outlier (Q3 + 1.5 x IQR)        number, 2 marks, follow-through (b), (c)

The interquartile range is drawn even, so 1.5 times it is whole and every answer is a whole number.

## Checks (on every question)

The values are re-read from the rendered text (``_reread``), and the quartiles are found a second way with
``statistics.median`` on the halves. The two must agree with the keys.
"""

from __future__ import annotations

import random
import statistics

from ..core import VerificationError, attempts, check, generator
from ._exd_x_sc4_kit import count_part
from .f10_stem_and_leaf_build import _draw, _key, _plot_text, _reread, _rows

# stems: the range of stems used; item and val describe the data; stmt introduces the plot.
_CONTEXTS = (
    dict(
        stems=(1, 6),
        stmt="The stem-and-leaf plot shows the number of text messages that {n} students sent yesterday.",
    ),
    dict(
        stems=(1, 5),
        stmt="The stem-and-leaf plot shows how many minutes each of {n} customers waited in a queue.",
    ),
    dict(
        stems=(3, 9),
        stmt="The stem-and-leaf plot shows the points a basketball team scored in each of its {n} games.",
    ),
    dict(
        stems=(2, 7),
        stmt="The stem-and-leaf plot shows the lengths in centimetres of {n} fish caught in a fishing competition.",
    ),
    dict(
        stems=(1, 8),
        stmt="The stem-and-leaf plot shows the number of runs a batter scored in each of {n} innings.",
    ),
)


def _ordinal(k: int) -> str:
    """3 -> "3rd", 11 -> "11th", 12 -> "12th", 21 -> "21st"."""
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


_TAIL = (
    "Leaves in each row are in order, smallest first. When finding the quartiles, leave the median out of both halves.",
    "The leaves in each row are written smallest first. Leave the median out of both halves when finding the quartiles.",
)


def _build(rng: random.Random, ctx: dict) -> dict:
    for _ in attempts(600, "a plot with an even IQR and distinct, interior quartiles"):
        n = rng.choice((11, 15))
        values = _draw(rng, ctx["stems"], rng.randint(4, min(6, ctx["stems"][1] - ctx["stems"][0] + 1)), n)
        half = n // 2
        q1, med, q3 = values[half // 2], values[half], values[half + 1 + half // 2]
        iqr = q3 - q1
        if iqr <= 0 or iqr % 2 or not (values[0] < q1 < med < q3 < values[-1]):
            continue
        fence = q3 + 3 * iqr // 2
        if len({q1, q3, iqr, fence, values[0], values[-1], med}) < 7:
            continue
        break
    rows = _rows(values)
    key = _key(rng, values, {q1, q3})
    stmt = ctx["stmt"].format(n=n)
    text = f"{stmt} {key} Plot, rows written as stem | leaves: {_plot_text(rows)}. {rng.choice(_TAIL)}"
    seen = sorted(_reread(text))
    check(seen == values, "the plot text disagrees with the values")
    lower, upper = seen[: n // 2], seen[n // 2 + 1 :]
    check((statistics.median(lower), statistics.median(upper)) == (q1, q3), "the quartiles disagree")
    check(iqr == q3 - q1 and fence == q3 + 1.5 * iqr, "the boundary disagrees")

    qa = half // 2
    part_a = count_part(
        "Find the lower quartile, Q1.",
        q1,
        2,
        [
            f"There are {n} values, so the median is the {_ordinal(half + 1)} and the lower half is the first {half}.",
            f"Q1 is the middle of the lower half, the {_ordinal(qa + 1)} value: {q1}.",
        ],
        [
            (values[0], "gives-the-smallest-value-as-the-lower-quartile"),
            (values[qa - 1], "quartile-one-position-out"),
            (values[qa + 1], "quartile-one-position-out"),
            (q1 % 10, "reads-the-leaf-alone"),
        ],
        lo=1,
    )
    part_b = count_part(
        "Find the upper quartile, Q3.",
        q3,
        2,
        [
            f"The upper half is the last {half} values. Q3 is its middle value, the {_ordinal(half + qa + 2)} overall: {q3}."
        ],
        [
            (values[-1], "gives-the-largest-value-as-the-upper-quartile"),
            (values[half + qa], "quartile-one-position-out"),
            (values[half + qa + 2], "quartile-one-position-out"),
            (q3 % 10, "reads-the-leaf-alone"),
        ],
        lo=1,
    )
    part_c = count_part(
        "Using your answers to (a) and (b), find the interquartile range.",
        iqr,
        1,
        [f"IQR = Q3 - Q1 = {q3} - {q1} = {iqr}."],
        [
            (values[-1] - values[0], "gives-the-range-as-the-iqr"),
            (q3 - med, "subtracts-the-median-from-q3"),
            (q3 + q1, "adds-the-quartiles"),
            (med - q1, "subtracts-q1-from-the-median"),
        ],
        follow_through="part_b-part_a",
        lo=1,
    )
    part_d = count_part(
        "A value is treated as an outlier if it is more than 1.5 times the interquartile range above Q3. "
        "Using your answers to (b) and (c), find the number above which a value is an outlier.",
        fence,
        2,
        [f"Boundary = Q3 + 1.5 × IQR = {q3} + 1.5 × {iqr} = {q3} + {3 * iqr // 2} = {fence}."],
        [
            (q3 + iqr, "uses-one-times-the-iqr"),
            (3 * iqr // 2, "leaves-out-q3-from-the-boundary"),
            (q1 - 3 * iqr // 2, "uses-the-lower-boundary"),
            (q3 + 3 * iqr, "uses-three-times-the-iqr"),
        ],
        follow_through="part_b+3*part_c/2",
        lo=1,
    )
    return dict(text=text, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_fc_stem_and_leaf", level=10, presets={"exam": {"tier": "exam"}})
def exd_x_fc_stem_and_leaf(rng: random.Random, tier: str = "exam"):
    """Q1, Q3, the interquartile range and the outlier boundary read from one stem-and-leaf plot."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(60, "a plot with enough distinct wrong answers"):
        try:
            return _build(rng, ctx)
        except VerificationError:
            continue
    raise AssertionError("unreachable")
