"""Exam chain for a back-to-back stem plot: medians, interquartile ranges and which group is more consistent.

(a) the median of the first group                                              number, 1 mark
(b) the interquartile range of the first group                                 number, 2 marks
(c) the interquartile range of the second group                                number, 2 marks
(d) how much larger the less consistent group's IQR is (follow-through (b), (c))  number, 1 mark

Each answer is recomputed from the values parsed back out of the stem text.
"""

from __future__ import annotations

from ..core import check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4g_kit import number_part, number_text, retry
from .r4g_back_to_back_stem_build import QUARTILE_NOTE, _places, _setup, iqr, median, quartiles


@distinct_part_answers
def _build(rng):
    ctx, base, a, b = _setup(rng)
    ga, gb = ctx["groups"]
    val = ctx["val"]
    check(median(a) != iqr(a) and iqr(a) != iqr(b), "two parts would share an answer")
    ia, ib = iqr(a), iqr(b)
    gap = abs(ia - ib)
    text = f"{base} {QUARTILE_NOTE}"
    s = sorted(a)
    sb = sorted(b)
    ma = median(a)
    ra, rb = max(a) - min(a), max(b) - min(b)
    parts = [
        number_part(
            f"Find the median {val} for {ga}.",
            ma,
            _places(ma),
            1,
            [f"{ga} in order: {', '.join(str(v) for v in s)}; the median is {number_text(ma, _places(ma))}."],
            [
                (median(b), "r4g-other-group-read"),
                (sum(a) / len(a), "r4g-mean-for-median"),
                (min(a) + (max(a) - min(a)) / 2, "r4g-midrange-for-median"),
                (s[len(s) // 2 - 1], "r4g-one-middle-value-only"),
            ],
        ),
        number_part(
            f"Find the interquartile range of the {val}s for {ga}.",
            ia,
            _places(ia),
            2,
            [
                f"Q1 = {number_text(quartiles(a)[0], 1)}, Q3 = {number_text(quartiles(a)[1], 1)}, so IQR = {number_text(ia, _places(ia))}."
            ],
            [
                (ra, "box-iqr-as-range"),
                (quartiles(a)[1], "box-quartile-for-iqr"),
                (quartiles(a)[1] - ma, "box-iqr-from-median"),
                (ib, "r4g-other-group-read"),
            ],
        ),
        number_part(
            f"Find the interquartile range of the {val}s for {gb}.",
            ib,
            _places(ib),
            2,
            [
                f"Q1 = {number_text(quartiles(b)[0], 1)}, Q3 = {number_text(quartiles(b)[1], 1)}, so IQR = {number_text(ib, _places(ib))}."
            ],
            [
                (rb, "box-iqr-as-range"),
                (quartiles(b)[1], "box-quartile-for-iqr"),
                (quartiles(b)[1] - median(b), "box-iqr-from-median"),
                (ia, "r4g-other-group-read"),
            ],
        ),
        number_part(
            f"The group with the smaller interquartile range has the more consistent {val}s. "
            "By how much is the interquartile range of the less consistent group greater than that of the more consistent group?",
            gap,
            _places(gap),
            1,
            [
                f"{ga if ia < ib else gb} is more consistent. Difference = {number_text(max(ia, ib), 1)} - {number_text(min(ia, ib), 1)} = {number_text(gap, _places(gap))}."
            ],
            [
                (abs(median(a) - median(b)), "box-iqr-difference-of-medians"),
                (abs(ra - rb), "box-iqr-difference-of-ranges"),
                (ia + ib, "box-iqr-difference-summed"),
                (max(ia, ib), "r4g-one-iqr-not-the-difference"),
            ],
            follow_through="Abs(part_b - part_c)",
        ),
    ]
    return dict(text=text, parts=parts)


@generator("exd_x_r4g_back_to_back_stem", level=11, presets={"exam": {}})
def exd_x_r4g_back_to_back_stem(rng):
    return retry(_build, rng)
