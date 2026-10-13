"""Exam chain for smoothing a time series: moving means, a moving median and the effect of a spike.

(a) the 3-point moving mean at a stated point                                 number, 1 mark
(b) the 5-point moving mean at the same point                                 number, 2 marks
(c) the 3-point moving median at the same point                               number, 2 marks
(d) how much greater the 3-point moving mean is than the moving median        number, 1 mark, follow-through (a), (c)

The point is a spike (well above both neighbours), so the mean is pulled up and the median is not.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4g_kit import number_part, number_text, retry
from .r4g_moving_average_smoothing_build import _places, draw_series, mean_at, median_at, series_text


@distinct_part_answers
def _build(rng):
    ctx, s, day = draw_series(rng, spike=True)
    i = day - 1
    when = ctx["when"]
    m3, m5, md = mean_at(s, i, 3), mean_at(s, i, 5), median_at(s, i, 3)
    gap = m3 - md
    w3, w5 = s[i - 1 : i + 2], s[i - 2 : i + 3]
    text = (
        f"{series_text(ctx, s)} Smooth the series using centred moving values; the smoothed value for {when} {day} "
        f"uses the observations on either side of {when} {day}."
    )
    parts = [
        number_part(
            f"Find the 3-point moving mean for {when} {day}.",
            m3,
            _places(m3),
            1,
            [f"({' + '.join(map(str, w3))}) / 3 = {number_text(m3, _places(m3))}."],
            [
                (mean_at(s, i - 1, 3), "r4g-window-shifted-one-step"),
                (mean_at(s, i + 1, 3), "r4g-window-shifted-one-step"),
                (Fraction(sum(w3), 4), "r4g-window-sum-divided-by-wrong-count"),
                (Fraction(s[i]), "r4g-raw-value-for-smoothed"),
            ],
        ),
        number_part(
            f"Find the 5-point moving mean for {when} {day}.",
            m5,
            _places(m5),
            2,
            [f"({' + '.join(map(str, w5))}) / 5 = {number_text(m5, _places(m5))}."],
            [
                (m3, "r4g-wrong-window-length"),
                (Fraction(sum(w5), 4), "r4g-window-sum-divided-by-wrong-count"),
                (Fraction(sum(w5), 3), "r4g-window-sum-divided-by-wrong-count"),
                (Fraction(sorted(w5)[2]), "r4g-median-instead-of-mean-smoothing"),
            ],
        ),
        number_part(
            f"Find the 3-point moving median for {when} {day}.",
            md,
            0,
            2,
            [f"In order: {', '.join(map(str, sorted(w3)))}; the middle value is {number_text(md, 0)}."],
            [
                (m3, "r4g-mean-instead-of-median-smoothing"),
                (Fraction(s[i]), "r4g-raw-value-for-smoothed"),
                (Fraction(w3[1]), "r4g-middle-in-time-not-in-size"),
                (median_at(s, i + 1, 3), "r4g-window-shifted-one-step"),
            ],
        ),
        number_part(
            f"The observation for {when} {day} is unusually high. By how much is your 3-point moving mean from (a) greater than your 3-point moving median from (c)?",
            gap,
            _places(gap),
            1,
            [f"{number_text(m3, _places(m3))} - {number_text(md, 0)} = {number_text(gap, _places(gap))}."],
            [
                (m3 - s[i], "r4g-mean-minus-raw-value"),
                (Fraction(s[i]) - md, "r4g-raw-minus-median"),
                (m5 - md, "r4g-wrong-window-length"),
                (m3 + md, "r4g-smoothed-values-added"),
            ],
            follow_through="part_a - part_c",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=text, parts=parts)


@generator("exd_x_r4g_moving_average_smoothing", level=11, presets={"exam": {}})
def exd_x_r4g_moving_average_smoothing(rng):
    return retry(_build, rng)
