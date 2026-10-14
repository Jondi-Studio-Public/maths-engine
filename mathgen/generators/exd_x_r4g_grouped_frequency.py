"""Exam chain for a grouped frequency table with class intervals.

(a) the estimated sum of all the values, from class midpoints                  number, 2 marks
(b) the estimated mean (follow-through from (a))                               number, 1 mark
(c) the lower limit of the class that contains the median                      number, 2 marks
(d) the estimated mean after some more values are added (follow-through (a))   number, 2 marks

Each answer is re-derived from the table parsed back out of the stem. The follow-through for (b) and (d) uses the
learner's own (a), so a wrong sum still earns the later marks for working from it correctly.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4g_kit import clear_of_tie, number_part, number_text, retry
from .r4g_grouped_frequency_build import _intro, _reread, _table_sentence, draw_table


@distinct_part_answers
def _build(rng):
    for _ in attempts(500, "a table with a clean mean before and after extra values"):
        ctx, t = draw_table(rng, classes=(5, 6), n_range=(30, 60), same_median_class=True)
        extra_n = rng.choice((5, 8, 10))
        extra_mean = Fraction(rng.randint(int(t.mids[0]), int(t.mids[-1])))
        new_mean = (t.total + extra_n * extra_mean) / (t.n + extra_n)
        if clear_of_tie(t.mean, 1) and clear_of_tie(new_mean, 1) and extra_mean != t.mean:
            break
    total, n, mean = t.total, t.n, t.mean
    mi = t.median_index()
    text = (
        f"{_intro(ctx)} {_table_sentence(ctx, t)} "
        f"Another {extra_n} values are collected later; their mean is {int(extra_mean)} {ctx['unit']}."
    )
    lows, width, freqs = _reread(text)
    check(sum(f * (Fraction(2 * lo + width, 2)) for f, lo in zip(freqs, lows)) == total, "sum re-derivation differs")
    check(sum(freqs) == n, "total frequency re-derivation differs")
    places_a = 0 if total.denominator == 1 else 1
    ta = number_text(total, places_a)
    combined_total = total + extra_n * extra_mean
    parts = [
        number_part(
            "Use the class midpoints to estimate the sum of all the values in the table.",
            total,
            places_a,
            2,
            [
                "Add frequency x midpoint for each class: "
                + " + ".join(f"{f} x {number_text(m, 1)}" for f, m in zip(t.freqs, t.mids))
                + f" = {ta}."
            ],
            [
                (sum(t.mids), "r4g-midpoints-summed-without-frequencies"),
                (sum(f * lo for f, lo in zip(t.freqs, t.lows)), "r4g-lower-limits-used-as-midpoints"),
                (sum(f * (lo + t.width) for f, lo in zip(t.freqs, t.lows)), "r4g-upper-limits-used-as-midpoints"),
                (n * t.mids[t.mode_index()], "r4g-modal-midpoint-times-total-frequency"),
            ],
        ),
        number_part(
            f"Hence estimate the mean of the {n} values, correct to 1 decimal place.",
            mean,
            1,
            1,
            [f"Mean = {ta} / {n} = {number_text(mean, 1)}."],
            [
                (total / len(t.freqs), "r4g-divided-by-number-of-classes"),
                (total / (n + 1), "r4g-divided-by-wrong-count"),
                (sum(t.mids) / len(t.mids), "r4g-midpoints-averaged-without-frequencies"),
                (Fraction(n, len(t.freqs)), "r4g-mean-of-the-frequencies"),
            ],
            follow_through=f"part_a/{n}",
        ),
        number_part(
            f"Find the lower limit of the class that contains the median, in {ctx['unit']}.",
            t.lows[mi],
            0,
            2,
            [
                f"The median is at position {(n + 1) / 2:g}; the cumulative frequency first reaches it in the class starting at {t.lows[mi]}."
            ],
            [
                (t.lows[t.mode_index()], "hist-modal-class-for-median-class"),
                (t.lows[len(t.lows) // 2], "hist-middle-class-for-median-class"),
                (t.lows[min(mi + 1, len(t.lows) - 1)], "r4g-median-class-off-by-one"),
                (t.lows[max(mi - 1, 0)], "r4g-median-class-off-by-one"),
            ],
        ),
        number_part(
            f"Use your sum from (a) to estimate the mean of all {n + extra_n} values together, correct to 1 decimal place.",
            new_mean,
            1,
            2,
            [
                f"Combined total = {ta} + {extra_n} x {int(extra_mean)} = {number_text(combined_total, places_a)}; divide by {n + extra_n}."
            ],
            [
                ((mean + extra_mean) / 2, "r4g-two-means-averaged"),
                (combined_total / n, "r4g-divided-by-wrong-count"),
                (total / (n + extra_n), "r4g-new-values-left-out-of-the-total"),
                (total + extra_n * extra_mean, "r4g-total-given-as-mean"),
            ],
            follow_through=f"(part_a + {extra_n * int(extra_mean)})/{n + extra_n}",
        ),
    ]
    return dict(text=text, parts=parts)


@generator("exd_x_r4g_grouped_frequency", level=11, presets={"exam": {}})
def exd_x_r4g_grouped_frequency(rng):
    return retry(_build, rng)
