"""Generates `p16a_group_centre_difference` and `p16a_trend_value` questions: the difference between two groups'
centres and a value read from a fitted trend.

Senior statistics computations, part 1.

Two generators, both senior ``acara-senior-v8-4``, domain
``statistics-probability``, ``response_family: scalar_expression``:

- ``p16a_group_centre_difference``
  (``senior-implement-the-statistical-investigation-process-to-answer-questions``,
  ACMGM033) -- given two stated, generated datasets for a numerical variable
  measured across two named groups, compute the difference between the two
  groups' means or (if the prompt says so) medians. Answer: ``fraction``
  (exact -- means/medians of small integer datasets are always exact
  rationals). Genuinely different from the shipping ``calculate-mean`` /
  ``calculate-median`` / ``mean-median-mode-one-set`` skills, which each
  report ONE dataset's own statistic: this skill's graded value is always
  the SIGNED DIFFERENCE between two groups' statistics, never a single
  group's value alone.

- ``p16a_trend_value``
  (``senior-implement-the-statistical-investigation-process-to-answer-questions-acmgm093``,
  ACMGM093) -- given a stated linear trend for a time series, compute the
  trend value at a stated point. Answer: ``number``, to 1 decimal place.
  Deliberately builds ONLY the "trend value" half of the skill's scope ("a
  moving average of a stated window width OR a linear trend value"); the
  moving-average half already ships as ``moving-average``
  (``mathgen/generators/curriculum_expansion.py::moving_average``), so no
  question here ever asks for a moving average, and no tier here computes
  a mean over a window at all.
"""

from __future__ import annotations

import statistics
from fractions import Fraction

from ..core import (
    check,
    exact_decimal_text,
    generator,
    numeric_answer,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)

# --------------------------------------------------------------------------
# p16a_group_centre_difference (ACMGM033)
# --------------------------------------------------------------------------

_CONTEXTS = [
    ("Year 11", "Year 12", "resting heart rate (beats per minute)"),
    ("the home team", "the away team", "goals scored per match"),
    ("the morning shift", "the afternoon shift", "number of items packed per hour"),
    ("Group A", "Group B", "reaction time (hundredths of a second)"),
    ("the northern orchard", "the southern orchard", "apples per tree"),
    ("Class 11A", "Class 11B", "hours of sleep per night"),
]


def _verify_mean_diff(values_a: list[int], values_b: list[int], diff: Fraction) -> None:
    """Construction sums each group with a manual accumulator loop; this
    check recomputes both totals with Python's built-in ``sum`` -- a
    different code path over the same data, catching a mistranscribed list
    or an off-by-one in the accumulator."""
    mean_a = Fraction(sum(values_a), len(values_a))
    mean_b = Fraction(sum(values_b), len(values_b))
    check(mean_a - mean_b == diff, "sum()-based mean difference disagrees with construction")


def _verify_median_diff(values_a: list[int], values_b: list[int], diff: Fraction) -> None:
    """Independent route: Python's stdlib ``statistics.median`` (a different
    implementation from this module's own manual sort-and-pick-middle)."""
    med_a = Fraction(statistics.median(values_a)).limit_denominator(1000)
    med_b = Fraction(statistics.median(values_b)).limit_denominator(1000)
    check(med_a - med_b == diff, "statistics.median-based difference disagrees with construction")


def _manual_median(values: list[int]) -> Fraction:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return Fraction(ordered[mid])
    return Fraction(ordered[mid - 1] + ordered[mid], 2)


def _measure_working(measure_word, name_a, name_b, values_a, values_b):
    """One line per group showing how its mean or median comes from the stated values."""
    lines = []
    for name, values in ((name_a, values_a), (name_b, values_b)):
        name = name[0].upper() + name[1:]
        if measure_word == "mean":
            total = sum(values)
            mean = rational_text(Fraction(total, len(values)))
            lines.append(f"{name}: the values add to {total}, so the mean is {total} ÷ {len(values)} = {mean}.")
        else:
            ordered = sorted(values)
            mid = len(ordered) // 2
            med = rational_text(_manual_median(values))
            if len(ordered) % 2:
                how = f"the middle value is {ordered[mid]}"
            else:
                how = f"the middle two values are {ordered[mid - 1]} and {ordered[mid]}"
            lines.append(
                f"{name} median: the sorted values are {', '.join(str(v) for v in ordered)}; {how}"
                + (
                    f", and ({ordered[mid - 1]} + {ordered[mid]}) ÷ 2 = {med}."
                    if len(ordered) % 2 == 0
                    else f", so the median is {med}."
                )
            )
    return lines


def _verify_weighted_mean_diff(table_a: dict, table_b: dict, diff: Fraction) -> None:
    """Independent route for the hard (frequency-table) tier: EXPAND each
    table into a flat list of observations and call ``statistics.mean`` --
    a different approach from construction's own weighted
    sum(value*freq)/sum(freq)."""
    flat_a = [v for v, f in table_a.items() for _ in range(f)]
    flat_b = [v for v, f in table_b.items() for _ in range(f)]
    mean_a = Fraction(statistics.mean(flat_a)).limit_denominator(10000)
    mean_b = Fraction(statistics.mean(flat_b)).limit_denominator(10000)
    check(mean_a - mean_b == diff, "expand-then-average disagrees with weighted-sum construction")


def _weighted_mean(table: dict) -> Fraction:
    total_n = sum(table.values())
    total = sum(v * f for v, f in table.items())
    return Fraction(total, total_n)


@generator(
    "p16a_group_centre_difference",
    level=12,
    presets={
        "easy": dict(variant="mean_plain"),
        "medium": dict(variant="median_plain"),
        "hard": dict(variant="mean_table"),
        "exam": dict(variant="implicit_measure"),
    },
)
def p16a_group_centre_difference(rng, variant="mean_plain"):
    """Two stated, generated datasets for a numerical variable measured
    across two named groups: compute the difference between the two groups'
    means or (if the prompt says so) medians. The graded value is always
    the SIGNED difference (Group 1 minus Group 2), never a single group's
    own statistic."""
    name_a, name_b, quantity = rng.choice(_CONTEXTS)
    # Every context name is written for mid-sentence use ("the morning
    # shift"); capitalise a copy for the templates that put it at the start
    # of a sentence, so "the morning shift recorded ..." never opens a
    # sentence lowercase.
    name_a_cap = name_a[0].upper() + name_a[1:]
    name_b_cap = name_b[0].upper() + name_b[1:]

    if variant == "mean_plain":
        n = 5
        values_a = [rng.randint(4, 40) for _ in range(n)]
        values_b = [rng.randint(4, 40) for _ in range(n)]
        mean_a = Fraction(sum(values_a), n)
        mean_b = Fraction(sum(values_b), n)
        diff = mean_a - mean_b
        _verify_mean_diff(values_a, values_b, diff)
        measure_word = "mean"
        measure_phrase = "the mean"

        wrong_order = mean_b - mean_a
        sum_diff = Fraction(sum(values_a) - sum(values_b))  # forgot to divide by n
        only_a = mean_a

    elif variant == "median_plain":
        n = 6  # even, so the median is the average of the two middle values
        values_a = [rng.randint(4, 40) for _ in range(n)]
        values_b = [rng.randint(4, 40) for _ in range(n)]
        med_a = _manual_median(values_a)
        med_b = _manual_median(values_b)
        diff = med_a - med_b
        _verify_median_diff(values_a, values_b, diff)
        measure_word = "median"
        measure_phrase = "the median"

        wrong_order = med_b - med_a
        # uses the mean instead of the median
        mean_confusion = Fraction(sum(values_a), n) - Fraction(sum(values_b), n)
        only_a = med_a

    elif variant == "mean_table":
        # A small frequency table per group: {value: frequency}. Requires a
        # weighted mean, not a plain sum/n -- a genuinely more complex
        # computation than "mean_plain" above.
        def draw_table():
            distinct = sorted(rng.sample(range(2, 12), 4))
            return {v: rng.randint(2, 6) for v in distinct}

        table_a = draw_table()
        table_b = draw_table()
        mean_a = _weighted_mean(table_a)
        mean_b = _weighted_mean(table_b)
        diff = mean_a - mean_b
        _verify_weighted_mean_diff(table_a, table_b, diff)
        measure_word = "mean"
        measure_phrase = "the mean"

        wrong_order = mean_b - mean_a
        # unweighted mean of the DISTINCT values, ignoring frequency
        unweighted_a = Fraction(sum(table_a.keys()), len(table_a))
        unweighted_b = Fraction(sum(table_b.keys()), len(table_b))
        unweighted_diff = unweighted_a - unweighted_b
        only_a = mean_a
        # raw weighted-sum difference (forgets to divide by the total count)
        weighted_sum_diff = sum(v * f for v, f in table_a.items()) - sum(v * f for v, f in table_b.items())
        # forgot to divide by the total count AND forgot group B: reports
        # Class A's raw weighted total on its own. Always a strictly
        # positive integer (every value/frequency drawn is >= 2), so this
        # never coincides with a zero (or otherwise fractional) true
        # difference the way weighted_sum_diff occasionally does when both
        # groups' raw totals happen to match.
        raw_total_a = sum(v * f for v, f in table_a.items())

    elif variant == "implicit_measure":
        # Neither "mean" nor "median" nor any periphrasis of either is
        # named: the prompt states a REASON that determines which measure
        # applies (outlier resistance), and the data are built so the
        # mean-difference and median-difference genuinely DIFFER -- so
        # picking the wrong measure is a real, distinct wrong answer, not
        # just a relabelled version of the right one. This is the step no
        # smaller tier requires: recognising which measure the reasoning
        # calls for, not just computing a named one.
        use_median = rng.random() < 0.5
        if use_median:
            # One group carries a genuine outlier, so its mean is dragged
            # away from its median -- exactly the situation the median is
            # for. Redraw (bounded) on the rare coincidence where the
            # median difference and the mean-confusion difference land on
            # the same value anyway, so a wrong-measure pick is always a
            # genuinely different, wrong answer.
            for _ in range(50):
                n = 6
                values_a = [rng.randint(4, 20) for _ in range(n - 1)]
                outlier_a = rng.randint(55, 95)
                values_a.append(outlier_a)
                values_b = [rng.randint(4, 20) for _ in range(n)]
                med_a = _manual_median(values_a)
                med_b = _manual_median(values_b)
                diff = med_a - med_b
                mean_confusion_value = Fraction(sum(values_a), n) - Fraction(sum(values_b), n)
                if diff != mean_confusion_value:
                    break
            else:
                raise ValueError("could not separate the median difference from the mean difference")
            _verify_median_diff(values_a, values_b, diff)
            measure_word = "median"
            reason_sentence = f"{name_a_cap}'s data contain an outlier ({outlier_a})."
            instruction_sentence = (
                "Compare the two groups' typical values using the measure that is not affected by an extreme value"
            )
            wrong_order = med_b - med_a
            confusion = mean_confusion_value  # picked the wrong (outlier-sensitive) measure
            only_a = med_a
            sum_diff = sum(values_a) - sum(values_b)
        else:
            # No outlier anywhere: every value sits in a tight band, so
            # there is nothing for the median to resist and the ordinary
            # average is the intended measure. Same bounded redraw as
            # above, for the same reason.
            for _ in range(50):
                n = 5
                lo = rng.randint(10, 20)
                values_a = [rng.randint(lo, lo + 8) for _ in range(n)]
                values_b = [rng.randint(lo, lo + 8) for _ in range(n)]
                mean_a = Fraction(sum(values_a), n)
                mean_b = Fraction(sum(values_b), n)
                diff = mean_a - mean_b
                median_confusion_value = _manual_median(values_a) - _manual_median(values_b)
                if diff != median_confusion_value:
                    break
            else:
                raise ValueError("could not separate the mean difference from the median difference")
            _verify_mean_diff(values_a, values_b, diff)
            measure_word = "mean"
            reason_sentence = "None of the data in either group is an outlier."
            instruction_sentence = "Compare the two groups' typical values using the ordinary average of all the data"
            wrong_order = mean_b - mean_a
            confusion = median_confusion_value  # picked the wrong (outlier-resistant) measure
            only_a = mean_a
            sum_diff = sum(values_a) - sum(values_b)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    answer_text = rational_text(diff)

    if variant == "mean_table":
        rows_a = ", ".join(f"{v} ({f} times)" for v, f in table_a.items())
        rows_b = ", ".join(f"{v} ({f} times)" for v, f in table_b.items())
        question_text = (
            f"{name_a_cap} and {name_b} were surveyed for {quantity}. Results for {name_a}: {rows_a}. "
            f"Results for {name_b}: {rows_b}. Find {measure_phrase} of {name_a} minus "
            f"{measure_phrase} of {name_b}, as a fraction in lowest terms."
        )
        working = [
            f"{name_a} mean = {rational_text(mean_a)}. {name_b} mean = {rational_text(mean_b)}.",
            f"Difference = {rational_text(mean_a)} - {rational_text(mean_b)} = {answer_text}.",
        ]
        candidates = [
            str(raw_total_a),
            str(weighted_sum_diff),
            rational_text(wrong_order),
            rational_text(unweighted_diff),
            rational_text(only_a),
        ]
    elif variant == "implicit_measure":
        list_a = ", ".join(str(v) for v in values_a)
        list_b = ", ".join(str(v) for v in values_b)
        question_text = (
            f"{name_a_cap} recorded {quantity}: {list_a}. {name_b_cap} recorded {quantity}: "
            f"{list_b}. {reason_sentence} {instruction_sentence}: find that measure for "
            f"{name_a} minus the same measure for {name_b}, as a fraction in lowest terms."
        )
        working = [
            *_measure_working(measure_word, name_a, name_b, values_a, values_b),
            f"Difference ({name_a} - {name_b}) = {answer_text}.",
        ]
        # sum_diff first: guarantees an integer-shaped option ("used the
        # group totals instead of the appropriate measure") whenever the
        # true difference happens to be an integer too, so the answer is
        # never the lone option of its shape.
        candidates = [
            str(sum_diff),
            rational_text(confusion),
            rational_text(wrong_order),
            rational_text(only_a),
        ]
    else:
        list_a = ", ".join(str(v) for v in values_a)
        list_b = ", ".join(str(v) for v in values_b)
        question_text = phrase(
            rng,
            [
                "{name_a_cap} recorded {quantity}: {list_a}. {name_b_cap} recorded {quantity}: "
                "{list_b}. Find {measure_phrase} of {name_a} minus {measure_phrase} of {name_b}, "
                "as a fraction in lowest terms.",
                "A study compared {quantity} for {name_a} ({list_a}) and {name_b} ({list_b}). "
                "Find {measure_phrase} of {name_a} minus {measure_phrase} of {name_b}, "
                "as a fraction in lowest terms.",
                "{name_a_cap}'s {quantity}: {list_a}. {name_b_cap}'s {quantity}: {list_b}. "
                "By how much does {measure_phrase} of {name_a} exceed {measure_phrase} of "
                "{name_b}? Give your answer as {measure_phrase} of {name_a} minus "
                "{measure_phrase} of {name_b}, as a fraction in lowest terms.",
            ],
            name_a=name_a,
            name_b=name_b,
            name_a_cap=name_a_cap,
            name_b_cap=name_b_cap,
            quantity=quantity,
            list_a=list_a,
            list_b=list_b,
            measure_phrase=measure_phrase,
        )
        working = [
            *_measure_working(measure_word, name_a, name_b, values_a, values_b),
            f"Difference ({name_a} - {name_b}) = {answer_text}.",
        ]
        candidates = [rational_text(wrong_order), rational_text(only_a)]
        if variant == "median_plain":
            candidates.append(rational_text(mean_confusion))
        else:
            candidates.append(rational_text(sum_diff))

    distractors = pick_distractors(answer_text, candidates, n=3, unordered=False)
    return dict(
        text=question_text,
        latex=question_text,
        answer={"p": diff.numerator, "q": diff.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# --------------------------------------------------------------------------
# p16a_trend_value (ACMGM093)
# --------------------------------------------------------------------------

_TIME_CONTEXTS = [
    ("a cafe's quarterly coffee sales (cups, thousands)", "period"),
    ("a gym's monthly new memberships", "month"),
    ("a factory's weekly output (units)", "week"),
    ("a clinic's yearly patient visits (hundreds)", "year"),
]

# The period_label and seasonal_two_point variants group a sub-annual unit
# into calendar years, so they only ever draw a context whose unit is
# genuinely sub-annual ("year" itself has no such grouping) -- and the
# group size must be the REAL number of that unit per year, not a
# once-hardcoded 4 that was only ever true for "period" (quarters).
_PERIOD_LABEL_CONTEXTS = [c for c in _TIME_CONTEXTS if c[1] != "year"]
_PER_GROUP_BY_UNIT = {"period": 4, "month": 12, "week": 52}
_LABEL_WORD_BY_UNIT = {"period": "quarter", "month": "month", "week": "week"}


def _t_to_label(t: int, per_group: int) -> tuple[int, int]:
    """Inverse of `t = (year - 1) * per_group + sub`: the (year, sub) a
    given t falls in."""
    year, sub0 = divmod(t - 1, per_group)
    return year + 1, sub0 + 1


def _verify_trend_value(a: Fraction, b: Fraction, t: Fraction, value: Fraction) -> None:
    """Independent route: recompute via the point-slope form anchored at
    t=0 shifted to t=1 (T(t) = T(1) + b*(t-1)) instead of the direct
    T(t) = a + b*t construction uses -- a different arithmetic path to the
    same value."""
    t1 = a + b  # T(1)
    recomputed = t1 + b * (t - 1)
    check(recomputed == value, "point-slope recomputation disagrees with a + b*t construction")


def _step(base, slope, t) -> str:
    """ "base + 1.5×t" or "base - 0.5×t" with the slope as a plain decimal, never a fraction or a bracketed negative."""
    sign = "+" if slope >= 0 else "-"
    return f"{base} {sign} {exact_decimal_text(abs(slope))}×{t}"


@generator(
    "p16a_trend_value",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="period_label"),
        "hard": dict(variant="two_points"),
        "exam": dict(variant="seasonal_two_point"),
    },
)
def p16a_trend_value(rng, variant="direct"):
    """A linear trend T(t) = a + b*t has been fitted to a time series (t
    counts periods from 1). Compute the trend value at a stated point.
    Never asks for a moving average -- that half of the scope already
    ships as the ``moving-average`` skill."""
    if variant in ("period_label", "seasonal_two_point"):
        context, unit = rng.choice(_PERIOD_LABEL_CONTEXTS)
    else:
        context, unit = rng.choice(_TIME_CONTEXTS)

    def _draw_ab():
        half_steps = rng.choice([v for v in range(-8, 9) if v != 0])  # b*2, nonzero
        return Fraction(half_steps, 2), Fraction(rng.randint(10, 80))

    # Every _TIME_CONTEXTS quantity is a count (coffee sales, memberships,
    # factory output, patient visits): a negative trend value is not a
    # meaningful answer for any of them. Redraw (a, b) -- bounded -- until
    # every value this variant DISPLAYS (any given point plus the target)
    # rounds to non-negative, rather than only checking the final answer.

    if variant == "direct":
        for _ in range(200):
            b, a = _draw_ab()
            t_star = rng.randint(9, 14)
            value = a + b * t_star
            if round_half_up(value, 1) >= 0:
                break
        else:
            raise ValueError("could not construct a non-negative trend value (direct)")
        _verify_trend_value(a, b, Fraction(t_star), value)
        answer = numeric_answer(round_half_up(value, 1))
        answer_text = stated_precision_string(round_half_up(value, 1), 1)
        b_str = f"({b})" if b < 0 else str(b)
        question_text = (
            f"For {context}, a linear trend has been fitted: T(t) = {a} + {b_str}\u00d7t, where t "
            f"counts each {unit} starting from t = 1. Find the trend value T({t_star}), to 1 "
            f"decimal place."
        )
        working = [f"T({t_star}) = {_step(a, b, t_star)} = {answer_text}."]
        wrong_a = a - b * t_star  # sign error on b
        wrong_shift = a + b * (t_star + 1)  # off-by-one in t
        wrong_swap = b + a * t_star  # swapped roles of a and b

    elif variant == "period_label":
        # Compound period label ("quarter 3 of Year 5") that must first be
        # converted to a single t index before evaluating -- an extra
        # reasoning step beyond direct substitution. per_group is the REAL
        # count of that unit per year (4 quarters, 12 months, 52 weeks),
        # never a hardcoded 4.
        per_group = _PER_GROUP_BY_UNIT[unit]
        label_word = _LABEL_WORD_BY_UNIT[unit]
        for _ in range(200):
            b, a = _draw_ab()
            year = rng.randint(2, 5)
            sub = rng.randint(1, per_group)
            t_star = (year - 1) * per_group + sub
            value = a + b * t_star
            if round_half_up(value, 1) >= 0:
                break
        else:
            raise ValueError("could not construct a non-negative trend value (period_label)")
        _verify_trend_value(a, b, Fraction(t_star), value)
        answer = numeric_answer(round_half_up(value, 1))
        answer_text = stated_precision_string(round_half_up(value, 1), 1)
        b_str = f"({b})" if b < 0 else str(b)
        question_text = (
            f"For {context}, a linear trend has been fitted: T(t) = {a} + {b_str}\u00d7t, where t = 1 "
            f"is the 1st {unit} of Year 1 and each Year has {per_group} {unit}s numbered in order. "
            f"Find the trend value for {label_word} {sub} of Year {year}, to 1 decimal place."
        )
        working = [
            f"{label_word} {sub} of Year {year} is t = {(year - 1) * per_group} + {sub} = {t_star}.",
            f"T({t_star}) = {_step(a, b, t_star)} = {answer_text}.",
        ]
        wrong_a = a + b * (year)  # uses the year number as t
        wrong_shift = a + b * (t_star - per_group)  # off-by-one year
        wrong_swap = a + b * sub  # ignores the year entirely, uses only the sub-period
        wrong_swap = a + b * sub  # ignores the year entirely, uses only the sub-period

    elif variant == "two_points":
        # Two points on the trend line are given instead of the equation --
        # the learner must derive a and b via the two-point line fit first.
        for _ in range(200):
            b, a = _draw_ab()
            t1, t2 = 2, 2 + rng.randint(2, 6)
            y1 = a + b * t1
            y2 = a + b * t2
            t_star = t2 + rng.randint(3, 8)
            value = a + b * t_star
            if round_half_up(y1, 1) >= 0 and round_half_up(y2, 1) >= 0 and round_half_up(value, 1) >= 0:
                break
        else:
            raise ValueError("could not construct non-negative trend values (two_points)")
        _verify_trend_value(a, b, Fraction(t_star), value)
        answer = numeric_answer(round_half_up(value, 1))
        answer_text = stated_precision_string(round_half_up(value, 1), 1)
        y1_text = stated_precision_string(y1, 1)
        y2_text = stated_precision_string(y2, 1)
        question_text = (
            f"For {context}, the trend value in {unit} {t1} is {y1_text} and in {unit} {t2} is "
            f"{y2_text}. The trend is linear. Find the trend value in {unit} {t_star}, to 1 "
            f"decimal place."
        )
        b_recomputed = (y2 - y1) / (t2 - t1)
        a_recomputed = y1 - b_recomputed * t1
        check(b_recomputed == b and a_recomputed == a, "two-point line fit did not recover the construction's own a, b")
        b_recomputed_str = paren_negative(b_recomputed)
        working = [
            f"slope = ({y2_text} - {y1_text}) ÷ ({t2} - {t1}) = {exact_decimal_text(b_recomputed)}.",
            f"intercept = {_step(y1_text, -b_recomputed, t1)} = {a_recomputed}.",
            f"T({t_star}) = {_step(a_recomputed, b_recomputed, t_star)} = {answer_text}.",
        ]
        wrong_a = y2 + b_recomputed * (t_star - t2)  # anchors at (t2, y2) but reuses t_star wrongly (double-shift)
        wrong_shift = a_recomputed + b_recomputed * (t_star - 1)  # off-by-one
        wrong_swap = y1 + (y2 - y1) * (t_star - t1)  # uses raw point difference, not the per-unit slope

    elif variant == "seasonal_two_point":
        # Compounds BOTH smaller-tier reasoning steps: two NON-CONSECUTIVE
        # trend values given by compound period label (medium's own
        # label-to-t conversion), from which a and b must be derived by a
        # two-point line fit (hard's own step), then evaluated at a THIRD
        # period label that must also be converted. Genuinely harder than
        # hard (which hands t1, t2 as bare numbers with no conversion) and
        # was replaced from an earlier "read off the common difference"
        # design that was, in fact, easier than hard.
        per_group = _PER_GROUP_BY_UNIT[unit]
        label_word = _LABEL_WORD_BY_UNIT[unit]
        for _ in range(200):
            b, a = _draw_ab()
            year1 = rng.randint(2, 3)
            sub1 = rng.randint(1, per_group)
            t1 = (year1 - 1) * per_group + sub1
            gap = rng.randint(2, 6)
            t2 = t1 + gap
            year2, sub2 = _t_to_label(t2, per_group)
            t_star = t2 + rng.randint(3, 8)
            year3, sub3 = _t_to_label(t_star, per_group)

            y1 = a + b * t1
            y2 = a + b * t2
            value = a + b * t_star
            if round_half_up(y1, 1) >= 0 and round_half_up(y2, 1) >= 0 and round_half_up(value, 1) >= 0:
                break
        else:
            raise ValueError("could not construct non-negative trend values (seasonal_two_point)")
        _verify_trend_value(a, b, Fraction(t_star), value)
        answer = numeric_answer(round_half_up(value, 1))
        answer_text = stated_precision_string(round_half_up(value, 1), 1)
        y1_text = stated_precision_string(y1, 1)
        y2_text = stated_precision_string(y2, 1)
        question_text = (
            f"For {context}, the trend value for {label_word} {sub1} of Year {year1} is "
            f"{y1_text} and the trend value for {label_word} {sub2} of Year {year2} is "
            f"{y2_text}. The trend is linear, with t = 1 the 1st {unit} of Year 1 and "
            f"{per_group} {unit}s per Year. Find the trend value for {label_word} {sub3} "
            f"of Year {year3}, to 1 decimal place."
        )
        b_recomputed = (y2 - y1) / (t2 - t1)
        a_recomputed = y1 - b_recomputed * t1
        check(
            b_recomputed == b and a_recomputed == a,
            "two-point line fit (seasonal) did not recover the construction's own a, b",
        )
        b_recomputed_str = paren_negative(b_recomputed)
        y1_text_paren = f"({y1_text})" if y1 < 0 else y1_text
        working = [
            f"{label_word} {sub1} of Year {year1} is t = {t1}; {label_word} {sub2} of Year {year2} is t = {t2}.",
            f"slope = ({y2_text} - {y1_text_paren}) ÷ ({t2} - {t1}) = {exact_decimal_text(b_recomputed)}.",
            f"intercept = {_step(y1_text_paren, -b_recomputed, t1)} = {a_recomputed}.",
            f"{label_word} {sub3} of Year {year3} is t = {t_star}.",
            f"T({t_star}) = {_step(a_recomputed, b_recomputed, t_star)} = {answer_text}.",
        ]
        # forgot to convert the TARGET label to t: used the within-year
        # position alone, dropping the (year - 1) * per_group offset
        wrong_a = a_recomputed + b_recomputed * sub3
        wrong_shift = a_recomputed + b_recomputed * (t_star - 1)  # off-by-one in the final t
        wrong_swap = y1 + (y2 - y1) * (t_star - t1)  # uses the raw point difference, not the per-unit slope
    else:
        raise ValueError(f"unknown variant {variant!r}")

    def fmt(v: Fraction) -> str:
        return stated_precision_string(round_half_up(v, 1), 1)

    candidates = [fmt(wrong_a), fmt(wrong_shift), fmt(wrong_swap)]
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=Fraction(1, 20))
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
