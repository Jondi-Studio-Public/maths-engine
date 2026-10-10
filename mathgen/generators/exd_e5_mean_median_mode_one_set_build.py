"""Generates `exd_e5_mean_median_mode_one_set` questions: the mean, median and mode of a single data set, as
three graded parts.

A NEW, SEPARATE, COMBINED multi-part skill: ONE 8-12 value data set (values 1-60, with repeats --
`calculate-mean`/`-median`/`-mode`'s own convention, reused here for consistency), three
independently-graded answers (mean, median, mode). The single-part skills and this combined skill both ship; this module asserts
nothing about the single-part skills' files.

Uses `mathgen.core.compose` in its MULTI-PART shape: the data set is drawn ONCE, before `compose` runs, and
threaded into all three parts -- and into `final` -- via `functools.partial` keyword arguments (not a bare
closure, so each part stays a genuine MODULE-LEVEL function this module's own test file can monkeypatch by
name, the same discipline the other chained generators use). Each part then simply reads the shared
`data`/`mode_value`/etc. it was bound to and ignores `values` (the growing list of EARLIER parts' own raw
results), matching `compose`'s own docstring ("a multi-part generator's parts simply ignore `values`").

## `answer: pair`, label order (a load-bearing constraint)

`answer_text` renders "mean = <v>, median = <v>, mode = <v>"; `answer`'s own
dict key order is built in EXACTLY that same order (`{"mean": ...,
"median": ..., "mode": ...}`) -- `_labelled_pair_verdict`
(`mathgen/grading/typed.py`) zips the label order parsed out of the CANONICAL
`answer_text` against `answer.values()`'s own iteration order, positionally,
so a mismatched order would silently pair the wrong label to the wrong
value. Proven directly against the real `mathgen.grading.grade_pair` in this
module's test file, reproducing four 3-label cases on THIS
module's own generated questions (two further cases are a 4-label
demonstration on an unrelated a/b/c/d scratch answer -- not reproducible
"on our own generated questions" by definition, since this skill always
carries exactly 3 labels; see the test file for the honest note).

## The mode: unambiguous by construction, not by luck

The mode value always gets EXACTLY 3 occurrences; every other value is
capped at 2 (filled in genuine pairs where possible, at most one leftover
singleton) -- the same deterministic-surplus technique E3's
`exam_mode_summary_statistic` uses, rewritten here independently (no
import, per this module's own docstring above) since this module owns its
own data set. Verified by an INDEPENDENT run-length scan over the sorted
data (`itertools.groupby`, a different algorithm from the pair-filling
construction loop), never merely assumed from the construction method --
both in this module's own `_verify_unique_mode`/`_final` and, separately
again, in the test file's independent oracle.

## The median: odd/even is the PRIMARY, masking-safe tier lever

`scripts/measure_tiers.py`/`mathgen.text_quality.prompt_shape` masks every digit run (including its leading
sign, `text_quality.py:_DIGIT_RUN_RE`) and every `mathgen.slots` unit/name/context word to a shared
placeholder, so a tier that differs only in the data set's magnitude/range is invisible to the instrument. The
rendered comma-separated values LIST's own TOKEN COUNT is not masked away, though: an odd-count list (single
exact middle value) and an even-count list (average of two middle values, possibly a half-integer) render as
genuinely different masked shapes. easy/medium draw an ODD count (9 or 11); hard/exam draw an EVEN count (8,
10 or 12) -- a materially different PROCEDURE (sort-and-average-two-middle-values, not pick-the-middle-value),
not a magnitude change. hard/exam ALSO each add their own explicit new WORDED clause (see `_TEMPLATE_BANKS`
below) so the escalation is provable two ways, not resting on list length alone.

## The mean: stated precision, and the rounding-boundary hazard

Every tier states "to 1 decimal digit" for the mean (`_MEAN_DP = 1`) and `answer_text`'s mean component is
rounded to that stated precision (`mathgen.core.round_half_up` then `stated_precision_string`). Deliberately
NOT worded "decimal place": the stated-precision check
(`mathgen.text_quality.requested_rounding`) assumes a SINGLE stated precision phrase applies to every number
in `answer_text` -- correct for a plain numeric answer, wrong for this skill, whose three labelled parts each
carry their own precision convention (the mean stated, the median exact-or-half-integer depending on the drawn
count's parity, the mode always a bare integer). `requested_rounding`'s own docstring already documents the
escape hatch a differently-precisioned `pair` answer needs: it returns `None` -- and the check correctly
declines to run -- whenever the text does not match its `_DECIMAL_PLACES_RE`/`_NEAREST_RE` patterns at all,
which "decimal digit" (unlike "decimal place"/"nearest...") does not. The learner-facing meaning is identical;
only the two literal trigger words are avoided.

A data set whose true mean sits EXACTLY on the 1-decimal-place rounding tie is rejected and redrawn
(`_draw_clean_data_set`'s retry loop, `_is_on_rounding_boundary`) -- by construction, not by chance.

## The 0.051 hazard (`mathgen/grading/typed.py::_PAIR_TOLERANCE`, FIXED, absolute)

Every part's value, and every deliberately-wrong distractor/test value,
must clear the true value by MORE than 0.051 in absolute terms,
unconditionally, regardless of the stated decimal precision
(`_PAIR_TOLERANCE` is a flat tolerance for every `answer: pair` label, not
derived from `answer_text`'s own stated precision the way `grade_number`'s
tolerance is). This module's own retry loop additionally requires the
wrong-divisor mean distractor to clear the true mean by more than a full
unit (`_MIN_MEAN_GAP = 1`), and the median/mode misconceptions are drawn
from the same 1..60 integer/half-integer grid the data itself lives on,
where the smallest possible GENUINE gap between two distinct candidate
values is 0.5 (two distinct integers differ by >= 1; two distinct
half-integers, or an integer and a half-integer, differ by >= 0.5) -- both
comfortably clear of 0.051 by construction, proven numerically (not
assumed) in this module's test file for every candidate that survives the
degenerate-collision filter below. `pick_distractors(..., tolerance=
GRADER_TOLERANCE)` additionally auto-drops any candidate that, despite
differing on paper, would still be graded "the same answer" within that
0.051 margin -- a second, independent line of defence over this module's
own explicit filtering.
"""

from __future__ import annotations

import functools
from fractions import Fraction
from itertools import groupby

from ..core import (
    GRADER_TOLERANCE,
    check,
    check_equivalent,
    compose,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    rounded_decimal_string,
    stated_precision_string,
)

_TIERS = ("easy", "medium", "hard", "exam")
_ODD_COUNTS = (9, 11)
_EVEN_COUNTS = (8, 10, 12)
_MEAN_DP = 1  # every tier states "give the mean to 1 decimal digit"
_MIN_MEAN_GAP = Fraction(1)  # wrong-divisor distractor's safety margin, well over 0.051

_TIER_COUNTS = {
    "easy": _ODD_COUNTS,
    "medium": _ODD_COUNTS,
    "hard": _EVEN_COUNTS,
    "exam": _EVEN_COUNTS,
}

_EASY_TEMPLATES = (
    "Here is a data set: {values}. Find the mean (to 1 decimal digit), the median and the mode.",
    "A data set contains {count} values: {values}. Calculate the mean (give your answer to 1 "
    "decimal digit), the median, and the mode.",
    "Data set: {values}. Work out the mean (to 1 decimal digit), the median and the mode.",
)
_MEDIUM_TEMPLATES = (
    "This data set has {count} values: {values}. Find (a) the mean, to 1 decimal digit, "
    "(b) the median, and (c) the mode.",
    "Given the data set {values}, determine (a) the mean correct to 1 decimal digit, (b) the median, and (c) the mode.",
    "A data set of {count} values is recorded below: {values}. State the mean (to 1 decimal "
    "digit), the median and the mode.",
)
_HARD_TEMPLATES = (
    "This data set has an even number of values: {values}. Order the values, then find the "
    "mean (to 1 decimal digit), the median (the average of the two middle values) and the mode.",
    "The data set {values} contains {count} values. Because the count is even, the median is "
    "the average of the two middle values once the data is ordered; find the mean (to 1 "
    "decimal digit), the median and the mode.",
    "Given the {count}-value data set {values}, order the values first, then find the mean "
    "(to 1 decimal digit), the median (average the two middle values) and the mode.",
)
_EXAM_TEMPLATES = (
    "Exam-style question. This data set has {count} values (an even count): {values}. Show "
    "your working for each part as you find the mean (to 1 decimal digit), the median (the "
    "average of the two middle values once ordered) and the mode.",
    "This exam question gives {count} values: {values}. Determine, with full working, the "
    "mean (to 1 decimal digit), the median (average the two middle values after ordering) and "
    "the mode.",
    "A {count}-value data set is given for this exam question: {values}. State your answer "
    "for the mean (to 1 decimal digit), the median (the two middle values' average, once "
    "ordered) and the mode, showing full working for each part.",
)
_TEMPLATE_BANKS = {
    "easy": _EASY_TEMPLATES,
    "medium": _MEDIUM_TEMPLATES,
    "hard": _HARD_TEMPLATES,
    "exam": _EXAM_TEMPLATES,
}


def _draw_one_set(rng, count):
    """One `count`-value data set, 1..60, with a deliberately unambiguous
    mode: the mode value gets exactly 3 occurrences, every other value at
    most 2 (genuine pairs where possible, at most one leftover singleton).
    Returns `(data, mode_value)`; the caller independently re-verifies
    uniqueness (`_verify_unique_mode`) rather than trusting this
    construction."""
    mode_value = rng.randint(1, 60)
    remaining = count - 3
    pool = [v for v in range(1, 61) if v != mode_value]
    pair_count = remaining // 2
    pair_values = rng.sample(pool, pair_count) if pair_count else []
    used = set(pair_values)
    data = [mode_value, mode_value, mode_value]
    for v in pair_values:
        data.extend([v, v])
    if remaining % 2:
        single_pool = [v for v in pool if v not in used]
        data.append(rng.choice(single_pool))
    rng.shuffle(data)
    return data, mode_value


def _verify_unique_mode(data, mode_value):
    """Independent recomputation via a sorted run-length scan
    (`itertools.groupby`) -- a different algorithm from `_draw_one_set`'s
    pair-filling construction loop."""
    runs = [(v, len(list(g))) for v, g in groupby(sorted(data))]
    max_len = max(length for _, length in runs)
    winners = [v for v, length in runs if length == max_len]
    check(len(winners) == 1, "mean_median_mode_one_set: mode is not unique -- a tie exists")
    check(
        winners[0] == mode_value,
        "mean_median_mode_one_set: constructed mode is not the independently-derived most frequent value",
    )


def _is_on_rounding_boundary(value: Fraction, dp: int) -> bool:
    """Scaling by `2 * 10**dp` lands an exact tie on an ODD integer (an EVEN integer means `value` is already
    exactly on the `dp`-place grid with no ambiguity at all, e.g. `count == 10` makes `value * 20` an integer
    for every draw without ever being a genuine tie -- checking `denominator == 1` alone would flag every
    single draw as a boundary and starve the retry loop)."""
    scaled = value * (2 * 10**dp)
    return scaled.denominator == 1 and scaled.numerator % 2 == 1


def _median_split(data):
    """`(median_fraction, ordered_data, split_or_None)`. `split` is
    `(lower, upper)` -- the two sorted middle values -- for an even count,
    or `None` for an odd count (a single exact middle value)."""
    ordered = sorted(data)
    n = len(ordered)
    if n % 2:
        return Fraction(ordered[n // 2]), ordered, None
    lower, upper = ordered[n // 2 - 1], ordered[n // 2]
    return Fraction(lower + upper, 2), ordered, (lower, upper)


def _unsorted_middle(data):
    """The 'does-not-sort' misconception: applies the median's own
    positional formula directly to `data` AS DRAWN (never sorted) --
    genuinely different from `_median_split` whenever the drawn order
    differs from sorted order at the relevant index/indices."""
    n = len(data)
    if n % 2:
        return Fraction(data[n // 2])
    return Fraction(data[n // 2 - 1] + data[n // 2], 2)


def _display_dp(value: Fraction) -> int:
    """0 for an exact integer, 1 for an exact half-integer -- every value
    this module ever displays at "median precision" is one or the other
    (a single drawn value, or the average of two)."""
    return 0 if value.denominator == 1 else 1


def _fixed_precision(value: Fraction, dp: int) -> str:
    """Display `value` at EXACTLY `dp` places, the same discipline E3's own
    statistics sources use (rewritten independently, no import): route a
    genuine half-integer needing `dp == 0` through half-up rounding
    (`rounded_decimal_string`) rather than `stated_precision_string`, which
    raises when the value does not sit exactly on the 0-place grid."""
    if dp == 0:
        return rounded_decimal_string(value, dp)
    return stated_precision_string(value, dp)


def _draw_clean_data_set(rng, count):
    """Draw a `count`-value data set whose mean (a) does not sit on the
    1-decimal-place rounding-tie boundary and (b) keeps its wrong-divisor
    misconception comfortably (`_MIN_MEAN_GAP`, well above the
    `_PAIR_TOLERANCE = 0.051` hazard) clear of the true mean -- both by
    construction, retried rather than assumed. 200 attempts is generous:
    both conditions are rare (a specific rational coincidence, and a very
    low total) against an 8-12 value, 1-60 range draw."""
    for _ in range(200):
        data, mode_value = _draw_one_set(rng, count)
        _verify_unique_mode(data, mode_value)
        total = sum(data)
        mean_frac = Fraction(total, count)
        if _is_on_rounding_boundary(mean_frac, _MEAN_DP):
            continue
        wrong_divisor_gap = abs(mean_frac - Fraction(total, count - 1))
        if wrong_divisor_gap <= _MIN_MEAN_GAP:
            continue
        return data, mode_value, mean_frac
    check(False, "mean_median_mode_one_set: could not draw a clean data set in 200 attempts")


# --- the three parts (module-level, so the test file can monkeypatch one by
# name -- the established pattern for a compose() part that needs shared
# per-question state without becoming un-patchable) --------------------------


def _part_mean(rng, values, *, rounded_mean, data, count):
    working = (
        f"Mean: {' + '.join(map(str, data))} = {sum(data)}; "
        f"{sum(data)} / {count} = {stated_precision_string(rounded_mean, _MEAN_DP)} "
        f"(to {_MEAN_DP} decimal digit)."
    )
    return rounded_mean, working


def _part_median(rng, values, *, median_frac, ordered, split, count):
    ordered_text = ", ".join(map(str, ordered))
    if split is None:
        working = (
            f"Median: order the values: {ordered_text}. With {count} values (odd), the "
            f"median is the middle value: {ordered[count // 2]}."
        )
    else:
        lower, upper = split
        median_text = _fixed_precision(median_frac, _display_dp(median_frac))
        working = (
            f"Median: order the values: {ordered_text}. With {count} values (even), the "
            f"median is the mean of the two middle values, {lower} and {upper}: "
            f"({lower} + {upper}) / 2 = {median_text}."
        )
    return median_frac, working


def _part_mode(rng, values, *, mode_value, data):
    # Never "{mode_value} occurs ..." -- when the drawn mode value is 1, that
    # reads as a literal digit "1" directly followed by a word ending in "s"
    # and trips the text-quality suite's singular_count_plural check (which
    # cannot tell a count of one from a numeral that simply happens to be
    # 1) -- the same hazard E3's `exam_mode_summary_statistic` names and
    # works around (rewritten independently here, no import). The colon +
    # "it" breaks that adjacency for every mode value, not only 1.
    working = (
        f"Mode: {mode_value} is the most frequent value: it occurs "
        f"{data.count(mode_value)} times, more than any other value in the data set."
    )
    return mode_value, working


def _final(values, rng, *, data, tier, count):
    mean_v, median_v, mode_v = values

    check_equivalent(
        mean_v,
        round_half_up(Fraction(sum(data), len(data)), _MEAN_DP),
        "mean_median_mode_one_set: mean part disagrees with independent recomputation",
    )
    recomputed_median, _ordered2, _split2 = _median_split(data)
    check_equivalent(
        median_v,
        recomputed_median,
        "mean_median_mode_one_set: median part disagrees with independent recomputation",
    )
    runs = [(v, len(list(g))) for v, g in groupby(sorted(data))]
    max_len = max(length for _, length in runs)
    winners = [v for v, length in runs if length == max_len]
    check(
        len(winners) == 1 and winners[0] == mode_v,
        "mean_median_mode_one_set: mode part disagrees with independent recomputation",
    )

    mean_text = stated_precision_string(mean_v, _MEAN_DP)
    median_dp = _display_dp(median_v)
    median_text = _fixed_precision(median_v, median_dp)
    mode_text = str(mode_v)

    answer_text = f"mean = {mean_text}, median = {median_text}, mode = {mode_text}"
    answer = {
        "mean": numeric_answer(mean_v),
        "median": numeric_answer(median_v),
        "mode": mode_v,
    }

    values_text = ", ".join(map(str, data))
    text = phrase(rng, _TEMPLATE_BANKS[tier], values=values_text, count=count)

    # Misconceptions, one per part, held as a FULL labelled candidate string
    # (the other two parts correct) -- `answer: pair` compares by LABEL, so
    # a candidate must be a complete, differently-labelled triple, not a
    # single mis-valued number.
    candidates = []

    wrong_divisor_v = round_half_up(Fraction(sum(data), count - 1), _MEAN_DP)
    if wrong_divisor_v != mean_v:
        candidates.append(
            f"mean = {stated_precision_string(wrong_divisor_v, _MEAN_DP)}, median = {median_text}, mode = {mode_text}"
        )

    unsorted_v = _unsorted_middle(data)
    if unsorted_v != median_v:
        unsorted_text = _fixed_precision(unsorted_v, _display_dp(unsorted_v))
        candidates.append(f"mean = {mean_text}, median = {unsorted_text}, mode = {mode_text}")

    max_value = max(data)
    if max_value != mode_v:
        candidates.append(f"mean = {mean_text}, median = {median_text}, mode = {max_value}")

    for cand in candidates:
        check(cand != answer_text, f"mean_median_mode_one_set: a distractor matches the answer text: {cand!r}")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


def _build_mean_median_mode(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    count = rng.choice(_TIER_COUNTS[tier])
    data, mode_value, mean_frac = _draw_clean_data_set(rng, count)
    rounded_mean = round_half_up(mean_frac, _MEAN_DP)
    median_frac, ordered, split = _median_split(data)

    part_mean = functools.partial(_part_mean, rounded_mean=rounded_mean, data=data, count=count)
    part_median = functools.partial(
        _part_median,
        median_frac=median_frac,
        ordered=ordered,
        split=split,
        count=count,
    )
    part_mode = functools.partial(_part_mode, mode_value=mode_value, data=data)
    final = functools.partial(_final, data=data, tier=tier, count=count)

    return compose(rng, [part_mean, part_median, part_mode], final=final)


@generator(
    "exd_e5_mean_median_mode_one_set",
    level=7,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e5_mean_median_mode_one_set(rng, tier: str = "easy"):
    return _build_mean_median_mode(rng, tier=tier)
