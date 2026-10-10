"""Exam-tier sources for the three single-source summary-statistic skills: ``calculate-median``,
``calculate-mean``, ``calculate-mode``. All three currently have exactly ONE source, the shared
``summary_statistic`` topic in ``mathgen/generators/curriculum_expansion.py`` (count=5 odd median, count=5
mean, hi=30 mode) -- untouched here.

Each skill gets ONE new ``exam``-difficulty source on a NEW topic in this
NEW ``_build`` module, so ``scripts/skill_contract.py`` treats it as a build topic (held to the
learner-text bar in ``mathgen/text_quality.py``) -- the shared
topic never was, since its module name does not end ``_build``.

Three separate topics (one per skill), not one topic with a ``kind``
switch: prompt-variety checks and
``scripts/measure_tiers.py`` both iterate ``(topic, difficulty)`` pairs
straight from ``mathgen.core.PRESETS`` -- they never pass a skill's own
source ``params`` through. A single shared topic selected only by an
explicit ``kind`` param would leave two of the three kinds invisible to
those generic sweeps (they would only ever exercise the function's
default ``kind``). Three topics, each with its own trivial ``"exam"``
preset (an empty dict -- no params to override, just registering the label
so the generic sweeps exercise the exact seeded stream every skill's YAML
source declares), keeps every kind visible to both.

Each source is a genuine escalation in kind over its skill's existing
``easy`` tier, not a bigger-number restyle of the same formula
:

- ``calculate-median``'s ``easy`` tier is an ODD count with a single
  middle value (``summary_statistic``'s ``median`` branch,
  ``values[len(values) // 2]``). ``exam_median_summary_statistic`` draws an EVEN
  count (8/10/12) with at least one repeat, forcing the "average the two
  middle values" formula -- a different formula, not the same one on a
  longer list. The average is an exact half-integer roughly half the
  time; when it is not a whole number the prompt states "to 1 decimal
  place" and every option on offer (answer and distractors alike) is
  rendered at that same stated precision, never a shorter/longer form
  that would make the correct option identifiable by its own shape
  (the ``midpoint_coordinate_scalar`` defect this convention exists to
  avoid).
- ``calculate-mean``'s ``easy`` tier is 5 values built from one balanced
  pair around a centre. ``exam_mean_summary_statistic`` is a full 8-12 value data
  set (with genuine, unforced repeats) whose sum is constructed to be
  exactly divisible by the count, keeping this skill's existing
  exact-integer-answer convention (documented choice, not the "state a
  decimal precision" alternative that was also available -- the
  skill's current tier is exact, and nothing in its contract asks for a
  rounded mean).
- ``calculate-mode``'s ``easy`` tier gives the mode value a fixed 3-vs-1
  surplus over a small fixed set of neighbours. ``exam_mode_summary_statistic``
  draws a full 8-12 value set from the wider 1-60 range, where an
  un-managed draw has a real chance of tying two values for most
  frequent -- the surplus is constructed deterministically (the mode
  value always gets exactly 3 occurrences; every other value is capped
  at 2) and independently re-verified by a run-length scan over the
  sorted values, never merely assumed from the construction method.
"""

from __future__ import annotations

import heapq
from fractions import Fraction
from itertools import groupby

from ..core import (
    check,
    generator,
    numeric_answer,
    phrase,
    pick_distractors,
    rounded_decimal_string,
    stated_precision_string,
    stated_precision_tolerance,
)

# ---------------------------------------------------------------------------
# calculate-median: exam_median_summary_statistic
# ---------------------------------------------------------------------------


def _fixed_precision(value: Fraction, dp: int) -> str:
    """Display `value` at EXACTLY `dp` decimal places, matching whatever the question's own canonical answer
    states, so no option (the answer or a distractor) is identifiable.

    `stated_precision_string` alone is not safe here: every candidate value
    this module ever builds has denominator 1 or 2, so it is always exactly
    representable at `dp == 1` (a whole number is exactly `x.0`) but NOT
    always representable at `dp == 0` (a genuine half-integer, e.g. an
    unordered-middle average landing on `x.5`, has no exact zero-decimal
    form) -- `stated_precision_string` raises in that case by design. Route
    through `rounded_decimal_string` (half-up rounding to the nearest whole
    number) only for that direction; `stated_precision_string` is exact and
    preferred whenever it can succeed."""
    if dp == 0:
        return rounded_decimal_string(value, dp)
    return stated_precision_string(value, dp)


_MEDIAN_COUNTS = (8, 10, 12)

_MEDIAN_TEMPLATES = (
    "Find the median of this data set: {values}.",
    "This data set has {count} values: {values}. What is the median?",
    "A data set contains the values {values}. Work out the median.",
)


def _verify_median_middle(values: list[int], lower: int, upper: int) -> None:
    """Independent recomputation of the two sorted middle values via
    `heapq.nsmallest` partial selection -- a different algorithm from the
    full `sorted()` the construction uses, so a construction bug that
    silently picked the wrong pair out of an otherwise-correctly-sorted
    list is still caught here."""
    upper_rank = len(values) // 2  # 0-indexed rank of the upper middle value
    smallest = heapq.nsmallest(upper_rank + 1, values)
    check(smallest[-1] == upper, "independent partial-selection check: upper middle value mismatch")
    check(smallest[-2] == lower, "independent partial-selection check: lower middle value mismatch")


@generator("exam_median_summary_statistic", level=7, presets={"exam": dict()})
def exam_median_summary_statistic(rng):
    """Even-count median: sort, answer = mean of the two middle values."""
    count = rng.choice(_MEDIAN_COUNTS)
    values = [rng.randint(1, 60) for _ in range(count)]
    if len(set(values)) == count:  # force "at least one repeat"
        i = rng.randrange(count)
        j = rng.randrange(count)
        while j == i:
            j = rng.randrange(count)
        values[j] = values[i]

    ordered = sorted(values)
    lower_idx, upper_idx = count // 2 - 1, count // 2
    lower, upper = ordered[lower_idx], ordered[upper_idx]
    _verify_median_middle(values, lower, upper)

    answer_frac = Fraction(lower + upper, 2)
    dp = 0 if answer_frac.denominator == 1 else 1

    values_text = ", ".join(map(str, values))
    precision_note = " Give your answer to 1 decimal place." if dp else ""
    text = phrase(rng, _MEDIAN_TEMPLATES, values=values_text, count=count) + precision_note
    answer_text = stated_precision_string(answer_frac, dp)

    # Misconception candidates (unordered-formatted, then rounded/formatted
    # at this question's own dp so no option is identifiable by its shape):
    #  - does-not-order: averages the two middle values of the data AS
    #    LISTED (never sorted) -- the skill's own declared misconception.
    #  - single-middle (lower/upper): applies the odd-count skill's own
    #    "pick the single middle value" formula (`values[len(values)//2]`
    #    in `summary_statistic`) straight to an even count, forgetting the
    #    average step this tier exists to require.
    #  - chooses the maximum/minimum value instead of the middle.
    # Not every candidate is a genuinely different VALUE on every draw --
    # e.g. when a forced repeat lands exactly on both middle positions,
    # `single_middle_lower`/`single_middle_upper` legitimately equal the
    # answer, not a construction bug -- so this filters degenerate
    # candidates out (exact `Fraction` equality, before any rounding)
    # rather than hard-failing the whole question on a genuine coincidence;
    # `pick_distractors` is the second, format-aware line of defence for
    # any near-duplicate that survives filtering (e.g. two candidates that
    # differ before rounding but land on the same displayed string).
    does_not_order = Fraction(values[lower_idx] + values[upper_idx], 2)
    single_middle_lower = Fraction(lower)
    single_middle_upper = Fraction(upper)
    picks_max = Fraction(max(values))
    picks_min = Fraction(min(values))
    raw_candidates = [
        c for c in (does_not_order, single_middle_lower, single_middle_upper, picks_max, picks_min) if c != answer_frac
    ]

    tolerance = stated_precision_tolerance(dp) if dp else Fraction(0)
    candidates = [_fixed_precision(c, dp) for c in raw_candidates]
    return dict(
        text=text,
        answer=numeric_answer(answer_frac),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Order the values: {', '.join(map(str, ordered))}.",
            f"With {count} values (even), the median is the mean of the two "
            f"middle values, {lower} and {upper}: ({lower} + {upper}) / 2 = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, candidates, tolerance=tolerance),
    )


# ---------------------------------------------------------------------------
# calculate-mean: exam_mean_summary_statistic
# ---------------------------------------------------------------------------

_MEAN_TEMPLATES = (
    "Find the mean of this data set: {values}.",
    "This data set has {count} values: {values}. Calculate the arithmetic mean.",
    "A data set contains the values {values}. What is the average (mean)?",
)


def _verify_mean_total(values: list[int], count: int, mean_value: int) -> None:
    """Independent recomputation of the total via ascending-sorted-order
    accumulation -- a traversal distinct from the construction loop's own
    insertion order -- cross-checked against the drawn-order total and
    against the target mean by exact `Fraction` division."""
    total_sorted = 0
    for v in sorted(values):
        total_sorted += v
    check(total_sorted == sum(values), "sorted-order and drawn-order totals disagree")
    check(Fraction(total_sorted, count) == mean_value, "total / count does not equal the constructed mean")


@generator("exam_mean_summary_statistic", level=7, presets={"exam": dict()})
def exam_mean_summary_statistic(rng):
    """8-12 values, repeats allowed, sum constructed divisible by the
    count so the mean is an exact integer (this skill's existing
    exactness convention)."""
    count = rng.randint(8, 12)
    mean_value = rng.randint(15, 45)
    total = mean_value * count

    values: list[int] = []
    remaining_total = total
    for i in range(count - 1):
        remaining_slots = count - i - 1
        lo = max(1, remaining_total - 60 * remaining_slots)
        hi = min(60, remaining_total - remaining_slots)
        v = rng.randint(lo, hi)
        values.append(v)
        remaining_total -= v
    values.append(remaining_total)
    check(1 <= remaining_total <= 60, "final constructed value fell outside 1..60")
    rng.shuffle(values)

    _verify_mean_total(values, count, mean_value)

    values_text = ", ".join(map(str, values))
    text = phrase(rng, _MEAN_TEMPLATES, values=values_text, count=count)
    answer_text = stated_precision_string(Fraction(mean_value), 0)

    # Misconception candidates, all plain integers (matching the exact
    # answer's own no-decimal-point shape):
    #  - forgets-divide: reports the raw total (the skill's own declared
    #    misconception). Never legitimately equals the mean itself, since
    #    total = mean * count with count >= 8.
    #  - divides by count - 1: forgets one observation.
    #  - chooses the maximum value instead of averaging.
    #  - confuses mean with median (the middle value(s) of the sorted set).
    # As in `exam_median_summary_statistic`, a candidate that CAN legitimately equal
    # the answer (e.g. every drawn value happens to equal the mean itself,
    # a degenerate but reachable uniform draw) is filtered by exact
    # `Fraction` equality rather than hard-failing the question.
    total_drawn = sum(values)
    off_by_one = Fraction(total_drawn, count - 1)
    picks_max = Fraction(max(values))
    picks_min = Fraction(min(values))
    sorted_vals = sorted(values)
    mid = count // 2
    if count % 2 == 0:
        median_like = Fraction(sorted_vals[mid - 1] + sorted_vals[mid], 2)
    else:
        median_like = Fraction(sorted_vals[mid])
    mean_frac = Fraction(mean_value)
    raw_candidates = [c for c in (total_drawn, off_by_one, picks_max, picks_min, median_like) if c != mean_frac]

    candidates = [rounded_decimal_string(Fraction(c), 0) for c in raw_candidates]
    return dict(
        text=text,
        answer=numeric_answer(Fraction(mean_value)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Total: {' + '.join(map(str, values))} = {total_drawn}.",
            f"Divide by the count of values: {total_drawn} / {count} = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


# ---------------------------------------------------------------------------
# calculate-mode: exam_mode_summary_statistic
# ---------------------------------------------------------------------------

_MODE_TEMPLATES = (
    "Find the mode of this data set: {values}.",
    "This data set has {count} values: {values}. Which value occurs most often?",
    "A data set contains the values {values}. What is the mode?",
)


def _verify_unique_mode(values: list[int], mode_value: int) -> None:
    """Independent recomputation via a sorted run-length scan
    (`itertools.groupby` over `sorted(values)`) -- a different algorithm
    from the `Counter`-based bookkeeping the construction loop uses -- to
    confirm the mode value both is unique and is genuinely the most
    frequent."""
    runs = [(value, len(list(group))) for value, group in groupby(sorted(values))]
    max_len = max(length for _, length in runs)
    winners = [value for value, length in runs if length == max_len]
    check(len(winners) == 1, "mode is not unique: a tie exists (independent run-length check)")
    check(winners[0] == mode_value, "constructed mode value is not the independently-derived most frequent value")


@generator("exam_mode_summary_statistic", level=7, presets={"exam": dict()})
def exam_mode_summary_statistic(rng):
    """8-12 values to 60 with a deterministic surplus (mode value gets
    exactly 3 occurrences; every other value capped at 2), so the mode is
    genuinely unique -- verified, not assumed. As many of the remaining
    slots as possible are filled in genuinely-repeating PAIRS (never more
    than 2, so the mode's surplus of 3 stays unique) rather than leaving
    every non-mode value to a single, independent `rng.randint` draw: with
    only 5-9 non-mode slots, an independent-draw fill left a real chance
    (seen directly in a 200-seed scan) that every non-mode value drawn was
    a distinct singleton, so the only real "also repeats" distractor
    (`second_most_frequent` below) degenerated to an arbitrary tie-break
    among values that do not actually repeat at all, and regularly
    collided with `picks_max`/`picks_min` down to a single surviving
    distractor. Forcing genuine pairs keeps that distractor meaningful and
    keeps the candidate pool spread across more distinct values."""
    count = rng.randint(8, 12)
    mode_value = rng.randint(1, 60)
    remaining = count - 3
    pool = [v for v in range(1, 61) if v != mode_value]

    pair_count = remaining // 2
    pair_values = rng.sample(pool, pair_count) if pair_count else []
    other_counts: dict[int, int] = {v: 2 for v in pair_values}
    values = [mode_value, mode_value, mode_value]
    for v in pair_values:
        values.extend([v, v])
    if remaining % 2:  # one leftover slot: a genuine singleton
        single_pool = [v for v in pool if v not in other_counts]
        single_value = rng.choice(single_pool)
        other_counts[single_value] = 1
        values.append(single_value)
    rng.shuffle(values)

    _verify_unique_mode(values, mode_value)

    values_text = ", ".join(map(str, values))
    text = phrase(rng, _MODE_TEMPLATES, values=values_text, count=count)
    answer_text = str(mode_value)

    # Misconception candidates, all plain integers:
    #  - chooses-largest: the maximum value (the skill's own declared
    #    misconception). Can legitimately equal `mode_value` itself when the
    #    mode happens to be drawn at the top of the 1..60 range -- filtered
    #    by exact equality rather than hard-failing, same as
    #    `exam_median_summary_statistic`/`exam_mean_summary_statistic`.
    #  - chooses the minimum value (same caveat, at the bottom of the range).
    #  - second/third-most-frequent: genuinely different values that also
    #    repeat (twice, from a forced pair), but not as often as the true
    #    mode (three times) -- a learner who counted correctly but stopped
    #    at the first repeat they noticed. `other_counts` never holds
    #    `mode_value` as a key (excluded at construction), so these can
    #    never coincide with the answer.
    #  - confuses the mode with the mean: rounds the data set's own
    #    arithmetic mean to the nearest integer instead of counting
    #    frequencies. A fifth candidate, uncorrelated with the small pool
    #    the four candidates above draw from, kept as a buffer against
    #    those collapsing into fewer than 3 distinct options (seen
    #    directly in a seed scan when the non-mode pool is small).
    picks_max = max(values)
    picks_min = min(values)
    repeaters = sorted((v for v, c in other_counts.items() if c == 2))
    second_most_frequent = repeaters[0] if repeaters else None
    third_most_frequent = repeaters[1] if len(repeaters) > 1 else None
    mean_like = round(Fraction(sum(values), count))
    raw_candidates = [
        c
        for c in (picks_max, picks_min, second_most_frequent, third_most_frequent, mean_like)
        if c is not None and c != mode_value
    ]

    candidates = [str(c) for c in raw_candidates]
    return dict(
        text=text,
        answer=mode_value,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            # Never "{mode_value} occurs ..." -- when the drawn mode value is
            # 1, that reads as a literal digit "1" directly followed by a
            # word ending in "s" and trips the text-quality suite's
            # singular_count_plural check (which cannot tell a count of one
            # from a numeral that simply happens to be 1). The colon breaks
            # that adjacency for every mode value, not only 1.
            f"The most frequent value is {mode_value}: it occurs "
            f"{values.count(mode_value)} times, more than any other value.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )
