"""Generates outlier-fence questions: apply the 1.5*IQR rule to a stated Q1 and Q3 to identify the single
outlier, as `answer: list`.

`answer: list`, a single-element list holding the one outlier value.
Q1 and Q3 are STATED in the prompt (never computed there); the learner's own
job is to apply the standard 1.5*IQR fence rule and identify which single
stated value lies more than 1.5*IQR below Q1 or above Q3.

## Construction (n = 10, half size m = 5, always odd -> Q1/Q3 are literal
integer data values, no averaging)

The full sorted 10-value dataset is built directly from a chosen (Q1, Q3, IQR) triple:

- `Q1`, `Q3` are literal elements at the middle position of each 5-element
  half (`_median_of5`) -- always true because m = 5 is odd.
- One additional slot sits below Q1 (present only when the true outlier is
  NOT already occupying that half) and two additional slots sit above Q3
  (or vice versa, mirrored, when the outlier is high) -- see `_build` for the
  exact slot layout in each branch. Every slot's value is an explicit
  `Q +/- offset`, so every rank position is controlled directly (construct
  to target, not draw-and-hope): `_offset` maps a difficulty-appropriate
  MODE ("safe" | "loose" | "tight" | "exact") to a magnitude relative to
  `k = IQR // 2`, so a "loose"/"tight"/"exact" slot lands beyond 1*IQR but at
  or within 1.5*IQR of its quartile by construction, never by chance.
- Four more slots strictly between Q1 and Q3 are filled with `rng.sample`
  from the open integer interval `(Q1, Q3)` -- pure padding, never offered
  as a distractor.
- The true outlier sits comfortably beyond its fence (offset `3k + drawn
  extra`), on `easy` always below Q1 (stated below) and on the other three
  tiers drawn 50/50.

Because every non-outlier slot's *value* is an explicit `Q +/- offset` with
`offset <= k` (safe) or `2k < offset <= 3k` (loose/tight/exact), and the four
mid-fillers are strictly inside `(Q1, Q3)`, the ranges are disjoint by
construction -- no resampling loop is needed; a violation would be this
module's own construction bug and is checked directly (`_verify`).

### Independent-check proof

`_verify` re-slices the full 10-value sorted list into lower/upper halves by
BOTH the exclusive method (median-of-half excluding nothing, since n = 10 is
even there is nothing to exclude) and the Tukey/inclusive method, and asserts
both agree with each other and with the constructed `(Q1, Q3)` via
`check_equivalent`, calling Python's own `statistics.median` -- a different
piece of code from the direct indexing `_median_of5` construction uses. It
also independently recomputes both fences and counts how many of the ten
values lie strictly beyond either fence, asserting the count is exactly one
and that the flagged value is the constructed outlier -- the central
"exactly one outlier" guarantee is checked from scratch, not assumed.

## Difficulty progression (genuine reasoning escalation)

- easy: sorted data, outlier always below Q1 (easy is allowed to be
  one-sided), placed with a large, unambiguous margin.
  Both above-Q3 slots and the below-Q1 slot are "safe" (comfortably inside
  their fence, nowhere near the 1*IQR line). DECISION: apply the fence rule
  once, on the one side that matters.
- medium: unsorted data; the outlier is drawn low or high 50/50, so the
  learner must compute and check BOTH fences. The slot on the SAME side as
  the outlier is "loose": beyond 1*IQR from its quartile but within 1.5*IQR
  -- a genuine "not far enough" decoy. DECISION: which side, and does
  "beyond 1*IQR" already mean outside the fence (no -- 1.5*IQR is the rule).
- hard: unsorted; the slot on the OPPOSITE side from the true outlier is
  "tight" -- within 1 unit of that side's own fence, so it cannot be
  eyeballed as safe; the fence must actually be computed. DECISION: compute
  BOTH fences exactly enough to separate a near-miss from the real outlier.
- exam: unsorted; the opposite-side slot sits EXACTLY on its fence (an exact
  integer fence, by construction) -- not an outlier, since "more than" is
  strict. DECISION: apply the strict inequality correctly at the one point
  where equality and "beyond" are hardest to tell apart.

## Misconceptions (every candidate a genuinely different integer, checked
directly every seed)

- `opposite-extreme`: the most extreme non-outlier value on the side
  OPPOSITE the true outlier (on hard/exam this sits within 1 unit of, or
  exactly on, that side's own fence).
- `near-quartile-value`: a value near the SAME quartile as the true outlier
  (on medium, deliberately beyond 1*IQR but within 1.5*IQR of that quartile).
- `inner-filler-value`: an ordinary value between the relevant quartile and
  the extreme end of the data on the non-outlier side.
"""

from __future__ import annotations

import statistics

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

TOPIC = "p19b_outlier_fence"
_TIERS = ("easy", "medium", "hard", "exam")

_RULE = (
    "An outlier is a value that lies more than 1.5 times the interquartile range (IQR = Q3 - Q1) below Q1 or above Q3."
)

_CONTEXTS = ("delivery times (minutes)", "daily step counts", "temperature readings (degrees C)")


# ============================================================================
# Quartile arithmetic (n = 10 always, half size m = 5, always odd)
# ============================================================================


def _median_of5(half: list[int]) -> int:
    return half[2]


def _slice_halves(sorted_vals: list[int]):
    n = len(sorted_vals)
    mid = n // 2
    return sorted_vals[:mid], sorted_vals[mid:]


def _verify(sorted_vals: list[int], q1: int, q3: int, outlier: int, lower_fence: int, upper_fence: int):
    """Independent route: re-slice (exclusive AND Tukey/inclusive -- identical
    for even n, asserted anyway) and call `statistics.median`, a different
    piece of code from `_median_of5`'s direct indexing."""
    n = len(sorted_vals)
    mid = n // 2
    excl_lower, excl_upper = sorted_vals[:mid], sorted_vals[mid:]
    incl_lower, incl_upper = sorted_vals[:mid], sorted_vals[mid:]  # n even: coincide
    check_equivalent(
        q1, statistics.median(excl_lower), "p19b_outlier_fence: Q1 disagrees with exclusive-method recomputation"
    )
    check_equivalent(
        q3, statistics.median(excl_upper), "p19b_outlier_fence: Q3 disagrees with exclusive-method recomputation"
    )
    check_equivalent(
        q1, statistics.median(incl_lower), "p19b_outlier_fence: Q1 disagrees with Tukey/inclusive-method recomputation"
    )
    check_equivalent(
        q3, statistics.median(incl_upper), "p19b_outlier_fence: Q3 disagrees with Tukey/inclusive-method recomputation"
    )
    check_equivalent(
        lower_fence,
        q1 - 3 * ((q3 - q1) // 2),
        "p19b_outlier_fence: lower fence disagrees with independent recomputation",
    )
    check_equivalent(
        upper_fence,
        q3 + 3 * ((q3 - q1) // 2),
        "p19b_outlier_fence: upper fence disagrees with independent recomputation",
    )
    beyond = [v for v in sorted_vals if v < lower_fence or v > upper_fence]
    check(beyond == [outlier], f"p19b_outlier_fence: expected exactly one outlier, found {beyond}")


def _offset(mode: str, k: int, rng) -> int:
    if mode == "safe":
        return rng.randint(2, k)
    if mode == "loose":
        return rng.randint(2 * k + 1, 3 * k - 1)
    if mode == "tight":
        return 3 * k - 1
    if mode == "exact":
        return 3 * k
    raise ValueError(f"unknown mode {mode!r}")


# ============================================================================
# Templates
# ============================================================================

_EASY_TEMPLATES = (
    "A dataset of {label}, already sorted from smallest to largest, is: {values}. Its quartiles are "
    "Q1 = {q1} and Q3 = {q3}. {rule} Which single value in the dataset is the outlier?",
    "Here are 10 {label}, listed from smallest to largest: {values}. For this dataset, Q1 = {q1} and "
    "Q3 = {q3}. {rule} Identify the one outlier value.",
    "The following {label} have already been sorted: {values}. For this dataset, Q1 = {q1} and "
    "Q3 = {q3}. {rule} State the outlier.",
)
_MEDIUM_TEMPLATES = (
    "A dataset of {label} was recorded in this order: {values}. Its quartiles are Q1 = {q1} and "
    "Q3 = {q3}. {rule} Which single value in the dataset is the outlier?",
    "Here are 10 {label}, in the order they were collected: {values}. For this dataset, Q1 = {q1} and "
    "Q3 = {q3}. {rule} Identify the one outlier value.",
    "The following {label} were recorded in no particular order: {values}. For this dataset, Q1 = {q1} "
    "and Q3 = {q3}. {rule} State the outlier.",
)
_HARD_TEMPLATES = (
    "A dataset of {label} was recorded in this order: {values}. Its quartiles are Q1 = {q1} and "
    "Q3 = {q3}. {rule} Compute both fences carefully, then state the single value that is the outlier.",
    "Here are 10 {label}, in the order they were collected: {values}. For this dataset, Q1 = {q1} and "
    "Q3 = {q3}. {rule} Work out both fences exactly, then identify the one outlier value.",
    "The following {label} were recorded in no particular order: {values}. For this dataset, Q1 = {q1} "
    "and Q3 = {q3}. {rule} Check both ends carefully, then state the outlier.",
)
_EXAM_TEMPLATES = (
    "Exam-style question. A dataset of {label} was recorded in this order: {values}. Its quartiles "
    "are Q1 = {q1} and Q3 = {q3}. {rule} A value exactly on a fence is NOT an outlier. State the "
    "single value that is the outlier.",
    "Exam-style question. Here are 10 {label}, in the order they were collected: {values}. For this "
    "dataset, Q1 = {q1} and Q3 = {q3}. {rule} Remember that 'more than' is strict. Identify the one "
    "outlier value.",
    "Exam-style question. The following {label} were recorded in no particular order: {values}. "
    "For this dataset, Q1 = {q1} and Q3 = {q3}. {rule} A value lying exactly on a fence does not "
    "count. State the outlier.",
)
_TEMPLATES = {"easy": _EASY_TEMPLATES, "medium": _MEDIUM_TEMPLATES, "hard": _HARD_TEMPLATES, "exam": _EXAM_TEMPLATES}


def _values_text(vals):
    return ", ".join(str(v) for v in vals)


# ============================================================================
# Build
# ============================================================================


def _build(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    label = rng.choice(_CONTEXTS)
    q1 = rng.randint(60, 90)
    iqr = rng.choice([8, 10, 12])
    k = iqr // 2
    q3 = q1 + iqr
    lower_fence = q1 - 3 * k
    upper_fence = q3 + 3 * k

    if tier == "easy":
        want_low = True
    else:
        want_low = rng.choice([True, False])

    mid_fillers = sorted(rng.sample(range(q1 + 1, q3), 4))

    if want_low:
        outlier = lower_fence - rng.randint(5, 2 * k + 10)

        # below-Q1 slot (same side as outlier): loose only on medium.
        below_mode = "loose" if tier == "medium" else "safe"
        below_val = q1 - _offset(below_mode, k, rng)

        # above-Q3 slots (opposite side): q_inner (always safe) < q_outer
        # (decoy on hard/exam, otherwise safe -- the "extreme other side").
        outer_mode = "tight" if tier == "hard" else "exact" if tier == "exam" else "safe"
        if outer_mode == "safe":
            lo_off, hi_off = sorted(rng.sample(range(2, k + 1), 2))
        else:
            lo_off = rng.randint(2, k)
            hi_off = _offset(outer_mode, k, rng)
        q_inner = q3 + lo_off
        q_outer = q3 + hi_off

        sorted_vals = sorted([outlier, below_val, q1, *mid_fillers, q3, q_inner, q_outer])
        opposite_extreme, near_quartile, inner_filler = q_outer, below_val, q_inner
    else:
        outlier = upper_fence + rng.randint(5, 2 * k + 10)

        above_mode = "loose" if tier == "medium" else "safe"
        above_val = q3 + _offset(above_mode, k, rng)

        outer_mode = "tight" if tier == "hard" else "exact" if tier == "exam" else "safe"
        if outer_mode == "safe":
            lo_off, hi_off = sorted(rng.sample(range(2, k + 1), 2))
        else:
            lo_off = rng.randint(2, k)
            hi_off = _offset(outer_mode, k, rng)
        r_inner = q1 - lo_off
        r_outer = q1 - hi_off

        sorted_vals = sorted([outlier, above_val, q1, *mid_fillers, q3, r_inner, r_outer])
        opposite_extreme, near_quartile, inner_filler = r_outer, above_val, r_inner

    check(len(set(sorted_vals)) == 10, "p19b_outlier_fence: dataset values collided")

    q1_check, q3_check = _median_of5(sorted_vals[:5]), _median_of5(sorted_vals[5:])
    check_equivalent(q1, q1_check, "p19b_outlier_fence: constructed Q1 disagrees with its own slot")
    check_equivalent(q3, q3_check, "p19b_outlier_fence: constructed Q3 disagrees with its own slot")
    _verify(sorted_vals, q1, q3, outlier, lower_fence, upper_fence)

    presented = list(sorted_vals)
    if tier != "easy":
        tries = 0
        while True:
            rng.shuffle(presented)
            tries += 1
            if presented != sorted_vals or tries >= 10:
                break

    answer_text = str(outlier)
    candidates = [str(opposite_extreme), str(near_quartile), str(inner_filler)]
    check_pairwise_distinct(
        outlier,
        opposite_extreme,
        near_quartile,
        inner_filler,
        msg="p19b_outlier_fence: a distractor collides with the outlier or another option",
    )

    text = phrase(rng, _TEMPLATES[tier], label=label, values=_values_text(presented), q1=q1, q3=q3, rule=_RULE)
    working = [
        f"Sorted: {_values_text(sorted_vals)}.",
        f"Lower fence = Q1 - 1.5 times IQR = {lower_fence}. Upper fence = Q3 + 1.5 times IQR = {upper_fence}.",
        f"Only {outlier} lies beyond a fence.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=[outlier],
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=pick_distractors(answer_text, candidates),
        working=working,
    )


@generator(
    TOPIC,
    level=19,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19b_outlier_fence(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
