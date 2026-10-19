"""Generates `p12d_five_number_summary` and `p12d_outlier_effect_mean` questions: a five-number summary of nine
values, and the effect of an outlier on the mean.

Two senior statistics skills.

Two independent topics, sharing only this module and its imports from
`mathgen.core` -- neither reads the other's data.

## `p12d_five_number_summary` (skill `senior-complete-a-five-number-summary-
for-different-datasets`, ACMEM058)

`answer: pair`, a 5-label `answer_text` ("min = .., Q1 = .., median = ..,
Q3 = .., max = .."). `n = 9` fixed, every tier. Uses the Tukey/"inclusive"
method: sort the nine integers `v1 <= ... <= v9`; `min = v1`, `Q1 = v3`
(0-indexed `sorted[2]`), `median = v5` (`sorted[4]`), `Q3 = v7`
(`sorted[6]`), `max = v9` (`sorted[8]`). Every one of the five is a literal
element of the (integer) dataset, so every value is an integer automatically
-- no interpolation ever enters the computation.

### Independent-check proof (Tukey/inclusive method, odd n=9)

Construction indexes `sorted[2]`, `sorted[4]`, `sorted[6]` directly. The
independent check instead splits `lower = sorted[:5]` (indices 0..4) and
`upper = sorted[4:]` (indices 4..8), and takes `statistics.median` of each,
plus `sorted[4]` for the median directly. Proof this is the identical value
for odd n=9: `lower` is `sorted[:5]`, already in the SAME ascending order as
the full list (a prefix of a sorted list is sorted), so `statistics.median`
of these 5 elements (Python's `statistics.median` returns the literal middle
element for an odd-length input, not an interpolation) is `lower[2]`, which
is `sorted[2]` -- the same element, same index, same value. Symmetrically
`upper = sorted[4:]` is `sorted[4], sorted[5], sorted[6], sorted[7],
sorted[8]`, and its median (index 2 of `upper`) is `sorted[4 + 2] =
sorted[6]`. So `statistics.median(lower) == sorted[2] == Q1` and
`statistics.median(upper) == sorted[6] == Q3` algebraically, not
coincidentally -- proven here, once, rather than merely asserted; the test
file additionally proves the check actually FIRES by corrupting one
constructed value and confirming `check()` raises.

### Difficulty progression

easy: 9 integers already sorted ascending in the presented text -- apply the rule with no sorting step.
medium: the same draw, unsorted (shuffled) -- sorting becomes a genuinely new required step. hard: unsorted
AND at least one repeated value is FORCED to land at, or directly adjacent to, one of the three
summary-statistic index positions (2, 4 or 6, 0-indexed) on every single seed (`_draw_dataset`'s `t`/direction
choice) -- confirmed by a sweep test, not left to chance. exam: a genuine escalation in KIND, not magnitude --
at least one NEGATIVE integer is forced into the dataset, so the learner must sort correctly across the
positive/negative boundary (a comparison hard never requires: hard's numbers are all drawn non-negative).

### Misconceptions (every candidate a genuinely different Fraction/int value,
verified directly every seed, not a display variant)

- `excludes-median-from-halves`: the EXCLUSIVE method (Q1/Q3 computed from
  v1..v4 / v6..v9, excluding the overall median) -- for n=9 this changes the
  index and can produce a half-integer. Only offered when it actually
  differs from the inclusive-method answer (checked directly; the rare hard-
  tier seed where the forced duplicate happens to sit exactly where this
  collides is dropped ("verify directly and drop on the
  rare seed it collides").
- `forgets-to-sort`: applies the v3/v5/v7 index rule to the UNSORTED
  presentation order directly, instead of sorting first (min/max are
  invariant to order, so only Q1/median/Q3 change). Never offered on easy
  (sorted presentation == sorted order, so this would coincide with the
  answer); offered on medium/hard/exam only when it actually differs.
- `swaps-q1-q3`: Q1 and Q3 exchanged in the labelled string. Offered only
  when Q1 != Q3 (checked directly; would coincide only in a degenerate
  dataset this generator never draws).

## `p12d_outlier_effect_mean` (skill `senior-investigate-the-effect-of-
outliers-on-the-mean`, ACMEM052)

`answer: number`, SIGNED `mean_with_outlier - mean_without_outlier`, stated
to 1 decimal place always (`stated_precision_string(value, 1)`). The
subtraction order is stated in WORDS in every question (never inferred).

### The construct-then-adjust recipe (never relies on natural rounding)

Draw the `n - 1` non-outlier values freely from the context's range, then
adjust the LAST one so their sum is an exact multiple of `n - 1`
(`_adjust_last_to_multiple`) -- `mean_without_outlier` is then an exact
integer, never a rounded approximation. Draw the outlier `x` freely from its
own (far-away) range, then adjust it the same way so the FULL n-value sum is
an exact multiple of `n` -- `mean_with_outlier` is then an exact integer
too. Both means are therefore exact integers by construction; no rounding
of an intermediate value ever happens, and `sum(all) % n == 0` /
`sum(without_outlier) % (n - 1) == 0` are asserted directly over a swept
sample of seeds in the test file, every tier.

### Independent check -- a genuinely different route from construction

Construction sums the full list and the list-minus-outlier separately.
The check instead derives `mean_with - mean_without` algebraically from `n`,
the outlier `x`, and `mean_without` ALONE, never touching the raw list a
second time the same way construction did:

    sum_without = (n - 1) * mean_without
    sum_with = sum_without + x
    mean_with = sum_with / n
    diff = mean_with - mean_without
         = [(n - 1) * mean_without + x] / n - mean_without
         = [(n - 1) * mean_without + x - n * mean_without] / n
         = (x - mean_without) / n

So `diff == (x - mean_without) / n` in closed form -- the check computes this directly from `x` and
`mean_without` and asserts it equals construction's own `mean_with - mean_without` to the exact `Fraction`.

### Plausibility -- ONE context across every tier, admitting negatives

Every tier uses the SAME context, "a small business's daily profit or loss
($)" -- genuinely admits negative values (a loss day) and an occasional
large outlier (an unusually big sale, or an unusually bad day), unlike a
context such as test scores or heights where a negative or extreme value
would be nonsensical. Regular days are drawn from a bounded, stated
"ordinary" range; the designated outlier (and, hard/exam, the second
extreme) are drawn from a disjoint, more extreme range on the same context,
so every value stays plausible FOR THIS CONTEXT by construction -- swept and
asserted directly over >= 200 seeds per tier in the test file, not merely
constructed and hoped.

### Difficulty progression

easy (n=5): one value is explicitly labelled "the outlier"; the magnitude
gap between it and the rest is large. medium (n=7): the outlier is still
explicitly labelled, but the magnitude gap is smaller -- the learner must
actually compute, not eyeball the sign. hard (n=9): a SECOND, genuinely
different extreme value is also present, but only one is designated "the
outlier" in the text -- the learner must remove the STATED one. exam (n=9):
the designated outlier is NEGATIVE against an otherwise-positive dataset (so
removing it INCREASES the mean, the sign learners most often get backwards),
combined with hard's second-extreme-value trap.
"""

from __future__ import annotations

import statistics
from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    numeric_answer,
    paren_negative,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

# ============================================================================
# Topic 1: p12d_five_number_summary
# ============================================================================

_FNS_TIERS = ("easy", "medium", "hard", "exam")

_FNS_CONTEXTS = ("quiz", "steps", "bare")
_FNS_RANGE = {"quiz": (40, 100), "steps": (2000, 15000), "bare": (1, 200)}
_FNS_EXAM_RANGE = (-15, 20)

_FNS_METHOD = (
    "Use the median-of-the-lower/upper-half method (the Tukey / inclusive method): sort the "
    "values, then Q1 is the median of the lower half of the sorted list (including the overall "
    "middle value), and Q3 is the median of the upper half (also including the overall middle "
    "value)."
)

# The exam tier states the quartile convention as a fact about the summary, not as steps to follow.
_FNS_EXAM_CONVENTION = (
    "Quartiles follow the median-of-the-lower/upper-half (Tukey, inclusive) convention: Q1 is the median of "
    "the lower half of the sorted values (including the overall middle value) and Q3 is the median of the "
    "upper half (also including it)."
)

_FNS_EASY_TEMPLATES = {
    "quiz": "A class's nine quiz scores (out of 100), already sorted from smallest to largest, "
    "are: {values}. {method} State the five-number summary: min, Q1, median, Q3, max.",
    "steps": "One person's daily step counts over nine days, listed from smallest to largest, "
    "were: {values}. {method} Find the five-number summary (min, Q1, median, Q3, max).",
    "bare": "Here are nine numbers, already arranged from smallest to largest: {values}. "
    "{method} Give the five-number summary: min, Q1, median, Q3 and max.",
}
_FNS_MEDIUM_TEMPLATES = {
    "quiz": "A class's nine quiz scores (out of 100) were recorded in this order: {values}. "
    "{method} Sort them first, then state the five-number summary: min, Q1, median, Q3, max.",
    "steps": "One person's daily step counts over nine days were recorded in this order: "
    "{values}. {method} Sort the values first, then find the five-number summary (min, Q1, "
    "median, Q3, max).",
    "bare": "Here are nine numbers, in the order they were collected: {values}. {method} Sort "
    "them first, then give the five-number summary: min, Q1, median, Q3 and max.",
}
_FNS_HARD_TEMPLATES = {
    "quiz": "A class's nine quiz scores (out of 100), including one score achieved twice, were "
    "recorded in this order: {values}. {method} Sort the values (keeping every repeated score "
    "in place) and give the five-number summary: min, Q1, median, Q3, max.",
    "steps": "One person's daily step counts over nine days, with one day's count repeated, "
    "were recorded in this order: {values}. {method} Sort the values, keeping every repeat, "
    "then find the five-number summary (min, Q1, median, Q3, max).",
    "bare": "Here are nine numbers, in the order collected, with one value appearing twice: "
    "{values}. {method} Sort them, keeping both copies of the repeated value, then give the "
    "five-number summary: min, Q1, median, Q3 and max.",
}
_FNS_EXAM_TEMPLATES = (
    "A weather station recorded nine daily temperature changes (degrees C), some positive and "
    "some negative, in this order: {values}. {method} Give the five-number summary: min, Q1, "
    "median, Q3, max.",
    "Nine daily temperature change readings (degrees C) -- some rises, "
    "some falls -- were recorded in this order: {values}. {method} Find the five-number summary "
    "(min, Q1, median, Q3, max).",
    "Nine daily temperature changes (degrees C), including negative readings, were recorded "
    "in this order: {values}. {method} State the five-number summary: min, Q1, median, Q3 and "
    "max.",
)


def _fns_draw_dataset(rng, tier: str):
    """Returns (context_label, sorted_list, presented_list). `sorted_list`
    may carry a forced duplicate (hard tier only); `presented_list` is what
    the question text shows (ascending for easy, shuffled otherwise)."""
    if tier == "exam":
        low, high = _FNS_EXAM_RANGE
        neg_pool = list(range(low, 0))
        forced_negative = rng.choice(neg_pool)
        remaining = [v for v in range(low, high + 1) if v != forced_negative]
        rest = rng.sample(remaining, 8)
        base = rest + [forced_negative]
        sorted_list = sorted(base)
        presented = list(sorted_list)
        rng.shuffle(presented)
        return "temp", sorted_list, presented

    context = rng.choice(_FNS_CONTEXTS)
    low, high = _FNS_RANGE[context]
    sorted_list = sorted(rng.sample(range(low, high + 1), 9))

    if tier == "easy":
        return context, sorted_list, list(sorted_list)

    if tier == "medium":
        presented = list(sorted_list)
        tries = 0
        while True:
            rng.shuffle(presented)
            tries += 1
            if presented != sorted_list or tries >= 10:
                break
        return context, sorted_list, presented

    # hard: force a duplicate at, or adjacent to, one of the three
    # summary-statistic index positions (2, 4, 6), never at the endpoints.
    t = rng.choice((2, 4, 6))
    d = rng.choice((-1, 1))
    sorted_list = list(sorted_list)
    sorted_list[t + d] = sorted_list[t]
    sorted_list.sort()
    presented = list(sorted_list)
    rng.shuffle(presented)
    return context, sorted_list, presented


def _fns_stat(v) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else stated_precision_string(v, 1)


def _fns_labelled(mn, q1, md, q3, mx) -> str:
    return (
        f"min = {_fns_stat(mn)}, Q1 = {_fns_stat(q1)}, median = {_fns_stat(md)}, "
        f"Q3 = {_fns_stat(q3)}, max = {_fns_stat(mx)}"
    )


def _fns_verify(sorted_list, mn, q1, md, q3, mx):
    """Independent route: split lower/upper halves (each INCLUDING the
    overall median) and take statistics.median of each, plus the direct
    middle element for the median itself -- see module docstring for the
    algebraic-equivalence proof."""
    lower = sorted_list[:5]
    upper = sorted_list[4:]
    check(mn == sorted_list[0], "p12d_five_number_summary: min mismatch")
    check(mx == sorted_list[8], "p12d_five_number_summary: max mismatch")
    check(md == sorted_list[4], "p12d_five_number_summary: median mismatch")
    check(
        q1 == statistics.median(lower),
        "p12d_five_number_summary: Q1 disagrees with independent lower-half median",
    )
    check(
        q3 == statistics.median(upper),
        "p12d_five_number_summary: Q3 disagrees with independent upper-half median",
    )


def _build_five_number_summary(rng, tier: str = "easy"):
    if tier not in _FNS_TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    context, sorted_list, presented = _fns_draw_dataset(rng, tier)

    mn, q1, md, q3, mx = (
        sorted_list[0],
        sorted_list[2],
        sorted_list[4],
        sorted_list[6],
        sorted_list[8],
    )
    _fns_verify(sorted_list, mn, q1, md, q3, mx)

    answer_text = _fns_labelled(mn, q1, md, q3, mx)
    answer = {"min": mn, "Q1": q1, "median": md, "Q3": q3, "max": mx}

    values_text = ", ".join(str(v) for v in presented)
    if tier == "exam":
        text = phrase(rng, _FNS_EXAM_TEMPLATES, values=values_text, method=_FNS_EXAM_CONVENTION)
    else:
        template = {
            "easy": _FNS_EASY_TEMPLATES,
            "medium": _FNS_MEDIUM_TEMPLATES,
            "hard": _FNS_HARD_TEMPLATES,
        }[tier][context]
        text = template.format(values=values_text, method=_FNS_METHOD)

    candidates = []
    seen = {answer_text}

    def _offer(cand: str):
        # Dropped, not a hard failure, whenever a candidate coincides with the
        # answer OR with an already-offered candidate -- two misconceptions
        # can land on the same wrong value by coincidence on a given seed
        # (e.g. forgets-to-sort and excludes-median-from-halves both moving
        # Q1 to the same neighbouring element); offering it twice would ship
        # a duplicate option, not a defect worth failing question generation
        # over.
        if cand in seen:
            return
        seen.add(cand)
        candidates.append(cand)

    # excludes-median-from-halves: the EXCLUSIVE method (v1..v4 / v6..v9,
    # dropping the overall median from both halves).
    lower_excl = sorted_list[0:4]
    upper_excl = sorted_list[5:9]
    q1_alt = Fraction(lower_excl[1] + lower_excl[2], 2)
    q3_alt = Fraction(upper_excl[1] + upper_excl[2], 2)
    if not (q1_alt == q1 and q3_alt == q3):
        _offer(_fns_labelled(mn, q1_alt, md, q3_alt, mx))

    # forgets-to-sort: apply the index rule to the UNSORTED presentation
    # order directly (min/max are order-invariant, so only Q1/median/Q3
    # change) -- never offered on easy, where presented == sorted.
    if tier != "easy":
        u_q1, u_md, u_q3 = presented[2], presented[4], presented[6]
        if not (u_q1 == q1 and u_md == md and u_q3 == q3):
            _offer(_fns_labelled(mn, u_q1, u_md, u_q3, mx))

    # swaps-q1-q3
    if q1 != q3:
        _offer(_fns_labelled(mn, q3, md, q1, mx))

    check(len(candidates) >= 1, "p12d_five_number_summary: no distractor survived")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates),
        working=[f"Sorted: {', '.join(str(v) for v in sorted_list)}.", answer_text],
    )


@generator(
    "p12d_five_number_summary",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12d_five_number_summary(rng, tier: str = "easy"):
    return _build_five_number_summary(rng, tier=tier)


# ============================================================================
# Topic 2: p12d_outlier_effect_mean
# ============================================================================

_OEM_TIERS = ("easy", "medium", "hard", "exam")
_OEM_N = {"easy": 5, "medium": 7, "hard": 9, "exam": 9}
_OEM_DP = 1

# "a small business's daily profit or loss ($)" -- the one context used at
# every tier; genuinely admits both negative (a loss day) and an unusually
# large (positive or negative) value (an outlier day), never test scores /
# heights / counts, where a negative or extreme reading would be nonsensical.
_OEM_ORDINARY = (-50, 150)  # a typical day's profit/loss, $
_OEM_OUTLIER_LARGE = (500, 900)  # easy/hard/exam's designated-outlier magnitude
_OEM_OUTLIER_SMALL = (250, 400)  # medium's designated outlier / hard's 2nd extreme
_OEM_EXAM_ORDINARY = (10, 150)  # exam's REST is positive-only, so a negative
# outlier visibly pulls the mean down


def _oem_adjust_last_to_multiple(rng, values: list[int], modulus: int, lo: int, hi: int):
    """Draw-then-adjust: `values` was drawn freely; nudge the LAST entry so
    `sum(values) % modulus == 0`, preferring the smaller in-range nudge.
    Returns the adjusted list, or None if no in-range nudge exists (caller
    retries with a fresh draw)."""
    remainder = sum(values) % modulus
    if remainder == 0:
        return values
    up = values[-1] + (modulus - remainder)
    down = values[-1] - remainder
    if lo <= up <= hi:
        values[-1] = up
        return values
    if lo <= down <= hi:
        values[-1] = down
        return values
    return None


def _oem_draw_clean(rng, count: int, lo: int, hi: int, modulus: int, forced_index=None, forced_value=None):
    """Draw `count` free values in `[lo, hi]`, then adjust the LAST one so
    the sum is a multiple of `modulus`. `forced_index`/`forced_value`
    overwrite one (never the last) index BEFORE the sum-adjustment runs, so
    the modulus invariant is established AFTER, not undone by, the
    override -- inserting a second extreme value after the adjustment would
    silently break `sum(rest) % modulus == 0`, the bug this ordering
    avoids."""
    for _ in range(200):
        values = [rng.randint(lo, hi) for _ in range(count)]
        if forced_index is not None:
            values[forced_index] = forced_value
        adjusted = _oem_adjust_last_to_multiple(rng, values, modulus, lo, hi)
        if adjusted is not None:
            return adjusted
    check(False, "p12d_outlier_effect_mean: could not draw a sum-clean sample in 200 attempts")


_OEM_MARKER = " (The stated outlier value is {outlier}.)"
_OEM_EASY_TEMPLATES = tuple(
    t + _OEM_MARKER
    for t in (
        "A small business recorded its daily profit or loss ($) for {n} days: {values}. One value, "
        "{outlier}, is the outlier. Find the mean with the outlier included, minus the mean with the "
        "outlier removed. Give your answer to 1 decimal place.",
        "Over {n} days, a small business's profit or loss ($) was: {values}, where {outlier} is "
        "explicitly the outlier. Subtract the mean WITHOUT the outlier from the mean WITH the "
        "outlier included, to 1 decimal place.",
        "Here are {n} days of a small business's daily profit or loss ($): {values}. The value "
        "{outlier} is the outlier. Find the mean with the outlier included, minus the mean with the "
        "outlier removed, to 1 decimal place.",
    )
)
_OEM_MEDIUM_TEMPLATES = tuple(
    t + _OEM_MARKER
    for t in (
        "A small business recorded its daily profit or loss ($) for {n} days: {values}. The value "
        "{outlier} is the outlier, though it is closer to the rest than usual -- compute the exact "
        "means. Find the mean with the outlier included, minus the mean with the outlier removed, "
        "to 1 decimal place.",
        "To 1 decimal place, find the mean with the outlier included, minus the mean with the "
        "outlier removed, for this {n}-day profit or loss ($) record: {values} (the outlier is "
        "{outlier}).",
        "Over {n} days a small business's profit or loss ($) was {values}, with {outlier} recorded "
        "as the outlier. Subtract the mean without the outlier from the mean with the outlier "
        "included; give your answer to 1 decimal place.",
    )
)
_OEM_HARD_TEMPLATES = tuple(
    t + _OEM_MARKER
    for t in (
        "A small business recorded its daily profit or loss ($) for {n} days: {values}. Two values "
        "here look unusual, but only {outlier} is called the outlier. Find the mean with the "
        "outlier included, minus the mean with the outlier removed, to 1 decimal place.",
        "This {n}-day profit or loss ($) record has more than one unusual value: {values}. The "
        "outlier -- and only the outlier -- is {outlier}. Subtract the mean without the outlier "
        "from the mean with the outlier included, to 1 decimal place.",
        "Over {n} days, a small business's profit or loss ($) was {values}. Although a second value "
        "also stands out, the outlier is specifically {outlier}. Find the mean with the outlier "
        "included, minus the mean with the outlier removed, to 1 decimal place.",
    )
)
_OEM_EXAM_TEMPLATES = tuple(
    t + _OEM_MARKER
    for t in (
        "A small business recorded its daily profit or loss ($) for {n} days: "
        "{values}. Two values here look unusual, but only {outlier} -- a negative, loss-making day "
        "-- is called the outlier. Find the mean with the outlier included, minus the mean with the "
        "outlier removed, to 1 decimal place.",
        "A {n}-day profit or loss ($) record has more than one unusual "
        "value: {values}. The outlier, a loss of {outlier}, pulls an otherwise positive record down "
        "-- subtract the mean without the outlier from the mean with the outlier included, to 1 "
        "decimal place.",
        "Over {n} days, a small business's profit or loss ($) was {values}, an otherwise positive "
        "record with a single loss-making outlier, {outlier}, and a second unusually large value "
        "that is NOT the outlier. Find the mean with the outlier included, minus the mean with the "
        "outlier removed, to 1 decimal place.",
    )
)
_OEM_TEMPLATE_BANKS = {
    "easy": _OEM_EASY_TEMPLATES,
    "medium": _OEM_MEDIUM_TEMPLATES,
    "hard": _OEM_HARD_TEMPLATES,
    "exam": _OEM_EXAM_TEMPLATES,
}


def _oem_draw_bounded_multiple(rng, lo: int, hi: int, forced_sign: int, n: int, base_sum: int) -> int:
    """Draw the outlier `x` with magnitude in `[lo, hi]` (sign `forced_sign`
    if non-zero, else drawn randomly each attempt), then adjust it so
    `(base_sum + x) % n == 0` -- retrying with a FRESH `x` draw whenever
    neither in-band adjustment (`+`/`- remainder`) keeps `x` inside its
    stated magnitude band, rather than ever accepting an out-of-band value.
    This is what keeps the outlier plausible for its context at every seed
    (the earlier construction pushed `x` past its stated band whenever the
    modulus adjustment needed more room than the band's edge allowed --
    found directly by the >= 200-seed plausibility sweep, not assumed)."""
    for _ in range(200):
        sign = forced_sign if forced_sign else rng.choice((1, -1))
        magnitude = rng.randint(lo, hi)
        x = sign * magnitude
        remainder = (base_sum + x) % n
        if remainder == 0:
            return x
        for candidate in (x + (n - remainder), x - remainder):
            if sign > 0 and lo <= candidate <= hi:
                return candidate
            if sign < 0 and -hi <= candidate <= -lo:
                return candidate
    check(False, "p12d_outlier_effect_mean: could not draw an in-band outlier in 200 attempts")


def _oem_fmt_distractor(value: Fraction, dp: int) -> str:
    """Format a distractor value at EXACTLY `dp` decimal places, same as the
    canonical answer's own `stated_precision_string` -- so the answer is
    never the only option of its displayed shape (`grade_number` cares
    only about the numeric value, but a distractor that happens to land on
    an exact integer must still show ".0", or it becomes visually pickable
    without doing the arithmetic). Rounds half-up to `dp` first when the
    value does not already sit on the exact `10**-dp` grid -- unlike
    `rounded_decimal_string`, which under-displays an exact value (e.g.
    `-96` instead of `-96.0`)."""
    scaled = value * (10**dp)
    if scaled.denominator != 1:
        value = round_half_up(value, dp)
    return stated_precision_string(value, dp)


def _oem_verify(x, mean_without, n, diff):
    """Independent, closed-form route -- never re-sums the raw list the
    way construction did. See module docstring for the derivation."""
    check_equivalent(
        diff,
        (Fraction(x) - Fraction(mean_without)) / n,
        "p12d_outlier_effect_mean: diff disagrees with the closed-form recomputation",
    )


def _build_outlier_effect(rng, tier: str = "easy"):
    if tier not in _OEM_TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    n = _OEM_N[tier]

    ordinary_lo, ordinary_hi = _OEM_EXAM_ORDINARY if tier == "exam" else _OEM_ORDINARY

    second_extreme = None
    forced_index = None
    if tier in ("hard", "exam"):
        # a genuinely different second extreme value, present but never
        # designated -- drawn from the SMALL outlier magnitude band so it
        # is clearly distinguishable from the designated outlier below.
        # Inserted BEFORE the sum-to-multiple-of-(n-1) adjustment runs (at
        # an index that is never the LAST one, which the adjustment step
        # itself may still rewrite), so the modulus invariant this
        # generator relies on is established AFTER the override, not
        # silently broken by it.
        se_lo, se_hi = _OEM_OUTLIER_SMALL
        second_extreme = rng.randint(se_lo, se_hi)
        if rng.random() < 0.5 and tier == "hard":
            second_extreme = -second_extreme
        forced_index = rng.randrange(n - 2)  # never the last index of n-1 values

    rest = _oem_draw_clean(
        rng,
        n - 1,
        ordinary_lo,
        ordinary_hi,
        n - 1,
        forced_index=forced_index,
        forced_value=second_extreme,
    )
    mean_without = Fraction(sum(rest), n - 1)

    if tier == "medium":
        lo, hi = _OEM_OUTLIER_SMALL
    else:  # easy, hard, exam all use the LARGE magnitude band for the designated outlier
        lo, hi = _OEM_OUTLIER_LARGE
    forced_sign = -1 if tier == "exam" else 0  # exam's outlier must stay negative

    x = _oem_draw_bounded_multiple(rng, lo, hi, forced_sign, n, sum(rest))
    if tier == "exam":
        check(x < 0, "p12d_outlier_effect_mean: exam's designated outlier must stay negative")

    mean_with = Fraction(sum(rest) + x, n)
    diff = mean_with - mean_without
    _oem_verify(x, mean_without, n, diff)
    check(diff != 0, "p12d_outlier_effect_mean: true difference is zero")

    diff_text = stated_precision_string(diff, _OEM_DP)
    data = rest + [x]
    rng.shuffle(data)
    values_text = ", ".join(str(v) for v in data)
    text = phrase(rng, _OEM_TEMPLATE_BANKS[tier], n=n, values=values_text, outlier=x)

    tol = stated_precision_tolerance(_OEM_DP)
    candidates = []

    reversed_sign = -diff
    if abs(reversed_sign - diff) > tol:
        candidates.append(_oem_fmt_distractor(reversed_sign, _OEM_DP))

    wrong_denom_mean_without = Fraction(sum(rest), n)
    wrong_denom_diff = mean_with - wrong_denom_mean_without
    if abs(wrong_denom_diff - diff) > tol:
        candidates.append(_oem_fmt_distractor(wrong_denom_diff, _OEM_DP))

    if second_extreme is not None:
        rest_without_second = [v for v in rest if v != second_extreme]
        # remove exactly one occurrence
        rest_minus_se = list(rest)
        rest_minus_se.remove(second_extreme)
        mean_without_other = Fraction(sum(rest_minus_se) + x, n - 1)
        mean_with_full = mean_with
        removes_wrong_value_diff = mean_with_full - mean_without_other
        if abs(removes_wrong_value_diff - diff) > tol:
            candidates.append(_oem_fmt_distractor(removes_wrong_value_diff, _OEM_DP))

    for c in candidates:
        check(c != diff_text, f"p12d_outlier_effect_mean: a distractor matches the answer: {c!r}")
    check_distinct(diff_text, *candidates, msg="p12d_outlier_effect_mean: distractor equals answer")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(diff),
        answer_text=diff_text,
        answer_latex=latex_answer(diff_text),
        distractors=pick_distractors(diff_text, candidates),
        working=[
            f"Mean with outlier: ({' + '.join(paren_negative(v) for v in data)}) / {n} = "
            f"{stated_precision_string(mean_with, _OEM_DP)}.",
            f"Mean without outlier ({paren_negative(x)}): sum / {n - 1} = "
            f"{stated_precision_string(mean_without, _OEM_DP)}.",
            f"Difference: {stated_precision_string(mean_with, _OEM_DP)} - "
            f"{('(' + stated_precision_string(mean_without, _OEM_DP) + ')') if mean_without < 0 else stated_precision_string(mean_without, _OEM_DP)} "
            f"= {diff_text}.",
        ],
    )


@generator(
    "p12d_outlier_effect_mean",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12d_outlier_effect_mean(rng, tier: str = "easy"):
    return _build_outlier_effect(rng, tier=tier)
