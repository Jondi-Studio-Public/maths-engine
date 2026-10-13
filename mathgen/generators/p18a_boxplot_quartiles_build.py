"""Generates box-plot quartile questions: Q1, Q3 and the IQR of a stated data set by the exclusive (Moore-McCabe)
method, as a three-label pair.

`answer: pair`, a 3-label `answer_text` ("Q1 = .., Q3 = .., IQR = .."), graded
by the `pair` grader with labels -- same discipline as
`senior-complete-a-five-number-summary-for-different-datasets`
(`mathgen/generators/p12d_stats_summary_build.py`). Every tier emits this
same triple; only the reasoning required to get there escalates.

## The exclusive (Moore-McCabe) method, pinned in every prompt in words

Sort the data. If the data set size is odd, EXCLUDE the overall median from
both halves; Q1 is the median of the lower half, Q3 is the median of the
upper half. For an even-sized data set there is no median to exclude, so the
exclusive and Tukey/inclusive methods coincide (the split point is the same);
they can only genuinely differ when the data set size is odd -- confirmed
directly in `_fq_inclusive` below (identical to `_fq_construct` for even n).

Every quartile therefore sits on the half-integer grid (the median of an
odd-length integer list is a literal element; of an even-length list, the
average of two integers, which ends in .0 or .5) -- exactly the "differ by
>= 0.5" invariant, checked on every candidate
unconditionally (`_fq_check_min_gap`).

### Independent-check proof

Construction (`_fq_construct`) computes Q1/Q3 by direct index arithmetic: it
slices the sorted list into `lower`/`upper` (excluding the middle element
from BOTH when `n` is odd) and takes the literal middle element of an
odd-length half, or the average of the two middle elements of an even-length
half, via its own small `_median_of` helper -- never calling a library
function. `_fq_verify` instead re-slices independently and calls Python's
own `statistics.median` (which returns the literal middle element for an
odd-length input and the mean of the two middle elements for an even-length
one -- the same rule, from a different piece of code) -- a genuinely
different route to the same two numbers, proven equal every seed by
`check_equivalent`, and shown to actually fire on a corrupted value in the
test file.

## Difficulty progression (genuine reasoning escalation, not size)

- easy: an already-sorted list (n in 7..12, odd n on >= 40% of draws
  per tier by construction -- `_draw_n` biases 60% odd). DECISION: apply the
  halving rule directly; no sorting step is needed.
- medium: the same kind of list, presented unsorted. DECISION: the learner
  must sort first, and -- for odd n -- decide which single value (the
  overall median) to drop from BOTH halves before halving; that decision is
  simply unavailable on easy, where presented order already IS sorted order.
- hard: the data is given as a frequency table (value: count pairs, in an
  order that is not sorted by value) instead of a raw list. DECISION: the
  learner must first decode/expand the table into the full ordered list --
  a new representation-decoding step neither easy nor medium requires --
  before the same halving rule as medium applies.
- exam: two independently-drawn, named groups' data (each unsorted, same
  presentation as medium/hard-shaped); the prompt asks for the triple of
  whichever group has the LARGER IQR. DECISION: the learner must fully solve
  BOTH datasets and compare their IQRs to choose which one to report -- a
  comparison/judgement step that does not exist at any lower tier, where
  there is only ever one dataset to resolve.

## Misconceptions (every candidate a genuinely different (Q1, Q3, IQR)
Fraction triple, checked directly every seed -- never a display variant)

- `tukey-inclusive`: computes Q1/Q3 by the Tukey/inclusive method (median of
  each half INCLUDING the overall median for odd n) instead of the exclusive
  method the prompt states. Offered only when it actually differs (never,
  by construction, for even n).
- `swaps-q1-q3`: Q1 and Q3 exchanged, so the offered IQR is the (negative)
  Q1 - Q3 rather than Q3 - Q1. Offered whenever Q1 != Q3 (always true here
  in practice; checked directly).
- `iqr-as-range`: correct Q1/Q3, but IQR replaced by max - min (the range)
  instead of Q3 - Q1. Offered whenever the range and the true IQR genuinely
  differ.
- `positional-no-averaging`: takes the literal 25th-percentile-position and
  75th-percentile-position elements (`floor(0.25*(n-1))` /
  `floor(0.75*(n-1))`, 0-indexed) directly, with no averaging step at all,
  even where the true method calls for one. Offered whenever it differs.
- `off-by-one-split`: shifts the lower/upper split boundary one index to the
  right of the correct one before taking each half's median. Offered
  whenever it differs.
- exam only, `wrong-group`: reports the OTHER (smaller-IQR) group's
  correctly-computed triple -- the one specific, on-target mistake this
  tier's own genuinely new decision (which group to report) invites.
  Offered whenever the two groups' triples differ (guaranteed: construction
  redraws group B until the two IQRs differ).
"""

from __future__ import annotations

import statistics
from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    stated_precision_string,
)

TOPIC = "p18a_boxplot_quartiles"
_TIERS = ("easy", "medium", "hard", "exam")

_METHOD = (
    "Use the exclusive method: sort the values, and if there is an odd number of "
    "values, exclude the overall median from both halves; Q1 is the median of the "
    "lower half and Q3 is the median of the upper half."
)

# The exam tier states the quartile convention as a fact, not as steps to follow.
_EXAM_METHOD = (
    "Quartiles follow the exclusive method: if there is an odd number of values, the overall median is "
    "excluded from both halves; Q1 is the median of the lower half and Q3 is the median of the upper half."
)

_CONTEXTS = ("heights", "times", "scores")
_RANGE = {"heights": (120, 190), "times": (5, 90), "scores": (30, 100)}
_CONTEXT_LABEL = {
    "heights": "seedling heights (cm)",
    "times": "delivery times (minutes)",
    "scores": "test scores (out of 100)",
}


# ============================================================================
# Quartile arithmetic
# ============================================================================


def _median_of(values: list[int]) -> Fraction:
    m = len(values)
    if m % 2 == 1:
        return Fraction(values[m // 2])
    return Fraction(values[m // 2 - 1] + values[m // 2], 2)


def _fq_construct(sorted_vals: list[int]):
    """Direct index-arithmetic route: slice, then `_median_of` each half."""
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        lower, upper = sorted_vals[:mid], sorted_vals[mid:]
    else:
        lower, upper = sorted_vals[:mid], sorted_vals[mid + 1 :]
    return _median_of(lower), _median_of(upper)


def _fq_verify(sorted_vals: list[int], q1: Fraction, q3: Fraction):
    """Independent route: re-slice and call `statistics.median` (a different
    piece of code from `_median_of`, though the same rule) -- see module
    docstring for why the two agree on every seed."""
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        lower, upper = sorted_vals[:mid], sorted_vals[mid:]
    else:
        lower, upper = sorted_vals[:mid], sorted_vals[mid + 1 :]
    check_equivalent(
        q1, Fraction(statistics.median(lower)), "p18a_boxplot_quartiles: Q1 disagrees with independent recomputation"
    )
    check_equivalent(
        q3, Fraction(statistics.median(upper)), "p18a_boxplot_quartiles: Q3 disagrees with independent recomputation"
    )


def _fq_inclusive(sorted_vals: list[int]):
    """Tukey/inclusive method: include the overall median in BOTH halves for
    odd n. Identical to `_fq_construct` for even n (no median to include)."""
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        lower, upper = sorted_vals[:mid], sorted_vals[mid:]
    else:
        lower, upper = sorted_vals[: mid + 1], sorted_vals[mid:]
    return _median_of(lower), _median_of(upper)


def _fq_positional_no_averaging(sorted_vals: list[int]):
    n = len(sorted_vals)
    i1 = int(0.25 * (n - 1))
    i3 = int(0.75 * (n - 1))
    return Fraction(sorted_vals[i1]), Fraction(sorted_vals[i3])


def _fq_off_by_one(sorted_vals: list[int]):
    n = len(sorted_vals)
    split = n // 2 + 1
    lower, upper = sorted_vals[:split], sorted_vals[split:]
    if not lower or not upper:
        return None
    return _median_of(lower), _median_of(upper)


def _fq_check_min_gap(answer, candidate, msg: str):
    """Every distractor triple must differ from the answer in AT LEAST one
    component by >= 0.5 -- the half-integer-grid invariant,
    checked unconditionally, not merely assumed from the grid claim."""
    check(any(abs(a - c) >= Fraction(1, 2) for a, c in zip(answer, candidate)), msg)


def _stat(v) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else stated_precision_string(v, 1)


def _labelled(q1, q3, iqr) -> str:
    return f"Q1 = {_stat(q1)}, Q3 = {_stat(q3)}, IQR = {_stat(iqr)}"


# ============================================================================
# Drawing
# ============================================================================


def _draw_n(rng, lo=7, hi=12):
    """n in [lo, hi], odd on ~60% of draws (comfortably clears the required
    >= 40% floor over any reasonably sized sweep)."""
    parity = 1 if rng.random() < 0.6 else 0
    pool = [v for v in range(lo, hi + 1) if v % 2 == parity]
    if not pool:
        pool = list(range(lo, hi + 1))
    return rng.choice(pool)


def _draw_plain_list(rng, context, n):
    lo, hi = _RANGE[context]
    return sorted(rng.sample(range(lo, hi + 1), n))


def _draw_frequency_table(rng, context):
    """A small distinct-value/count table whose expansion has n in [8, 13].
    Returns (sorted_expanded_list, presented_pairs) where `presented_pairs`
    is a list of (value, count) in a NON-value-sorted display order."""
    lo, hi = _RANGE[context]
    n = _draw_n(rng, lo=8, hi=13)
    k = min(rng.randint(4, 6), n - 1)
    values = sorted(rng.sample(range(lo, hi + 1), k))
    counts = [1] * k
    remaining = n - k
    for _ in range(remaining):
        counts[rng.randrange(k)] += 1
    expanded = []
    for v, c in zip(values, counts):
        expanded.extend([v] * c)
    expanded.sort()
    pairs = list(zip(values, counts))
    rng.shuffle(pairs)
    return expanded, pairs


# ============================================================================
# Templates
# ============================================================================

_EASY_TEMPLATES = (
    "A sample of {label}, already sorted from smallest to largest, is: {values}. {method} "
    "State Q1, Q3 and the interquartile range (IQR).",
    "Here are {n} {label}, listed from smallest to largest: {values}. {method} Give Q1, Q3 and the IQR.",
    "The following {label} have already been arranged in ascending order: {values}. {method} Find Q1, Q3 and the IQR.",
)
_MEDIUM_TEMPLATES = (
    "A sample of {label} was recorded in this order: {values}. {method} Sort the values "
    "first, then state Q1, Q3 and the interquartile range (IQR).",
    "Here are {n} {label}, in the order they were collected: {values}. {method} Sort them "
    "first, then give Q1, Q3 and the IQR.",
    "The following {label} were recorded in no particular order: {values}. {method} Order "
    "them first, then find Q1, Q3 and the IQR.",
)
_HARD_TEMPLATES = (
    "A sample of {label} is given as a frequency table (value: count): {pairs}. {method} "
    "Expand the table into the full ordered list of {n} values, then state Q1, Q3 and the "
    "interquartile range (IQR).",
    "The following frequency table (value: count) records {n} {label}: {pairs}. {method} "
    "List all {n} values in order first, then give Q1, Q3 and the IQR.",
    "Here is a frequency table (value: count) of {n} {label}: {pairs}. {method} Write out "
    "every value in order, then find Q1, Q3 and the IQR.",
)
_EXAM_TEMPLATES = (
    "Two groups' {label} were recorded. Group A: {values_a}. Group B: {values_b}. {method} "
    "Find Q1, Q3 and the IQR for whichever group has the LARGER interquartile range.",
    "{label_cap}, group A: {values_a}; group B: {values_b}. {method} "
    "State Q1, Q3 and the IQR for the group with the larger IQR only.",
    "Two groups' {label} are compared. Group A: {values_a}. Group B: "
    "{values_b}. {method} Give Q1, Q3 and the IQR for the group whose IQR is larger.",
)


def _values_text(vals):
    return ", ".join(str(v) for v in vals)


def _pairs_text(pairs):
    return ", ".join(f"{v}: {c}" for v, c in pairs)


# ============================================================================
# Distractor construction (shared)
# ============================================================================


def _fq_candidates(sorted_vals, q1, q3, iqr):
    answer = (q1, q3, iqr)
    raw = []

    incl_q1, incl_q3 = _fq_inclusive(sorted_vals)
    incl = (incl_q1, incl_q3, incl_q3 - incl_q1)
    if incl != answer:
        raw.append(incl)

    swap = (q3, q1, q1 - q3)
    if swap != answer:
        raw.append(swap)

    rng_val = Fraction(max(sorted_vals) - min(sorted_vals))
    if rng_val != iqr:
        raw.append((q1, q3, rng_val))

    pos_q1, pos_q3 = _fq_positional_no_averaging(sorted_vals)
    pos = (pos_q1, pos_q3, pos_q3 - pos_q1)
    if pos != answer:
        raw.append(pos)

    off = _fq_off_by_one(sorted_vals)
    if off is not None:
        off_q1, off_q3 = off
        off_triple = (off_q1, off_q3, off_q3 - off_q1)
        if off_triple != answer:
            raw.append(off_triple)

    return raw


def _finalise(text, answer_triple, candidate_triples, working, n_distractors: int = 3):
    q1, q3, iqr = answer_triple
    answer_text = _labelled(q1, q3, iqr)
    answer = {"Q1": q1, "Q3": q3, "IQR": iqr}

    seen = {answer_triple}
    formatted = []
    for cand in candidate_triples:
        if cand in seen:
            continue
        seen.add(cand)
        _fq_check_min_gap(
            answer_triple, cand, f"p18a_boxplot_quartiles: candidate {cand} too close to answer {answer_triple}"
        )
        formatted.append(_labelled(*cand))

    check(len(formatted) >= 1, "p18a_boxplot_quartiles: no distractor survived")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, formatted, n=n_distractors),
        working=working,
    )


# ============================================================================
# Build
# ============================================================================


def _build(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    context = rng.choice(_CONTEXTS)
    label = _CONTEXT_LABEL[context]

    if tier == "easy":
        n = _draw_n(rng)
        sorted_vals = _draw_plain_list(rng, context, n)
        q1, q3 = _fq_construct(sorted_vals)
        _fq_verify(sorted_vals, q1, q3)
        iqr = q3 - q1
        text = phrase(rng, _EASY_TEMPLATES, label=label, n=n, values=_values_text(sorted_vals), method=_METHOD)
        candidates = _fq_candidates(sorted_vals, q1, q3, iqr)
        working = [f"Sorted: {_values_text(sorted_vals)}.", _labelled(q1, q3, iqr)]
        return _finalise(text, (q1, q3, iqr), candidates, working)

    if tier == "medium":
        n = _draw_n(rng)
        sorted_vals = _draw_plain_list(rng, context, n)
        presented = list(sorted_vals)
        tries = 0
        while True:
            rng.shuffle(presented)
            tries += 1
            if presented != sorted_vals or tries >= 10:
                break
        q1, q3 = _fq_construct(sorted_vals)
        _fq_verify(sorted_vals, q1, q3)
        iqr = q3 - q1
        text = phrase(rng, _MEDIUM_TEMPLATES, label=label, n=n, values=_values_text(presented), method=_METHOD)
        candidates = _fq_candidates(sorted_vals, q1, q3, iqr)
        working = [f"Sorted: {_values_text(sorted_vals)}.", _labelled(q1, q3, iqr)]
        return _finalise(text, (q1, q3, iqr), candidates, working)

    if tier == "hard":
        sorted_vals, pairs = _draw_frequency_table(rng, context)
        n = len(sorted_vals)
        q1, q3 = _fq_construct(sorted_vals)
        _fq_verify(sorted_vals, q1, q3)
        iqr = q3 - q1
        text = phrase(rng, _HARD_TEMPLATES, label=label, n=n, pairs=_pairs_text(pairs), method=_METHOD)
        candidates = _fq_candidates(sorted_vals, q1, q3, iqr)
        working = [f"Expanded and sorted: {_values_text(sorted_vals)}.", _labelled(q1, q3, iqr)]
        return _finalise(text, (q1, q3, iqr), candidates, working)

    # exam: two named groups, report the one with the larger IQR.
    for _ in range(200):
        n_a = _draw_n(rng)
        n_b = _draw_n(rng)
        vals_a = _draw_plain_list(rng, context, n_a)
        vals_b = _draw_plain_list(rng, context, n_b)
        q1_a, q3_a = _fq_construct(vals_a)
        q1_b, q3_b = _fq_construct(vals_b)
        iqr_a, iqr_b = q3_a - q1_a, q3_b - q1_b
        if iqr_a != iqr_b:
            break
    else:
        check(False, "p18a_boxplot_quartiles: could not draw two groups with different IQRs")

    _fq_verify(vals_a, q1_a, q3_a)
    _fq_verify(vals_b, q1_b, q3_b)

    if iqr_a > iqr_b:
        winner_q1, winner_q3, winner_iqr, winner_vals = q1_a, q3_a, iqr_a, vals_a
        loser_q1, loser_q3, loser_iqr = q1_b, q3_b, iqr_b
    else:
        winner_q1, winner_q3, winner_iqr, winner_vals = q1_b, q3_b, iqr_b, vals_b
        loser_q1, loser_q3, loser_iqr = q1_a, q3_a, iqr_a

    presented_a = list(vals_a)
    presented_b = list(vals_b)
    tries = 0
    while True:
        rng.shuffle(presented_a)
        tries += 1
        if presented_a != vals_a or tries >= 10:
            break
    tries = 0
    while True:
        rng.shuffle(presented_b)
        tries += 1
        if presented_b != vals_b or tries >= 10:
            break

    text = phrase(
        rng,
        _EXAM_TEMPLATES,
        label=label,
        label_cap=label.capitalize(),
        values_a=_values_text(presented_a),
        values_b=_values_text(presented_b),
        method=_EXAM_METHOD,
    )
    candidates = [(loser_q1, loser_q3, loser_iqr)] + _fq_candidates(
        winner_vals,
        winner_q1,
        winner_q3,
        winner_iqr,
    )
    working = [
        f"Group A sorted: {_values_text(vals_a)}. Group B sorted: {_values_text(vals_b)}.",
        f"IQR(A) = {_stat(iqr_a)}, IQR(B) = {_stat(iqr_b)}.",
        _labelled(winner_q1, winner_q3, winner_iqr),
    ]
    return _finalise(
        text,
        (winner_q1, winner_q3, winner_iqr),
        candidates,
        working,
        n_distractors=4,
    )


@generator(
    TOPIC,
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p18a_boxplot_quartiles(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
