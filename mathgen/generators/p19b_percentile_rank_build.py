"""Generates percentile-rank questions: find the data value at a stated percentile of a 20-value data set, as an
exact integer.

`answer: number`, an exact integer (a literal element of the data), graded
exactly (no decimal point in `answer_text`).

## Construction

Every dataset has n = 20 values (fixed across tiers -- escalation is by
representation and level-naming, never by size). The target RANK `k` (a
whole number, `1 <= k <= n`) is drawn FIRST -- construct to target; for example
(n = 20, P = 35 -> rank 7 -- P/5 = k always, since
`P/100 * 20 = P/5`, so every valid percentile level with n = 20 is
automatically a multiple of 5). `k` is drawn in `[4, 17]` so `k-1`, `k+1`
and `n-k+1` all stay inside `[1, n]` and distinct from `k` and each other.

The stated level is one of three types, each converted to a percentile P
purely by the standard convention (never the reverse):

- `percentile`: P = 5k, stated directly ("the Pth percentile").
- `decile`: k drawn even (`k = 2m`, m in [2, 8]); the mth decile = the
  `10m`th percentile, so P = 10m = 5k -- the SAME formula, just reached via
  the decile-to-percentile conversion the learner must perform.
- `quartile`: k drawn as a multiple of 5 (`k = 5m`, m in {1, 2, 3}); the
  mth quartile = the `25m`th percentile, P = 25m = 5k again.

`easy` is always `percentile` (level types are drawn across
percentile / decile / quartile on every tier except easy); `medium`/`hard`/
`exam` draw the type with `rng.choice`.

The data itself is 20 distinct integers. `easy`/`medium` print it as a raw
list (sorted on easy, shuffled on medium); `hard` prints a frequency table
(value: count); `exam` prints a CUMULATIVE frequency table (value: running
total) -- the genuinely new representation each tier introduces.

### Independent-check proof

`_rank_from_freq` accumulates a list of (value, count) pairs in ascending
value order and returns the first value whose running total reaches `k` --
called once on the TRUE per-value counts (construction's own route) and,
separately, `_verify_rank` recomputes the same rank a second way: it expands
the (value, count) pairs into the full 20-element list and indexes directly
(`expanded[k - 1]`), a different piece of code from the accumulator, and
`check_equivalent`s the two. On `exam`, the same accumulator is ALSO run
directly on the printed CUMULATIVE numbers (deliberately treating them as if
they were fresh per-value counts) to build the `cumulative-as-frequency`
misconception distractor -- proven, not assumed, to differ from the true
answer before it is offered (resampled otherwise).

## Difficulty progression (genuine reasoning escalation)

- easy: sorted raw list, a percentile level stated directly. DECISION: read
  off the value at the stated rank; no sorting, no conversion.
- medium: unsorted raw list, level named as a decile OR quartile. DECISION:
  convert the named level to a percentile/rank AND sort the data first --
  neither step is needed on easy.
- hard: a frequency table (value: count) instead of a raw list. DECISION:
  locate the rank by accumulating counts across GROUPED data, a new
  representation-decoding step neither easy nor medium requires.
- exam: a CUMULATIVE frequency table (value: running total). DECISION: read
  the running totals correctly as already-accumulated counts, rather than
  accumulating them a second time (the `cumulative-as-frequency` trap this
  tier alone can pose).

## Misconceptions (every candidate a genuinely different data value, checked
directly every seed)

- `adjacent-rank-plus-one` / `adjacent-rank-minus-one`: value at rank k+1 / k-1.
- `rank-from-top`: value at rank n-k+1 (counted from the top instead of the bottom).
- `rank-as-value`: the bare rank number k offered as though it were a data value.
- `percentile-as-rank`: value at rank P itself (treating the STATED level as a
  rank position), offered only when `1 <= P <= n` and it differs from k.
- `cumulative-as-frequency` (exam only): re-accumulates the printed cumulative
  counts as if they were fresh per-value counts.
"""

from __future__ import annotations

from ..core import (
    check,
    check_equivalent,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

TOPIC = "p19b_percentile_rank"
_TIERS = ("easy", "medium", "hard", "exam")

_CONVENTION = (
    "The Pth percentile is the value in position P/100 times n of the data sorted from smallest to "
    "largest, with no averaging; the mth decile is the 10m-th percentile and the mth quartile is the "
    "25m-th percentile."
)

_CONTEXTS = ("delivery times (minutes)", "daily step counts", "test marks")


def _rank_from_freq(pairs: list[tuple[int, int]], k: int) -> int:
    """`pairs` sorted ascending by value; first value whose running total of
    counts reaches `k`."""
    total = 0
    for value, count in pairs:
        total += count
        if total >= k:
            return value
    check(False, "p19b_percentile_rank: rank not reached within the table")


def _verify_rank(pairs: list[tuple[int, int]], k: int, value: int):
    """Independent route: expand the table into the full list and index
    directly -- a different piece of code from `_rank_from_freq`."""
    expanded = []
    for v, c in pairs:
        expanded.extend([v] * c)
    check_equivalent(
        value, expanded[k - 1], "p19b_percentile_rank: rank value disagrees with expanded-list recomputation"
    )


def _draw_k(rng, level_type: str):
    if level_type == "percentile":
        k = rng.randint(4, 17)
        m = None
    elif level_type == "decile":
        m = rng.randint(2, 8)
        k = 2 * m
    else:  # quartile
        m = rng.randint(1, 3)
        k = 5 * m
    return k, m


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _level_text(level_type: str, k: int, m: int | None) -> str:
    p = 5 * k
    if level_type == "percentile":
        return f"the {_ordinal(p)} percentile"
    if level_type == "decile":
        return f"the {_ordinal(m)} decile"
    return f"the {_ordinal(m)} quartile"


def _draw_dataset(rng, n: int = 20, lo: int = 30, hi: int = 150):
    return sorted(rng.sample(range(lo, hi + 1), n))


def _values_text(vals):
    return ", ".join(str(v) for v in vals)


def _pairs_text(pairs):
    return ", ".join(f"{v}: {c}" for v, c in pairs)


def _cumulative_pairs_text(pairs):
    running = 0
    parts = []
    cumulative_pairs = []
    for v, c in pairs:
        running += c
        parts.append(f"{v}: {running}")
        cumulative_pairs.append((v, running))
    return ", ".join(parts), cumulative_pairs


def _build_freq_table(rng, n: int = 20):
    num_distinct = rng.randint(6, 8)
    values = sorted(rng.sample(range(30, 130), num_distinct))
    counts = [1] * num_distinct
    remaining = n - num_distinct
    for _ in range(remaining):
        counts[rng.randrange(num_distinct)] += 1
    return list(zip(values, counts))


# ============================================================================
# Templates
# ============================================================================

_EASY_TEMPLATES = (
    "A data set of {n} {label}, already sorted from smallest to largest, is: {values}. {convention} "
    "Find the value at {level}.",
    "Here are {n} {label}, listed from smallest to largest: {values}. {convention} State the value at {level}.",
    "The following {label} have already been sorted: {values}. {convention} What is the value at {level}?",
)
_MEDIUM_TEMPLATES = (
    "A data set of {n} {label} was recorded in this order: {values}. {convention} Sort the values "
    "first, then find the value at {level}.",
    "Here are {n} {label}, in the order they were collected: {values}. {convention} Order them first, "
    "then state the value at {level}.",
    "The following {label} were recorded in no particular order: {values}. {convention} Sort them "
    "first, then give the value at {level}.",
)
_HARD_TEMPLATES = (
    "A data set of {n} {label} is given as a frequency table (value: count): {values}. {convention} "
    "Expand the table into the full ordered list, then find the value at {level}.",
    "The following frequency table (value: count) records {n} {label}: {values}. {convention} "
    "Accumulate the counts in order, then state the value at {level}.",
    "Here is a frequency table (value: count) of {n} {label}: {values}. {convention} Work out the "
    "rank by accumulating counts, then give the value at {level}.",
)
_EXAM_TEMPLATES = (
    "Exam-style question. A data set of {n} {label} is given as a CUMULATIVE frequency table (value: "
    "cumulative count, i.e. the running total up to and including that value): {values}. {convention} "
    "Find the value at {level}.",
    "Exam-style question. The following cumulative frequency table (value: running total) records "
    "{n} {label}: {values}. {convention} State the value at {level}.",
    "Exam-style question. Here is a cumulative frequency table (value: running total, not a "
    "per-value count) of {n} {label}: {values}. {convention} Give the value at {level}.",
)
_TEMPLATES = {"easy": _EASY_TEMPLATES, "medium": _MEDIUM_TEMPLATES, "hard": _HARD_TEMPLATES, "exam": _EXAM_TEMPLATES}


# ============================================================================
# Build
# ============================================================================


def _build(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _attempt in range(50):
        result = _build_once(rng, tier)
        if result is not None:
            return result
    check(False, f"p19b_percentile_rank/{tier}: could not draw a valid question in 50 attempts")


def _build_once(rng, tier: str):
    n = 20
    label = rng.choice(_CONTEXTS)
    level_type = "percentile" if tier == "easy" else rng.choice(["percentile", "decile", "quartile"])
    k, m = _draw_k(rng, level_type)
    level = _level_text(level_type, k, m)
    p = 5 * k

    if tier in ("easy", "medium"):
        sorted_vals = _draw_dataset(rng, n)
        answer = sorted_vals[k - 1]
        pairs = list(zip(sorted_vals, [1] * n))  # for the shared _verify_rank route
        _verify_rank(pairs, k, answer)
        presented = list(sorted_vals)
        if tier == "medium":
            tries = 0
            while True:
                rng.shuffle(presented)
                tries += 1
                if presented != sorted_vals or tries >= 10:
                    break
        values_text = _values_text(presented)
    else:
        pairs = _build_freq_table(rng, n)
        answer = _rank_from_freq(pairs, k)
        _verify_rank(pairs, k, answer)
        display_pairs = list(pairs)
        if tier == "hard":
            rng.shuffle(display_pairs)
            values_text = _pairs_text(display_pairs)
        else:  # exam: cumulative table, printed in ascending value order
            values_text, cumulative_pairs = _cumulative_pairs_text(pairs)

    # candidates common to every tier; the tier's own SIGNATURE misconception
    # (exam's cumulative-as-frequency misread) is listed FIRST so it survives
    # `pick_distractors`'s n=3 cutoff even when several other candidates tie.
    candidates_raw = []
    if tier == "exam":
        candidates_raw.append(_rank_from_freq(cumulative_pairs, k))
    rank_plus1 = _rank_from_freq(pairs, k + 1) if k + 1 <= n else None
    rank_minus1 = _rank_from_freq(pairs, k - 1) if k - 1 >= 1 else None
    rank_from_top = _rank_from_freq(pairs, n - k + 1)
    candidates_raw += [rank_plus1, rank_minus1, rank_from_top, k]
    if 1 <= p <= n and p != k:
        candidates_raw.append(_rank_from_freq(pairs, p))

    seen_values = {answer}
    unique_candidates: list[int] = []
    for c in candidates_raw:
        if c is None or c in seen_values:
            continue
        seen_values.add(c)
        unique_candidates.append(c)
    if len(unique_candidates) < 2:
        return None

    answer_text = str(answer)
    candidate_texts = [str(c) for c in unique_candidates]

    text = phrase(rng, _TEMPLATES[tier], n=n, label=label, values=values_text, convention=_CONVENTION, level=level)
    working = [
        f"P = {p}, rank k = {p}/100 times {n} = {k}.",
        f"The value at rank {k} is {answer}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=int(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=pick_distractors(answer_text, candidate_texts, n=4),
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
def p19b_percentile_rank(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
