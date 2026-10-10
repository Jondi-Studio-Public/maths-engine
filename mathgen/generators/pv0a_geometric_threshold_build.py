"""Generates `pv0a_geometric_threshold` questions (ACMMM076): when a geometric growth or decay first passes a
stated threshold.

Scope: "Given a geometric growth or decay context
with a stated first value and rate per period, find the first period in which
the value passes a stated threshold." The answer is always a whole-number
period -- never the value at a stated period (that is the sharp neighbours
``compound-interest`` and ``senior-use-the-formula-t-n-r-n-1``, which run the
computation the other way) and never a linear model (that is
``senior-use-arithmetic-sequences-in-contexts-involving-discrete-linear``).

Every question states its own convention explicitly (period 0 -- "now"/
"initially", value at period n is V*R^n -- or period 1 -- "in the first
year/hour the value is V", value at period m is V*R^(m-1)) and its own
direction ("first EXCEEDS" for growth, "first FALLS BELOW" for decay).

Construct-to-target: the key period ``k_target`` is drawn FIRST, uniformly
over a wide range; the threshold is then placed at the exact log-space point
``V * factor ** (k_target - 1 + f)`` with ``f`` drawn from [0.15, 0.85] so
the true crossing period sits clear of both neighbouring integers, rounded to
a plausible display value and RE-CHECKED (on the rounded value) against the
mandatory [0.05, 0.95] log-space margin -- this is what keeps the
key period off the boundary where a rounded rate or calculator step could
flip the answer.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import (
    check,
    check_pairwise_distinct,
    generator,
    pick_distractors,
)

# Rates for every DOLLAR-denominated context (compound interest, a machine's dollar value, an
# investment) -- capped 2%-12% p.a. at every tier: a stated rate above 12% is implausible for money,
# however plausible it is for a population/colony. Never used for a non-money context.
_GROWTH_MONEY_RATES = (2, 3, 4, 5, 6, 8, 10, 12)
_DECAY_MONEY_RATES = (2, 4, 6, 8, 10, 12)
#: Non-money contexts (a colony, an insect population, a town, a radioactive
#: sample's mass, a two-stage population) may use a wider, more realistic
#: range; never applied where the rendered text carries a "$".
_GROWTH_POP_RATES = (5, 8, 10, 12, 15, 20)
_DECAY_POP_RATES = (5, 8, 10, 15, 20, 25, 30, 40)
_DECAY_DRUG_RATES = (5, 8, 10, 12, 15, 20)
_POP_TWO_STAGE_RATES = (5, 6, 8, 10, 12, 15, 18, 20, 25)


def _factor(rate_pct, growth: bool) -> Fraction:
    return Fraction(100 + rate_pct, 100) if growth else Fraction(100 - rate_pct, 100)


def _first_crossing(first_value, factor: Fraction, threshold, growth: bool) -> int:
    """Smallest k >= 0 with value(k) = first_value * factor**k strictly past
    `threshold` (greater for growth, less for decay). Exact `Fraction`
    arithmetic throughout -- no logs."""
    k = 0
    val = Fraction(first_value)
    threshold = Fraction(threshold)
    while True:
        if growth:
            if val > threshold:
                return k
        else:
            if val < threshold:
                return k
        val *= factor
        k += 1
        check(k < 300, "runaway search: threshold never crossed")


def _first_crossing_two_stage(first_value, r1: Fraction, m: int, r2: Fraction, threshold, growth: bool) -> int:
    """Like `_first_crossing`, but the factor is `r1` for the first `m`
    periods and `r2` after -- the exam tier's rate change."""
    k = 0
    val = Fraction(first_value)
    threshold = Fraction(threshold)
    while True:
        if growth:
            if val > threshold:
                return k
        else:
            if val < threshold:
                return k
        val *= r1 if k < m else r2
        k += 1
        check(k < 300, "runaway two-stage search: threshold never crossed")


def _first_crossing_linear(first_value, step: Fraction, threshold, growth: bool) -> int:
    """The simple-interest-style misconception: value(k) = first_value +/-
    step*k, a constant amount added/removed every period, never compounding."""
    k = 0
    val = Fraction(first_value)
    threshold = Fraction(threshold)
    while True:
        if growth:
            if val > threshold:
                return k
        else:
            if val < threshold:
                return k
        val += step if growth else -step
        k += 1
        check(k < 3000, "runaway linear search: threshold never crossed")


def _log_frac(threshold: float, first_value: float, factor: float) -> float:
    return (math.log(threshold / first_value) / math.log(factor)) % 1.0


def _round_to(exact: float, places: int) -> Fraction:
    if places == 0:
        return Fraction(int(round(exact)))
    scale = 10**places
    return Fraction(int(round(exact * scale)), scale)


def _draw_threshold(rng, first_value, factor: Fraction, k_target: int, *, places: int, min_value=None):
    """Construct-to-target: place a
    threshold whose exact log-space crossing sits strictly between
    k_target-1 and k_target, round to a plausible display value and
    re-verify the [0.05, 0.95] margin against the ROUNDED value -- retried
    (bounded) since rounding can occasionally push it onto the boundary.
    ``min_value`` rejects an implausibly small display threshold (a plausible
    figure, never "the machine's value first falls below $22")."""
    for _ in range(80):
        f = rng.uniform(0.15, 0.85)
        exponent = k_target - 1 + f
        exact = float(first_value) * (float(factor) ** exponent)
        threshold = _round_to(exact, places)
        if threshold == Fraction(first_value) or threshold <= 0:
            continue
        if min_value is not None and threshold < min_value:
            continue
        margin = _log_frac(float(threshold), float(first_value), float(factor))
        if not (0.05 <= margin <= 0.95):
            continue
        return threshold, margin
    raise ValueError("could not place a threshold clear of the log-space boundary")


def _decay_k_max(first_value, factor: Fraction, min_value, k_lo: int = 3, k_hi: int = 25) -> int:
    """Largest `k_target` for a DECAYING sequence that keeps even the
    worst-case (f=0.85) exact threshold at or above `min_value` -- shared by
    every decay branch (medium's dollar floor, hard's percent floor) so
    `_draw_threshold`'s min_value retry loop is never asked to satisfy an
    impossible request (a plausible stated figure, never "$22" or
    "0% of its starting size")."""
    k = k_lo
    while float(first_value) * (float(factor) ** (k - 1 + 0.85)) >= min_value and k < k_hi:
        k += 1
    return max(k_lo, k - 1)


def _dedupe_ints(answer: int, *raw) -> list[int]:
    out, seen = [], {answer}
    for v in raw:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# =============================================================================
# easy: growth, period-0 convention, absolute threshold, compound interest
# =============================================================================

_EASY_TEMPLATES = (
    "An investment of ${V} is invested now at {r}% p.a., compounded annually. "
    "After how many whole years will its value first exceed ${T}? Give your "
    "answer as a whole number of years.",
    "A savings account starts today with a balance of ${V} and earns {r}% "
    "interest per year, compounded annually. After how many whole years "
    "does the balance first exceed ${T}? Give your answer as a whole "
    "number of years.",
    "${V} is invested at {r}% p.a. compounded annually, starting now. Find "
    "the first whole number of years after which the investment's value "
    "exceeds ${T}.",
)


def _easy(rng):
    k_target = rng.randint(3, 25)
    rate_pct = rng.choice(_GROWTH_MONEY_RATES)
    factor = _factor(rate_pct, growth=True)
    V = rng.randint(2, 20) * 500
    T, _margin = _draw_threshold(rng, V, factor, k_target, places=0, min_value=100)

    n_correct = _first_crossing(V, factor, T, growth=True)
    check(n_correct >= 2, "search resolved too quickly to be a genuine search")

    template = rng.choice(_EASY_TEMPLATES)
    text = template.format(V=V, r=rate_pct, T=int(T))

    step = Fraction(V) * rate_pct / 100
    d_minus = n_correct - 1
    d_plus = n_correct + 1
    d_linear = _first_crossing_linear(V, step, T, growth=True)
    raw = _dedupe_ints(n_correct, d_minus, d_plus, d_linear)
    check_pairwise_distinct(n_correct, *raw, msg="easy-tier distractor collides")
    distractors = pick_distractors(str(n_correct), [str(v) for v in raw], n=3)

    return dict(
        text=text,
        answer=n_correct,
        answer_text=str(n_correct),
        answer_latex=str(n_correct),
        distractors=distractors,
        working=[
            f"Applying ${V} × (1 + {rate_pct}/100)^n period by period, the balance "
            f"first exceeds ${int(T)} at n = {n_correct}."
        ],
    )


# =============================================================================
# medium: decay, period-1 convention, absolute threshold, depreciating machine
# =============================================================================

_MEDIUM_TEMPLATES = (
    "At the end of its first year, a machine is worth ${V}. It then loses "
    "{r}% of its value each year after that. In which year is its value "
    "first below ${T}? Give the year as a whole number.",
    "A machine is worth ${V} at the end of year 1, then loses {r}% of its "
    "value every following year. Find the first whole year in which its "
    "value falls below ${T}.",
    "A piece of equipment is valued at ${V} at the end of its first year of "
    "use. Each year after that, its value falls by {r}%. In which year does "
    "its value first drop below ${T}? Give the year as a whole number.",
)


def _medium(rng):
    rate_pct = rng.choice(_DECAY_MONEY_RATES)  # dollar context -- capped 2%-12%
    factor = _factor(rate_pct, growth=False)
    V = rng.randint(5, 60) * 1000
    k_target = rng.randint(3, _decay_k_max(V, factor, 100))
    T, _margin = _draw_threshold(rng, V, factor, k_target, places=0, min_value=100)

    k_correct = _first_crossing(V, factor, T, growth=False)
    n_correct = k_correct + 1  # period-1 convention: V is t_1, so k=0 is year 1
    check(k_correct >= 2, "search resolved too quickly to be a genuine search")

    template = rng.choice(_MEDIUM_TEMPLATES)
    text = template.format(V=V, r=rate_pct, T=int(T))

    step = Fraction(V) * rate_pct / 100
    d_minus = n_correct - 1
    d_plus = n_correct + 1
    d_linear = _first_crossing_linear(V, step, T, growth=False) + 1
    d_conv = k_correct  # the other period convention (period-0 slip)
    raw = _dedupe_ints(n_correct, d_minus, d_plus, d_linear, d_conv)
    check_pairwise_distinct(n_correct, *raw, msg="medium-tier distractor collides")
    distractors = pick_distractors(str(n_correct), [str(v) for v in raw], n=3)

    return dict(
        text=text,
        answer=n_correct,
        answer_text=str(n_correct),
        answer_latex=str(n_correct),
        distractors=distractors,
        working=[
            f"Applying t_m = {V} × (1 - {rate_pct}/100)^(m-1) period by period, the value "
            f"first falls below ${int(T)} at m = {n_correct}."
        ],
    )


# =============================================================================
# hard: relative threshold -- the first value is stated but cancels
# =============================================================================

_HARD_GROWTH_POP_P0 = (
    "A colony of {V} bacteria grows by {r}% every hour, starting now. After "
    "how many whole hours does the colony's size first exceed {pct}% of its "
    "starting size? Give your answer as a whole number of hours.",
    "A colony of {V} insects, counted now, grows by {r}% each week. Find "
    "the first whole number of weeks after which the colony's size exceeds "
    "{pct}% of the size it started at.",
)
_HARD_GROWTH_POP_P1 = (
    "In its first hour, a bacteria colony contains {V} organisms. It then "
    "grows by {r}% every hour after that. In which hour does its size first "
    "exceed {pct}% of that first-hour size? Give the hour as a whole number.",
)
_HARD_GROWTH_MONEY_P0 = (
    "An investment of ${V} grows by {r}% every year, starting now. After "
    "how many whole years does its value first exceed {pct}% of the amount "
    "originally invested?",
)
_HARD_DECAY_DRUG_P0 = (
    "A radioactive sample begins at {V} mg now and loses {r}% of its mass "
    "every hour. After how many whole hours does its mass first fall below "
    "{pct}% of the amount it started with?",
)
_HARD_DECAY_DRUG_P1 = (
    "In its first hour of decay, a radioactive sample has a mass of {V} mg. "
    "It then loses {r}% of its mass every subsequent hour. In which hour "
    "does its mass first fall below {pct}% of that first-hour mass? Give "
    "the hour as a whole number.",
)
_HARD_DECAY_POP_P0 = (
    "A town's population is {V} people now and falls by {r}% every year. "
    "After how many whole years does the population first fall below {pct}% "
    "of its current size?",
)


def _hard(rng):
    growth = rng.random() < 0.5
    period1 = rng.random() < 0.5
    if growth:
        flavour = rng.choice(("pop", "money")) if not period1 else "pop"
        if flavour == "money":
            rate_pct = rng.choice(_GROWTH_MONEY_RATES)  # dollar context -- capped 2%-12%
            V = rng.randint(2, 20) * 500
            templates = _HARD_GROWTH_MONEY_P0
        else:
            rate_pct = rng.choice(_GROWTH_POP_RATES)
            V = rng.randint(200, 5000)
            templates = _HARD_GROWTH_POP_P1 if period1 else _HARD_GROWTH_POP_P0
        # Growth ratios stay a plausible (< 4-digit) percent out to k=16 at
        # the largest rate offered (1.15**16 ~ 936%; the money pool's 12% is
        # milder still).
        k_target = rng.randint(3, 16)
    else:
        flavour = rng.choice(("drug", "pop")) if not period1 else "drug"
        if flavour == "pop":
            rate_pct = rng.choice(_DECAY_POP_RATES)  # non-money -- wider range
            V = rng.randint(2, 20) * 500
            templates = _HARD_DECAY_POP_P0
        else:
            rate_pct = rng.choice(_DECAY_DRUG_RATES)  # non-money (mg) -- wider range
            V = rng.randint(200, 900)
            templates = _HARD_DECAY_DRUG_P1 if period1 else _HARD_DECAY_DRUG_P0
        # A decay ratio at the steepest rate offered rounds a percent
        # threshold to 0% well before k=25 -- cap dynamically per the
        # ACTUAL drawn rate (a plausible percent, never "0%").
        k_target = rng.randint(3, _decay_k_max(100, _factor(rate_pct, growth=False), 5))

    factor = _factor(rate_pct, growth)
    pct_threshold, _margin = _draw_threshold(rng, 100, factor, k_target, places=0, min_value=None if growth else 5)

    k_correct = _first_crossing(100, factor, pct_threshold, growth)
    n_correct = k_correct + 1 if period1 else k_correct
    check(k_correct >= 2, "search resolved too quickly to be a genuine search")

    template = rng.choice(templates)
    text = template.format(V=V, r=rate_pct, pct=int(pct_threshold))

    step = Fraction(100) * rate_pct / 100
    d_minus = n_correct - 1
    d_plus = n_correct + 1
    d_linear_k = _first_crossing_linear(100, step, pct_threshold, growth)
    d_linear = d_linear_k + 1 if period1 else d_linear_k
    d_conv = k_correct if period1 else k_correct + 1
    raw = _dedupe_ints(n_correct, d_minus, d_plus, d_linear, d_conv)
    check_pairwise_distinct(n_correct, *raw, msg="hard-tier distractor collides")
    distractors = pick_distractors(str(n_correct), [str(v) for v in raw], n=3)

    return dict(
        text=text,
        answer=n_correct,
        answer_text=str(n_correct),
        answer_latex=str(n_correct),
        distractors=distractors,
        working=[
            f"The starting value cancels: solving (1 {'+' if growth else '-'} "
            f"{rate_pct}/100)^k against {int(pct_threshold)}% gives k = {k_correct}, "
            f"reported as {n_correct}."
        ],
    )


# =============================================================================
# exam: two-stage rate -- the learner decides which stage the crossing is in
# =============================================================================

_EXAM_POP_TEMPLATES = (
    "A population of {V} organisms now grows by {r1}% each year for the "
    "first {m} years. After that, a change in conditions means the growth "
    "rate becomes {r2}% each year. After how many whole years does the "
    "population first exceed {T}? Give your answer as a whole number of "
    "years.",
    "A colony of {V} insects, counted now, grows by {r1}% each week for the "
    "first {m} weeks, while conditions are favourable. After that, the "
    "growth rate changes to {r2}% each week. After how many whole weeks "
    "does the colony first exceed {T} insects?",
)
_EXAM_MONEY_TEMPLATES = (
    "An investment of ${V}, made now, grows at {r1}% p.a. for the first "
    "{m} years. After that, the rate changes to {r2}% p.a. for every year "
    "after. After how many whole years does the investment's value first "
    "exceed ${T}? Give your answer as a whole number of years.",
)


def _exam(rng):
    m = rng.randint(3, 6)
    money = rng.random() < 0.5
    # Dollar context -- capped 2%-12% p.a. throughout, both stages; a non-money population/colony
    # may use the wider pool.
    rate_pool = _GROWTH_MONEY_RATES if money else _POP_TWO_STAGE_RATES
    r1_pct, r2_pct = rng.sample(rate_pool, 2)
    r1, r2 = _factor(r1_pct, growth=True), _factor(r2_pct, growth=True)
    V = rng.randint(2, 20) * 500 if money else rng.randint(200, 5000)

    stage2 = rng.random() < 0.75
    if stage2:
        extra_target = rng.randint(3, 20)
        v1 = Fraction(V) * r1**m
        T, _margin = _draw_threshold(rng, v1, r2, extra_target, places=0, min_value=100)
    else:
        # k_target >= 2 (never 1): the trivial "already past the threshold
        # after a single period" case is not a genuine search (the
        # "k roughly 3..25" spirit, scaled down to fit within m periods).
        k_target = rng.randint(2, m)
        T, _margin = _draw_threshold(rng, V, r1, k_target, places=0, min_value=100)

    n_correct = _first_crossing_two_stage(V, r1, m, r2, T, growth=True)
    check(n_correct >= 2, "search resolved too quickly to be a genuine search")
    if n_correct <= m:
        margin = _log_frac(float(T), float(V), float(r1))
    else:
        v1_check = Fraction(V) * r1**m
        margin = _log_frac(float(T), float(v1_check), float(r2))
    check(0.05 <= margin <= 0.95, "exam-tier crossing sits on the log-space boundary")

    templates = _EXAM_MONEY_TEMPLATES if money else _EXAM_POP_TEMPLATES
    template = rng.choice(templates)
    text = template.format(V=V, r1=r1_pct, m=m, r2=r2_pct, T=int(T))

    d_minus = n_correct - 1
    d_plus = n_correct + 1
    d_r1_throughout = _first_crossing(V, r1, T, growth=True)
    d_r2_throughout = _first_crossing(V, r2, T, growth=True)
    raw = _dedupe_ints(n_correct, d_minus, d_plus, d_r1_throughout, d_r2_throughout)
    check_pairwise_distinct(n_correct, *raw, msg="exam-tier distractor collides")
    distractors = pick_distractors(str(n_correct), [str(v) for v in raw], n=3)

    return dict(
        text=text,
        answer=n_correct,
        answer_text=str(n_correct),
        answer_latex=str(n_correct),
        distractors=distractors,
        working=[
            f"For the first {m} periods the rate is {r1_pct}%; after that it is {r2_pct}%. "
            f"Iterating period by period, the value first exceeds {int(T)} at period "
            f"{n_correct}."
        ],
    )


_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}


@generator("pv0a_geometric_threshold", level=13, presets=_PRESETS)
def geometric_threshold(rng, variant="easy"):
    if variant == "easy":
        return _easy(rng)
    if variant == "medium":
        return _medium(rng)
    if variant == "hard":
        return _hard(rng)
    if variant == "exam":
        return _exam(rng)
    raise ValueError(f"unknown variant: {variant!r}")
