"""Exam source for `senior-use-arithmetic-sequences-to-model-and-analyse-practical`: a four-part chain on one
arithmetic model.

A quantity that changes by the same amount each period is known at two periods. Then:

- (a) the common difference (1 mark);
- (b) the first term (1 mark, ``follow_through`` from (a));
- (c) the total over the first ``N`` periods, the sum of an arithmetic series (2 marks, ``follow_through`` from
  (a) and (b));
- (d) the first period in which the quantity passes a stated threshold (3 marks, ``follow_through`` from (a)
  and (b)).

## Construction

The common difference and first term are drawn first; the two given terms follow from them. The threshold is
placed strictly between two consecutive terms, so the first period beyond it is unique and a reading of "exactly
reaches" cannot arise. The first period is found by search over the terms, then confirmed against the closed form
the follow-through expression uses. The quantities are per-period amounts (visitors a day, litres a day), so
adding the first ``N`` of them is meaningful.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, pick_distractors


@dataclass(frozen=True)
class _Context:
    rising: bool
    period: str
    stem: str  # slots: vp, vq, p, q
    first: str
    total: str  # slot: N
    threshold: str  # slot: L
    a_range: tuple[int, int, int]  # lo, hi, step of the first term
    d_range: tuple[int, int]  # lo, hi of |common difference|
    n_range: tuple[int, int] = (8, 16)  # periods in the series total
    extra_range: tuple[int, int] = (3, 25)  # periods from the end of the series to the threshold


_CONTEXTS = [
    _Context(
        True,
        "day",
        "The number of visitors to a museum exhibition is {vp} on day {p} and {vq} on day {q}. The number of "
        "visitors changes by the same amount each day.",
        "Find the number of visitors on day 1.",
        "Find the total number of visitors over the first {N} days.",
        "Find the first day on which more than {L} visitors attend.",
        (150, 400, 10),
        (10, 45),
    ),
    _Context(
        True,
        "week",
        "A runner's training distance for a fun run is {vp} km in week {p} and {vq} km in week {q}. The distance "
        "changes by the same number of kilometres each week.",
        "Find the training distance in week 1, in kilometres.",
        "Find the total distance, in kilometres, that the runner trains over the first {N} weeks.",
        "Find the first week in which the runner trains more than {L} km.",
        (8, 14, 1),
        (2, 3),
        (8, 10),
        (3, 5),
    ),
    _Context(
        True,
        "month",
        "A student saves ${vp} in month {p} and ${vq} in month {q}. The amount saved changes by the same number "
        "of dollars each month.",
        "Find the amount saved in month 1, in dollars.",
        "Find the total amount saved over the first {N} months, in dollars.",
        "Find the first month in which the student saves more than ${L}.",
        (100, 300, 10),
        (5, 15),
        (8, 14),
        (3, 10),
    ),
    _Context(
        False,
        "day",
        "A household's water use during restrictions is {vp} litres on day {p} and {vq} litres on day {q}. The "
        "daily use changes by the same number of litres each day.",
        "Find the household's water use on day 1, in litres.",
        "Find the total water used over the first {N} days, in litres.",
        "Find the first day on which the household uses fewer than {L} litres.",
        (700, 1200, 10),
        (8, 30),
    ),
    _Context(
        False,
        "week",
        "Weekly sales of a magazine are {vp} copies in week {p} and {vq} copies in week {q}. Sales change by the "
        "same number of copies each week.",
        "Find the number of copies sold in week 1.",
        "Find the total number of copies sold over the first {N} weeks.",
        "Find the first week in which fewer than {L} copies are sold.",
        (2500, 4000, 50),
        (40, 120),
    ),
]


def _tag(value: int, tag: str) -> Distractor:
    return Distractor(str(value), tag, value)


def _first_period_past(a: int, d: int, rising: bool, limit: int) -> int:
    """First period n with the term beyond `limit` (above if rising, below if falling), by search."""
    for n in range(1, 500):
        term = a + (n - 1) * d
        if (term > limit) if rising else (term < limit):
            return n
    raise AssertionError("the sequence never passes the threshold")


@generator("exd_x_r2e_arith_seq_chain", level=12, presets={"exam": dict(tier="exam")})
def arith_seq_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Common difference, first term, series total and the first period past a threshold."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(4000, "an arithmetic chain with a clear threshold"):
        lo, hi, step = ctx.a_range
        a = rng.randrange(lo, hi + 1, step)
        d_abs = rng.randint(*ctx.d_range)
        d = d_abs if ctx.rising else -d_abs
        p = rng.randint(2, 4)
        q = p + rng.randint(2, 5)
        n_terms = rng.randint(*ctx.n_range)
        m = n_terms + rng.randint(*ctx.extra_range)  # the first period past the threshold
        if d_abs < 2:
            continue
        before, at = a + (m - 2) * d, a + (m - 1) * d
        limit = min(before, at) + rng.randint(1, d_abs - 1)  # strictly between the terms before and at period m
        if limit <= a * 0.25 or at <= 0:
            continue
        if _first_period_past(a, d, ctx.rising, limit) != m:
            continue
        break

    vp, vq = a + (p - 1) * d, a + (q - 1) * d
    total = n_terms * (2 * a + (n_terms - 1) * d) // 2
    check(total == sum(a + k * d for k in range(n_terms)), "series total disagrees with the sum of terms")
    check(math.floor((limit - a) / d) + 2 == m, "closed form for the first period disagrees with the search")

    gap = q - p
    last = a + (n_terms - 1) * d
    a_cands = [
        _tag(-d, "sign-error-common-difference"),
        _tag(round((vq - vp) / (gap + 1)), "counts-terms-not-gaps") if (vq - vp) % (gap + 1) == 0 else None,
        _tag(vq - vp, "forgot-to-divide-by-gap"),
        _tag(round((vq - vp) / q), "counts-terms-not-gaps") if (vq - vp) % q == 0 else None,
    ]
    b_cands = [
        _tag(vp - p * d, "uses-n-not-n-minus-one"),
        _tag(vp + (p - 1) * d, "sign-error-common-difference"),
        _tag(vp, "takes-given-term-as-first"),
        _tag(vq - q * d, "uses-n-not-n-minus-one"),
    ]
    c_cands = [
        _tag(n_terms * (a + last), "forgot-half-in-series-sum"),
        _tag(n_terms * last, "series-total-as-n-times-last-term"),
        _tag((n_terms + 1) * (2 * a + n_terms * d) // 2, "off-by-one-count"),
        _tag(n_terms * (2 * a + n_terms * d) // 2, "uses-n-not-n-minus-one"),
    ]
    d_cands = [
        _tag(m - 1, "rounds-down-when-must-exceed"),
        _tag(m + 1, "adds-a-year-after-rounding-up"),
        _tag(math.floor((limit - vp) / d) + 2, "takes-given-term-as-first"),
    ]
    picks = []
    for key, cands in ((d, a_cands), (a, b_cands), (total, c_cands), (m, d_cands)):
        picks.append(pick_distractors(str(key), [c for c in cands if c is not None], 3))
    check(all(len(x) >= 2 for x in picks), "a part has too few distractors")

    return dict(
        text=ctx.stem.format(vp=vp, vq=vq, p=p, q=q),
        parts=[
            dict(
                prompt="Find the common difference of the sequence.",
                answer_kind="number",
                answer=d,
                answer_text=str(d),
                answer_latex=latex_answer(str(d)),
                marks=1,
                working=[f"d = ({vq} - {vp}) / ({q} - {p}) = {d}."],
                distractors=picks[0],
            ),
            dict(
                prompt=ctx.first,
                answer_kind="number",
                answer=a,
                answer_text=str(a),
                answer_latex=latex_answer(str(a)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{vp} - {p - 1}*part_a",
                working=[f"First term = {vp} - {p - 1} x ({d}) = {a}."],
                distractors=picks[1],
            ),
            dict(
                prompt=ctx.total.format(N=n_terms),
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=latex_answer(str(total)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{n_terms}*(2*part_b + {n_terms - 1}*part_a)/2",
                working=[f"S = {n_terms}/2 x (2 x {a} + {n_terms - 1} x ({d})) = {total}."],
                distractors=picks[2],
            ),
            dict(
                prompt=ctx.threshold.format(L=limit),
                answer_kind="number",
                answer=m,
                answer_text=str(m),
                answer_latex=latex_answer(str(m)),
                marks=3,
                sensible_range=(0, None),
                follow_through=f"floor(({limit} - part_b)/part_a) + 2",
                working=[
                    f"Term n = {a} + (n - 1)({d}) passes {limit} when (n - 1) > ({limit} - {a}) / ({d}).",
                    f"So n = {m}.",
                ],
                distractors=picks[3],
            ),
        ],
    )
