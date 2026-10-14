"""Generates `exd_x_r3b_arith_seq`: a multi-part exam question on an arithmetic sequence read from two terms.

The stem gives two non-adjacent terms of a quantity that changes by a constant amount each period (loaves baked,
litres pumped, minutes run, visitors, dollars saved; none of them interest). The chain is the one a methods or
general-mathematics paper sets:

(a) the common difference d                                     number, 1 mark
(b) the first term                                              number, 2 marks, t_p - (p - 1) part_a
(c) the total over the first N periods                          number, 2 marks, N (2 part_b + (N - 1) part_a) / 2
(d) the first period in which the quantity passes a threshold   number, 2 marks, floor((T - part_b) / part_a) + 2

The sequence is drawn first (first term, difference, the two positions) and the stated terms read off it, so the
difference is a whole number. T lies strictly between two consecutive terms, so the crossing period is unique.
The independent route builds the whole sequence term by term with an accumulation loop and sums and searches it.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase, rational_text
from .exd_x_r3b_common import number_part
from ._exd_distinct_parts import distinct_part_answers

# stem uses {A}, {B} (already with unit text), {p}, {q}; sign is the direction of change.
_CONTEXTS = (
    dict(
        stem="A bakery bakes {A} loaves on day {p} of the month and {B} loaves on day {q}. The number of loaves "
        "baked changes by the same amount each day.",
        first="Find the number of loaves baked on day 1 of the month.",
        total="Find the total number of loaves baked on the first {N} days of the month.",
        threshold="On which day of the month does the bakery first bake {cmp} {T} loaves?",
        sign=1,
        d=(6, 22),
        t1=(180, 420),
        money=False,
    ),
    dict(
        stem="A pump removes {A} litres of water from a tank in hour {p} and {B} litres in hour {q}. The amount "
        "removed changes by the same amount each hour.",
        first="Find the number of litres removed in hour 1.",
        total="Find the total number of litres removed during the first {N} hours.",
        threshold="In which hour does the pump first remove {cmp} {T} litres?",
        sign=-1,
        d=(8, 30),
        t1=(900, 1800),
        money=False,
    ),
    dict(
        stem="In a training plan, Mia runs for a total of {A} minutes in week {p} and {B} minutes in week {q}. The time "
        "she runs each week changes by the same amount from one week to the next.",
        first="Find the number of minutes Mia runs in week 1.",
        total="Find the total number of minutes Mia runs during the first {N} weeks.",
        threshold="In which week does Mia first run for {cmp} {T} minutes?",
        sign=1,
        d=(3, 8),
        t1=(60, 140),
        money=False,
    ),
    dict(
        stem="A school holiday program has {A} visitors on day {p} and {B} visitors on day {q}. The number of "
        "visitors changes by the same amount each day.",
        first="Find the number of visitors on day 1 of the program.",
        total="Find the total number of visitors on the first {N} days of the program.",
        threshold="On which day does the program first have {cmp} {T} visitors?",
        sign=-1,
        d=(6, 18),
        t1=(400, 800),
        money=False,
    ),
    dict(
        stem="Ethan saves ${A} in week {p} of the year and ${B} in week {q}. The amount he saves changes by the "
        "same amount each week.",
        first="Find the amount, in dollars, that Ethan saves in week 1.",
        total="Find the total amount, in dollars, that Ethan saves during the first {N} weeks of the year.",
        threshold="In which week does Ethan first save {cmp} ${T}?",
        sign=1,
        d=(3, 12),
        t1=(40, 120),
        money=True,
    ),
    dict(
        stem="A school fete sells {A} raffle tickets on day {p} of the ticket drive and {B} on day {q}. The number "
        "sold changes by the same amount each day.",
        first="Find the number of raffle tickets sold on day 1 of the drive.",
        total="Find the total number of raffle tickets sold on the first {N} days of the drive.",
        threshold="On which day of the ticket drive are {cmp} {T} tickets first sold?",
        sign=1,
        d=(5, 15),
        t1=(100, 260),
        money=False,
    ),
)


def _term(t1: int, d: int, k: int) -> int:
    return t1 + (k - 1) * d


def _build(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(2000, "a sequence with a unique crossing period"):
        d = ctx["sign"] * rng.randint(*ctx["d"])
        t1 = rng.randint(*ctx["t1"])
        p = rng.randint(2, 4)
        q = p + rng.randint(3, 6)
        n_total = rng.randint(8, 14)
        n_th = n_total + rng.randint(2, 12)
        low, high = sorted((_term(t1, d, n_th - 1), _term(t1, d, n_th)))
        if min(_term(t1, d, k) for k in (1, n_th, q, n_total)) <= abs(d):
            continue
        inside = [v for v in range(low + 1, high)]
        if not inside:
            continue
        round_ones = [v for v in inside if v % 10 == 0 or v % 5 == 0] or inside
        threshold = rng.choice(round_ones)
        break
    a_val, b_val = _term(t1, d, p), _term(t1, d, q)

    # Independent route: build the sequence term by term and search and sum it.
    seq, value = [], t1
    for _ in range(n_th + 2):
        seq.append(value)
        value += d
    check(seq[p - 1] == a_val and seq[q - 1] == b_val, "stated terms disagree with the sequence")
    check((b_val - a_val) % (q - p) == 0 and (b_val - a_val) // (q - p) == d, "difference is not a whole number")
    total = sum(seq[:n_total])
    check(total == n_total * (2 * t1 + (n_total - 1) * d) // 2, "sum formula disagrees with the running total")
    crossing = next(k + 1 for k, v in enumerate(seq) if (v > threshold if d > 0 else v < threshold))
    check(crossing == n_th, "crossing period disagrees")

    cmp_word = "more than" if d > 0 else "fewer than"
    stem = phrase(rng, (ctx["stem"],), A=a_val, B=b_val, p=p, q=q)
    t_n = _term(t1, d, n_total)
    gap = q - p
    part_a = number_part(
        "Find the common difference of the arithmetic sequence formed by these values.",
        Fraction(d),
        0,
        1,
        [f"d = ({b_val} - {a_val}) / ({q} - {p}) = {d}."],
        [
            ("forgot-to-divide-by-gap", Fraction(b_val - a_val)),
            ("counted-terms-inclusively-for-gap", Fraction(b_val - a_val, gap + 1)),
            ("sign-error-common-difference", Fraction(-d)),
            ("divided-by-position-sum", Fraction(b_val - a_val, p + q)),
        ],
    )
    part_b = number_part(
        ctx["first"],
        Fraction(t1),
        0,
        2,
        [f"First value = {a_val} - ({p} - 1) × ({d}) = {t1}."],
        [
            ("stated-term-taken-as-first-term", Fraction(a_val)),
            ("off-by-one-nth-term", Fraction(a_val - p * d)),
            ("wrong-term-paired-with-position", Fraction(b_val - (p - 1) * d)),
            ("sign-error-common-difference", Fraction(a_val + (p - 1) * d)),
        ],
        follow_through=f"{a_val} - {p - 1}*part_a",
        sensible_range=(0, None),
    )
    part_c = number_part(
        ctx["total"].format(N=n_total),
        Fraction(total),
        0,
        2,
        [
            f"Sum = {n_total}/2 × (2 × {t1} + {n_total - 1} × ({d})) = {total}.",
        ],
        [
            ("sum-formula-not-halved", Fraction(n_total * (2 * t1 + (n_total - 1) * d))),
            ("total-as-count-times-last-term", Fraction(n_total * t_n)),
            ("reported-term-not-sum", Fraction(t_n)),
            ("off-by-one-in-sum", Fraction(n_total * (2 * t1 + n_total * d), 2)),
        ],
        follow_through=f"{n_total}*(2*part_b + {n_total - 1}*part_a)/2",
        sensible_range=(0, None),
    )
    part_d = number_part(
        ctx["threshold"].format(cmp=cmp_word, T=threshold),
        Fraction(n_th),
        0,
        2,
        [
            f"{t1} + (n - 1) × ({d}) {'>' if d > 0 else '<'} {threshold}, so n - 1 > {rational_text(Fraction(threshold - t1, d))}.",
            f"The first whole number is n = {n_th}.",
        ],
        [
            ("first-index-one-early", Fraction(n_th - 1)),
            ("first-index-one-late", Fraction(n_th + 1)),
            ("reported-the-value-not-the-period", Fraction(seq[n_th - 1])),
        ],
        follow_through=f"floor(({threshold} - part_b)/part_a) + 2",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r3b_arith_seq", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_arith_seq(rng: random.Random, tier: str = "exam") -> dict:
    """An arithmetic sequence from two terms: its difference, its first term, a total, and a threshold period."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
