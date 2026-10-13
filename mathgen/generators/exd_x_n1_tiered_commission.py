"""Exam source: a two-rate commission scheme, worked forwards and then backwards.

One scenario: a salesperson earns one percentage on the first T dollars of monthly sales and a higher percentage
on every dollar above T.

(a) the commission on the first T dollars;
(b) the commission on the sales above T;
(c) the total commission, from (a) and (b);
(d) the monthly sales needed to earn a stated higher commission, from (a): the learner works backwards through the
    second tier only.

Rates are multiples of 0.5% and sales are multiples of $10, so every commission is an exact number of cents and
(d) is a whole number of dollars.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators.exd_x_n1_support import money, number_part, show

_STEM = (
    "A sales representative is paid commission of {r1}% on the first ${T} of sales made in a month, and {r2}% on "
    "all sales above ${T}. In one month the representative's sales total ${S}.",
    "A car yard pays its salespeople {r1}% commission on the first ${T} of monthly sales and {r2}% on the part "
    "of the month's sales above ${T}. One salesperson sells ${S} worth of cars in a month.",
    "A real-estate agency pays {r1}% on the first ${T} of a consultant's monthly sales and {r2}% on any sales "
    "over ${T}. A consultant's sales for the month are ${S}.",
)


def _pct(halves: int) -> str:
    return f"{halves // 2}" if halves % 2 == 0 else f"{halves / 2:.1f}"


@generator("exd_x_n1_tiered_commission", level=11, presets={"exam": dict(tier="exam")})
def exd_x_n1_tiered_commission(rng, tier="exam"):
    for _ in attempts(1000, "a commission draw with distinct tiers"):
        h1 = rng.randint(3, 8)  # first rate, in half-percents
        h2 = rng.randint(h1 + 2, 20)
        t = Fraction(rng.randrange(8000, 30001, 500))
        s = t + Fraction(rng.randrange(3000, 40001, 10))
        s2 = s + Fraction(rng.randrange(3000, 30001, 10))
        r1, r2 = Fraction(h1, 200), Fraction(h2, 200)
        if h1 == h2:
            continue
        break

    first = r1 * t
    above = r2 * (s - t)
    total = first + above
    target_commission = first + r2 * (s2 - t)
    check((first + above).denominator == 1 or (100 * (first + above)).denominator == 1, "commission is not exact cents")
    check(100 * target_commission == int(100 * target_commission), "target is not exact cents")
    check(t + (target_commission - first) / r2 == s2, "backwards route disagrees")

    stem = phrase(rng, _STEM, r1=_pct(h1), r2=_pct(h2), T=money(t, 0), S=money(s, 0))
    f = Fraction
    part_a = number_part(
        f"Find the commission on the first ${money(t, 0)} of sales, in dollars.",
        first,
        2,
        1,
        [f"{_pct(h1)}% of {money(t, 0)} = ${show(first, 2)}."],
        [(r2 * t, None), (r1 * s, None), (r1 * t * 10, None)],
        sensible_range=(0, None),
    )
    part_b = number_part(
        "Find the commission on the sales above this amount, in dollars.",
        above,
        2,
        2,
        [
            f"Sales above ${money(t, 0)}: {money(s, 0)} - {money(t, 0)} = {money(s - t, 0)}.",
            f"{_pct(h2)}% of {money(s - t, 0)} = ${show(above, 2)}.",
        ],
        [(r2 * s, None), (r1 * (s - t), None), (r2 * t, None)],
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Using your answers to (a) and (b), find the total commission, in dollars.",
        total,
        2,
        1,
        [f"Total = {show(first, 2)} + {show(above, 2)} = ${show(total, 2)}."],
        [(r2 * s, None), (r1 * s, None), ((r1 + r2) / 2 * s, None), (above, None)],
        follow_through="part_a + part_b",
        sensible_range=(0, None),
    )
    part_d = number_part(
        f"In the following month the commission earned is ${money(target_commission, 2)}. Using your answer to (a), find the "
        "sales for that month, in dollars.",
        s2,
        0,
        3,
        [
            f"Commission above the first tier = {money(target_commission, 2)} - {show(first, 2)} = ${show(target_commission - first, 2)}.",
            f"Sales above ${money(t, 0)} = {show(target_commission - first, 2)} ÷ {_pct(h2)}% = {money(s2 - t, 0)}.",
            f"Sales = {money(t, 0)} + {money(s2 - t, 0)} = ${money(s2, 0)}.",
        ],
        [
            ((target_commission - first) / r2, None),
            (target_commission / r2, None),
            (t + (target_commission - first) / r1, None),
            (target_commission / r2 + t, None),
        ],
        follow_through=f"{int(t)} + ({target_commission.numerator}/{target_commission.denominator} - part_a)/({h2}/200)",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
