"""Generates `exd_x_r2b_total_change_chain` questions: a multi-part exam chain on total change from a rate.

Skill: senior-calculate-total-change-by-integrating-instantaneous-or-marginal (exam tier).

A quantity changes at the rate `r(t) = k t^p + c` (per unit time), where `c` is a stated baseline and the constant `k`
is not. The stem gives the total change over the first `a` units of time:

(a) find k from the total change over [0, a]                      fraction, 2 marks
(b) the total change over [a, b], to 2 decimal places             number,   2 marks, k (b^P - a^P) / P + c (b - a)
(c) the average rate of change over [a, b]                        number,   1 mark,  (b) / (b - a)
(d) how much the rate at t = b exceeds that average rate          number,   1 mark,  r(b) - (c)

## Construction

`k`, `p`, `c`, `a` and `b` are drawn, with `k a^(p+1)` a multiple of `p + 1` so that the stated total change is a whole
number. The total over [a, b] is computed from the antiderivative in exact fractions and again by SymPy's definite
integral; the keys are drawn only clear of a rounding tie and the routes through the rounded (b) and (c) must give
the same last digit.
"""

from __future__ import annotations

from fractions import Fraction

import sympy
from sympy import Rational, symbols

from ..core import check, generator, join_terms, phrase, round_half_up
from .exd_x_r2b_support import Redraw, clear_of_boundary, draw, fraction_part, number_part

t = symbols("t")
_CONTEXTS = (
    ("Water flows into a tank", "litres", "minute", "minutes"),
    ("Cars enter a car park", "cars", "hour", "hours"),
    ("A market stall earns money", "dollars", "day", "days"),
    ("A factory uses electricity", "kilowatt-hours", "hour", "hours"),
)
_STEMS = (
    "{subject} at the rate r(t) = {rate} {unit} per {singular}, where t is the time in {plural} from the start and "
    "k is a constant. Over the first {a} {plural}, the total is {total} {unit}.",
    "The rate at which {unit} accumulate is r(t) = {rate} {unit} per {singular}, where t is the time in {plural} "
    "from the start and k is a constant. The total over the first {a} {plural} is {total} {unit}.",
)


def _build(rng):
    p = rng.choice([1, 1, 2, 2, 3])
    big_p = p + 1
    k = rng.randint(1, 5)
    c = rng.randint(2, 10)
    a = rng.randint(2, 5)
    b = a + rng.randint(2, 3)
    if (k * a**big_p) % big_p:
        raise Redraw("the stated total would not be a whole number")
    total = Fraction(k * a**big_p, big_p) + c * a

    # ---- (b) the total over [a, b]: antiderivative in exact fractions, and SymPy's definite integral --------------
    amount = Fraction(k * (b**big_p - a**big_p), big_p) + c * (b - a)
    antiderivative = sympy.Poly(k * t**p + c, t).integrate().as_expr()
    direct = antiderivative.subs(t, b) - antiderivative.subs(t, a)
    check(direct == Rational(amount.numerator, amount.denominator), "total change disagrees with SymPy's integral")
    check(antiderivative.subs(t, a) - antiderivative.subs(t, 0) == total, "the stated total is not the integral")
    avg = amount / (b - a)
    rate_b = Fraction(k * b**p + c)
    gap = rate_b - avg
    for key in (amount, avg, gap):
        if not clear_of_boundary(key, 2):
            raise Redraw("a key too close to a rounding tie")
    amount_r, avg_r = round_half_up(amount, 2), round_half_up(avg, 2)
    if round_half_up(amount_r / (b - a), 2) != round_half_up(avg, 2):
        raise Redraw("rounding (b) first changes the last digit of (c)")
    if round_half_up(rate_b - avg_r, 2) != round_half_up(gap, 2):
        raise Redraw("rounding (c) first changes the last digit of (d)")

    subject, unit, singular, plural = rng.choice(_CONTEXTS)
    stem = phrase(
        rng,
        _STEMS,
        subject=subject,
        rate=join_terms([(1, f"kt^{p}" if p > 1 else "kt"), (c, "")]),
        unit=unit,
        singular=singular,
        plural=plural,
        a=a,
        total=int(total),
    )

    # (a) k
    a_wrong = [
        (Fraction(int(total) * big_p, a**big_p), "dropped-constant-term"),  # the baseline term forgotten
        (Fraction(int(total) - c * a, a**big_p), "integration-power-not-divided"),  # not divided by p + 1
        (Fraction(int(total) - c, a**p), "rate-for-total"),  # the total read as the rate at t = a
    ]
    part_a = fraction_part(
        "Find the value of k.",
        k,
        2,
        [
            f"Total = integral of ({join_terms([(k, f't^{p}' if p > 1 else 't'), (c, '')])}) from 0 to {a} "
            f"= {Fraction(k, big_p)} × {a}^{big_p} + {c * a} = {total}.",
            f"So k = {k}.",
        ],
        a_wrong,
        sensible_range=(0, None),
    )

    # (b) total over [a, b]
    b_wrong = [
        (Fraction(k * b**big_p, big_p) + c * b, "lower-limit-taken-as-zero"),
        (Fraction(k * (b**big_p - a**big_p)) + c * (b - a), "integration-power-not-divided"),
        (Fraction(k * (b**big_p - a**big_p), big_p), "dropped-constant-term"),  # the baseline c lost
        (Fraction(k * (b**p - a**p)), "difference-of-rates-for-total"),
    ]
    part_b = number_part(
        f"Find the total change, in {unit}, between t = {a} and t = {b}.",
        amount,
        2,
        2,
        [f"Total = [{Fraction(k, big_p)}t^{big_p} + {c}t] from {a} to {b} = {amount_r}."],
        b_wrong,
        follow_through=f"part_a*({b}**{big_p} - {a}**{big_p})/{big_p} + {c}*({b} - {a})",
        sensible_range=(0, None),
    )

    # (c) average rate
    c_wrong = [
        (amount, "total-not-divided-by-width"),
        (amount / b, "divided-by-endpoint-not-width"),
        (rate_b, "instantaneous-for-average"),
    ]
    part_c = number_part(
        f"Hence find the average rate of change, in {unit} per {singular}, between t = {a} and t = {b}.",
        avg,
        2,
        1,
        [f"Average rate = {amount_r} ÷ {b - a} = {avg_r}."],
        c_wrong,
        follow_through=f"part_b/({b} - {a})",
    )

    # (d) how much the rate at t = b exceeds the average
    d_wrong = [
        (-gap, "sign-reversed-result"),
        (rate_b, "difference-not-taken"),
        (avg, "difference-not-taken"),
        (Fraction(k * a**p + c) - avg, "evaluated-at-the-wrong-x"),
    ]
    part_d = number_part(
        f"Find how much the rate at t = {b} exceeds this average rate, in {unit} per {singular}.",
        gap,
        2,
        1,
        [f"r({b}) = {rate_b}, so the rate exceeds the average by {rate_b} - {avg_r} = {round_half_up(gap, 2)}."],
        d_wrong,
        follow_through=f"part_a*{b}**{p} + {c} - part_c",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2b_total_change_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_total_change_chain(rng, tier: str = "exam"):
    """A rate with an unknown constant: k from a total, the total over a later period, an average, the excess."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
