"""Exam tier for "use first-order linear recurrence relations to model" a practical situation.

The practice tiers give the recurrence u(n+1) = a u(n) + b and ask for the value it settles to. Here only three
recorded values are given and the learner has to recover the recurrence, find its limit and then judge how fast it
is approached, which is how a numerical or graphical recurrence investigation is set in a senior exam:

(a) the retained fraction a = (u2 - u1) / (u1 - u0);
(b) the added amount b = u1 - a u0, from (a);
(c) the long-term value b / (1 - a), from (a) and (b);
(d) the first period n for which u(n) is within 1 unit of the long-term value, from (a) and (c).

Built backwards from a whole-number limit L and a one-decimal retained fraction a = k / 10, with u0 = L +/- 10 j so
that u1 = L + a (u0 - L) and u2 = L + a^2 (u0 - L) are exact decimals. (d) is found by iterating the distance to the
limit; draws where the distance lands exactly on 1 are rejected so "within" is never a tie.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator, rational_text, stated_precision_string
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3g_kit import clashes, number_part

_SITUATIONS = (
    ("mass of chlorine in a swimming pool", "g", "day"),
    ("mass of a pollutant in a lake", "kg", "week"),
    ("amount of medicine in a patient's bloodstream", "mg", "hour"),
    ("volume of water in a farm dam", "kL", "month"),
)
_STEMS = (
    "The {thing} follows a recurrence of the form u(n+1) = a u(n) + b for constants a and b, where u(n) is the "
    "{thing} in {unit} at the end of {period} n. The recorded values are u(0) = {u0}, u(1) = {u1} and u(2) = {u2}.",
    "A model of the {thing} uses u(n+1) = a u(n) + b, with u(n) measured in {unit} at the end of {period} n. "
    "Three consecutive records are u(0) = {u0}, u(1) = {u1} and u(2) = {u2}.",
    "Records of the {thing} (in {unit}) at the end of {period}s 0, 1 and 2 are {u0}, {u1} and {u2}. "
    "They satisfy u(n+1) = a u(n) + b for fixed constants a and b.",
)


def _dec(value: Fraction, places: int) -> str:
    return stated_precision_string(value, places)


@generator("exd_x_r3g_recurrence_fit", level=11, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3g_recurrence_fit(rng: random.Random, tier: str = "exam") -> dict:
    """From three recorded values: a, b, the long-term value and the time to come within 1 unit of it."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a recurrence with a clear approach to its limit"):
        k = rng.randint(2, 9)
        a = Fraction(k, 10)
        limit = rng.randint(4, 60) * 5
        j = rng.randint(1, 12)
        gap = rng.choice((-1, 1)) * 10 * j  # u0 - limit
        u0 = limit + gap
        if u0 <= 0 or (a**3 * abs(gap)).denominator == 0:
            continue
        u1, u2 = limit + a * gap, limit + a * a * gap
        b = limit * (1 - a)
        n, distance = 0, Fraction(abs(gap))
        while distance >= 1:
            distance *= a
            n += 1
            check(n < 200, "the recurrence never comes within 1 unit")
        if distance == 0 or n < 3 or n > 40 or any(a**m * abs(gap) == 1 for m in range(1, n + 1)):
            continue
        thing, unit, period = rng.choice(_SITUATIONS)
        stem = rng.choice(_STEMS).format(
            thing=thing, unit=unit, period=period, u0=_dec(Fraction(u0), 1), u1=_dec(u1, 1), u2=_dec(u2, 1)
        )
        pd = f"Find the first {period} n for which u(n) is within 1 {unit} of this long-term value."
        keys = (stated_precision_string(a, 1), stated_precision_string(b, 1), limit, n)
        if clashes(keys, stem, pd):
            continue
        break

    check(u1 == a * u0 + b and u2 == a * u1 + b, "the records do not satisfy the recurrence")
    check(limit == b / (1 - a), "the limit is not b / (1 - a)")
    inv_ratio = 1 / a
    parts = [
        number_part(
            "Find the constant a, correct to 1 decimal place.",
            a,
            1,
            2,
            [
                f"a = (u(2) - u(1)) / (u(1) - u(0)) = ({_dec(u2, 1)} - {_dec(u1, 1)}) / ({_dec(u1, 1)} - {_dec(Fraction(u0), 1)}) = {a}."
            ],
            [
                (u1 / u0, "recurrence-ratio-of-consecutive-values"),
                (inv_ratio, "rate-and-dose-swapped"),
                (1 - a, "recurrence-retained-and-lost-swapped"),
                (u2 / u1, "recurrence-ratio-of-consecutive-values"),
            ],
            sensible_range=(0, 1),
        ),
        number_part(
            "Find the constant b, correct to 1 decimal place.",
            b,
            1,
            2,
            [f"b = u(1) - a u(0) = {_dec(u1, 1)} - ({a})({_dec(Fraction(u0), 1)}) = {_dec(b, 1)}."],
            [
                (u1 - u0, "simple-difference-for-added-amount"),
                (u1 - a * u1, "recurrence-added-amount-from-wrong-term"),
                (u2 - u1, "simple-difference-for-added-amount"),
                (u0 * (1 - a), "recurrence-added-amount-from-lost-fraction"),
            ],
            follow_through=f"{_dec(u1, 1)} - part_a*{u0}",
        ),
        number_part(
            "Find the long-term value that u(n) approaches, correct to the nearest whole number.",
            limit,
            0,
            2,
            [f"L = b / (1 - a) = {_dec(b, 1)} / {1 - a} = {limit}."],
            [
                (b / a, "divides-by-retained-fraction"),
                (b, "reports-added-amount"),
                (u2, "mistakes-recorded-value-for-longterm"),
                (b / (1 + a), "sign-slip-in-formula"),
            ],
            follow_through="part_b/(1 - part_a)",
            sensible_range=(0, None),
        ),
        number_part(
            pd,
            n,
            0,
            3,
            [
                f"The distance from the limit shrinks by a factor of {a} each {period}: it starts at {abs(gap)}.",
                f"{abs(gap)} x {a}^{n} = {float(abs(gap) * a**n):.3g} < 1 but {abs(gap)} x {a}^{n - 1} = "
                f"{float(abs(gap) * a ** (n - 1)):.3g} > 1, so n = {n}.",
            ],
            [
                (n - 1, "off-by-one-count"),
                (n + 1, "off-by-one-count"),
                (
                    math.ceil(math.log(1 / u0) / math.log(float(a))) if u0 > 1 else None,
                    "recurrence-time-from-the-start-not-the-gap",
                ),
                (2 * n, "dropped-factor-of-two"),
            ],
            follow_through=f"ceiling(log(1/abs({u0} - part_c))/log(part_a))",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
