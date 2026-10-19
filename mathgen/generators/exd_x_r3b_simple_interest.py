"""Generates `exd_x_r3b_simple_interest`: a multi-part exam question on a simple-interest investment held for months.

The stem gives a principal, an annual rate and a term in months that is not a whole number of years, so the
time has to be converted. The chain is the one a general-mathematics paper sets:

(a) the interest earned over the term, in dollars         number, 2 marks
(b) the total value of the investment at the end          number, 1 mark,  P + part_a
(c) the annual rate needed to earn that interest in a shorter term   number, 2 marks, part_a * 1200 / (P * m2)

The principal is a multiple of $100 and the rate a multiple of 0.5%, and a draw is kept only when the interest is a
whole number of cents, so (a) and (b) are exact. The rate in (c) is rounded once, to 2 decimal places, and the
draw is kept only when that rounding is not close to a boundary.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase, round_half_up
from .exd_x_r3b_common import number_part, number_text
from ._exd_distinct_parts import distinct_part_answers

_NAMES = ("Chloe", "Ava", "Mia", "Liam", "Noah", "Isla", "Ethan", "Grace", "Lucas", "Ruby", "Jack", "Olivia")
_PRODUCTS = ("a term deposit", "a fixed-term savings account", "a term investment")
_STEMS = (
    "{name} invests ${p} in {product} that pays {rate}% p.a. simple interest. The money stays invested for {m} months.",
    "{name} puts ${p} into {product} paying simple interest at {rate}% p.a. for a term of {m} months.",
    "For {m} months, {name}'s ${p} earns simple interest at a rate of {rate}% p.a. in {product}.",
)
_MONTHS = (6, 9, 15, 18, 21, 30, 42)
_SHORTER = (4, 5, 6, 8, 9, 10, 12)
_MARGIN = Fraction(1, 5)  # keep the unrounded rate this far (in hundredths) from a rounding boundary


def _money(value: Fraction) -> str:
    return number_text(round_half_up(value, 2), 2)


def _rate_text(r2: int) -> str:
    return str(r2 // 2) if r2 % 2 == 0 else f"{r2 // 2}.5"


def _build(rng: random.Random) -> dict:
    for _ in attempts(2000, "an investment with whole-cent interest and a stable rate"):
        p = rng.randrange(20, 201) * 100
        r2 = rng.randrange(5, 15)  # the rate in half-percents: 2.5% .. 7%
        m = rng.choice(_MONTHS)
        m2 = rng.choice([v for v in _SHORTER if v < m])
        interest = Fraction(p * r2 * m, 2400)
        if (interest * 100).denominator != 1:
            continue
        rate2 = Fraction(r2 * m, 2 * m2)  # per cent per annum
        hundredths = rate2 * 100
        offset = hundredths - (hundredths.numerator // hundredths.denominator)
        if abs(offset - Fraction(1, 2)) < _MARGIN or rate2 > 10:
            continue
        break
    rate = Fraction(r2, 2)
    check(interest == p * rate / 100 * Fraction(m, 12), "interest disagrees with the I = PRT route")
    check(p * (1 + rate / 100 * Fraction(m, 12)) == p + interest, "total value disagrees")
    check(interest * 1200 / (p * m2) == rate2, "rate for the shorter term disagrees")

    total = p + interest
    name = rng.choice(_NAMES)
    stem = phrase(rng, _STEMS, name=name, p=f"{p:,}", rate=_rate_text(r2), m=m, product=rng.choice(_PRODUCTS))
    one_year = Fraction(p) * rate / 100
    growth = (1 + float(rate) / 100) ** (m / 12)
    compound = Fraction(round(p * (growth - 1) * 100), 100)
    int_fmt = _money(interest)
    pt = f"{p:,}"
    int_c = f"{int(round_half_up(interest, 2)):,}" + _money(interest)[len(str(int(round_half_up(interest, 2)))) :]
    total_c = f"{int(round_half_up(total, 2)):,}" + _money(total)[len(str(int(round_half_up(total, 2)))) :]
    part_a = number_part(
        "Find the interest earned over the term, in dollars, correct to 2 decimal places.",
        interest,
        2,
        2,
        [f"{m} months = {m}/12 years.", f"I = PRT = {pt} × {_rate_text(r2)}/100 × {m}/12 = ${int_c}."],
        [
            ("months-used-as-years", Fraction(p) * rate / 100 * m),
            ("omitted-time-in-interest", one_year),
            ("used-compound-instead-of-simple", compound),
            ("percentage-not-converted-to-decimal", Fraction(p) * rate * Fraction(m, 12)),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        "Find the total value of the investment at the end of the term, in dollars, correct to 2 decimal places.",
        total,
        2,
        1,
        [f"Total value = ${pt} + ${int_c} = ${total_c}."],
        [
            ("reported-interest-not-total-value", interest),
            ("subtracted-interest-from-principal", p - interest),
            ("months-used-as-years", p + Fraction(p) * rate / 100 * m),
            ("omitted-time-in-interest", p + one_year),
        ],
        follow_through=f"{p} + part_a",
        sensible_range=(0, None),
    )
    part_c = number_part(
        f"Find the annual simple-interest rate that would earn the same interest on ${pt} in {m2} months "
        "instead, as a percentage correct to 2 decimal places.",
        rate2,
        2,
        2,
        [
            f"{m2} months = {m2}/12 years.",
            f"r = I / (P × T) = {int_fmt} / ({pt} × {m2}/12) = {_money(round_half_up(rate2, 2))}%.",
        ],
        [
            ("time-ratio-inverted", Fraction(r2, 2) * Fraction(m2, m)),
            ("kept-original-rate", Fraction(r2, 2)),
            ("months-used-as-years", interest * 100 / (p * m2)),
            ("forgot-percent-conversion", interest * 12 / (p * m2)),
        ],
        follow_through=f"part_a*1200/({p}*{m2})",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])


@generator("exd_x_r3b_simple_interest", level=10, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_simple_interest(rng: random.Random, tier: str = "exam") -> dict:
    """Simple interest over a term in months: the interest, the total value, then the rate for a shorter term."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
