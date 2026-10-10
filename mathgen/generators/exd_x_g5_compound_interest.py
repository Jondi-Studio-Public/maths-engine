"""Exam tier for compound interest: three problems the single-formula practice questions do not reach.

  phases   -- the rate changes part-way and a further deposit is made at the changeover, so the balance is
              built in two compounding stages (the deposit itself then earns the later rate).
  rate     -- the starting amount, the number of years and the final balance are given; find the annual
              rate as a whole percent (the learner must undo the power).
  periods  -- a nominal annual rate compounded half-yearly or quarterly, so both the rate per period and
              the number of periods change.

Every balance is computed in exact `Fraction` arithmetic with no intermediate rounding and rounded half-up to
the cent once, at the end. The rate question is built from a whole-percent rate, so its answer is unique.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, round_half_up
from .exd_x_g5_common import number_fields, usd

_NAMES = ("Ava", "Ben", "Chloe", "Dev", "Elena", "Femi")


def _cents(value: Fraction) -> Fraction:
    return round_half_up(value, 2)


def _clear_of_tie(value: Fraction) -> bool:
    """True unless `value` sits within a tenth of a cent of a rounding boundary."""
    scaled = value * 100
    frac = scaled - (scaled.numerator // scaled.denominator)
    return abs(frac - Fraction(1, 2)) > Fraction(1, 10)


def _phases(rng: random.Random) -> dict:
    while True:
        principal = rng.randrange(2000, 12001, 500)
        deposit = rng.randrange(500, 3001, 500)
        r1, r2 = rng.sample(range(3, 10), 2)
        n1, n2 = rng.randint(2, 4), rng.randint(2, 3)
        mid = Fraction(principal) * Fraction(100 + r1, 100) ** n1
        answer = (mid + deposit) * Fraction(100 + r2, 100) ** n2
        if _clear_of_tie(answer):
            break
    name = rng.choice(_NAMES)
    text = (
        f"{name} invests {usd(principal)} at {r1}% p.a., compounded annually. After {n1} years the rate changes "
        f"to {r2}% p.a. (still compounded annually) and {name} adds a further {usd(deposit)} at that moment. "
        f"Find the balance {n2} years after the change, to the nearest cent. Do not round until the end."
    )
    total = _cents(answer)
    p = Fraction(principal)
    wrong = [
        mid * Fraction(100 + r2, 100) ** n2,  # the extra deposit left out
        mid * Fraction(100 + r2, 100) ** n2 + deposit,  # deposit added at the end, earning nothing
        (p + deposit) * Fraction(100 + r1, 100) ** (n1 + n2),  # first rate kept throughout, deposit at the start
        (p * (1 + Fraction(r1 * n1, 100)) + deposit) * (1 + Fraction(r2 * n2, 100)),  # simple interest each stage
    ]
    working = [
        f"After {n1} years: {principal} x {1 + Fraction(r1, 100):g}^{n1} = {float(mid):.2f}.",
        f"Add the deposit: {float(mid):.2f} + {deposit} = {float(mid + deposit):.2f}.",
        f"Then {float(mid + deposit):.2f} x {1 + Fraction(r2, 100):g}^{n2} = {usd(total)}.",
    ]
    return number_fields(text, total, 2, wrong, working)


def _rate(rng: random.Random) -> dict:
    principal = rng.randrange(1000, 9001, 500)
    rate = rng.randint(3, 12)
    years = rng.randint(3, 6)
    final = _cents(Fraction(principal) * Fraction(100 + rate, 100) ** years)
    name = rng.choice(_NAMES)
    text = (
        f"{name} puts {usd(principal)} into an account that pays a fixed annual rate, compounded annually. "
        f"After {years} years the balance is {usd(final)}. What is the annual interest rate, as a whole-number "
        f"percentage?"
    )
    # The rate is unique: the balance is strictly increasing in the rate, and a one-point change moves it by
    # far more than the cent of rounding.
    check(_cents(Fraction(principal) * Fraction(100 + rate, 100) ** years) == final, "rate does not reproduce")
    ratio = final / principal

    # Wrong period counts: the root is irrational in general, so it is evaluated in floating point and only
    # ever used as a (rounded) wrong value.
    def root_rate(periods: int) -> Fraction:
        return Fraction((float(ratio) ** (1 / periods) - 1) * 100).limit_denominator(10**6)

    wrong = [
        (ratio - 1) * 100 / years,  # growth spread evenly over the years, as if simple interest
        (ratio - 1) * 100,  # total growth read as the annual rate
        root_rate(years + 1),  # one compounding period too many
        root_rate(years - 1),  # one compounding period too few
        (ratio - 1) * 100 / (years + 1),  # growth spread over one year too many
        (final - principal) / years,  # average dollars of interest per year read as the rate
    ]
    working = [
        f"{usd(final)} = {usd(principal)} x (1 + r)^{years}, so (1 + r)^{years} = {float(ratio):.4f}.",
        f"1 + r = {float(ratio) ** (1 / years):.4f}, so r = {rate}%.",
    ]
    return number_fields(text, Fraction(rate), 0, wrong, working)


def _periods(rng: random.Random) -> dict:
    while True:
        principal = rng.randrange(2000, 10001, 500)
        per_year = rng.choice((2, 4))
        nominal = rng.choice((4, 6, 8, 10, 12) if per_year == 2 else (4, 8, 12))
        years = rng.randint(2, 4)
        step = Fraction(nominal, 100 * per_year)
        answer = Fraction(principal) * (1 + step) ** (per_year * years)
        if _clear_of_tie(answer):
            break
    name = rng.choice(_NAMES)
    word = "half-yearly" if per_year == 2 else "quarterly"
    text = (
        f"{name} invests {usd(principal)} at a nominal rate of {nominal}% p.a., compounded {word}. "
        f"Find the balance after {years} years, to the nearest cent."
    )
    total = _cents(answer)
    p = Fraction(principal)
    wrong = [
        p * (1 + Fraction(nominal, 100)) ** years,  # compounded annually instead
        p * (1 + Fraction(nominal, 100)) ** (per_year * years),  # full annual rate applied every period
        p * (1 + step * per_year * years),  # simple interest at the nominal rate
        p * (1 + step) ** years,  # per-period rate but only one period per year
    ]
    working = [
        f"Rate per period = {nominal}% / {per_year} = {float(step * 100):g}%.",
        f"Number of periods = {per_year} x {years} = {per_year * years}.",
        f"Balance = {principal} x {float(1 + step):g}^{per_year * years} = {usd(total)}.",
    ]
    return number_fields(text, total, 2, wrong, working)


_VARIANTS = {"phases": _phases, "rate": _rate, "periods": _periods}


@generator("exd_x_g5_compound_interest", level=11, presets={"exam": dict(tier="exam")})
def exd_x_g5_compound_interest(rng: random.Random, tier: str = "exam") -> dict:
    """Two-stage balance, rate from a final balance, or periodic compounding."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _VARIANTS[rng.choice(tuple(_VARIANTS))](rng)
