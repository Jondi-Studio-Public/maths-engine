"""Exam tier for flat-interest loans: the repayment formula run backwards, or extended with a deposit and a fee.

  rate     -- the monthly repayment, amount borrowed and term are given; find the flat annual rate
              (whole percent). The learner rebuilds the total repaid, strips out the principal, and divides
              by the amount borrowed and the years.
  term     -- the monthly repayment, amount borrowed and rate are given; find the term in whole years.
  deposit  -- a purchase price with a percentage deposit, flat interest on the amount financed and a fee added
              to the total repayable; find the monthly repayment.

Rate and term are built backwards from whole values; the given repayment is the exact figure rounded to the
cent, and a one-unit change in the answer moves the repayment by many dollars, so the answer is unique.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, round_half_up
from .exd_x_g5_common import number_fields, usd

_ITEMS = ("a car", "a boat", "a caravan", "a motorbike", "a kitchen renovation")


def _repayment(principal: int, rate: int, years: int) -> Fraction:
    return round_half_up(Fraction(principal) * (100 + rate * years) / (100 * 12 * years), 2)


def _rate(rng: random.Random) -> dict:
    principal = rng.randrange(8000, 40001, 1000)
    rate = rng.randint(3, 11)
    years = rng.randint(2, 6)
    monthly = _repayment(principal, rate, years)
    months = 12 * years
    text = (
        f"A loan of {usd(principal)} is repaid in equal monthly instalments of {usd(monthly)} over {years} years. "
        f"Flat interest is charged on the original amount borrowed for each year of the loan. "
        f"What is the flat annual interest rate, as a whole-number percentage?"
    )
    total = monthly * months
    solved = (total - principal) * 100 / (principal * years)
    check(round_half_up(solved, 0) == rate and abs(solved - rate) < Fraction(1, 10), "rate does not reproduce")
    wrong = [
        (total - principal) * 100 / principal,  # total interest read as an annual rate
        (total - principal) * 100 / (principal * months),  # the months used as the number of years
        (total - principal) * 100 / (principal * 12),  # total interest over twelve, as if per month
        (total - principal) * 100 / (principal * (years - 1)),  # one year too few
        (total - principal) * 100 / (principal * (years + 1)),  # one year too many
        total * 100 / principal,  # total repaid as a percentage of the amount borrowed
    ]
    working = [
        f"Total repaid = {months} x {usd(monthly)} = {usd(total)}.",
        f"Interest = {usd(total)} - {usd(principal)} = {usd(total - principal)}.",
        f"Interest per year = {usd(round_half_up((total - principal) / years, 2))}; as a percentage of {usd(principal)} this is {rate}%.",
    ]
    return number_fields(text, Fraction(rate), 0, wrong, working)


def _term(rng: random.Random) -> dict:
    principal = rng.randrange(8000, 40001, 1000)
    rate = rng.randint(3, 11)
    years = rng.randint(2, 7)
    monthly = _repayment(principal, rate, years)
    text = (
        f"A loan of {usd(principal)} carries flat interest of {rate}% of the amount borrowed per year. It is "
        f"repaid in equal monthly instalments of {usd(monthly)}. Over how many years is the loan repaid?"
    )
    # Total repaid = 12*M*y = P + P*r*y/100  =>  y = P / (12*M - P*r/100).
    solved = Fraction(principal) / (12 * monthly - Fraction(principal * rate, 100))
    check(abs(solved - years) < Fraction(1, 10), "term does not reproduce")
    wrong = [
        Fraction(principal) / (12 * monthly),  # interest ignored
        Fraction(principal * (100 + rate), 100) / (12 * monthly),  # one year's interest only
        Fraction(12 * years),  # the term in months
        Fraction(principal) / (12 * monthly - Fraction(principal * rate, 1200)),  # annual rate used as monthly
        Fraction(years + 1),  # the first year counted twice
        Fraction(years - 1),  # the last year left out
    ]
    working = [
        f"Total repaid = 12 x {usd(monthly)} x y = {float(12 * monthly):g}y.",
        f"Total repayable = {principal} + {principal} x {rate / 100:g} x y.",
        f"{float(12 * monthly):g}y = {principal} + {principal * rate / 100:g}y, so y = {years}.",
    ]
    return number_fields(text, Fraction(years), 0, wrong, working)


def _deposit(rng: random.Random) -> dict:
    price = rng.randrange(18000, 60001, 1000)
    pct = rng.choice((10, 15, 20, 25))
    rate = rng.randint(4, 10)
    years = rng.randint(2, 5)
    fee = rng.randrange(100, 501, 50)
    financed = Fraction(price) * (100 - pct) / 100
    total = financed * (100 + rate * years) / 100 + fee
    answer = round_half_up(total / (12 * years), 2)
    item = rng.choice(_ITEMS)
    text = (
        f"{item[0].upper() + item[1:]} costs {usd(price)}. A deposit of {pct}% of the price is paid up front and "
        f"the rest is borrowed for {years} years at {rate}% flat interest per year (charged on the amount "
        f"borrowed). A {usd(fee)} fee is then added to the total repayable. Find the monthly repayment, to the "
        f"nearest cent."
    )
    p = Fraction(price)
    n = 12 * years
    wrong = [
        (p * (100 + rate * years) / 100 + fee) / n,  # interest charged on the full price
        (financed * (100 + rate * years) / 100) / n,  # the fee left out
        (financed * (100 + rate * years) / 100 + fee) / years,  # divided by years, not months
        (financed + fee) / n * (1 + Fraction(rate, 100)),  # one year's interest on the loan and the fee
    ]
    working = [
        f"Amount borrowed = {usd(price)} - {pct}% = {usd(financed)}.",
        f"Total repayable = {usd(financed)} x (1 + {rate / 100:g} x {years}) + {usd(fee)} = {usd(round_half_up(total, 2))}.",
        f"Monthly repayment = {usd(round_half_up(total, 2))} / {n} = {usd(answer)}.",
    ]
    return number_fields(text, answer, 2, wrong, working)


_VARIANTS = {"rate": _rate, "term": _term, "deposit": _deposit}


@generator("exd_x_g5_flat_loan", level=11, presets={"exam": dict(tier="exam")})
def exd_x_g5_flat_loan(rng: random.Random, tier: str = "exam") -> dict:
    """Flat-loan rate or term from the repayment, or a repayment with a deposit and a fee."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _VARIANTS[rng.choice(tuple(_VARIANTS))](rng)
