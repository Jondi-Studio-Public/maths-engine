"""Exam tier for simple interest: two problems where `I = Prt` has to be used as an equation, not a formula
to evaluate.

  split   -- one sum is divided between two accounts paying different simple-interest rates; the total
             interest after some years is given, and the learner finds how much was placed in the first
             account (two unknowns tied by the total and by the interest, so one linear equation).
  months  -- the interest earned over a number of months is given and the learner finds the annual rate
             (the time must be converted to years first).

Both are built backwards from chosen whole-dollar amounts and whole-percent rates, so the answer is an exact
integer and is the only solution (the two rates always differ).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator
from .exd_x_g5_common import number_fields, usd

_NAMES = ("Ava", "Ben", "Chloe", "Dev", "Elena", "Femi")


def _split(rng: random.Random) -> dict:
    total = rng.randrange(6000, 20001, 1000)
    low = rng.randint(2, 6)
    high = low + rng.randint(2, 4)
    first_is_high = rng.random() < 0.5
    first_rate, second_rate = (high, low) if first_is_high else (low, high)
    first = rng.choice([x for x in range(1000, total - 999, 500) if 2 * x != total])
    second = total - first
    years = rng.randint(2, 5)
    interest = Fraction(first * first_rate + second * second_rate, 100) * years
    name = rng.choice(_NAMES)
    text = (
        f"{name} divides {usd(total)} between two accounts. The first pays {first_rate}% p.a. simple interest and "
        f"the second pays {second_rate}% p.a. simple interest. After {years} years the total interest earned is "
        f"{usd(interest)}. How many dollars were placed in the first account?"
    )
    # Independent solve of  first_rate*x + second_rate*(total - x) = 100*interest/years.
    solved = (100 * interest / years - second_rate * total) / (first_rate - second_rate)
    check(solved == first, "split does not reproduce")
    t = Fraction(total)
    yearly = 100 * interest / years
    wrong = [
        Fraction(second),  # the other account
        t / 2,  # an equal split assumed
        yearly / first_rate,  # all of the interest attributed to the first account
        yearly / Fraction(first_rate + second_rate, 2),  # the whole sum assumed to earn the average rate
        (100 * interest - second_rate * total) / (first_rate - second_rate),  # the years left out
    ]
    working = [
        f"Let x be the amount in the first account. Interest per year = ({first_rate}x + {second_rate}({total} - x)) / 100.",
        f"Total interest: {years} x ({first_rate}x + {second_rate}({total} - x)) / 100 = {float(interest):g}.",
        f"{first_rate - second_rate}x + {second_rate * total} = {float(yearly):g}, so x = {first}.",
    ]
    return number_fields(text, Fraction(first), 0, wrong, working)


def _months(rng: random.Random) -> dict:
    principal = 100 * rng.randint(20, 120)
    rate = rng.randint(3, 11)
    months = rng.choice((6, 9, 15, 18, 21, 30))
    interest = Fraction(principal * rate * months, 1200)
    check(interest * 100 % 1 == 0, "interest is not a whole number of cents")
    name = rng.choice(_NAMES)
    text = (
        f"{name} invests {usd(principal)} for {months} months at a fixed simple-interest rate and earns "
        f"{usd(interest)} in interest. What is the annual interest rate, as a percentage?"
    )
    check(interest * 1200 / (principal * months) == rate, "months does not reproduce")
    wrong = [
        interest * 100 / (principal * months),  # the months used as years
        interest * 100 / principal,  # the time left out
        interest * 1200 / (principal * (months + 12)),  # an extra year added to the time
        Fraction(rate * months, 12),  # the interest for the period read back as an annual rate
    ]
    working = [
        f"Time in years = {months} / 12 = {float(Fraction(months, 12)):g}.",
        f"I = Prt, so r = I / (Pt) = {float(interest):g} / ({principal} x {float(Fraction(months, 12)):g}) = {rate / 100:g}.",
        f"The annual rate is {rate}%.",
    ]
    return number_fields(text, Fraction(rate), 0, wrong, working)


_VARIANTS = {"split": _split, "months": _months}


@generator("exd_x_g5_simple_interest", level=11, presets={"exam": dict(tier="exam")})
def exd_x_g5_simple_interest(rng: random.Random, tier: str = "exam") -> dict:
    """A sum split between two simple-interest accounts, or an annual rate from interest over months."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _VARIANTS[rng.choice(tuple(_VARIANTS))](rng)
