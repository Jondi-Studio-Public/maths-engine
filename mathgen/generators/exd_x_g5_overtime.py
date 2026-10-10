"""Exam tier for overtime pay: a weekly wage with two overtime bands, asked three ways.

The skill's practice questions apply one overtime multiplier to one block of hours. Here the pay structure
has two bands (time and a half for the first two overtime hours, double time after that), so the learner
must split the overtime hours before pricing them, and the unknown changes with the variant:

  pay     -- the hours are given; find the gross weekly pay.
  hours   -- the gross pay is given; find the total hours worked (invert a piecewise pay function).
  rate    -- the hours and the gross pay are given; find the ordinary hourly rate.

Every variant is built backwards from whole-dollar rates and a chosen overtime-hour count of at least 3,
so both bands are always used and the inverse questions have exactly one answer.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator
from .exd_x_g5_common import number_fields, usd

_BAND_ONE_HOURS = 2  # overtime hours paid at time and a half before double time starts


def _gross(rate: Fraction, ordinary: int, overtime: int) -> Fraction:
    """Gross pay for `overtime` hours beyond `ordinary`: first two at 1.5x, the rest at 2x."""
    check(overtime >= _BAND_ONE_HOURS, "both overtime bands must be used")
    return rate * ordinary + Fraction(3, 2) * rate * _BAND_ONE_HOURS + 2 * rate * (overtime - _BAND_ONE_HOURS)


def _structure(ordinary: int) -> str:
    return (
        f"Hours beyond {ordinary} in a week are overtime: the first {_BAND_ONE_HOURS} overtime hours are paid "
        f"at time and a half and every overtime hour after that at double time"
    )


def _pay(rng: random.Random) -> dict:
    rate = Fraction(rng.randint(42, 94), 2)
    ordinary = rng.choice((36, 38, 40))
    overtime = rng.randint(3, 9)
    total = ordinary + overtime
    name = rng.choice(("Sam", "Priya", "Tomas", "Mei", "Alana", "Kofi"))
    answer = _gross(rate, ordinary, overtime)
    text = (
        f"{name}'s ordinary rate is {usd(rate)} per hour. {_structure(ordinary)}. "
        f"In one week {name} works {total} hours. Find {name}'s gross pay for the week, to the nearest cent."
    )
    r = rate
    wrong = [
        r * ordinary + 2 * r * overtime,  # double time on every overtime hour
        r * ordinary + Fraction(3, 2) * r * overtime,  # time and a half on every overtime hour
        r * ordinary + 3 * r + 2 * r * (total - _BAND_ONE_HOURS),  # the total hours treated as overtime hours
        r * ordinary + 2 * r * _BAND_ONE_HOURS + Fraction(3, 2) * r * (overtime - _BAND_ONE_HOURS),  # bands swapped
    ]
    working = [
        f"Overtime hours = {total} - {ordinary} = {overtime}.",
        f"Pay = {ordinary} x {float(rate):g} + {_BAND_ONE_HOURS} x 1.5 x {float(rate):g} + {overtime - _BAND_ONE_HOURS} x 2 x {float(rate):g}.",
        f"Gross pay = {usd(answer)}.",
    ]
    return number_fields(text, answer, 2, wrong, working)


def _hours(rng: random.Random) -> dict:
    rate = Fraction(rng.randint(42, 94), 2)
    ordinary = rng.choice((36, 38, 40))
    overtime = rng.randint(3, 9)
    name = rng.choice(("Sam", "Priya", "Tomas", "Mei", "Alana", "Kofi"))
    pay = _gross(rate, ordinary, overtime)
    text = (
        f"{name}'s ordinary rate is {usd(rate)} per hour. {_structure(ordinary)}. "
        f"One week {name}'s gross pay was {usd(pay)}. How many hours did {name} work that week?"
    )
    r = rate
    wrong = [
        ordinary + (pay - r * ordinary) / (Fraction(3, 2) * r),  # one band at time and a half
        ordinary + (pay - r * ordinary) / (2 * r),  # one band at double time
        pay / r,  # no overtime loading at all
        pay / (Fraction(3, 2) * r),  # every hour paid at time and a half
        pay / (2 * r),  # every hour paid at double time
        Fraction(ordinary + _BAND_ONE_HOURS),  # only the time-and-a-half hours counted
    ]
    working = [
        f"Ordinary pay = {ordinary} x {float(rate):g} = {usd(ordinary * rate)}; the first {_BAND_ONE_HOURS} overtime hours "
        f"add {usd(Fraction(3, 2) * rate * _BAND_ONE_HOURS)}.",
        f"The remaining {usd(pay - r * ordinary - 3 * r)} is paid at {usd(2 * rate)} per hour, "
        f"so there are {overtime - _BAND_ONE_HOURS} further overtime hour(s).",
        f"Total hours = {ordinary} + {_BAND_ONE_HOURS} + {overtime - _BAND_ONE_HOURS} = {ordinary + overtime}.",
    ]
    check((pay - r * ordinary - 3 * r) / (2 * r) == overtime - _BAND_ONE_HOURS, "hours inverse does not close")
    return number_fields(text, Fraction(ordinary + overtime), 0, wrong, working)


def _rate(rng: random.Random) -> dict:
    rate = Fraction(rng.randint(42, 94), 2)
    ordinary = rng.choice((36, 38, 40))
    overtime = rng.randint(3, 9)
    total = ordinary + overtime
    name = rng.choice(("Sam", "Priya", "Tomas", "Mei", "Alana", "Kofi"))
    pay = _gross(rate, ordinary, overtime)
    text = (
        f"{name} works {total} hours in a week and is paid {usd(pay)} gross. Hours beyond {ordinary} are overtime: "
        f"the first {_BAND_ONE_HOURS} overtime hours are paid at time and a half and every overtime hour after "
        f"that at double time. What is {name}'s ordinary hourly rate, to the nearest cent?"
    )
    paid_hours = Fraction(ordinary) + Fraction(3, 2) * _BAND_ONE_HOURS + 2 * (overtime - _BAND_ONE_HOURS)
    check(pay / paid_hours == rate, "rate inverse does not close")
    wrong = [
        pay / total,  # every hour at the ordinary rate
        pay / (ordinary + Fraction(3, 2) * overtime),  # one band at time and a half
        pay / (ordinary + 2 * overtime),  # one band at double time
        pay / (ordinary + 3 + overtime - _BAND_ONE_HOURS),  # bands counted as hours, not rate multiples
    ]
    working = [
        f"Paid at the ordinary rate: {ordinary} + {_BAND_ONE_HOURS} x 1.5 + {overtime - _BAND_ONE_HOURS} x 2 "
        f"= {paid_hours:g} hours' worth.",
        f"Rate = {usd(pay)} / {paid_hours:g} = {usd(rate)} per hour.",
    ]
    return number_fields(text, rate, 2, wrong, working)


_VARIANTS = {"pay": _pay, "hours": _hours, "rate": _rate}


@generator("exd_x_g5_overtime", level=11, presets={"exam": dict(tier="exam")})
def exd_x_g5_overtime(rng: random.Random, tier: str = "exam") -> dict:
    """Two-band overtime: find the pay, the hours worked, or the ordinary rate."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _VARIANTS[rng.choice(tuple(_VARIANTS))](rng)
