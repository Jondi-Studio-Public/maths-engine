"""Exam tier for "use formulas to determine unknown values": a three-part chain on one situation.

A learner who can substitute into a formula is asked to use it, then to use its result in a second formula,
then to rearrange for a different unknown. Three situations, each a chain whose later parts carry
``follow_through`` from the earlier ones:

- speed: speed from a distance and a time, the distance in a new time, the time for a new distance;
- simple interest: interest over a term, the total value, the term needed to earn a different amount;
- electricity: energy used from power and time, the cost at a tariff, the cost over several days.

Every question is built backwards from the chosen answers, so each key is exact. The part answers are shown
at a stated number of decimal places and are exact at that precision.
"""

from __future__ import annotations

import random
from functools import partial
from fractions import Fraction

from ..core import check, draw_until, generator, phrase
from ._exd_x_sc4_kit import num_part as _num_part, txt

_NAMES = ["Priya", "Tom", "Mei", "Jack", "Aisha", "Liam", "Sofia", "Noah"]

num_part = partial(_num_part, key_places=True)  # every option in the key's own format


def _plain(value: Fraction) -> str:
    """A decimal with no trailing zeros, for a figure quoted in a stem."""
    for places in range(4):
        scaled = value * 10**places
        if scaled.denominator == 1:
            whole, frac = divmod(abs(int(scaled)), 10**places)
            return f"{whole}.{str(frac).zfill(places)}" if places else str(whole)
    raise ValueError(f"{value} has more than three decimal places")


def _speed(rng: random.Random):
    who, speed_lo, speed_hi, t1s, vehicle = rng.choice(
        [
            ("A cyclist rides", 14, 32, [Fraction(3, 2), 2, Fraction(5, 2), 3], "cyclist"),
            ("A coach travels", 70, 95, [2, Fraction(5, 2), 3, Fraction(7, 2), 4], "coach"),
            ("A train travels", 80, 120, [2, Fraction(5, 2), 3, Fraction(7, 2)], "train"),
        ]
    )
    v = Fraction(rng.randint(speed_lo, speed_hi))
    t1 = Fraction(rng.choice(t1s))
    d1 = v * t1
    t2 = Fraction(draw_until(rng, lambda r: r.randint(3, 10), lambda n: Fraction(n, 2) != t1), 2)
    t3 = Fraction(draw_until(rng, lambda r: r.randint(3, 10), lambda n: Fraction(n, 2) not in (t1, t2)), 2)
    d3 = v * t3
    d2 = v * t2
    check(d1.denominator <= 2 and d2.denominator <= 2 and d3.denominator <= 2, "distances must have one decimal")
    stem = f"{who} {_plain(d1)} km in {_plain(t1)} hours at a constant speed."
    stem += " Use speed = distance ÷ time."
    inverse_a = t1 / d1
    parts = [
        num_part(
            "Find the speed, in km/h.",
            v,
            0,
            2,
            [f"speed = {_plain(d1)} ÷ {_plain(t1)} = {txt(v, 0)} km/h."],
            [(d1 * t1, "multiplied-instead-of-divided"), (inverse_a, "inverted-ratio"), (d1 - t1, None)],
        ),
        num_part(
            f"At this speed, how far would it go in {_plain(t2)} hours? Give your answer in km, "
            "correct to 1 decimal place.",
            d2,
            1,
            2,
            [f"distance = speed × time = {txt(v, 0)} × {_plain(t2)} = {txt(d2, 1)} km."],
            [(v + t2, "added-instead-of-multiplied"), (v / t2, "divided-instead-of-multiplied"), (d1, None)],
            follow_through=f"part_a*{_plain(t2)}",
        ),
        num_part(
            f"How long, in hours, would it take to go {_plain(d3)} km at this speed? "
            "Give your answer correct to 1 decimal place.",
            t3,
            1,
            2,
            [f"time = distance ÷ speed = {_plain(d3)} ÷ {txt(v, 0)} = {txt(t3, 1)} hours."],
            [(d3 * v, "multiplied-instead-of-divided"), (v / d3, "inverted-ratio"), (d3 - v, None)],
            follow_through=f"{_plain(d3)}/part_a",
        ),
    ]
    return stem, parts


def _interest(rng: random.Random):
    name = rng.choice(_NAMES)
    principal = Fraction(rng.randrange(2000, 20001, 500))
    rate = Fraction(rng.randrange(250, 651, 25), 100)
    term = rng.randint(2, 9)
    other = draw_until(rng, lambda r: r.randint(2, 12), lambda n: n != term)
    interest = principal * rate * term / 100
    total = principal + interest
    target = principal * rate * other / 100
    stem = phrase(
        rng,
        (
            "{name} invests ${P} for {T} years at {r}% per annum simple interest. Use I = PRT/100.",
            "{name} puts ${P} into a term deposit paying {r}% per annum simple interest for {T} years. "
            "Use I = PRT/100.",
        ),
        name=name,
        P=txt(principal, 0),
        T=term,
        r=_plain(rate),
    )
    yearly = principal * rate / 100
    compound = principal * (1 + rate / 100) ** term
    parts = [
        num_part(
            "Find the interest earned over the term, in dollars, correct to 2 decimal places.",
            interest,
            2,
            2,
            [f"I = {txt(principal, 0)} × {_plain(rate)} × {term} ÷ 100 = ${txt(interest, 2)}."],
            [(yearly, "forgot-to-multiply-by-time"), (principal * rate * term, "percent-used-as-whole-number")],
        ),
        num_part(
            "Hence find the total value of the investment at the end of the term, in dollars, "
            "correct to 2 decimal places.",
            total,
            2,
            1,
            [f"Total = {txt(principal, 0)} + {txt(interest, 2)} = ${txt(total, 2)}."],
            [(interest, "interest-not-total"), (compound, "compound-for-simple"), (principal + yearly, None)],
            follow_through=f"{txt(principal, 0)} + part_a",
        ),
        num_part(
            f"A second investment has the same principal and rate. For how many years must it be invested to earn "
            f"${txt(target, 2)} in interest?",
            Fraction(other),
            0,
            2,
            [
                f"Interest per year = {txt(interest, 2)} ÷ {term} = {txt(yearly, 2)}.",
                f"Years = {txt(target, 2)} ÷ {txt(yearly, 2)} = {other}.",
            ],
            [
                (target / interest, "treated-total-as-annual"),
                (target * 100 / principal, "ignored-rate-in-interest"),
                (Fraction(other * term), "treated-total-as-annual"),
                (Fraction(term), None),
            ],
            follow_through=f"{txt(target, 2)}*{term}/part_a",
        ),
    ]
    return stem, parts


def _energy(rng: random.Random):
    appliance, watts_choices = rng.choice(
        [
            ("an air conditioner", range(1200, 3101, 100)),
            ("an electric heater", range(1000, 2401, 100)),
            ("a clothes dryer", range(2000, 4001, 100)),
            ("an oven", range(1800, 3401, 100)),
        ]
    )
    watts_choices = list(watts_choices)

    def draw(r):
        return (
            Fraction(r.choice(watts_choices)),
            Fraction(r.randint(3, 12), 2),
            Fraction(r.randint(26, 42)),
            r.choice([7, 14, 28, 30, 60, 90]),
        )

    def fine(values):
        p, t, rate, _ = values
        energy = p * t / 1000
        return energy.denominator <= 100 and (energy * rate).denominator == 1 and energy * rate % 1 == 0

    watts, hours, rate, days = draw_until(rng, draw, fine, what="power, time and tariff on whole cents")
    energy = watts * hours / 1000
    cost = energy * rate / 100
    total = cost * days
    stem = (
        f"{appliance[0].upper()}{appliance[1:]} is rated at {txt(watts, 0)} W. It is run for {_plain(hours)} hours. "
        "Use energy (kWh) = power (W) × time (h) ÷ 1000."
    )
    parts = [
        num_part(
            "Find the energy used, in kWh, correct to 2 decimal places.",
            energy,
            2,
            2,
            [f"E = {txt(watts, 0)} × {_plain(hours)} ÷ 1000 = {txt(energy, 2)} kWh."],
            [(watts * hours, "conversion-power-of-ten-slip"), (watts * hours / 100, "conversion-power-of-ten-slip")],
        ),
        num_part(
            f"Electricity costs {txt(rate, 0)} cents per kWh. Find the cost of running it for this time, in dollars, "
            "correct to 2 decimal places.",
            cost,
            2,
            2,
            [f"Cost = {txt(energy, 2)} × {txt(rate, 0)} ÷ 100 = ${txt(cost, 2)}."],
            [(energy * rate, "cents-read-as-dollars"), (energy + rate, "added-instead-of-multiplied")],
            follow_through=f"part_a*{txt(rate, 0)}/100",
        ),
        num_part(
            f"It is used for the same time every day for {days} days. Find the total cost, in dollars, "
            "correct to 2 decimal places.",
            total,
            2,
            1,
            [f"Total = {txt(cost, 2)} × {days} = ${txt(total, 2)}."],
            [
                (cost + days, "added-instead-of-multiplied"),
                (energy * days, "gave-energy-not-cost"),
                (energy * rate * days, "cents-read-as-dollars"),
            ],
            follow_through=f"part_b*{days}",
        ),
    ]
    return stem, parts


_SCENARIOS = (_speed, _interest, _energy)


@generator("exd_x_sc4_formula_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc4_formula_chain(rng, tier="exam"):
    """Use a formula, use its result in a second step, then rearrange it. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    stem, parts = rng.choice(_SCENARIOS)(rng)
    return dict(text=stem, parts=parts)
