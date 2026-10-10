"""Generates `exd_x_journey_cost` exam questions: the total cost of a two-stage delivery run.

Each stage has its own distance, average speed and fuel consumption (litres per 100 km). The driver is paid
by the hour and fuel is paid by the litre, so the learner must find the time of each stage (distance over
speed), the fuel of each stage (per-100 km scaling) and combine the two costs. Nothing in the prompt says
which quantities to use; the average-speed shortcut is the trap.

Built from chosen stage durations: each distance is speed x time, so the times are exact. The total is
recomputed a second way (per-kilometre consumption and a single pass over minutes) and a draw is kept only
when the total is a whole number of dollars.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, check_distinct, generator, pick_distractors

_HIGHWAY_SPEEDS = [80, 90, 100, 110]
_TOWN_SPEEDS = [40, 50, 60]
_HIGHWAY_HOURS = [Fraction(h, 2) for h in (2, 3, 4, 5, 6)]
_TOWN_HOURS = [Fraction(h, 2) for h in (1, 2, 3, 4)]
_HIGHWAY_USE = [6, 7, 8, 9]
_TOWN_USE = [9, 10, 12]
_PRICES = [Fraction(9, 5), Fraction(2), Fraction(11, 5)]
_WAGES = [20, 24, 30, 36, 40]
_VEHICLES = ["van", "truck", "minibus"]


def _cost(d1, v1, k1, d2, v2, k2, wage, price):
    hours = Fraction(d1, v1) + Fraction(d2, v2)
    litres = Fraction(d1 * k1, 100) + Fraction(d2 * k2, 100)
    return wage * hours, price * litres


def _build(rng: random.Random) -> dict:
    for _ in range(2000):
        v1, v2 = rng.choice(_HIGHWAY_SPEEDS), rng.choice(_TOWN_SPEEDS)
        t1, t2 = rng.choice(_HIGHWAY_HOURS), rng.choice(_TOWN_HOURS)
        d1, d2 = v1 * t1, v2 * t2
        if d1.denominator != 1 or d2.denominator != 1:
            continue
        d1, d2 = int(d1), int(d2)
        k1, k2 = rng.choice(_HIGHWAY_USE), rng.choice(_TOWN_USE)
        price, wage = rng.choice(_PRICES), rng.choice(_WAGES)
        wages, fuel = _cost(d1, v1, k1, d2, v2, k2, wage, price)
        total = wages + fuel
        if total.denominator != 1:
            continue
        avg_speed = Fraction(v1 + v2, 2)
        avg_use = Fraction(k1 + k2, 2)
        candidates = {
            "averages-the-two-speeds": round(wage * Fraction(d1 + d2) / avg_speed + fuel),
            "omits-the-wages": round(fuel),
            "omits-the-fuel": round(wages),
            "averages-the-two-consumption-rates": round(wages + price * Fraction((d1 + d2) * avg_use, 100)),
        }
        if len({int(total), *candidates.values()}) == 5:
            break
    else:
        raise AssertionError("no whole-dollar journey found")

    # Second route: minutes driven and cost per kilometre.
    minutes = Fraction(d1 * 60, v1) + Fraction(d2 * 60, v2)
    per_km = Fraction(k1, 100) * price
    per_km2 = Fraction(k2, 100) * price
    check(Fraction(wage, 60) * minutes + d1 * per_km + d2 * per_km2 == total, "journey total mismatch")

    answer = int(total)
    check_distinct(answer, *candidates.values())

    price_text = f"{float(price):.2f}"
    vehicle = rng.choice(_VEHICLES)
    text = (
        f"A courier is paid ${wage} for every hour spent driving. On one delivery run the {vehicle} travels "
        f"{d1} km on a highway at an average speed of {v1} km/h, using {k1} litres of fuel per 100 km, and then "
        f"{d2} km through town at an average speed of {v2} km/h, using {k2} litres per 100 km. Fuel costs "
        f"${price_text} per litre. What is the total cost of the run, wages plus fuel, in whole dollars?"
    )
    answer_text = str(answer)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Time: {d1}/{v1} + {d2}/{v2} = {float(wages / wage):g} hours, so wages are ${float(wages):g}.",
            f"Fuel: {d1}/100 x {k1} + {d2}/100 x {k2} = {float(fuel / price):g} litres, so fuel is ${float(fuel):g}.",
            f"Total: ${answer}.",
        ],
        distractors=pick_distractors(answer_text, [str(v) for v in candidates.values()]),
    )


@generator(
    "exd_x_journey_cost",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_journey_cost(rng: random.Random, tier: str = "exam") -> dict:
    """Total wages-plus-fuel cost of a two-stage run. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
