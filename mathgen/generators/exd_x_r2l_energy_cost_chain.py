"""Exam question: household electricity use, then its cost, then the saving from a more efficient appliance.

The stem gives an appliance's power in watts, a daily run time in hours and minutes, and a tariff in cents per
kilowatt-hour. Part (a) finds the daily energy in kilowatt-hours (power converted to kilowatts, time to hours).
Part (b) turns that into a cost over a stated number of days, converting cents to dollars. Part (c) replaces the
appliance with a lower-power one used for the same time and asks for the fall in cost. Parts (b) and (c) carry
`follow_through`.

Built backwards from exact arithmetic: powers are multiples of 100 W and times multiples of 15 minutes, and a draw
is kept only when the daily energy is exact to two decimal places. The saving is also kept only when working from
the rounded cost of (b) (as follow-through does) gives the same cent as the exact route, so the key does not
depend on which route a learner takes.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

# (appliance, article, power range in W (multiples of 100), hours range)
_APPLIANCES = (
    ("electric heater", "an", (12, 24), (2, 6)),
    ("air conditioner", "an", (15, 30), (3, 8)),
    ("clothes dryer", "a", (20, 30), (1, 2)),
    ("electric oven", "an", (20, 30), (1, 2)),
    ("pool pump", "a", (7, 15), (4, 8)),
    ("dehumidifier", "a", (6, 12), (5, 10)),
)
_DAYS = (30, 60, 90)
_MINUTES = (15, 30, 45)


def _kwh(power_w: int, hours: int, minutes: int) -> Fraction:
    return Fraction(power_w, 1000) * (hours + Fraction(minutes, 60))


def _time_text(hours: int, minutes: int) -> str:
    return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minutes"


def _dollars(value: Fraction) -> Fraction:
    return round_half_up(value, 2)


def _build(rng: random.Random) -> dict:
    name, article, (p_lo, p_hi), (h_lo, h_hi) = rng.choice(_APPLIANCES)
    for _ in attempts(2000, "an exact daily energy and a consistent saving"):
        power = 100 * rng.randint(p_lo, p_hi)
        hours = rng.randint(h_lo, h_hi)
        minutes = rng.choice(_MINUTES)
        tariff = Fraction(rng.randint(220, 460), 10)  # cents per kWh, one decimal place
        days = rng.choice(_DAYS)
        new_power = power - 100 * rng.randint(3, max(4, power // 200))
        energy = _kwh(power, hours, minutes)
        if (energy * 100).denominator != 1 or new_power < 400 or new_power >= power:
            continue
        cost = energy * days * tariff / 100
        saving = (_kwh(power, hours, minutes) - _kwh(new_power, hours, minutes)) * days * tariff / 100
        # Follow-through works from the rounded (b); the exact saving must round to the same cent that route gives.
        via_rounded = _dollars(cost) * Fraction(power - new_power, power)
        if _dollars(saving) != _dollars(via_rounded):
            continue
        # Keep every key clear of a half-cent boundary so no route rounds differently.
        if any(abs((v * 100) % 1 - Fraction(1, 2)) < Fraction(1, 20) for v in (cost, saving)):
            continue
        break

    # Independent route: work in watt-hours and cents, then convert once at the end.
    minutes_total = hours * 60 + minutes
    wh_per_day = Fraction(power * minutes_total, 60)
    check(wh_per_day / 1000 == energy, "energy disagrees with the watt-hour route")
    cents = wh_per_day / 1000 * days * tariff
    check(cents / 100 == cost, "cost disagrees with the cents route")
    new_cents = Fraction(new_power * minutes_total, 60) / 1000 * days * tariff
    check((cents - new_cents) / 100 == saving, "saving disagrees with the cents route")

    a_val, b_val, c_val = energy, _dollars(cost), _dollars(saving)
    a_text = stated_precision_string(a_val, 2)
    b_text = stated_precision_string(b_val, 2)
    c_text = stated_precision_string(c_val, 2)
    tariff_text = f"{float(tariff):.1f}"
    time_text = _time_text(hours, minutes)

    def d2(value: Fraction) -> Fraction:
        return round_half_up(value, 2)

    def cand(value: Fraction, tag: str) -> Distractor:
        v = d2(value)
        return Distractor(stated_precision_string(v, 2), tag, v)

    t_hours = hours + Fraction(minutes, 60)
    a_cands = [
        cand(Fraction(power, 1000) * (hours + Fraction(minutes, 100)), "decimal-time-base-100"),
        cand(Fraction(power, 1000) * hours, "ignores-the-minutes"),
        cand(Fraction(power, 100) * t_hours, "wrong-watt-kilowatt-factor"),
        cand(power * t_hours, "forgot-watt-kilowatt-conversion"),
    ]
    day_cost = energy * tariff / 100
    new_cost = cost * Fraction(new_power, power)
    b_cands = [
        cand(energy * days * tariff, "cents-read-as-dollars"),
        cand(day_cost, "forgot-to-multiply-by-days"),
        cand(energy * days, "forgot-the-tariff"),
        cand(energy * days * tariff / 10000, "converted-cents-to-dollars-twice"),
    ]
    c_cands = [
        cand(new_cost, "reports-new-cost-not-saving"),
        cand(cost, "reports-the-original-cost"),
        cand(saving / days, "forgot-to-multiply-by-days"),
        cand(Fraction(power - new_power, 1000) * days * tariff / 100, "ignores-the-run-time"),
    ]
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(c_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(len(a_d) == len(b_d) == len(c_d) == 3, "too few distinct distractors")

    stem = (
        f"A household uses {article} {name} rated at {power} W for {time_text} each day. "
        f"Electricity costs {tariff_text} cents per kilowatt-hour (kWh)."
    )
    ratio = Fraction(power - new_power, power)
    parts = [
        dict(
            prompt="Find the energy the appliance uses each day, in kilowatt-hours, correct to 2 decimal places.",
            answer_kind="number",
            answer=numeric_answer(a_val),
            answer_text=a_text,
            answer_latex=latex_answer(a_text),
            marks=2,
            sensible_range=(0, None),
            working=[
                f"{power} W = {float(Fraction(power, 1000))} kW; {time_text} = {float(t_hours)} hours.",
                f"Energy = {float(Fraction(power, 1000))} x {float(t_hours)} = {a_text} kWh.",
            ],
            distractors=a_d,
        ),
        dict(
            prompt=f"Find the cost of running the appliance for {days} days, in dollars, correct to the nearest cent.",
            answer_kind="number",
            answer=numeric_answer(b_val),
            answer_text=b_text,
            answer_latex=latex_answer(b_text),
            marks=2,
            sensible_range=(0, None),
            follow_through=f"part_a*{days}*{tariff.numerator}/{tariff.denominator}/100",
            working=[
                f"Energy over {days} days = {a_text} x {days} kWh.",
                f"Cost = {a_text} x {days} x {tariff_text} cents = {b_text} dollars.",
            ],
            distractors=b_d,
        ),
        dict(
            prompt=(
                f"The household replaces the appliance with a more efficient one rated at {new_power} W, used for "
                f"the same time each day. Using your answer to (b), find how much less it costs to run the new "
                f"appliance for {days} days, in dollars, correct to the nearest cent."
            ),
            answer_kind="number",
            answer=numeric_answer(c_val),
            answer_text=c_text,
            answer_latex=latex_answer(c_text),
            marks=2,
            sensible_range=(0, None),
            follow_through=f"part_b*({power - new_power})/{power}",
            working=[
                f"The cost is in proportion to the power: the new cost is {new_power}/{power} of {b_text}.",
                f"Saving = {b_text} x ({power} - {new_power})/{power} = {c_text} dollars.",
            ],
            distractors=c_d,
        ),
    ]
    check(ratio > 0, "no reduction in power")
    return dict(text=stem, parts=parts)


@generator("exd_x_r2l_energy_cost_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_r2l_energy_cost_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Daily kWh from watts and a time, the cost over a period at a tariff, then the saving from a smaller appliance."""
    check(tier == "exam", "exd_x_r2l_energy_cost_chain: only the exam tier exists")
    return _build(rng)
