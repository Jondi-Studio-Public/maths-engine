"""Exam tier for "convert from one unit of energy to another": an appliance chain.

An appliance's power in watts and running time in hours and minutes are given. Part (a) turns them into energy in
kilowatt-hours (watts to kilowatts, minutes to hours, then power times time). Part (b) converts that energy to
kilojoules, which needs the fact that a kilowatt-hour is 3600 kilojoules rather than a thousand of anything, and part
(c) finds the cost at a price per kilowatt-hour. Parts (b) and (c) carry `follow_through`.

## Construction: draw the energy first

The power is a multiple of 100 W and the time a multiple of 6 minutes, so the energy in kWh is an exact multiple of
0.01, the kilojoule figure is an exact integer, and only the cost needs rounding (to the nearest cent, kept clear of
a rounding boundary).
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
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

_APPLIANCES = (
    "An electric heater is rated at {w} W and runs for {time}.",
    "A clothes dryer rated at {w} W runs for {time}.",
    "An air conditioner drawing {w} W is left on for {time}.",
    "A pool pump rated at {w} W operates for {time}.",
    "An electric oven rated at {w} W is used for {time}.",
)
_TARIFFS = (
    "Electricity costs {rate} cents per kilowatt-hour.",
    "The household pays {rate} c per kWh for electricity.",
    "The retailer charges {rate} cents for each kWh used.",
)


def _duration(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h} hour{'s' if h > 1 else ''} {m} minutes"
    if h:
        return f"{h} hour{'s' if h > 1 else ''}"
    return f"{m} minutes"


def _money(value: Fraction) -> Distractor:
    shown = round_half_up(value, 2)
    return Distractor(stated_precision_string(shown, 2), None, shown)


@generator("exd_x_m4_energy_units_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m4_energy_units_chain(rng: random.Random, tier: str = "exam") -> dict:
    """kWh from watts and time, then kilojoules, then a cost. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a power, time and tariff with a clear cost"):
        watts = 100 * rng.randint(5, 40)
        minutes = 6 * rng.randint(5, 100)
        rate = rng.randint(24, 58)
        energy = Fraction(watts * minutes, 60000)
        cost = energy * rate / 100
        if abs((cost * 100) % 1 - Fraction(1, 2)) <= Fraction(1, 50) or not 1 <= energy <= 40:
            continue
        e_text = stated_precision_string(energy, 2)
        kj = energy * 3600
        kj_text = str(int(kj))
        cost_shown = round_half_up(cost, 2)
        c_text = stated_precision_string(cost_shown, 2)
        stem = (
            rng.choice(_APPLIANCES).format(w=watts, time=_duration(minutes))
            + " "
            + rng.choice(_TARIFFS).format(rate=rate)
        )
        if all(t not in stem.split() for t in (e_text, kj_text, c_text)):
            break
    check(kj.denominator == 1, "kilojoules are whole")

    e_wrong = [energy * 1000, Fraction(watts * minutes, 1000), energy * 10]
    b_wrong = [energy * 1000, energy * Fraction(18, 5), energy * 60, energy * Fraction(5, 18)]
    c_wrong = [energy * rate, energy / rate * 100, cost / 10, kj * rate / 100]
    a_d = pick_distractors(
        e_text,
        [Distractor(stated_precision_string(round_half_up(v, 2), 2), None, round_half_up(v, 2)) for v in e_wrong],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    b_d = pick_distractors(
        kj_text,
        [Distractor(str(int(round_half_up(v))), None, round_half_up(v)) for v in b_wrong],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    c_d = pick_distractors(c_text, [_money(v) for v in c_wrong], 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "not enough distractors")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the energy used, in kilowatt-hours.",
                answer_kind="number",
                answer=numeric_answer(energy),
                answer_text=e_text,
                answer_latex=latex_answer(e_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{watts} W = {Fraction(watts, 1000)} kW and {minutes} min = {Fraction(minutes, 60)} h.",
                    f"Energy = {e_text} kWh.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Hence express this energy in kilojoules.",
                answer_kind="number",
                answer=int(kj),
                answer_text=kj_text,
                answer_latex=latex_answer(kj_text),
                marks=1,
                sensible_range=(0, None),
                follow_through="3600*part_a",
                working=[f"1 kWh = 1 kW x 3600 s = 3600 kJ, so {e_text} x 3600 = {kj_text} kJ."],
                distractors=b_d,
            ),
            dict(
                prompt="Hence find the cost of running the appliance, in dollars to the nearest cent.",
                answer_kind="number",
                answer=numeric_answer(cost_shown),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{rate}/100",
                working=[f"Cost = {e_text} x {rate} cents = {c_text} dollars (nearest cent)."],
                distractors=c_d,
            ),
        ],
    )
