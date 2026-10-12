"""Exam source for `senior-convert-between-grams-and-milligrams`: a prescription taken down to tablets and days.

The stem gives a body mass, a dose rate in mg per kg per day shared over a number of doses, a tablet strength in
GRAMS and the size of a box. The parts follow the pharmacist's steps:

(a) the total daily dose, in mg;
(b) the dose each time, in mg, using (a);
(c) tablets per dose, using (b): the tablet strength in grams is converted to mg here;
(d) the number of full days one box lasts, using (c).

Parts (b), (c) and (d) carry `follow_through`. The draw is built backwards from the number of tablets per dose, so
every part but (d) is a whole number by construction.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, exact_decimal_string, generator

from .exd_x_r3c_parts import number_part
from ._exd_r3_distinct import distinct_part_answers

_STRENGTHS_MG = (100, 125, 200, 250, 400, 500)
_RATES = (10, 15, 20, 25, 30, 40, 50, 60, 75, 100)
_BOXES = (20, 24, 28, 30, 36, 40, 50, 60, 84, 100, 120)
_MEDICINES = ("an antibiotic", "an anti-inflammatory medicine", "a pain-relief medicine", "an antihistamine")


def _tablets(n):
    return f"{n} tablet" if n == 1 else f"{n} tablets"


def _person(mass):
    if mass < 16:
        return "A toddler"
    if mass < 45:
        return "A child"
    return "A patient"


@generator("exd_x_r3c_dose_tablets", level=13, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_dose_tablets(rng, tier="exam"):
    """Mass and rate to a daily dose, a single dose, tablets per dose and days per box."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a prescription with a realistic body mass"):
        strength = rng.choice(_STRENGTHS_MG)
        tablets = rng.randint(1, 4)
        doses = rng.choice((2, 3, 4))
        rate = rng.choice(_RATES)
        daily = strength * tablets * doses
        if daily % rate:
            continue
        mass = daily // rate
        box = rng.choice(_BOXES)
        days = box // (tablets * doses)
        if 8 <= mass <= 90 and days >= 3 and box % (tablets * doses):
            break
    per_dose = strength * tablets
    strength_g = exact_decimal_string(Fraction(strength, 1000), 3)
    check(Fraction(per_dose, 1000) / Fraction(strength_g) == tablets, "tablet count disagrees after unit conversion")
    check(mass * rate == daily and daily == per_dose * doses, "doses disagree")

    stem = (
        f"{_person(mass)} with a body mass of {mass} kg is prescribed {rng.choice(_MEDICINES)} at {rate} mg per "
        f"kilogram of body mass each day, shared equally over {doses} doses. Each tablet contains {strength_g} g of "
        f"the medicine, and the tablets are sold in boxes of {box}."
    )
    part_a = number_part(
        "Find the total dose of the medicine to be taken each day, in mg.",
        daily,
        working=[f"{mass} × {rate} = {daily} mg."],
        wrongs=[
            (mass + rate, "added-instead-of-multiplied"),
            (per_dose, "reported-per-dose-for-daily"),
            (daily * doses, "shared-over-wrong-number-of-doses"),
        ],
    )
    part_b = number_part(
        "Using your answer to (a), find the mass of medicine in each dose, in mg.",
        per_dose,
        working=[f"{daily} ÷ {doses} = {per_dose} mg."],
        wrongs=[
            (daily, "forgot-to-share-over-doses"),
            (Fraction(daily, doses + 1), "shared-over-wrong-number-of-doses"),
            (daily * doses, "multiplied-by-doses-not-shared"),
            (Fraction(daily, doses - 1), "shared-over-wrong-number-of-doses"),
        ],
        follow_through=f"part_a/{doses}",
    )
    part_c = number_part(
        "Using your answer to (b), how many tablets are taken in each dose?",
        tablets,
        marks=2,
        working=[f"{strength_g} g = {strength} mg, so {per_dose} ÷ {strength} = {_tablets(tablets)}."],
        wrongs=[
            (tablets * 1000, "tablet-strength-not-converted"),
            (tablets * 10, "factor-of-100-error"),
            (tablets * doses, "reported-daily-for-per-dose"),
        ],
        follow_through=f"part_b/{strength}",
    )
    part_d = number_part(
        "Using your answer to (c), for how many full days does one box last?",
        days,
        marks=2,
        working=[
            f"{tablets} × {doses} = {_tablets(tablets * doses)} are taken each day.",
            f"{box} ÷ {tablets * doses} gives {days} full days.",
        ],
        wrongs=[
            (box // tablets, "forgot-doses-per-day"),
            (-(-box // (tablets * doses)), "rounded-up-instead-of-down"),
            (box // doses, "forgot-doses-per-day"),
        ],
        follow_through=f"floor({box}/(part_c*{doses}))",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
