"""Exam tier for "recognise the need for milligrams".

The practice tiers ask which mass unit makes a total a short whole number. Here the need for milligrams arises
inside a dosage calculation, as it does in a health-sciences or numeracy exam, and the learner has to move
between milligrams and grams twice:

(a) the mass of one dose in mg, from a rate in mg per kilogram of body mass;
(b) the mass given in a day in mg, from (a);
(c) the daily mass in grams, from (b), to 3 decimal places;
(d) the milligrams left in a tub supplied in grams after some days of this dosage, from (b).

Everything is a whole number except (c), which is a whole number of milligrams written in grams, so the 3 decimal
places are exact.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3g_kit import clashes, number_part

_NAMES = ("Mia", "Noah", "Ava", "Leo", "Isla", "Jack", "Ella", "Sam")
_DRUGS = ("an antibiotic suspension", "a liquid pain reliever", "an anti-inflammatory syrup", "a fever medicine")
_STEMS = (
    "{name} is prescribed {drug}. Each dose is {r} mg per kilogram of body mass, and {n} doses are given each day. "
    "{name} has a body mass of {w} kg.",
    "A doctor orders {drug} for {name}, who has a body mass of {w} kg. The order is {r} mg per kilogram of body mass "
    "for each dose, with {n} doses spread across the day.",
    "The label for {drug} says to give {r} mg per kilogram of body mass at each dose, {n} doses a day. "
    "{name}'s body mass is {w} kg.",
)


@generator("exd_x_r3g_dose_units", level=9, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3g_dose_units(rng: random.Random, tier: str = "exam") -> dict:
    """A paediatric dose: one dose in mg, the daily mass in mg and in g, and what is left of a tub."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a dosage with a tub that is never quite emptied"):
        r = rng.choice((5, 7, 8, 10, 12, 15, 20, 25))
        w = rng.randint(8, 42)
        n = rng.choice((2, 3, 4))
        days = rng.choice((5, 7, 10, 14))
        dose = r * w
        daily = n * dose
        used = days * daily
        tub = -(-used // 1000) + rng.randint(1, 8)  # grams, always more than is used
        if used > 60_000 or dose < 40:
            continue
        left = tub * 1000 - used
        name, drug = rng.choice(_NAMES), rng.choice(_DRUGS)
        stem = rng.choice(_STEMS).format(name=name, drug=drug, r=r, w=w, n=n)
        article = "an" if str(tub).startswith(("8", "11", "18")) else "a"
        pd = (
            f"The medicine is supplied as {article} {tub} g tub of powder. How many milligrams of powder remain in the tub "
            f"after {days} days of this dosage?"
        )
        if clashes((dose, daily, left), stem, pd):
            continue
        break

    check(left == tub * 1000 - days * daily and left > 0, "the tub is overdrawn")
    grams = Fraction(daily, 1000)
    parts = [
        number_part(
            "Find the mass of medicine in one dose, in mg.",
            dose,
            0,
            1,
            [f"{r} mg per kg x {w} kg = {dose} mg."],
            [
                (r + w, "added-instead-of-multiplied"),
                (daily, "dose-and-daily-total-confused"),
                (r, "ignores-body-mass"),
                (w, "dose-ignores-the-rate"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find the total mass of medicine given in one day, in mg.",
            daily,
            0,
            1,
            [f"{n} doses x {dose} mg = {daily} mg."],
            [
                (dose, "dose-and-daily-total-confused"),
                (dose + n, "added-instead-of-multiplied"),
                (r * n, "ignores-body-mass"),
                (dose * (n + 1), "off-by-one-count"),
            ],
            follow_through=f"{n}*part_a",
            sensible_range=(0, None),
        ),
        number_part(
            "Express the total mass given in one day in grams, correct to 3 decimal places.",
            grams,
            3,
            1,
            [f"1 g = 1000 mg, so {daily} mg = {daily} / 1000 = {float(grams):.3f} g."],
            [
                (daily, "skipped-unit-conversion"),
                (Fraction(daily, 100), "wrong-unit-conversion-factor"),
                (Fraction(daily, 10), "wrong-unit-conversion-factor"),
                (Fraction(daily, 1000) / n, "grams-reports-one-dose"),
            ],
            follow_through="part_b/1000",
            sensible_range=(0, None),
        ),
        number_part(
            pd,
            left,
            0,
            3,
            [
                f"Mass given in {days} days = {days} x {daily} mg = {used} mg.",
                f"The tub holds {tub} g = {tub * 1000} mg, so {tub * 1000} - {used} = {left} mg remain.",
            ],
            [
                (used, "tub-reports-the-mass-used"),
                (tub * 1000, "tub-ignores-the-mass-used"),
                (tub * 1000 - days * dose, "tub-uses-one-dose-a-day"),
                (tub * 100 - used, "wrong-unit-conversion-factor"),
            ],
            follow_through=f"{tub * 1000} - {days}*part_b",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
