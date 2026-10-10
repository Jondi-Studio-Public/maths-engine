"""Generates `exd_x_grams_dose` exam questions: tablets per dose from a dose rate per kilogram.

A prescription gives a dose in milligrams per kilogram of body mass per day, shared over several doses, and
the tablet strength is stated in GRAMS. The learner chooses the order of the steps: total daily milligrams
from the body mass, milligrams per dose, tablet strength in milligrams, then the number of tablets.

Built backwards from the number of tablets per dose, the tablet strength and the number of doses; the body
mass is then drawn from the masses that divide the daily total into a whole dose rate.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, check_distinct, generator, pick_distractors

_STRENGTHS_MG = [125, 250, 500, 750]
_DOSES = [2, 3, 4]
_MASSES = [8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 40, 45, 50, 60]
_PATIENTS = ["A child", "A patient", "A toddler"]


def _build(rng: random.Random) -> dict:
    for _ in range(500):
        tablets = rng.randint(1, 4)
        strength = rng.choice(_STRENGTHS_MG)
        doses = rng.choice(_DOSES)
        daily = tablets * strength * doses
        masses = [m for m in _MASSES if daily % m == 0 and 5 <= daily // m <= 100]
        if masses:
            break
    else:
        raise AssertionError("no dosage found")
    mass = rng.choice(masses)
    rate = daily // mass

    # Second route: milligrams per tablet from the stated grams, then the full course of tablets.
    strength_g = Fraction(strength, 1000)
    check(Fraction(rate * mass, doses) / (strength_g * 1000) == tablets, "tablet count mismatch")

    strength_text = f"{float(strength_g):g}"
    per_dose = Fraction(rate * mass, doses)
    candidates = {
        "treats-the-tablet-as-a-tenth-of-the-size": int(per_dose / (strength_g * 100)),
        "forgets-to-share-over-doses": int(Fraction(rate * mass) / strength),
        "multiplies-by-the-number-of-doses": int(Fraction(rate * mass * doses) / strength),
        "reads-the-grams-as-milligrams": int(per_dose / strength_g),
    }
    check_distinct(tablets, *candidates.values())

    who = rng.choice(_PATIENTS)
    text = (
        f"{who} with a body mass of {mass} kg is prescribed {rate} mg of a medicine per kilogram of body mass "
        f"each day, shared equally over {doses} doses. Each tablet contains {strength_text} g of the medicine. "
        f"How many tablets are taken in each dose?"
    )
    answer_text = str(tablets)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=tablets,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Daily dose: {rate} x {mass} = {rate * mass} mg; per dose: {rate * mass} / {doses} = {int(per_dose)} mg.",
            f"One tablet is {strength_text} g = {strength} mg, so {int(per_dose)} / {strength} = {tablets} tablets.",
        ],
        distractors=pick_distractors(answer_text, [str(v) for v in candidates.values()]),
    )


@generator(
    "exd_x_grams_dose",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_grams_dose(rng: random.Random, tier: str = "exam") -> dict:
    """Tablets per dose from a mg/kg/day rate and a tablet strength in grams. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
