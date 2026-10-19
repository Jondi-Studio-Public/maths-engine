"""Exam-tier source for `senior-estimate-the-mass-of-different-objects`: the mass of a whole load.

A supplier loads a stated number of identical objects, each with a measured mass in grams, onto a vehicle with a
payload limit in tonnes. This is an estimation skill, so the learner first rounds one object's mass to a stated
precision, scales that up to the whole load, moves between grams, kilograms and tonnes, and then compares the load
with the limit:

* (a) the estimated mass of the whole load in kilograms, from the rounded mass of one object (2 marks);
* (b) the same mass in tonnes, to 3 decimal places (1 mark, follow-through from (a));
* (c) how many more kilograms the vehicle can carry (2 marks, follow-through from (a)).

The load is drawn first: the object, the count (a multiple that makes the rounded load a whole number of kilograms)
and the payload limit. The measured mass is the rounded mass plus a small offset that is never a half step, so the
rounding is unambiguous. Everything else is computed from those with exact fractions.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    check,
    draw_until,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)

# (singular for the stem, plural, mass of one in grams once rounded, smallest count, largest count, step,
#  the unit the learner rounds to, largest offset of the measured mass from the rounded mass)
_OBJECTS = [
    ("a tin of baked beans", "tins of baked beans", 450, 200, 900, 20, 10, 4),
    ("a 1 L carton of long-life milk", "cartons of long-life milk", 1000, 100, 600, 20, 100, 45),
    ("a house brick", "house bricks", 2800, 100, 340, 5, 100, 45),
    ("a bag of sugar", "bags of sugar", 1000, 150, 800, 10, 100, 45),
    ("a bottle of water", "bottles of water", 600, 200, 1000, 5, 100, 45),
    ("a paving block", "paving blocks", 4500, 60, 180, 2, 100, 45),
    ("a watermelon", "watermelons", 6500, 40, 130, 2, 100, 45),
    ("a bag of cement", "bags of cement", 20000, 20, 60, 1, 100, 45),
]
_VEHICLES = [
    "a ute with a payload limit of {t} tonnes",
    "a small truck with a payload limit of {t} tonnes",
    "a van that can carry at most {t} tonnes",
]
_OPENINGS = [
    "A supplier loads {n} {plural} onto {vehicle}. {one} has a mass of {m} g.",
    "{one} has a mass of {m} g. A delivery driver loads {n} {plural} onto {vehicle}.",
    "On a delivery run, {vehicle} is loaded with {n} {plural}. {one} has a mass of {m} g.",
]


def _cands(*pairs) -> list[Distractor]:
    """Tagged distractors for `(value, misconception)` pairs, skipping any value that is not a short decimal."""
    out = []
    for value, tag in pairs:
        value = Fraction(value)
        try:
            text = exact_decimal_string(value, 3)
        except ValueError:
            continue
        out.append(Distractor(text, tag, exact_value(value)))
    return out


def _tonnes_text(kg: int) -> str:
    return stated_precision_string(Fraction(kg, 1000), 3)


@generator("exd_x_sc5_mass_load", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc5_mass_load(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: scale one object's mass up to a load, convert g -> kg -> t and compare with a payload limit."""
    check(tier == "exam", f"unknown tier {tier!r}")
    one, plural, grams, lo, hi, step, unit, spread = rng.choice(_OBJECTS)

    def draw(r: random.Random):
        n = r.randrange(lo, hi + 1, step)
        payload_tenths = r.randint(8, 20)
        return n, payload_tenths

    def acceptable(d) -> bool:
        n, payload_tenths = d
        total_g = n * grams
        if total_g % 1000:
            return False  # keep the load a whole number of kilograms
        total_kg = total_g // 1000
        return 100 <= total_kg and total_kg + 50 <= payload_tenths * 100

    n, payload_tenths = draw_until(rng, draw, acceptable, what="a load that fits the vehicle")
    total_kg = n * grams // 1000
    payload_kg = payload_tenths * 100
    spare_kg = payload_kg - total_kg
    check(spare_kg >= 50, "load does not fit under the payload limit")
    t_text = stated_precision_string(Fraction(payload_tenths, 10), 1)
    # The measured mass: the rounded mass plus a nonzero offset smaller than half the rounding unit.
    offset = rng.choice([o for o in range(-spread, spread + 1) if o and 2 * abs(o) < unit])
    measured = grams + offset
    exact_kg = Fraction(n * measured, 1000)

    vehicle = rng.choice(_VEHICLES).format(t=t_text)
    opening = rng.choice(_OPENINGS).format(
        n=n,
        plural=plural,
        m=measured,
        vehicle=vehicle,
        one=one[0].upper() + one[1:],
    )

    noun = one.split(" ", 1)[1]
    a_text = str(total_kg)
    b_text = _tonnes_text(total_kg)
    c_text = str(spare_kg)

    payload_t = Fraction(payload_tenths, 10)
    a_cands = _cands(
        (exact_kg, "used-exact-values-not-rounded"),
        (n * grams, "forgot-to-convert-units"),
        (total_kg * 10, "conversion-power-of-ten-slip"),
        (Fraction(grams, 1000), "gave-amount-for-one-item"),
        (total_kg * 1000, "converted-wrong-direction"),
    )
    b_cands = _cands(
        (Fraction(total_kg, 100), "conversion-power-of-ten-slip"),
        (total_kg, "forgot-to-convert-units"),
        (Fraction(total_kg, 10000), "conversion-power-of-ten-slip"),
        (total_kg * 1000, "converted-wrong-direction"),
    )
    c_cands = _cands(
        (payload_t - total_kg, "mixed-units-in-arithmetic"),
        (payload_kg + total_kg, "added-instead-of-subtracted"),
        (payload_tenths * 10 - total_kg, "conversion-power-of-ten-slip"),
        (total_kg - payload_kg, "subtracted-in-wrong-order"),
    )
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(c_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 2, "too few distractors survive")

    return dict(
        text=opening,
        parts=[
            dict(
                prompt=f"Round the mass of one {noun} to the nearest {unit} g, then use it to estimate the total mass of the load, in kilograms.",
                answer_kind="number",
                answer=total_kg,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{measured} g to the nearest {unit} g is {grams} g.",
                    f"Mass in grams = {n} × {grams} = {n * grams} g.",
                    f"1 kg = 1000 g, so {n * grams} g = {a_text} kg.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Write your estimate from (a) in tonnes, correct to 3 decimal places.",
                answer_kind="number",
                answer=numeric_answer(Fraction(total_kg, 1000)),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_a/1000",
                working=[f"1 t = 1000 kg, so {a_text} kg = {b_text} t."],
                distractors=b_d,
            ),
            dict(
                prompt="Using your estimate from (a), how many more kilograms can the vehicle carry before it reaches its payload limit?",
                answer_kind="number",
                answer=spare_kg,
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{payload_kg} - part_a",
                working=[
                    f"Payload limit = {t_text} t = {payload_kg} kg.",
                    f"{payload_kg} - {a_text} = {c_text} kg.",
                ],
                distractors=c_d,
            ),
        ],
    )
