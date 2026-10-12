"""Exam question: food-label energy per 100 g, the energy of a serve, and what is left of a daily target.

A three-part chain. Part (a) scales the per-100 g figure on a label to the serve size, part (b) converts that
to Calories (1 Calorie = 4.2 kJ), and part (c) takes several serves off a daily target that is stated in
kilojoules, so the budget must be converted too. Parts (b) and (c) carry `follow_through`.

Built backwards from a whole number of Calories per serve (a multiple of 5, so the kilojoules are a whole
number), the serve size, and a daily target in whole hundreds of Calories, so every key is an integer.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase

from .exd_x_m6_support import TooFewDistractors, number_part, shows_no_key, tagged

_SERVES = (20, 25, 30, 40, 50, 60, 75, 80)
_FOODS = ("muesli bar", "packet of crackers", "serve of dried fruit mix", "bag of roasted nuts")
_STEMS = (
    "The label on a {food} says it has {E} kJ per 100 g, and one serve is {s} g.",
    "A nutrition panel on a {food} lists {E} kJ in every 100 g. A single serve is {s} g.",
)
_BUDGET = (
    "A student's daily energy target is {X} kJ. In one day the student eats {n} serves of this food.",
    "A hiker plans on {X} kJ of energy a day and eats {n} serves of this food during it.",
)


@generator("exd_x_m6_energy_serve", level=12, presets={"exam": dict(tier="exam")})
def exd_x_m6_energy_serve(rng: random.Random, tier: str = "exam") -> dict:
    """Energy of a serve in kJ, in Calories, and the Calories left of a daily target given in kJ."""
    check(tier == "exam", "exd_x_m6_energy_serve: only the exam tier exists")
    for _ in attempts(1000, "a label whose figures work out exactly"):
        try:
            question = _attempt(rng)
        except TooFewDistractors:
            continue
        if question is not None:
            return question


def _attempt(rng: random.Random) -> dict | None:
    m = rng.randint(24, 80)
    cal = 5 * m  # Calories in one serve
    kj = 21 * m  # the same energy in kilojoules (1 Calorie = 4.2 kJ)
    s = rng.choice(_SERVES)
    if (kj * 100) % s:
        return None
    per100 = kj * 100 // s
    if not 900 <= per100 <= 2400:
        return None
    target_cal = rng.randrange(1800, 2700, 100)
    target_kj = target_cal * 21 // 5
    if target_cal * 21 % 5:
        return None
    n = rng.randint(2, 6)
    left = target_cal - n * cal
    if left < 300:
        return None
    food = rng.choice(_FOODS)
    stem = phrase(rng, _STEMS, food=food, E=per100, s=s) + " " + phrase(rng, _BUDGET, X=target_kj, n=n)
    prompts = (
        "Find the energy in one serve, in kilojoules.",
        "Use 1 Calorie = 4.2 kJ to find the energy in one serve, in Calories.",
        "How many Calories of the daily target are left after those serves?",
    )
    if not shows_no_key([stem, *prompts], [kj, cal, left]):
        return None

    a_cands = [
        tagged(Fraction(per100), "per-100-g-not-scaled"),
        tagged(Fraction(per100 * s), "unit-factor-wrong-power"),
        tagged(Fraction(per100 * s, 10), "unit-factor-wrong-power"),
    ]
    b_cands = [
        tagged(Fraction(kj * 42, 10), "unit-conversion-wrong-way", 0),
        tagged(Fraction(kj), "forgot-unit-conversion"),
        tagged(Fraction(per100 * 10, 42), "per-100-g-not-scaled", 0),
    ]
    c_cands = [
        tagged(Fraction(target_kj - n * cal), "mixed-units-in-subtraction"),
        tagged(Fraction(target_kj - n * kj), "forgot-unit-conversion"),
        tagged(Fraction(n * cal)),  # the Calories eaten, not the Calories left
        tagged(Fraction(target_cal - cal)),  # takes off one serve only
    ]
    return dict(
        text=stem,
        parts=[
            number_part(
                prompts[0],
                Fraction(kj),
                a_cands,
                marks=1,
                working=[f"{s} g is {s}/100 of 100 g.", f"Energy = {per100} × {s} ÷ 100 = {kj} kJ."],
            ),
            number_part(
                prompts[1],
                Fraction(cal),
                b_cands,
                marks=1,
                follow_through="part_a*5/21",
                working=[f"{kj} ÷ 4.2 = {cal} Calories."],
            ),
            number_part(
                prompts[2],
                Fraction(left),
                c_cands,
                marks=2,
                follow_through=f"{target_kj}*5/21 - {n}*part_b",
                working=[
                    f"Target in Calories = {target_kj} ÷ 4.2 = {target_cal}.",
                    f"Eaten = {n} × {cal} = {n * cal} Calories.",
                    f"Left = {target_cal} - {n * cal} = {left} Calories.",
                ],
            ),
        ],
    )
