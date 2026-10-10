"""Food energy units (Calories/kJ), senior ``acara-senior-v8-4``, domain ``measurement``, ``answer: number``.

One generator, ``food_energy_conversion`` (skill
``senior-use-units-of-energy-used-for-foods-including``, ACMEM032) --
converts between a food item's Calorie count and its kilojoule equivalent,
using the fixed rational factor ``Fraction(21, 5)`` (= 4.2) that the question
always states in its own text: "Using 1 Calorie is approximately 4.2
kilojoules". This is deliberately kept apart from the shipping neighbour
``senior-convert-from-one-unit-of-energy-to-another`` (``convert_energy_units``,
ACMEM034), which converts J/kJ/MJ/kWh and never mentions Calories or food.

``Fraction(21, 5)`` has denominator 5, so multiplying any integer Calorie
count by it and then scaling by 10 always yields an integer:
``C * 21/5 * 10 = C * 42``, an integer for integer ``C``. Every Calorie ->
kilojoule conversion in this module therefore terminates in EXACTLY 1
decimal place, never fewer -- displayed with ``stated_precision_string``
so the grading tolerance matches what is actually shown.

Three tiers, genuine reasoning escalation by DIRECTION and composition, not
magnitude:

- ``cal_to_kj`` (easy): Calories -> kilojoules, single multiplication.
- ``kj_to_cal`` (medium): kilojoules -> Calories, the reverse direction
  (division, or recognising the inverse relationship). The kJ figure shown
  is always derived from the SAME catalogue by the SAME multiplication, so
  dividing back is exact by construction.
- ``two_item_total`` (hard): two DIFFERENT food items' kJ figures given,
  asking for their COMBINED Calorie total -- convert-then-sum (or
  sum-then-convert), two conversions composed rather than one.

Calorie answers (medium and hard) are always integers and displayed with NO
decimal point (``fixed_point_string(..., 0)``, graded EXACTLY). The
kilojoule answer (easy) always displays exactly 1 decimal place. Every
misconception distractor is rounded to the SAME displayed precision as the
tier's own answer (``round_half_up`` then the matching formatter) so no
option is identifiable by its shape alone -- a raw, unrounded misconception
value can otherwise happen to land on an exact integer (e.g. the "divided
instead of multiplied" value for a Calorie count divisible by 21) and ship
as the only decimal-free option among decimal ones.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_pairwise_distinct,
    fixed_point_string,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

# Fixed rational conversion factor: 1 Calorie is approximately 4.2 kilojoules.
# Never a float, never a different rounding -- see module docstring for the
# algebraic proof this always terminates to exactly 1 decimal place.
FACTOR = Fraction(21, 5)

FACTOR_PHRASE = "Using 1 Calorie is approximately 4.2 kilojoules"

# The food-energy catalogue -- fixed, copied verbatim. Every entry is (name, calories); Calories are
# the construct-to-target quantity, kJ is always DERIVED by multiplying by FACTOR, never the other
# way around.
CATALOGUE: list[tuple[str, int]] = [
    ("a medium apple", 83),
    ("a medium banana", 105),
    ("a slice of white bread", 67),
    ("a can of regular soft drink (375 mL)", 140),
    ("a 50 g chocolate bar", 260),
    ("a boiled egg", 78),
    ("a cup of cooked rice", 160),
    ("a small baked potato", 85),
    ("a slice of cheese", 100),
    ("a tablespoon of peanut butter", 90),
    ("a cup of whole milk", 150),
    ("a small serving of hot chips", 260),
    ("a cooked chicken breast (100 g)", 165),
    ("a muesli bar", 120),
    ("a slice of pizza", 240),
    ("a cup of orange juice", 107),
    ("a doughnut", 240),
    ("a can of tuna in spring water", 107),
    ("a cup of ice cream", 260),
    ("an adult's approximate daily energy requirement", 2080),
]


def _capitalise(name: str) -> str:
    """Every catalogue name already carries its own article ("a medium
    apple", "an adult's ..."); capitalise the first letter only, never
    prefix "A "/"An " a second time."""
    return name[0].upper() + name[1:]


def _kj_of(cal: Fraction) -> Fraction:
    return cal * FACTOR


def _cal_from_kj_text(kj_text: str) -> Fraction:
    """Independent check helper: parse a RENDERED kJ figure (a string, not
    construction's own intermediate variable) back into a Fraction and
    divide by FACTOR -- the algebraic inverse of construction, which always
    multiplies. Exercised on the actual displayed text."""
    return Fraction(kj_text) / FACTOR


def _build_cal_to_kj(rng) -> dict:
    idx = rng.randrange(len(CATALOGUE))
    name, cal_int = CATALOGUE[idx]
    cal = Fraction(cal_int)
    kj = _kj_of(cal)
    answer_text = stated_precision_string(kj, 1)

    # Independent check: divide the RENDERED kJ text back by FACTOR and
    # recover the original integer Calorie count exactly.
    recomputed = _cal_from_kj_text(answer_text)
    check(recomputed == cal, f"cal_to_kj independent check failed: {recomputed} != {cal}")

    text = f"{_capitalise(name)} is about {cal_int} Calories. {FACTOR_PHRASE}, how many kilojoules is that?"
    latex = text

    forgot_to_convert = stated_precision_string(round_half_up(cal, 1), 1)
    divided_instead = stated_precision_string(round_half_up(cal / FACTOR, 1), 1)
    rounder_factor = stated_precision_string(round_half_up(cal * 4, 1), 1)

    check_pairwise_distinct(
        round_half_up(kj, 1),
        round_half_up(cal, 1),
        round_half_up(cal / FACTOR, 1),
        round_half_up(cal * 4, 1),
        msg="cal_to_kj misconception collides with the answer or another option",
    )

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(kj),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{cal_int} Calories x 4.2 = {answer_text} kilojoules.",
        ],
        distractors=pick_distractors(
            answer_text,
            [forgot_to_convert, divided_instead, rounder_factor],
            tolerance=stated_precision_tolerance(1),
        ),
    )


def _build_kj_to_cal(rng) -> dict:
    idx = rng.randrange(len(CATALOGUE))
    name, cal_int = CATALOGUE[idx]
    cal = Fraction(cal_int)
    kj = _kj_of(cal)
    kj_text = stated_precision_string(kj, 1)
    answer_text = fixed_point_string(cal_int, 0)

    # Independent check: divide the RENDERED kJ text back by FACTOR.
    recomputed = _cal_from_kj_text(kj_text)
    check(recomputed.denominator == 1, f"kj_to_cal independent check: {recomputed} is not an exact integer")
    check(recomputed == cal, f"kj_to_cal independent check failed: {recomputed} != {cal}")

    text = f"{_capitalise(name)} is about {kj_text} kilojoules of energy. {FACTOR_PHRASE}, how many Calories is that?"
    latex = text

    forgot_to_convert = fixed_point_string(int(round_half_up(kj, 0)), 0)
    multiplied_instead = fixed_point_string(int(round_half_up(kj * FACTOR, 0)), 0)
    rounder_factor = fixed_point_string(int(round_half_up(kj / 4, 0)), 0)

    check_pairwise_distinct(
        cal_int,
        int(round_half_up(kj, 0)),
        int(round_half_up(kj * FACTOR, 0)),
        int(round_half_up(kj / 4, 0)),
        msg="kj_to_cal misconception collides with the answer or another option",
    )

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(cal),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{kj_text} kilojoules / 4.2 = {answer_text} Calories.",
        ],
        distractors=pick_distractors(answer_text, [forgot_to_convert, multiplied_instead, rounder_factor]),
    )


def _build_two_item_total(rng) -> dict:
    i1, i2 = rng.sample(range(len(CATALOGUE)), 2)
    check(i1 != i2, "two_item_total drew the same catalogue item twice")
    name1, cal1_int = CATALOGUE[i1]
    name2, cal2_int = CATALOGUE[i2]
    cal1, cal2 = Fraction(cal1_int), Fraction(cal2_int)
    kj1, kj2 = _kj_of(cal1), _kj_of(cal2)
    kj1_text, kj2_text = stated_precision_string(kj1, 1), stated_precision_string(kj2, 1)
    total_cal = cal1 + cal2
    answer_text = fixed_point_string(int(total_cal), 0)

    # Independent check: divide BOTH rendered kJ figures back by FACTOR and
    # sum -- the algebraic inverse of "convert then sum" is "un-convert then
    # sum", exercised on the two displayed kJ strings.
    recomputed = _cal_from_kj_text(kj1_text) + _cal_from_kj_text(kj2_text)
    check(recomputed.denominator == 1, f"two_item_total independent check: {recomputed} is not an exact integer")
    check(recomputed == total_cal, f"two_item_total independent check failed: {recomputed} != {total_cal}")

    text = (
        f"{_capitalise(name1)} is about {kj1_text} kilojoules of energy and {name2} is about "
        f"{kj2_text} kilojoules of energy. {FACTOR_PHRASE}, what is the combined Calorie total of both?"
    )
    latex = text

    combined_kj = kj1 + kj2
    only_converted_one = fixed_point_string(int(round_half_up(cal1 + kj2, 0)), 0)
    summed_kj_forgot = fixed_point_string(int(round_half_up(combined_kj, 0)), 0)
    rounder_factor = fixed_point_string(int(round_half_up(combined_kj / 4, 0)), 0)

    check_pairwise_distinct(
        int(total_cal),
        int(round_half_up(cal1 + kj2, 0)),
        int(round_half_up(combined_kj, 0)),
        int(round_half_up(combined_kj / 4, 0)),
        msg="two_item_total misconception collides with the answer or another option",
    )

    return dict(
        text=text,
        latex=latex,
        answer=numeric_answer(total_cal),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{kj1_text} kilojoules / 4.2 = {int(cal1)} Calories; {kj2_text} kilojoules / 4.2 = {int(cal2)} Calories; "
            f"{int(cal1)} + {int(cal2)} = {answer_text} Calories.",
        ],
        distractors=pick_distractors(answer_text, [only_converted_one, summed_kj_forgot, rounder_factor]),
    )


@generator(
    "food_energy_conversion",
    level=11,
    presets={
        "easy": dict(mode="cal_to_kj"),
        "medium": dict(mode="kj_to_cal"),
        "hard": dict(mode="two_item_total"),
    },
)
def food_energy_conversion(rng, mode="cal_to_kj"):
    """Convert between a food item's Calorie count and its kilojoule
    equivalent using the fixed factor 1 Calorie ~= 4.2 kilojoules; see the
    module docstring for the three tiers' reasoning escalation."""
    if mode == "cal_to_kj":
        return _build_cal_to_kj(rng)
    if mode == "kj_to_cal":
        return _build_kj_to_cal(rng)
    if mode == "two_item_total":
        return _build_two_item_total(rng)
    raise ValueError(f"unknown mode: {mode!r}")
