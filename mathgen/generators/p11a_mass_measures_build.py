"""Mass estimation + mass-unit review (ACMEM026, ACMEM098). Both share ONE mass-object catalogue
(``_MACRO``/``_MICRO``/``_MEGA``/``_MILLI`` below) so the two skills are consistent with each other by
construction.

``estimate_mass_reference`` (``senior-estimate-the-mass-of-different-objects``,
mass: an mcq over a real-world reference object, the correctly-scaled numeral,
and three deliberately WRONG-ORDER-OF-MAGNITUDE options (x10, /10, x100 of the
same numeral, same unit). Difficulty is the asked UNIT:

- easy: a MACRO object (natural scale already kilograms) asked in KILOGRAMS.
- medium: a MICRO object (natural scale smaller than a kilogram) asked in
  GRAMS.
- hard: either half of the catalogue, asked in the unit that is NOT the
  object's own natural scale (grams for a macro object, kilograms for a
  micro one) -- a forced x1000/÷1000 conversion.

``review_metric_units_of_mass`` (``senior-eview-metric-units-of-mass-and-
weight-their``, ACMEM098) -- deliberately NOT a typed g<->kg conversion (that
is the shipping skill ``convert-metric-mass``'s own job). Instead an
``mcq`` skill over the WHOLE mass-unit ladder (mg, g, kg, t):

- easy (``unit_choice``): "Which unit is the most appropriate choice for
  measuring the mass of {object}?" -- a classification judgement over four
  qualitatively different scales, no conversion arithmetic. Construct-to-
  target: the correct unit is drawn UNIFORMLY from {mg, g, kg, t} first, then
  an object is drawn from that unit's own catalogue -- never the reverse,
  which would give kg/g roughly 45% each and mg/t under 10% each.
- medium (``adjacent_relation``): "How many {smaller} are there in {N}
  {larger}?" for one of the three ADJACENT pairs on the ladder (mg-g, g-kg,
  kg-t) -- a units-FACT question about the ladder itself (x1000), no object,
  no "Convert" verb.
- hard (``equivalence``): "Which of these has the same mass as {N} {unit}?"
  for one of the two NON-adjacent pairs (mg, kg) or (g, t), a x1,000,000
  factor, direction drawn 50/50 (small-unit value stated and big-unit
  equivalent asked, or vice versa).

No rendered row of ``review_metric_units_of_mass`` ever matches the shape
"Convert ... to ..." -- see ``tests/test_p11a_mass_measures_build.py``'s
direct substring assertion.

Every option in every tier of both skills is formatted to the SAME number of
decimal places (the smallest of the four candidate values, always the /10 or
tenth-scale one, decides how many places are needed) via the single shared
``_format_options`` helper -- ONE local helper this module shares between
both topics, matching ``mathgen/generators/p10b_estimate_lengths_build.py``'s
own ``_format_options``/``_exact_places_needed`` shape.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_conversion,
    check_pairwise_distinct,
    fixed_point_string,
    generator,
    latex_answer,
    pick_distractors,
)

# The shared mass-object catalogue. Every canonical value is stored ONCE. ``_MACRO``/``_MEGA`` are
# in kilograms; ``_MICRO``/``_MILLI`` are in grams (so each table's own natural unit needs no
# runtime conversion at that table's own tier). Fixed.

_MACRO: list[tuple[str, Fraction]] = [
    ("a bag of sugar", Fraction(1)),
    ("a house brick", Fraction("2.5")),
    ("a laptop", Fraction(2)),
    ("a bicycle", Fraction(12)),
    ("a microwave oven", Fraction(12)),
    ("an office chair", Fraction(15)),
    ("a bag of cement", Fraction(20)),
    ("a full school backpack", Fraction(8)),
    ("an adult cat", Fraction("4.5")),
    ("a medium dog", Fraction(18)),
    ("a newborn baby", Fraction("3.5")),
    ("an adult person", Fraction(70)),
    ("a car battery", Fraction(15)),
    ("a queen mattress", Fraction(30)),
    ("an empty bathtub", Fraction(25)),
    ("an adult kangaroo", Fraction(55)),
    ("an emu", Fraction(40)),
    ("a washing machine", Fraction(70)),
    ("an empty wheelbarrow", Fraction(25)),
    ("a golden retriever", Fraction(30)),
    ("a mountain bike", Fraction(14)),
    ("a bar fridge", Fraction(30)),
]

_MICRO: list[tuple[str, Fraction]] = [
    ("a smartphone", Fraction(180)),
    ("a paperback novel", Fraction(200)),
    ("an apple", Fraction(150)),
    ("a banana", Fraction(120)),
    ("a chocolate bar", Fraction(50)),
    ("a deck of cards", Fraction(100)),
    ("a tennis ball", Fraction(58)),
    ("a golf ball", Fraction(46)),
    ("a chicken egg", Fraction(60)),
    ("a slice of bread", Fraction(30)),
    ("a AA battery", Fraction(25)),
    ("a pair of scissors", Fraction(70)),
    ("a computer mouse", Fraction(100)),
    ("a wallet", Fraction(80)),
    ("a bar of soap", Fraction(100)),
    ("a metal teaspoon", Fraction(15)),
    ("a table tennis ball", Fraction(3)),
    ("a matchbox with matches", Fraction(20)),
    ("a compact disc", Fraction(16)),
    ("a smartphone case", Fraction(30)),
    ("an LED light bulb", Fraction(50)),
    ("a pack of chewing gum", Fraction(20)),
]

_MEGA: list[tuple[str, Fraction]] = [
    ("a family car", Fraction(1500)),
    ("a large SUV", Fraction(2500)),
    ("an adult elephant", Fraction(5000)),
    ("a loaded shipping container", Fraction(20000)),
    ("a city bus", Fraction(12000)),
    ("an orca (killer whale)", Fraction(5500)),
    ("a fire engine", Fraction(15000)),
    ("a loaded semi-trailer truck", Fraction(30000)),
    ("a double-decker bus", Fraction(12000)),
    ("an adult hippopotamus", Fraction(1500)),
    ("a garbage truck", Fraction(10000)),
    ("a small aeroplane", Fraction(1000)),
]

_MILLI: list[tuple[str, Fraction]] = [
    ("a grain of rice", Fraction(1, 50)),
    ("a mosquito", Fraction(1, 400)),
    ("an eyelash", Fraction(1, 1000)),
    ("a raindrop", Fraction(1, 20)),
]

# The metric mass ladder, mg per one of each unit -- ONE independent table
# shared by every `check_conversion` call in this module (both topics),
# deliberately never used by construction's own direct arithmetic (each
# `_build_*` function computes `correct` by explicit x1000/÷1000/x1e6
# multiplication, then re-derives it a second way through this table).
_MG_PER_UNIT = {
    "mg": Fraction(1),
    "g": Fraction(1_000),
    "kg": Fraction(1_000_000),
    "t": Fraction(1_000_000_000),
}

_UNIT_WORD_MASS = {"kg": "kilograms", "g": "grams"}


def _exact_places_needed(value: Fraction) -> int:
    """The fewest decimal places that display `value` exactly."""
    places = 0
    while (value * (10**places)).denominator != 1:
        places += 1
    return places


def _format_options(correct: Fraction, x10: Fraction, tenth: Fraction, x100: Fraction, suffix: str):
    """Format all four option values to the SAME number of decimal places --
    the count `tenth` (always the smallest, and the one most likely to need
    an extra digit) needs to display exactly -- so decimal-digit count never
    tells a learner which option is correct. Shared, module-scope, between
    `estimate_mass_reference` and `review_metric_units_of_mass`."""
    places = _exact_places_needed(tenth)

    def fmt(value: Fraction) -> str:
        scaled = value * (10**places)
        check(scaled.denominator == 1, f"{value} does not terminate exactly within {places} places")
        return f"{fixed_point_string(int(scaled), places)}{suffix}"

    return fmt(correct), fmt(x10), fmt(tenth), fmt(x100)


# ===========================================================================
# Skill A: estimate_mass_reference (ACMEM026)
# ===========================================================================


def _build_estimate_mass(rng, tier: str) -> dict:
    if tier == "easy":
        name, canonical = rng.choice(_MACRO)
        base_unit, ask_unit = "kg", "kg"
    elif tier == "medium":
        name, canonical = rng.choice(_MICRO)
        base_unit, ask_unit = "g", "g"
    elif tier == "hard":
        is_macro = rng.choice([True, False])
        if is_macro:
            name, canonical = rng.choice(_MACRO)
            base_unit, ask_unit = "kg", "g"
        else:
            name, canonical = rng.choice(_MICRO)
            base_unit, ask_unit = "g", "kg"
    else:
        raise ValueError(f"unknown tier {tier!r}")

    if base_unit == ask_unit:
        correct = canonical
    elif base_unit == "kg" and ask_unit == "g":
        correct = canonical * 1000
    else:  # base_unit == "g" and ask_unit == "kg"
        correct = canonical / Fraction(1000)

    check_conversion(canonical, base_unit, ask_unit, correct, _MG_PER_UNIT, label="estimate-mass unit conversion")

    x10 = correct * 10
    tenth = correct / 10
    x100 = correct * 100
    check_pairwise_distinct(correct, x10, tenth, x100, msg="an order-of-magnitude distractor collided with the answer")

    correct_text, x10_text, tenth_text, x100_text = _format_options(correct, x10, tenth, x100, f" {ask_unit}")
    distractors = pick_distractors(correct_text, [x10_text, tenth_text, x100_text])
    check(len(distractors) == 3, "an order-of-magnitude distractor collapsed onto another option")

    text = f"What is the best estimate for the mass of {name}? Give your answer in {_UNIT_WORD_MASS[ask_unit]}."
    latex = f"plain:{text}"
    working = [
        f"{name} has a mass of around {canonical} {base_unit}.",
        f"{canonical} {base_unit} expressed in {_UNIT_WORD_MASS[ask_unit]} is {correct_text}.",
    ]
    return dict(
        text=text,
        latex=latex,
        answer=correct_text,
        answer_text=correct_text,
        answer_latex=latex_answer(correct_text, plain=True),
        working=working,
        distractors=distractors,
    )


@generator(
    "estimate_mass_reference",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def estimate_mass_reference(rng, tier="easy"):
    """Best mcq estimate of a real-world object's mass, among the
    correctly-scaled numeral and its x10/x0.1/x100 order-of-magnitude
    distractors. `tier` selects the object pool and the asked unit -- see
    the module docstring."""
    return _build_estimate_mass(rng, tier)


# ===========================================================================
# Skill B: review_metric_units_of_mass (ACMEM098)
# ===========================================================================

_ALL_UNITS = ("mg", "g", "kg", "t")
_UNIT_CHOICE_SOURCE = {"mg": _MILLI, "g": _MICRO, "kg": _MACRO, "t": _MEGA}

# Adjacent pairs on the ladder (smaller, larger) -- always a x1000 factor.
_ADJACENT_PAIRS = [("mg", "g"), ("g", "kg"), ("kg", "t")]

# Non-adjacent (two-step) pairs -- always a x1,000,000 factor.
_EQUIVALENCE_PAIRS = [("mg", "kg"), ("g", "t")]


def _build_unit_choice(rng) -> dict:
    """Construct-to-target: draw the correct unit UNIFORMLY first, then draw
    an object from that unit's own catalogue -- never the reverse (module
    docstring)."""
    unit = rng.choice(_ALL_UNITS)
    name, _value = rng.choice(_UNIT_CHOICE_SOURCE[unit])

    distractors = [u for u in _ALL_UNITS if u != unit]
    check(len(distractors) == 3, "unit_choice distractor count wrong")
    check_pairwise_distinct(unit, *distractors, msg="unit_choice options collided")

    text = f"Which unit is the most appropriate choice for measuring the mass of {name}?"
    latex = f"plain:{text}"
    working = [f"{name} is best measured in {unit} -- its own natural scale on the metric mass ladder (mg, g, kg, t)."]
    return dict(
        text=text,
        latex=latex,
        answer=unit,
        answer_text=unit,
        answer_latex=latex_answer(unit, plain=True),
        working=working,
        distractors=distractors,
    )


def _build_adjacent_relation(rng) -> dict:
    smaller, larger = rng.choice(_ADJACENT_PAIRS)
    # Widened to {1..9} (3 x 9 = 27) -- still small single-digit multipliers (the tier's own
    # one-step-fact reasoning is unchanged), genuinely widening the draw space rather than padding a
    # test to accept 15 (widen the generator, do not pin the pool).
    n = rng.choice(range(1, 10))
    correct = Fraction(n) * 1000
    check_conversion(Fraction(n), larger, smaller, correct, _MG_PER_UNIT, label="adjacent mass-unit relation")

    x10 = correct * 10
    tenth = correct / 10
    x100 = correct * 100
    check_pairwise_distinct(correct, x10, tenth, x100, msg="adjacent-relation distractor collided with the answer")

    correct_text, x10_text, tenth_text, x100_text = _format_options(correct, x10, tenth, x100, f" {smaller}")
    distractors = pick_distractors(correct_text, [x10_text, tenth_text, x100_text])
    check(len(distractors) == 3, "an adjacent-relation distractor collapsed onto another option")

    text = f"How many {smaller} are there in {n} {larger}?"
    latex = f"plain:{text}"
    working = [
        f"1 {larger} = 1000 {smaller}.",
        f"{n} {larger} = {n} x 1000 {smaller} = {correct_text}.",
    ]
    return dict(
        text=text,
        latex=latex,
        answer=correct_text,
        answer_text=correct_text,
        answer_latex=latex_answer(correct_text, plain=True),
        working=working,
        distractors=distractors,
    )


def _build_equivalence(rng) -> dict:
    small_unit, big_unit = rng.choice(_EQUIVALENCE_PAIRS)
    n = rng.choice(range(100, 1000, 100))  # 100, 200, ..., 900
    big_value = Fraction(n, 1_000_000)
    check_conversion(Fraction(n), small_unit, big_unit, big_value, _MG_PER_UNIT, label="mass equivalence")

    direction = rng.choice(["small_given", "big_given"])
    if direction == "small_given":
        correct = big_value
        ask_unit = big_unit
        text = f"Which of these has the same mass as {n} {small_unit}?"
        working = [
            f"1 {big_unit} = 1,000,000 {small_unit}, so {n} {small_unit} = {n} / 1,000,000 {big_unit}.",
        ]
    else:
        scaled = big_value * 10_000
        check(scaled.denominator == 1, f"{big_value} does not terminate at 4 decimal places")
        big_value_text = f"{fixed_point_string(int(scaled), 4)} {big_unit}"
        correct = Fraction(n)
        ask_unit = small_unit
        text = f"Which of these has the same mass as {big_value_text}?"
        working = [
            f"1 {big_unit} = 1,000,000 {small_unit}, so {big_value_text} = {n} {small_unit}.",
        ]

    x10 = correct * 10
    tenth = correct / 10
    x100 = correct * 100
    check_pairwise_distinct(correct, x10, tenth, x100, msg="equivalence distractor collided with the answer")

    correct_text, x10_text, tenth_text, x100_text = _format_options(correct, x10, tenth, x100, f" {ask_unit}")
    distractors = pick_distractors(correct_text, [x10_text, tenth_text, x100_text])
    check(len(distractors) == 3, "an equivalence distractor collapsed onto another option")

    latex = f"plain:{text}"
    return dict(
        text=text,
        latex=latex,
        answer=correct_text,
        answer_text=correct_text,
        answer_latex=latex_answer(correct_text, plain=True),
        working=working,
        distractors=distractors,
    )


def _build_review_metric_units_of_mass(rng, tier: str) -> dict:
    if tier == "easy":
        return _build_unit_choice(rng)
    if tier == "medium":
        return _build_adjacent_relation(rng)
    if tier == "hard":
        return _build_equivalence(rng)
    raise ValueError(f"unknown tier {tier!r}")


@generator(
    "review_metric_units_of_mass",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def review_metric_units_of_mass(rng, tier="easy"):
    """Review metric units of mass across the whole mg/g/kg/t ladder: an
    appropriate-unit classification (easy), an adjacent-unit relation fact
    (medium), or a non-adjacent equivalence recognition (hard) -- never a
    typed g<->kg conversion (that is `convert-metric-mass`'s own kept
    kernel; see the module docstring)."""
    return _build_review_metric_units_of_mass(rng, tier)
