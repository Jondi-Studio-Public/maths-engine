"""Exam sources of `convert-metric-length`, `convert-metric-capacity` and `convert-metric-mass`: a total in one
unit shared out in portions given in another.

The practice tiers convert a value, or add two values given in different units. An exam question is a short
multi-step problem set in a real setting: (a) express the total in the smaller unit, (b) work out how many whole
portions it makes (a division, rounded down because only complete portions count), (c) find what is left over and
give it in the larger unit, which is a second conversion in the other direction.

All three topics share one builder; only the units and the settings differ.

## Construction

The portion size is drawn first, then a number of whole portions and a leftover, so the total is built backwards
and the leftover is never zero. The total is shown to at most 2 decimal places of the larger unit, so every
conversion is exact and the answers are exact decimals.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

# Each setting: (larger unit, smaller unit, units of the smaller per larger, intro, portion sentence,
#   part (a) question, part (b) question, part (c) question, portion sizes in the smaller unit)
_SETTINGS = {
    "capacity": [
        (
            "L",
            "mL",
            1000,
            "A school canteen urn holds {big} L of cordial.",
            "Each cup is filled with {s} mL.",
            "Write the capacity of the urn in millilitres.",
            "How many full cups can be filled?",
            "How much cordial is left in the urn after that, in litres?",
            [150, 180, 200, 220, 250, 300, 350, 400],
        ),
        (
            "L",
            "mL",
            1000,
            "A rainwater tank at a camp holds {big} L of water.",
            "Each drink bottle is filled with {s} mL.",
            "Write the amount of water in millilitres.",
            "How many full bottles can be filled?",
            "How much water is left in the tank after that, in litres?",
            [450, 500, 600, 750, 800],
        ),
        (
            "L",
            "mL",
            1000,
            "A pot of pumpkin soup holds {big} L.",
            "Each bowl takes {s} mL.",
            "Write the amount of soup in millilitres.",
            "How many full bowls can be served?",
            "How much soup is left in the pot after that, in litres?",
            [240, 250, 280, 300, 320, 350],
        ),
    ],
    "mass": [
        (
            "kg",
            "g",
            1000,
            "A baker has {big} kg of flour.",
            "Each loaf uses {s} g of flour.",
            "Write the mass of the flour in grams.",
            "How many whole loaves can be made?",
            "How much flour is left over, in kilograms?",
            [350, 400, 450, 500, 600, 750],
        ),
        (
            "kg",
            "g",
            1000,
            "A farmer picks {big} kg of mandarins.",
            "Each bag holds {s} g.",
            "Write the mass of the mandarins in grams.",
            "How many full bags can be packed?",
            "What mass of mandarins is left over, in kilograms?",
            [800, 900, 1200, 1500, 2500],
        ),
        (
            "kg",
            "g",
            1000,
            "A gardener buys {big} kg of potting mix.",
            "Each pot needs {s} g.",
            "Write the mass of the potting mix in grams.",
            "How many pots can be filled completely?",
            "What mass of potting mix is left over, in kilograms?",
            [280, 320, 350, 400, 450, 650],
        ),
    ],
    "length": [
        (
            "m",
            "cm",
            100,
            "A roll of ribbon is {big} m long.",
            "Each bow uses {s} cm of ribbon.",
            "Write the length of the ribbon in centimetres.",
            "How many whole bows can be made?",
            "How much ribbon is left over, in metres?",
            [35, 40, 45, 60, 75, 80],
        ),
        (
            "m",
            "cm",
            100,
            "A carpenter has a timber plank {big} m long.",
            "Each shelf piece is {s} cm long.",
            "Write the length of the plank in centimetres.",
            "How many whole shelf pieces can be cut?",
            "How much plank is left over, in metres?",
            [30, 45, 55, 65, 90, 120],
        ),
        (
            "cm",
            "mm",
            10,
            "A strip of sticky tape is {big} cm long.",
            "Each label needs {s} mm of tape.",
            "Write the length of the tape in millimetres.",
            "How many whole labels can be made?",
            "How much tape is left over, in centimetres?",
            [35, 45, 55, 65, 85, 95],
        ),
        (
            "km",
            "m",
            1000,
            "A bush walking track is {big} km long.",
            "It is divided into sections of {s} m.",
            "Write the length of the track in metres.",
            "How many whole sections are there?",
            "How long is the leftover part of the track, in kilometres?",
            [250, 300, 400, 450, 600, 750],
        ),
    ],
}


def _dec(value: Fraction) -> str:
    return exact_decimal_string(value, 6)


def _build(rng, kind: str, tier: str):
    check(tier == "exam", f"unknown tier {tier!r}")
    big_unit, small_unit, factor, intro, item, qa, qb, qc, sizes = rng.choice(_SETTINGS[kind])
    for _ in attempts(1000, "a total, portion and leftover with distinct answers"):
        s = rng.choice(sizes)
        portions = rng.randint(4, 40)
        leftover = rng.randint(1, s - 1)
        if leftover % 5:  # keep the total to two decimal places of the larger unit when the factor is 1000 or 100
            leftover -= leftover % 5
        if leftover <= 0:
            continue
        total_small = portions * s + leftover
        big = Fraction(total_small, factor)
        if big.denominator not in (1, 2, 4, 5, 10, 20, 25, 50, 100):
            continue
        big_text = _dec(big)
        if len(big_text.split(".")[-1]) > 2 and "." in big_text:
            continue
        key_a, key_b, key_c = total_small, portions, Fraction(leftover, factor)
        places = len(str(factor)) - 1  # the leftover is a whole number of the smaller unit: 3, 2 or 1 places
        shown = {big_text, str(s), "1", str(places)}
        if {str(key_a), str(key_b), stated_precision_string(key_c, places)} & shown:
            continue
        break

    a_text, b_text, c_text = str(key_a), str(key_b), stated_precision_string(key_c, places)
    stem = f"{intro.format(big=big_text)} {item.format(s=s)}"
    p_a, p_b = qa, f"Using your answer to (a), {qb[0].lower()}{qb[1:]}"
    p_c = f"Using your answers to (a) and (b), {qc[0].lower()}{qc[1:-1]}, correct to {places} decimal place{'s' if places > 1 else ''}?"

    def tagged(value, ident=None, places=0):
        """The wrong value shown to the key's own places (a different format would give the key away); None if it is 0."""
        shown_value = round_half_up(Fraction(value), places)
        if shown_value <= 0:
            return None
        text = stated_precision_string(shown_value, places)
        return Distractor(text, ident, shown_value) if ident else text

    a_cands = [
        tagged(key_a * 10, "unit-factor-wrong-power"),
        tagged(Fraction(key_a, 10), "unit-factor-wrong-power"),  # 100 per unit instead of 1000 (shown to whole units)
        tagged(key_a * 100, "unit-factor-wrong-power"),
    ]
    near = Fraction(key_a, s)
    b_cands = [
        Distractor(str(math.ceil(near)), "rounded-count-up", math.ceil(near)),
        str(round(near)),
        Distractor(
            str(math.floor(Fraction(key_a * 10, s))), "unit-factor-wrong-power", math.floor(Fraction(key_a * 10, s))
        ),
    ]
    c_cands = [
        tagged(leftover, "leftover-not-converted", places),
        tagged(Fraction(leftover, factor * 10), "unit-factor-wrong-power", places),
        tagged(Fraction(s - leftover, factor), None, places),  # what is still needed for one more portion
    ]
    a_cands = [c for c in a_cands if c is not None]
    c_cands = [c for c in c_cands if c is not None]
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=stated_precision_tolerance(0))
    b_d = pick_distractors(b_text, b_cands, 3)
    c_d = pick_distractors(c_text, c_cands, 3, tolerance=stated_precision_tolerance(places))
    check(len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 2, "too few distractors survive")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p_a,
                answer_kind="number",
                answer=key_a,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=1,
                # A total below one portion floors (b) to 0 and would hand (b) over as free follow-through, so
                # an (a) that small carries nothing forward. The key is at least four portions.
                sensible_range=(s - 1, None),
                working=[f"1 {big_unit} = {factor} {small_unit}, so {big_text} × {factor} = {a_text} {small_unit}."],
                distractors=a_d,
            ),
            dict(
                prompt=p_b,
                answer_kind="number",
                answer=key_b,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"floor(part_a/{s})",
                working=[f"{a_text} ÷ {s} = {float(near):.2f}.", f"Only whole portions count, so {b_text}."],
                distractors=b_d,
            ),
            dict(
                prompt=p_c,
                answer_kind="number",
                answer=numeric_answer(key_c),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"(part_a - {s}*part_b)/{factor}",
                working=[
                    f"{b_text} × {s} = {portions * s} {small_unit} used.",
                    f"{a_text} - {portions * s} = {leftover} {small_unit} left, which is {c_text} {big_unit}.",
                ],
                distractors=c_d,
            ),
        ],
    )


@generator("exd_x_m1_metric_length", level=6, presets={"exam": {"tier": "exam"}})
def exd_x_m1_metric_length(rng, tier: str = "exam"):
    return _build(rng, "length", tier)


@generator("exd_x_m1_metric_capacity", level=6, presets={"exam": {"tier": "exam"}})
def exd_x_m1_metric_capacity(rng, tier: str = "exam"):
    return _build(rng, "capacity", tier)


@generator("exd_x_m1_metric_mass", level=6, presets={"exam": {"tier": "exam"}})
def exd_x_m1_metric_mass(rng, tier: str = "exam"):
    return _build(rng, "mass", tier)
