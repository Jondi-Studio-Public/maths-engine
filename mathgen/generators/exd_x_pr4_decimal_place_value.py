"""Exam question on decimal place value: build a reading from its place-value description, then use it.

A digital scale reading is described out of order by what each place holds ("7 thousandths, 4 tens, 9 tenths
and 6 ones, and no hundredths"). Part (a) assembles the decimal, keeping the zero placeholder. Part (b) adds a
number of thousandths that carries into the hundredths. Part (c) asks how many whole hundredths the new reading
contains, which is a place-value regrouping rather than a read-off.

Built forwards from the digits, with every figure an exact `Fraction`, so each key is exact.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import floor

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    exact_decimal_string,
    generator,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)

_SETTINGS = (
    ("A digital freight scale at a Darwin depot", "a pallet of tiles"),
    ("A digital scale at a Perth hardware store", "a bag of cement"),
    ("A digital scale at a Hobart farm supply shop", "a sack of grain"),
    ("A digital scale at a Brisbane paint shop", "a drum of paint"),
    ("A digital scale at an Adelaide plumbing supplier", "a roll of copper pipe"),
)


_SINGULAR = {"tens": "ten", "ones": "one", "tenths": "tenth", "hundredths": "hundredth", "thousandths": "thousandth"}


def _count(n: int, unit: str) -> str:
    """ "1 ten" but "2 tens"."""
    return f"{n} {_SINGULAR[unit] if n == 1 else unit}"


def _dist(value: Fraction, misconception: str) -> Distractor:
    return Distractor(exact_decimal_string(value, 3), misconception, value)


@generator("exd_x_pr4_decimal_place_value", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr4_decimal_place_value(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "exd_x_pr4_decimal_place_value: only the exam tier exists")
    tens = rng.randint(1, 9)
    ones = rng.randint(1, 9)
    zero_place = rng.choice(("tenths", "hundredths"))
    tenths = 0 if zero_place == "tenths" else rng.randint(1, 9)
    hundredths = 0 if zero_place == "hundredths" else rng.randint(1, 9)
    thousandths = rng.randint(2, 9)
    for _ in attempts(1000, "a number of thousandths that carries"):
        extra = rng.randint(11, 98)
        if extra % 10 != 0 and (thousandths + extra) % 10 < thousandths:
            break

    clauses = [_count(thousandths, "thousandths"), _count(tens, "tens"), _count(ones, "ones")]
    if tenths:
        clauses.append(_count(tenths, "tenths"))
    if hundredths:
        clauses.append(_count(hundredths, "hundredths"))
    rng.shuffle(clauses)
    description = ", ".join(clauses[:-1]) + f" and {clauses[-1]}, and no {zero_place}"

    value = Fraction(10 * tens + ones) + Fraction(tenths, 10) + Fraction(hundredths, 100) + Fraction(thousandths, 1000)
    added = Fraction(extra, 1000)
    new_value = value + added
    hundredths_count = floor(new_value * 100)

    # Independent route: positional digits.
    check(Fraction(f"{tens}{ones}.{tenths}{hundredths}{thousandths}") == value, "assembled reading disagrees")
    check(hundredths_count == int(new_value * 100) and hundredths_count > 0, "hundredths count disagrees")

    a_text = stated_precision_string(value, 3)
    b_text = stated_precision_string(new_value, 3)
    c_text = str(hundredths_count)

    decimals = [tenths, hundredths, thousandths]
    decimals.remove(0)
    zero_dropped = Fraction(f"{tens}{ones}.{''.join(map(str, decimals))}")
    a_candidates = [
        _dist(zero_dropped, "zero-placeholder-dropped"),
        _dist(Fraction(f"{tens}{ones}.{hundredths}{tenths}{thousandths}"), "decimal-digits-in-wrong-places"),
        _dist(Fraction(f"{tens}{ones}.{thousandths}{hundredths}{tenths}"), "decimal-digits-in-wrong-places"),
    ]
    b_candidates = [
        _dist(value + Fraction(extra, 100), "thousandths-added-as-hundredths"),
        _dist(value + Fraction((thousandths + extra) % 10 - thousandths, 1000), "decimal-carry-lost"),
    ]
    frac_part = (new_value * 100) % 1
    c_candidates = [
        Distractor(str(floor(new_value * 10)), "counts-wrong-decimal-unit", floor(new_value * 10)),
        Distractor(str(floor(new_value * 1000)), "counts-wrong-decimal-unit", floor(new_value * 1000)),
        Distractor(str(floor(new_value * 100) % 10), "digit-read-for-count", floor(new_value * 100) % 10),
    ]
    if frac_part:
        c_candidates.append(
            Distractor(str(hundredths_count + 1), "rounds-instead-of-whole-units", hundredths_count + 1)
        )
    a_d = pick_distractors(a_text, a_candidates, 3)
    b_d = pick_distractors(b_text, b_candidates, 3)
    c_d = pick_distractors(c_text, c_candidates, 3)
    check(len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 2, "too few distractors")

    scale, load = rng.choice(_SETTINGS)
    stem = (
        f"{scale} shows the mass of {load} in kilograms. The reading has {description}. "
        f"Then {extra} thousandths of a kilogram are added to the load."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Write the first reading as a decimal number of kilograms.",
                answer_kind="number",
                answer=numeric_answer(value),
                answer_text=a_text,
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Tens {tens}, ones {ones}, tenths {tenths}, hundredths {hundredths}, thousandths {thousandths}.",
                    f"Reading = {a_text} kg.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Write the new reading as a decimal number of kilograms.",
                answer_kind="number",
                answer=numeric_answer(new_value),
                answer_text=b_text,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a + {extra}/1000",
                working=[f"{a_text} + {stated_precision_string(added, 3)} = {b_text} kg."],
                distractors=b_d,
            ),
            dict(
                prompt="How many whole hundredths of a kilogram are in the new reading?",
                answer_kind="number",
                answer=hundredths_count,
                answer_text=c_text,
                marks=1,
                sensible_range=(0, None),
                follow_through="floor(100*part_b)",
                working=[f"{b_text} kg is {new_value * 100} hundredths, so {c_text} whole hundredths."],
                distractors=c_d,
            ),
        ],
    )
