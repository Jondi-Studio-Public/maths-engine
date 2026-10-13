"""Exam question on rounding decimals: what true values could a rounded reading stand for?

A reading is shown rounded to the nearest tenth (or the nearest whole number), while the true value is stored
one place finer. Part (a) asks for the smallest stored value that rounds to the reading, part (b) the largest,
and part (c) uses the smallest to find the least a lighter or shorter second item could be.

The reading is drawn first and the bounds follow from it exactly: half a rounding unit below, and one finest
step short of half a unit above (the halfway value itself rounds up).
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

# (opening sentence, quantity noun, unit, name of the second item, difference sentence, plausible reading range)
_SETTINGS = (
    (
        "A digital scale at a Sydney post office shows a parcel as {x} kg, rounded to the nearest {place}.",
        "mass",
        "kg",
        "second parcel",
        "A second parcel is {d} kg lighter than the first.",
        (4, 60),
    ),
    (
        "A rain gauge near Cairns reports {x} mm for one day, rounded to the nearest {place}.",
        "rainfall",
        "mm",
        "next day's rainfall",
        "The next day's rainfall is {d} mm less than that day's.",
        (3, 95),
    ),
    (
        "A tape measure shows a timber plank as {x} m, rounded to the nearest {place}.",
        "length",
        "m",
        "second plank",
        "A second plank is {d} m shorter than the first.",
        (3, 9),
    ),
    (
        "A fuel pump in Alice Springs shows {x} L, rounded to the nearest {place}.",
        "volume",
        "L",
        "second tank",
        "A second tank holds {d} L less than the first.",
        (20, 120),
    ),
)


@generator("exd_x_pr4_round_decimal_bounds", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr4_round_decimal_bounds(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "exd_x_pr4_round_decimal_bounds: only the exam tier exists")
    places = rng.choice((0, 1))  # the reading is rounded to this many decimal places
    unit = Fraction(1, 10**places)
    finest = unit / 10
    place_name = "whole number" if places == 0 else "tenth"
    dp = places + 1  # the stored value has one more decimal place
    opening, noun, unit_word, second_noun, diff_sentence, (lo, hi) = rng.choice(_SETTINGS)
    for _ in attempts(1000, "a reading with room for the difference"):
        reading = unit * rng.randint(int(lo / unit), int(hi / unit))
        diff = Fraction(rng.randint(2, 9), 10) + (Fraction(rng.randint(0, 3), 1) if rng.random() < 0.5 else 0)
        if reading - unit / 2 - diff >= Fraction(1, 2):
            break
    lowest = reading - unit / 2
    highest = reading + unit / 2 - finest
    second = lowest - diff

    # Independent route: scan the stored grid and keep what rounds to the reading.
    grid = [finest * m for m in range(int((reading - unit) / finest), int((reading + unit) / finest) + 1)]
    rounds_to = [v for v in grid if round_half_up(v, places) == reading]
    check(min(rounds_to) == lowest and max(rounds_to) == highest, "bounds disagree with the scan")
    check(second > 0, "second value must be positive")

    reading_text = stated_precision_string(reading, places)
    low_text, high_text, second_text = (stated_precision_string(v, dp) for v in (lowest, highest, second))

    def d(v, misconception):
        return Distractor(stated_precision_string(v, dp), misconception, v)

    a_candidates = [
        d(reading, "rounded-value-taken-as-bound"),
        d(reading - unit, "bound-off-by-whole-unit"),
        d(reading - unit / 2 + finest, None),
        d(highest, "uses-wrong-bound"),
    ]
    b_candidates = [
        d(reading + unit / 2, "upper-bound-includes-boundary"),
        d(reading, "rounded-value-taken-as-bound"),
        d(reading + unit, "bound-off-by-whole-unit"),
        d(lowest, "uses-wrong-bound"),
    ]
    c_candidates = [
        d(reading - diff, "rounded-value-taken-as-bound"),
        d(highest - diff, "uses-wrong-bound"),
        d(lowest + diff, None),
    ]
    a_d = pick_distractors(low_text, a_candidates, 3, tolerance=finest / 2)
    b_d = pick_distractors(high_text, b_candidates, 3, tolerance=finest / 2)
    c_d = pick_distractors(second_text, c_candidates, 3, tolerance=finest / 2)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors")

    stem = (
        opening.format(x=reading_text, place=place_name)
        + f" The true {noun} is stored to {dp} decimal place{'s' if dp > 1 else ''}. "
        + diff_sentence.format(d=stated_precision_string(diff, 1))
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"What is the smallest true value, in {unit_word}, that the display would show as this reading?",
                answer_kind="number",
                answer=numeric_answer(lowest),
                answer_text=low_text,
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Half a rounding unit below {reading_text} is {low_text}, and {low_text} rounds up to {reading_text}."
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"What is the largest true value, in {unit_word}, that the display would show as this reading?",
                answer_kind="number",
                answer=numeric_answer(highest),
                answer_text=high_text,
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{stated_precision_string(reading + unit / 2, dp)} would round up to the next reading, "
                    f"so the largest is {high_text}."
                ],
                distractors=b_d,
            ),
            dict(
                prompt=f"Using your answer to part (a), what is the smallest possible true value of the {second_noun}, in {unit_word}?",
                answer_kind="number",
                answer=numeric_answer(second),
                answer_text=second_text,
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a - {diff.numerator}/{diff.denominator}",
                working=[f"{low_text} - {stated_precision_string(diff, 1)} = {second_text}."],
                distractors=c_d,
            ),
        ],
    )
