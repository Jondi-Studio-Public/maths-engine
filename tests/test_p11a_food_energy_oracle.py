"""Independent oracle for the `p11a_food_energy` questions: it re-derives each answer from the rendered question
text.

Parses ONLY the rendered question (``question.text``, ``question.answer_text``, ``question.distractors``) and
re-derives the answer from scratch. Imports nothing from ``mathgen.generators.p11a_food_energy_build`` -- not
the catalogue, not the conversion factor, not a formatting helper. The Calorie-to-kilojoule factor itself is
extracted from the rendered text by regex, never hardcoded to 21/5, so this oracle would disagree with a
generator that stated "4.2" but silently computed with a different number.

Two independent checks:

1. **Plausibility oracle**: a real-world Calorie range for every food item
   the skill can render, authored fresh from general nutrition knowledge in
   this file -- not copied from, or derived by reading, the generator's own
   catalogue.
2. **Value oracle**: re-derives the correct Calorie/kilojoule answer from the
   question text alone, via the SAME algebraic inverse
   (divide the stated kilojoule figure(s) by the stated factor), and asserts
   it matches the rendered ``answer_text`` exactly.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

TOPIC = "food_energy_conversion"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

_FACTOR_RE = re.compile(r"Using 1 Calorie is approximately ([\d.]+) kilojoules")

_EASY_RE = re.compile(
    r"^(.+?) is about (\d+) Calories\. Using 1 Calorie is approximately [\d.]+ kilojoules, "
    r"how many kilojoules is that\?$"
)
_MEDIUM_RE = re.compile(
    r"^(.+?) is about ([\d.]+) kilojoules of energy\. Using 1 Calorie is approximately [\d.]+ kilojoules, "
    r"how many Calories is that\?$"
)
_HARD_RE = re.compile(
    r"^(.+?) is about ([\d.]+) kilojoules of energy and (.+?) is about ([\d.]+) kilojoules of energy\. "
    r"Using 1 Calorie is approximately [\d.]+ kilojoules, what is the combined Calorie total of both\?$"
)

# Plausibility oracle: real-world Calorie ranges authored fresh from general
# nutrition knowledge, independent of the generator's own catalogue table.
_CALORIE_RANGES: dict[str, tuple[int, int]] = {
    "a medium apple": (55, 115),
    "a medium banana": (80, 135),
    "a slice of white bread": (45, 95),
    "a can of regular soft drink (375 mL)": (110, 175),
    "a 50 g chocolate bar": (200, 310),
    "a boiled egg": (55, 105),
    "a cup of cooked rice": (120, 210),
    "a small baked potato": (55, 115),
    "a slice of cheese": (75, 135),
    "a tablespoon of peanut butter": (65, 115),
    "a cup of whole milk": (110, 190),
    "a small serving of hot chips": (200, 330),
    "a cooked chicken breast (100 g)": (130, 210),
    "a muesli bar": (85, 165),
    "a slice of pizza": (190, 310),
    "a cup of orange juice": (80, 135),
    "a doughnut": (190, 310),
    "a can of tuna in spring water": (80, 135),
    "a cup of ice cream": (200, 330),
    "an adult's approximate daily energy requirement": (1700, 2500),
}


def _lookup(name: str) -> tuple[int, int]:
    key = name.strip()
    key = key[0].lower() + key[1:] if key else key
    assert key in _CALORIE_RANGES, f"no plausibility range authored for {key!r}"
    return _CALORIE_RANGES[key]


def _factor(text: str) -> Fraction:
    m = _FACTOR_RE.search(text)
    assert m, f"no stated conversion factor found in: {text!r}"
    return Fraction(m.group(1))


def test_plausibility_oracle_covers_every_rendered_item_every_tier_every_seed():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier)
            if tier == "easy":
                name, cal_text = _EASY_RE.match(q.text).groups()
                cal = int(cal_text)
            elif tier == "medium":
                name, _kj_text = _MEDIUM_RE.match(q.text).groups()
                cal = int(q.answer_text)
            else:
                name1, _kj1, name2, _kj2 = _HARD_RE.match(q.text).groups()
                # For the hard tier, plausibility-check each item's own
                # implied Calorie share is out of scope (only the combined
                # total is asked); check both names are recognised instead.
                _lookup(name1)
                _lookup(name2)
                checked += 1
                continue
            lo, hi = _lookup(name)
            assert lo <= cal <= hi, f"{tier} seed {seed}: {name!r} = {cal} Cal outside [{lo}, {hi}]"
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_easy_value_oracle_recomputes_kilojoules_from_the_question_alone():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        name, cal_text = _EASY_RE.match(q.text).groups()
        factor = _factor(q.text)
        expected_kj = Fraction(cal_text) * factor
        assert Fraction(q.answer_text) == expected_kj, (
            f"seed {seed}: {name!r} -- expected {expected_kj}, rendered {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_medium_value_oracle_recomputes_calories_from_the_question_alone():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        name, kj_text = _MEDIUM_RE.match(q.text).groups()
        factor = _factor(q.text)
        expected_cal = Fraction(kj_text) / factor
        assert expected_cal.denominator == 1, (
            f"seed {seed}: {name!r} -- {kj_text} kJ / {factor} is not an exact integer Calorie count"
        )
        assert int(q.answer_text) == int(expected_cal), (
            f"seed {seed}: {name!r} -- expected {expected_cal}, rendered {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_hard_value_oracle_recomputes_combined_calories_from_the_question_alone():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        name1, kj1_text, name2, kj2_text = _HARD_RE.match(q.text).groups()
        assert name1.lower() != name2.lower(), f"seed {seed}: same item drawn twice: {name1!r}"
        factor = _factor(q.text)
        expected_total = Fraction(kj1_text) / factor + Fraction(kj2_text) / factor
        assert expected_total.denominator == 1, f"seed {seed}: combined total {expected_total} is not an exact integer"
        assert int(q.answer_text) == int(expected_total), (
            f"seed {seed}: expected {expected_total}, rendered {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_factor_phrase_states_four_point_two_every_tier_every_seed():
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert _factor(q.text) == Fraction("4.2"), f"{tier} seed {seed}: stated factor is not 4.2"


def test_distractors_never_equal_the_rendered_answer():
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
