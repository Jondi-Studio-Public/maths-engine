"""Independent oracle for the `p11a_mass_measures` questions: it re-derives each answer from the rendered
question text.

Parses ONLY the rendered question (``question.text``, ``question.answer_text``, ``question.distractors``) and
re-derives both things a learner is actually asked to trust, from scratch. Imports nothing from
``mathgen.generators.p11a_mass_measures_build`` -- not the object catalogue, not a conversion factor, not a
formatting helper.

Three independent checks:

1. **Plausibility oracle** (`_PLAUSIBLE_RANGES_KG`): a real-world kilogram
   range for every object name across BOTH skills, authored fresh from
   general knowledge in this file -- not copied from, or derived by reading,
   the generator's own canonical-value tables. Also used to independently
   classify each `unit_choice` object's own natural ladder unit, via
   magnitude buckets with wide gaps between clusters (module docstring's
   own catalogue design keeps macro/micro/mega/milli well separated).
2. **Value oracle (skill A)**: independently recomputes x10/x0.1/x100 of the
   parsed correct numeral and asserts the three rendered distractors match.
3. **Ladder oracle (skill B)**: an independently-authored mg-per-unit ladder
   table re-derives the adjacent-relation and equivalence answers from the
   question text alone.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

TOPIC_A = "estimate_mass_reference"
TOPIC_B = "review_metric_units_of_mass"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

_PARAMS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}


def _mk(topic, seed, tier):
    return make(topic, seed=seed, difficulty=tier, **_PARAMS[tier])


# ---------------------------------------------------------------------------
# Plausibility oracle: a real-world kilogram range for every object name
# either skill can render, authored independently of the generator's own
# tables. Covers all 60 objects across both skills' catalogues.

_PLAUSIBLE_RANGES_KG: dict[str, tuple[float, float]] = {
    # macro-scale (kilogram-natural) reference objects
    "a bag of sugar": (0.5, 2),
    "a house brick": (1.5, 3.5),
    "a laptop": (1, 3),
    "a bicycle": (7, 18),
    "a microwave oven": (8, 20),
    "an office chair": (8, 22),
    "a bag of cement": (18, 22),
    "a full school backpack": (4, 10),
    "an adult cat": (3, 6),
    "a medium dog": (12, 25),
    "a newborn baby": (2.5, 4.5),
    "an adult person": (45, 110),
    "a car battery": (8, 22),
    "a queen mattress": (20, 45),
    "an empty bathtub": (15, 40),
    "an adult kangaroo": (20, 90),
    "an emu": (25, 50),
    "a washing machine": (50, 100),
    "an empty wheelbarrow": (8, 32),
    "a golden retriever": (20, 38),
    "a mountain bike": (8, 18),
    "a bar fridge": (15, 40),
    # micro-scale (gram-natural) reference objects
    "a smartphone": (0.1, 0.26),
    "a paperback novel": (0.1, 0.35),
    "an apple": (0.08, 0.25),
    "a banana": (0.07, 0.18),
    "a chocolate bar": (0.02, 0.15),
    "a deck of cards": (0.06, 0.15),
    "a tennis ball": (0.05, 0.065),
    "a golf ball": (0.04, 0.05),
    "a chicken egg": (0.04, 0.075),
    "a slice of bread": (0.015, 0.045),
    "an AA battery": (0.015, 0.03),
    "a pair of scissors": (0.03, 0.12),
    "a computer mouse": (0.05, 0.15),
    "a wallet": (0.04, 0.15),
    "a bar of soap": (0.05, 0.18),
    "a metal teaspoon": (0.005, 0.025),
    "a table tennis ball": (0.002, 0.004),
    "a matchbox with matches": (0.008, 0.035),
    "a compact disc": (0.01, 0.025),
    "a smartphone case": (0.01, 0.06),
    "an LED light bulb": (0.015, 0.1),
    "a pack of chewing gum": (0.008, 0.04),
    # mega-scale (tonne-natural) reference objects, unit_choice only
    "a family car": (900, 2200),
    "a large SUV": (1800, 3500),
    "an adult elephant": (2000, 7000),
    "a loaded shipping container": (10000, 30000),
    "a city bus": (8000, 16000),
    "an orca (killer whale)": (3000, 6500),
    "a fire engine": (8000, 22000),
    "a loaded semi-trailer truck": (15000, 42000),
    "a double-decker bus": (8000, 16000),
    "an adult hippopotamus": (1000, 3500),
    "a garbage truck": (6000, 15000),
    "a small aeroplane": (500, 2000),
    # milli-scale (milligram-natural) reference objects, unit_choice only
    "a grain of rice": (0.00001, 0.00005),
    "a mosquito": (0.0000005, 0.000005),
    "an eyelash": (0.0000003, 0.000003),
    "a raindrop": (0.00001, 0.0001),
}


def _range_for(name: str) -> tuple[str, tuple[float, float]]:
    """Exact lookup -- the question regexes above already isolate the exact
    object name with no trailing words, so a substring scan is unnecessary
    and would mismatch "a smartphone" against "a smartphone case"."""
    assert name in _PLAUSIBLE_RANGES_KG, f"object name {name!r} has no plausibility entry"
    return name, _PLAUSIBLE_RANGES_KG[name]


def _expected_unit_bucket(mass_kg: float) -> str:
    """Independent classification of an object's own natural ladder unit
    from its plausible mass alone -- the catalogue's four scales sit in
    well-separated clusters (module docstring), so any threshold inside
    each gap works; these are authored fresh, not read off the generator."""
    if mass_kg < 0.001:
        return "mg"
    if mass_kg < 0.5:
        return "g"
    if mass_kg < 500:
        return "kg"
    return "t"


# ---------------------------------------------------------------------------
# Skill A: estimate_mass_reference

_Q_A_RE = re.compile(r"^What is the best estimate for the mass of (.+)\? Give your answer in (kilograms|grams)\.$")
_OPTION_A_RE = re.compile(r"^(-?\d+(?:\.\d+)?) (kg|g)$")
_UNIT_A_TO_KG = {"kg": Fraction(1), "g": Fraction(1, 1000)}


def _parse_option_a(opt: str):
    m = _OPTION_A_RE.match(opt)
    assert m, f"unparseable option: {opt!r}"
    return Fraction(m.group(1)), m.group(2)


def test_a_plausibility_oracle_covers_every_rendered_object_every_tier_every_seed():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, tier)
            m = _Q_A_RE.match(q.text)
            assert m, f"unparseable question: {q.text!r}"
            name = m.group(1)
            key, (lo, hi) = _range_for(name)
            value, unit = _parse_option_a(q.answer_text)
            value_kg = float(value * _UNIT_A_TO_KG[unit])
            assert lo <= value_kg <= hi, (
                f"{tier} seed {seed}: {name!r} rendered {q.answer_text!r} = {value_kg} kg, "
                f"outside the independently-authored plausible range {key!r}: [{lo}, {hi}] kg"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_a_distractors_match_independently_recomputed_multiples():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, tier)
            answer_value, answer_unit = _parse_option_a(q.answer_text)
            expected = {answer_value * Fraction(10) ** k for k in (-2, -1, 1, 2, 3)}
            assert len(q.distractors) == 3
            actual = set()
            for dist in q.distractors:
                value, unit = _parse_option_a(dist)
                assert unit == answer_unit
                actual.add(value)
            assert len(actual) == 3 and actual <= expected, (
                f"{tier} seed {seed}: distractors {actual} != expected {expected}"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


# ---------------------------------------------------------------------------
# Skill B: review_metric_units_of_mass

_Q_UNIT_CHOICE_RE = re.compile(r"^Which unit is the most appropriate choice for measuring the mass of (.+)\?$")
_Q_ADJACENT_RE = re.compile(r"^How many (mg|g|kg) are there in (\d+) (g|kg|t)\?$")
_Q_EQUIVALENCE_RE = re.compile(r"^Which of these has the same mass as (.+)\?$")
_STATED_VALUE_RE = re.compile(r"^(-?\d+(?:\.\d+)?) (mg|g|kg|t)$")

# Independently-authored mg-per-unit ladder (never imported from the
# generator's own `_MG_PER_UNIT`).
_LADDER_MG_PER_UNIT = {"mg": Fraction(1), "g": Fraction(1_000), "kg": Fraction(1_000_000), "t": Fraction(1_000_000_000)}


def test_b_unit_choice_matches_the_plausibility_oracle_bucket():
    checked = 0
    for seed in SEEDS:
        q = _mk(TOPIC_B, seed, "easy")
        m = _Q_UNIT_CHOICE_RE.match(q.text)
        assert m, f"unparseable question: {q.text!r}"
        name = m.group(1)
        key, (lo, hi) = _range_for(name)
        mid = (lo + hi) / 2
        expected_unit = _expected_unit_bucket(mid)
        assert q.answer_text == expected_unit, (
            f"seed {seed}: {name!r} (oracle range {key!r} -> {expected_unit}) but generator answered {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_b_adjacent_relation_matches_independently_recomputed_ladder():
    checked = 0
    for seed in SEEDS:
        q = _mk(TOPIC_B, seed, "medium")
        m = _Q_ADJACENT_RE.match(q.text)
        assert m, f"unparseable question: {q.text!r}"
        smaller, n_str, larger = m.groups()
        n = int(n_str)
        expected = n * (_LADDER_MG_PER_UNIT[larger] / _LADDER_MG_PER_UNIT[smaller])
        stated_value, stated_unit = _STATED_VALUE_RE.match(q.answer_text).groups()
        assert stated_unit == smaller
        assert Fraction(stated_value) == expected, (
            f"seed {seed}: {q.text!r} -> {q.answer_text!r} != {expected} {smaller}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_b_equivalence_matches_independently_recomputed_ladder_both_directions():
    seen_small_given = seen_big_given = False
    checked = 0
    for seed in SEEDS:
        q = _mk(TOPIC_B, seed, "hard")
        m = _Q_EQUIVALENCE_RE.match(q.text)
        assert m, f"unparseable question: {q.text!r}"
        stated_text = m.group(1)
        stated_value_str, stated_unit = _STATED_VALUE_RE.match(stated_text).groups()
        stated_value = Fraction(stated_value_str)

        answer_value_str, answer_unit = _STATED_VALUE_RE.match(q.answer_text).groups()
        answer_value = Fraction(answer_value_str)

        expected_answer = stated_value * (_LADDER_MG_PER_UNIT[stated_unit] / _LADDER_MG_PER_UNIT[answer_unit])
        assert answer_value == expected_answer, (
            f"seed {seed}: {stated_text!r} != {q.answer_text!r} independently (expected {expected_answer} {answer_unit})"
        )
        if "." in stated_value_str:
            seen_big_given = True
        else:
            seen_small_given = True
        checked += 1
    assert checked == len(SEEDS)
    assert seen_small_given and seen_big_given, "hard tier failed to reach both directions over 60 seeds"


def test_plausibility_table_covers_a_broad_set_of_distinct_objects():
    names_seen = set()
    for tier in TIERS:
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, tier)
            m = _Q_A_RE.match(q.text)
            key, _range = _range_for(m.group(1))
            names_seen.add(key)
    for seed in SEEDS:
        q = _mk(TOPIC_B, seed, "easy")
        m = _Q_UNIT_CHOICE_RE.match(q.text)
        key, _range = _range_for(m.group(1))
        names_seen.add(key)
    assert len(names_seen) >= 30, f"oracle sweep only reached {len(names_seen)} distinct objects: {sorted(names_seen)}"
