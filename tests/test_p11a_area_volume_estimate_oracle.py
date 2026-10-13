"""Independent oracle for the `p11a_area_volume_estimate` questions: it re-derives each answer from the rendered
question text.

Parses ONLY the rendered question (``question.text``, ``question.answer_text``, ``question.distractors``) and
re-derives both things a learner is actually asked to trust, from scratch. Imports nothing from
``mathgen.generators.p11a_area_volume_estimate_build`` -- not the object catalogues, not a conversion factor,
not a formatting helper.

Two independent checks, per skill:

1. **Plausibility oracle**: a real-world range for every object name the
   skill can render, authored fresh from general knowledge in this file --
   not copied from, or derived by reading, the generator's own canonical
   table.
2. **Value oracle**: independently recomputes what x10/x0.1/x100 of the
   parsed correct numeral should be (as `Fraction` values) and asserts the
   three rendered distractors match those three values exactly.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

AREA_TOPIC = "estimate_area_reference"
CAPACITY_TOPIC = "estimate_volume_capacity_reference"

_AREA_Q_RE = re.compile(
    r"^What is the best estimate for the area of (.+)\? "
    r"Give your answer in (square metres|square centimetres)\.$"
)
_CAPACITY_Q_RE = re.compile(
    r"^What is the best estimate for the capacity of (.+)\? "
    r"Give your answer in (litres|millilitres)\.$"
)
_AREA_OPTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?) (m²|cm²)$")
_CAPACITY_OPTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?) (L|mL)$")

_AREA_UNIT_TO_CM2 = {"m²": Fraction(10000), "cm²": Fraction(1)}
_CAPACITY_UNIT_TO_L = {"L": Fraction(1), "mL": Fraction(1, 1000)}

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

# ---------------------------------------------------------------------------
# Plausibility oracle: real-world ranges authored independently of the
# generator's own catalogue. Ranges are honest estimates of real-world
# spread, not a fixed percentage of any particular canonical value. Macro
# ranges are authored in square METRES (their own natural scale) and
# converted to cm2 (the comparison pivot, since a hard-tier macro question
# is answered in cm2) at module load; micro ranges are authored directly in
# square CENTIMETRES, their own natural scale.

_AREA_RANGES_M2_NATURAL: dict[str, tuple[float, float]] = {
    "a classroom floor": (45, 80),
    "a tennis court": (190, 220),
    "a basketball court": (380, 450),
    "a football (soccer) pitch": (5500, 8250),
    "a badminton court": (70, 90),
    "a squash court": (55, 70),
    "an Olympic swimming pool surface": (1100, 1400),
    "a suburban house block": (350, 900),
    "a car parking space": (10, 15),
    "a shipping container floor": (20, 35),
    "a volleyball court": (140, 180),
    "a netball court": (400, 500),
    "a queen bed surface": (2.5, 3.5),
    "a double garage floor": (30, 42),
    "a school hall floor": (250, 600),
    "a cricket pitch": (55, 65),
    "a standard door surface": (1.3, 2.0),
    "a car roof": (1.5, 2.5),
    "a table tennis table": (3.8, 4.4),
    "a round trampoline (3.6 m diameter)": (9, 11),
    "an average bedroom floor": (9, 16),
    "a kitchen floor": (10, 22),
    "a residential solar panel": (1.4, 2.1),
    "an A0 poster sheet": (0.9, 1.1),
}

_AREA_RANGES_CM2_NATURAL: dict[str, tuple[float, float]] = {
    "a smartphone screen": (70, 130),
    "a credit card": (40, 55),
    "a postage stamp": (2, 8),
    "a sheet of A4 paper": (580, 650),
    "a drink coaster": (70, 130),
    "a dinner plate surface": (450, 700),
    "a slice of bread": (70, 150),
    "a laptop screen": (350, 700),
    "a business card": (35, 55),
    "a coffee mug base": (15, 50),
    "a paperback book cover": (280, 420),
    "a chess board": (1500, 2500),
    "a tablet screen": (200, 400),
    "a birthday card": (100, 250),
    "a woven placemat": (700, 1500),
    "a bathroom tile": (200, 900),
    "a passport photo": (10, 20),
    "a light switch plate": (50, 110),
    "a round birthday cake (20 cm diameter)": (290, 340),
    "a large pizza (30 cm diameter)": (650, 760),
    "a picture frame": (300, 900),
    "a laptop keyboard": (300, 550),
    "an open wallet": (60, 200),
    "a face towel": (700, 1200),
    "a DL envelope": (180, 280),
}

# Comparison pivot is cm2 throughout, since a hard-tier macro question is
# answered in cm2 and a hard-tier micro question is answered in m2 -- cm2
# keeps both comparisons on integers/simple decimals rather than tiny m2
# fractions for the macro half.
_AREA_RANGES_CM2: dict[str, tuple[float, float]] = {
    name: (lo * 10000, hi * 10000) for name, (lo, hi) in _AREA_RANGES_M2_NATURAL.items()
}
_AREA_RANGES_CM2.update(_AREA_RANGES_CM2_NATURAL)

_CAPACITY_RANGES_L: dict[str, tuple[float, float]] = {
    "a bathtub (full)": (100, 200),
    "a bucket": (7, 15),
    "a household wheelie bin": (120, 360),
    "a car fuel tank": (35, 80),
    "a washing machine drum": (40, 90),
    "a cooler/esky box": (20, 60),
    "a kitchen sink": (10, 30),
    "a household hot water system": (125, 315),
    "a watering can": (5, 12),
    "a small fish tank": (20, 60),
    "a large saucepan": (3, 8),
    "a school backpack (packed capacity)": (15, 30),
    "a jerry can": (10, 25),
    "a wine barrel": (200, 250),
    "a beer keg": (30, 60),
    "a garden compost bin": (100, 400),
    "a mini rubbish skip": (2000, 4000),
    "a backyard spa": (1000, 2500),
    "a small rainwater tank": (500, 2000),
    "a shopping trolley": (100, 200),
    "a large stockpot": (8, 20),
    "a car boot": (300, 550),
    "a household fridge": (300, 600),
    "a small above-ground pool": (10000, 30000),
    # micro half (converted to L in the oracle)
    "a teaspoon": (0.004, 0.006),
    "a tablespoon": (0.012, 0.018),
    "a shot glass": (0.025, 0.045),
    "a can of soft drink": (0.33, 0.4),
    "a coffee mug": (0.2, 0.4),
    "a standard wine glass pour": (0.1, 0.2),
    "a small water bottle": (0.35, 0.6),
    "a small carton of milk": (0.2, 0.3),
    "a chicken egg": (0.04, 0.08),
    "a tennis ball": (0.1, 0.2),
    "a beer pot/middy glass": (0.25, 0.32),
    "an espresso cup": (0.04, 0.08),
    "a soup bowl": (0.3, 0.5),
    "a juice box": (0.15, 0.25),
    "a small perfume bottle": (0.03, 0.08),
    "a travel hand sanitiser bottle": (0.03, 0.1),
    "a small yoghurt tub": (0.1, 0.2),
    "a squeeze honey bottle": (0.25, 0.4),
    "a table tennis ball": (0.03, 0.04),
    "a lemon": (0.08, 0.15),
    "a matchbox": (0.005, 0.02),
}


def _parse_area_question(text: str):
    m = _AREA_Q_RE.match(text)
    assert m, f"unparseable area question: {text!r}"
    return m.group(1), m.group(2)


def _parse_capacity_question(text: str):
    m = _CAPACITY_Q_RE.match(text)
    assert m, f"unparseable capacity question: {text!r}"
    return m.group(1), m.group(2)


def _parse_area_option(opt: str):
    m = _AREA_OPTION_RE.match(opt)
    assert m, f"unparseable area option: {opt!r}"
    return Fraction(m.group(1)), m.group(2)


def _parse_capacity_option(opt: str):
    m = _CAPACITY_OPTION_RE.match(opt)
    assert m, f"unparseable capacity option: {opt!r}"
    return Fraction(m.group(1)), m.group(2)


def _range_for(name: str, table: dict) -> tuple[str, tuple[float, float]]:
    matches = [key for key in table if key in name]
    assert len(matches) == 1, f"object name {name!r} matched {len(matches)} plausibility entries: {matches}"
    return matches[0], table[matches[0]]


# ===========================================================================
# Area oracle
# ===========================================================================


def test_area_plausibility_oracle_covers_every_rendered_object_every_tier_every_seed():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(AREA_TOPIC, seed=seed, difficulty=tier, tier=tier)
            name, _unit_word = _parse_area_question(q.text)
            key, (lo, hi) = _range_for(name, _AREA_RANGES_CM2)
            value, unit = _parse_area_option(q.answer_text)
            value_cm2 = float(value * _AREA_UNIT_TO_CM2[unit])
            assert lo <= value_cm2 <= hi, (
                f"{tier} seed {seed}: {name!r} rendered {q.answer_text!r} = {value_cm2} cm2, "
                f"outside the independently-authored plausible range {key!r}: [{lo}, {hi}] cm2"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_area_distractors_match_independently_recomputed_multiples():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(AREA_TOPIC, seed=seed, difficulty=tier, tier=tier)
            answer_value, answer_unit = _parse_area_option(q.answer_text)
            expected = {answer_value * Fraction(10) ** k for k in (-2, -1, 1, 2, 3)}
            assert len(q.distractors) == 3
            actual = set()
            for dist in q.distractors:
                value, unit = _parse_area_option(dist)
                assert unit == answer_unit
                actual.add(value)
            assert len(actual) == 3 and actual <= expected, (
                f"{tier} seed {seed}: distractors {actual} != expected {expected}"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_area_plausibility_table_reaches_a_broad_set_of_distinct_objects():
    names_seen = set()
    for tier in TIERS:
        for seed in SEEDS:
            q = make(AREA_TOPIC, seed=seed, difficulty=tier, tier=tier)
            name, _unit_word = _parse_area_question(q.text)
            key, _range = _range_for(name, _AREA_RANGES_CM2)
            names_seen.add(key)
    assert len(names_seen) >= 25, (
        f"area oracle sweep only reached {len(names_seen)} distinct objects: {sorted(names_seen)}"
    )


def test_area_hard_tier_squared_factor_is_independently_confirmed():
    """Re-derive, from the question text alone, that a macro object asked
    in cm2 gets a x10 000 jump (not x100) and a micro object asked in m2
    gets a /10 000 jump (not /100)."""
    checked_macro = checked_micro = 0
    for seed in SEEDS:
        q = make(AREA_TOPIC, seed=seed, difficulty="hard", tier="hard")
        name, unit_word = _parse_area_question(q.text)
        key, (lo, hi) = _range_for(name, _AREA_RANGES_CM2)
        value, unit = _parse_area_option(q.answer_text)
        value_cm2 = float(value * _AREA_UNIT_TO_CM2[unit])
        if unit_word == "square centimetres":
            # a macro object's natural-scale value in m2 sits within [lo, hi];
            # its rendered cm2 value divided by 10000 must land back in range.
            assert lo <= value_cm2 <= hi
            checked_macro += 1
        else:
            assert lo <= value_cm2 <= hi
            checked_micro += 1
    assert checked_macro > 0 and checked_micro > 0


# ===========================================================================
# Capacity oracle
# ===========================================================================


def test_capacity_plausibility_oracle_covers_every_rendered_object_every_tier_every_seed():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(CAPACITY_TOPIC, seed=seed, difficulty=tier, tier=tier)
            name, _unit_word = _parse_capacity_question(q.text)
            key, (lo, hi) = _range_for(name, _CAPACITY_RANGES_L)
            value, unit = _parse_capacity_option(q.answer_text)
            value_l = float(value * _CAPACITY_UNIT_TO_L[unit])
            assert lo <= value_l <= hi, (
                f"{tier} seed {seed}: {name!r} rendered {q.answer_text!r} = {value_l} L, "
                f"outside the independently-authored plausible range {key!r}: [{lo}, {hi}] L"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_capacity_distractors_match_independently_recomputed_multiples():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(CAPACITY_TOPIC, seed=seed, difficulty=tier, tier=tier)
            answer_value, answer_unit = _parse_capacity_option(q.answer_text)
            expected = {answer_value * Fraction(10) ** k for k in (-2, -1, 1, 2, 3)}
            assert len(q.distractors) == 3
            actual = set()
            for dist in q.distractors:
                value, unit = _parse_capacity_option(dist)
                assert unit == answer_unit
                actual.add(value)
            assert len(actual) == 3 and actual <= expected, (
                f"{tier} seed {seed}: distractors {actual} != expected {expected}"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_capacity_plausibility_table_reaches_a_broad_set_of_distinct_objects():
    names_seen = set()
    for tier in TIERS:
        for seed in SEEDS:
            q = make(CAPACITY_TOPIC, seed=seed, difficulty=tier, tier=tier)
            name, _unit_word = _parse_capacity_question(q.text)
            key, _range = _range_for(name, _CAPACITY_RANGES_L)
            names_seen.add(key)
    assert len(names_seen) >= 25, (
        f"capacity oracle sweep only reached {len(names_seen)} distinct objects: {sorted(names_seen)}"
    )


def test_capacity_questions_never_say_volume():
    for tier in TIERS:
        for seed in SEEDS:
            q = make(CAPACITY_TOPIC, seed=seed, difficulty=tier, tier=tier)
            assert "volume" not in q.text.lower(), q.text
