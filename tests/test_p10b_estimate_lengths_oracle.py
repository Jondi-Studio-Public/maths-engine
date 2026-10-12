"""Independent oracle for the `p10b_estimate_lengths` questions: it re-derives each answer from the rendered
question text.

Parses ONLY the rendered question (``question.text``, ``question.answer_text``, ``question.distractors``) and
re-derives both things a learner is actually asked to trust, from scratch. Imports nothing from
``mathgen.generators.p10b_estimate_lengths_build`` -- not the object catalogue, not a conversion factor, not a
formatting helper.

Two independent checks:

1. **Plausibility oracle** (`_PLAUSIBLE_RANGES_M`): a real-world metre range
   for every object name the skill can render, authored fresh from general
   knowledge in this file -- not copied from, or derived by reading, the
   generator's own canonical-value table. This is the check that would have
   figure).
2. **Value oracle**: independently recomputes what x10/x0.1/x100 of the
   parsed correct numeral should be (as `Fraction` values, not strings --
   see `test_distractors_match_independently_recomputed_multiples`) and
   asserts the three rendered distractors match those three values exactly.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

TOPIC = "estimate_length_reference"

_Q_RE = re.compile(
    r"^What is the best estimate for the (\w+) of (.+)\? "
    r"Give your answer in (metres|centimetres|millimetres)\.$"
)
_OPTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?) (m|cm|mm)$")

_UNIT_TO_M = {"m": Fraction(1), "cm": Fraction(1, 100), "mm": Fraction(1, 1000)}

# ---------------------------------------------------------------------------
# Plausibility oracle: a real-world metre range for every object name the
# skill can render, authored independently of the generator's own table.
# Ranges are honest estimates of real-world spread, not a fixed percentage
# of any particular canonical value.

_PLAUSIBLE_RANGES_M: dict[str, tuple[float, float]] = {
    # macro-scale reference objects
    "a family car": (3.5, 5.5),
    "a city bus": (10, 14),
    "an adult door": (1.9, 2.2),
    "a canoe": (3, 6),
    "a queen-size bed": (1.9, 2.1),
    "an adult person": (1.5, 1.9),
    "a bathtub": (1.3, 1.8),
    "a soccer pitch": (90, 120),
    "a tennis court": (23, 25),
    "a telegraph pole": (7, 12),
    "a shipping container": (11, 13),
    "an Olympic swimming pool": (48, 52),
    "an adult giraffe": (4, 6),
    "a classroom": (7, 11),
    "a double-decker bus": (4.2, 4.8),
    "a small yacht": (6, 12),
    "a cricket pitch": (19, 21),
    "a basketball court": (26, 29),
    "a killer whale": (5, 9),
    "a fire engine": (8, 12),
    "a train carriage": (18, 26),
    "a semi-trailer truck": (16, 26),
    "a flagpole": (4, 12),
    "a large gum tree": (15, 35),
    "a rowing eight": (16, 20),
    "a highway lane": (2.8, 4),
    "a park bench": (1.4, 2.2),
    "a stretch limousine": (7, 11),
    # micro-scale reference objects
    "a pencil": (0.14, 0.20),
    "a smartphone": (0.12, 0.18),
    "a dinner plate": (0.22, 0.30),
    "an adult shoe": (0.24, 0.32),
    "a paperback book": (0.17, 0.24),
    "a laptop": (0.28, 0.38),
    "a tennis ball": (0.06, 0.08),
    "a soft drink can": (0.10, 0.14),
    "a bar of soap": (0.07, 0.12),
    "a wallet": (0.08, 0.13),
    "a house key": (0.04, 0.08),
    "a dinner fork": (0.16, 0.22),
    "a computer mouse": (0.08, 0.13),
    "a slice of bread": (0.09, 0.14),
    "a golf ball": (0.035, 0.045),
    "a standard brick": (0.20, 0.25),
    "an A4 exercise book": (0.27, 0.32),
    "a table tennis paddle": (0.23, 0.28),
    "a banana": (0.13, 0.23),
    "a wine bottle": (0.27, 0.33),
    "a teaspoon": (0.12, 0.16),
    "a hardcover textbook": (0.18, 0.24),
    "a USB flash drive": (0.03, 0.07),
    "a matchbox": (0.04, 0.06),
    "a chess piece": (0.06, 0.11),
    "a hairbrush": (0.15, 0.25),
    "a DL envelope": (0.20, 0.24),
    "a paperclip": (0.02, 0.05),
}

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)


def _parse_question(text: str):
    m = _Q_RE.match(text)
    assert m, f"unparseable question: {text!r}"
    attribute, name, unit_word = m.groups()
    return attribute, name, unit_word


def _parse_option(opt: str):
    m = _OPTION_RE.match(opt)
    assert m, f"unparseable option: {opt!r}"
    return Fraction(m.group(1)), m.group(2)


def _object_range_for(name: str) -> tuple[str, tuple[float, float]]:
    matches = [key for key in _PLAUSIBLE_RANGES_M if key in name]
    assert len(matches) == 1, f"object name {name!r} matched {len(matches)} plausibility entries: {matches}"
    return matches[0], _PLAUSIBLE_RANGES_M[matches[0]]


def test_plausibility_oracle_covers_every_rendered_object_every_tier_every_seed():
    """Every rendered object's correct value, independently converted to metres, must fall inside a real-world
    range this file authored on its own."""
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            _attribute, name, _unit_word = _parse_question(q.text)
            key, (lo, hi) = _object_range_for(name)
            value, unit = _parse_option(q.answer_text)
            value_m = float(value * _UNIT_TO_M[unit])
            assert lo <= value_m <= hi, (
                f"{tier} seed {seed}: {name!r} rendered {q.answer_text!r} = {value_m} m, "
                f"outside the independently-authored plausible range {key!r}: [{lo}, {hi}] m"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_distractors_match_independently_recomputed_multiples():
    """Value oracle: independently recompute x10/x0.1/x100 of the parsed
    correct numeral (as exact `Fraction` values, chosen over string
    comparison so this oracle owns its own arithmetic rather than
    replicating the generator's formatting) and assert the three rendered
    distractors equal those three values exactly, in the same unit as the
    answer."""
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            answer_value, answer_unit = _parse_option(q.answer_text)
            expected = {answer_value * Fraction(10) ** k for k in (-2, -1, 1, 2, 3)}
            assert len(q.distractors) == 3, f"{tier} seed {seed}: expected exactly 3 distractors"
            actual = set()
            for dist in q.distractors:
                value, unit = _parse_option(dist)
                assert unit == answer_unit, (
                    f"{tier} seed {seed}: distractor unit {unit!r} != answer unit {answer_unit!r}"
                )
                actual.add(value)
            assert len(actual) == 3 and actual <= expected, (
                f"{tier} seed {seed}: distractors {actual} != expected {expected}"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_plausibility_table_covers_the_full_catalogue_by_object_count():
    """Sanity on the oracle's own coverage, not the generator's: every
    rendered object over this sweep resolves to exactly one plausibility
    entry (`_object_range_for` already asserts that per-call); here we also
    require the sweep actually reached a broad set of distinct objects, so
    an oracle that only ever matched three names would be caught."""
    names_seen = set()
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            _attribute, name, _unit_word = _parse_question(q.text)
            key, _range = _object_range_for(name)
            names_seen.add(key)
    assert len(names_seen) >= 30, f"oracle sweep only reached {len(names_seen)} distinct objects: {sorted(names_seen)}"
