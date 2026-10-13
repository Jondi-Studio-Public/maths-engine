"""Tests `estimate_length_reference`: the reference-object catalogues, order-of-magnitude options and an
independent check.

Estimate lengths build."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
import mathgen.generators.p10b_estimate_lengths_build as p10b_mod
from mathgen.generators.p10b_estimate_lengths_build import (
    _MACRO,
    _MICRO,
    _build,
    _exact_places_needed,
)

TOPIC = "estimate_length_reference"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

_PARAMS = {
    "easy": dict(tier="easy"),
    "medium": dict(tier="medium"),
    "hard": dict(tier="hard"),
}

_OPTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?) (m|cm|mm)$")
_UNIT_WORD_FOR_TIER_EASY_MEDIUM = {"easy": "metres", "medium": "centimetres"}


def _mk(seed, difficulty):
    return make(TOPIC, seed=seed, difficulty=difficulty, **_PARAMS[difficulty])


def _parse_option(opt: str) -> tuple[Fraction, str]:
    m = _OPTION_RE.match(opt)
    assert m, f"unparseable option: {opt!r}"
    return Fraction(m.group(1)), m.group(2)


# ===========================================================================
# Catalogue invariants
# ===========================================================================


def test_catalogue_sizes_clear_the_20_distinct_floor_independently():
    assert len(_MACRO) >= 20, f"macro half only has {len(_MACRO)} objects"
    assert len(_MICRO) >= 20, f"micro half only has {len(_MICRO)} objects"


def test_catalogue_names_are_unique():
    names = [name for name, _attr, _val in _MACRO + _MICRO]
    assert len(names) == len(set(names)), "duplicate object name in the catalogue"


def test_macro_canonical_values_are_at_least_one_metre():
    for name, _attr, value in _MACRO:
        assert value >= 1, f"{name}: {value} m is not macro-scale"


def test_micro_canonical_values_are_under_one_metre():
    for name, _attr, value in _MICRO:
        assert value < 1, f"{name}: {value} m is not micro-scale"


# ===========================================================================
# Shared behaviour across all three tiers
# ===========================================================================


class TestEstimateLengthReference:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = _mk(seed, difficulty)
            q2 = _mk(seed, difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == 3
            assert len(q.distractors) == len(set(q.distractors))

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, str(correct_index)).correct
            for dist in q.distractors:
                dist_index = options.index(dist)
                assert not grade_mcq(q, str(dist_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_a_deliberately_wrong_index(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            wrong_index = (correct_index + 1) % len(options)
            assert not grade_mcq(q, str(wrong_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_in_options(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            assert q.answer_text in mcq_options(q)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions(self, difficulty):
        texts = {_mk(seed, difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    @pytest.mark.parametrize("difficulty", ["easy", "medium"])
    def test_easy_medium_ask_in_the_stated_unit_word(self, difficulty):
        word = _UNIT_WORD_FOR_TIER_EASY_MEDIUM[difficulty]
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            assert f"in {word}." in q.text

    def test_easy_only_draws_macro_objects(self):
        macro_names = {n for n, _a, _v in _MACRO}
        for seed in SEEDS:
            q = _mk(seed, "easy")
            assert any(n in q.text for n in macro_names), q.text
            _, unit = _parse_option(q.answer_text)
            assert unit == "m"

    def test_medium_only_draws_micro_objects(self):
        micro_names = {n for n, _a, _v in _MICRO}
        for seed in SEEDS:
            q = _mk(seed, "medium")
            assert any(n in q.text for n in micro_names), q.text
            _, unit = _parse_option(q.answer_text)
            assert unit == "cm"

    def test_hard_always_asks_in_the_non_natural_unit(self):
        macro_names = {n for n, _a, _v in _MACRO}
        micro_names = {n for n, _a, _v in _MICRO}
        seen_macro = seen_micro = False
        for seed in SEEDS:
            q = _mk(seed, "hard")
            _, unit = _parse_option(q.answer_text)
            is_macro = any(n in q.text for n in macro_names)
            is_micro = any(n in q.text for n in micro_names)
            assert is_macro != is_micro, q.text  # exactly one half matches
            if is_macro:
                assert unit == "mm", f"macro object asked in its OWN natural unit: {q.text}"
                seen_macro = True
            else:
                assert unit == "m", f"micro object asked in its OWN natural unit: {q.text}"
                seen_micro = True
        assert seen_macro and seen_micro, "hard tier failed to reach both halves of the catalogue over 60 seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_options_share_the_same_decimal_place_count(self, difficulty):
        """No option is identifiable by its own decimal-digit count within a
        single question (the `midpoint_coordinate_scalar` defect class)."""
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            all_options = [q.answer_text] + list(q.distractors)
            places = [len(opt.split(" ")[0].split(".")[1]) if "." in opt else 0 for opt in all_options]
            assert len(set(places)) == 1, f"{difficulty} seed {seed}: mixed decimal-place shapes {all_options}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_are_exact_order_of_magnitude_multiples(self, difficulty):
        for seed in SEEDS:
            q = _mk(seed, difficulty)
            answer_val, answer_unit = _parse_option(q.answer_text)
            values = []
            for dist in q.distractors:
                val, unit = _parse_option(dist)
                assert unit == answer_unit
                values.append(val)
            expected = {answer_val * Fraction(10) ** k for k in (-2, -1, 1, 2, 3)}
            assert len(set(values)) == 3 and set(values) <= expected, (
                f"{difficulty} seed {seed}: {values} != {expected}"
            )

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-estimate-lengths"), None)
        assert skill is not None
        assert skill.domain == "measurement"
        assert skill.answer == "mcq"
        assert skill.prerequisites == [
            "convert-metric-length",
            "senior-review-metric-units-of-length-their-abbreviations-conversions",
        ]
        assert skill.placement_eligible is False
        assert len([s for s in skill.sources if s.difficulty != "exam"]) == 3


# ===========================================================================
# Independent-check plumbing, exercised directly
# ===========================================================================


class TestIndependentCheck:
    def test_conversion_check_fires_on_a_corrupted_factor_table(self, monkeypatch):
        """Drives the ACTUAL `_build` construction with `_MM_PER_UNIT`
        deliberately wired to disagree with `_DIRECT_FACTOR_FROM_M` -- the
        `check_conversion` call must raise."""
        import random

        monkeypatch.setattr(p10b_mod, "_MM_PER_UNIT", {"mm": Fraction(1), "cm": Fraction(999), "m": Fraction(1000)})
        raised = False
        for seed in range(20):
            try:
                _build(random.Random(seed), "medium")
            except VerificationError:
                raised = True
                break
        assert raised, "no seed exercised the corrupted-conversion mismatch"

    def test_exact_places_needed(self):
        assert _exact_places_needed(Fraction(18)) == 0
        assert _exact_places_needed(Fraction(18, 10)) == 1
        assert _exact_places_needed(Fraction(45, 100)) == 2

    def test_exact_places_needed_does_not_hang_on_a_real_tenth_value(self):
        # every real /10 distractor in this skill terminates within a handful
        # of places (canonical values have at most 2 decimal places, so /10
        # of a formatted option needs at most 3) -- guard the loop bound.
        for _name, _attr, canonical in _MACRO + _MICRO:
            for factor in (Fraction(1), Fraction(100), Fraction(1000)):
                tenth = canonical * factor / 10
                assert _exact_places_needed(tenth) <= 4
