"""Tests `estimate_area_reference` and `estimate_volume_capacity_reference`: catalogue sizes, magnitude bands and
independent checks.

Area volume estimate build."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
import mathgen.generators.p11a_area_volume_estimate_build as p11a_mod
from mathgen.generators.p11a_area_volume_estimate_build import (
    _MACRO_AREA,
    _MICRO_AREA,
    _MACRO_CAPACITY,
    _MICRO_CAPACITY,
    _build_area,
    _build_capacity,
    _exact_places_needed,
)

AREA_TOPIC = "estimate_area_reference"
CAPACITY_TOPIC = "estimate_volume_capacity_reference"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

_AREA_OPTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?) (m²|cm²)$")
_CAPACITY_OPTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?) (L|mL)$")


def _mk_area(seed, difficulty):
    return make(AREA_TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)


def _mk_capacity(seed, difficulty):
    return make(CAPACITY_TOPIC, seed=seed, difficulty=difficulty, tier=difficulty)


def _parse_area_option(opt: str) -> tuple[Fraction, str]:
    m = _AREA_OPTION_RE.match(opt)
    assert m, f"unparseable area option: {opt!r}"
    return Fraction(m.group(1)), m.group(2)


def _parse_capacity_option(opt: str) -> tuple[Fraction, str]:
    m = _CAPACITY_OPTION_RE.match(opt)
    assert m, f"unparseable capacity option: {opt!r}"
    return Fraction(m.group(1)), m.group(2)


# ===========================================================================
# Catalogue invariants
# ===========================================================================


def test_area_catalogue_sizes_clear_the_20_distinct_floor_independently():
    assert len(_MACRO_AREA) >= 20, f"macro area only has {len(_MACRO_AREA)} objects"
    assert len(_MICRO_AREA) >= 20, f"micro area only has {len(_MICRO_AREA)} objects"


def test_capacity_catalogue_sizes_clear_the_20_distinct_floor_independently():
    assert len(_MACRO_CAPACITY) >= 20, f"macro capacity only has {len(_MACRO_CAPACITY)} objects"
    assert len(_MICRO_CAPACITY) >= 20, f"micro capacity only has {len(_MICRO_CAPACITY)} objects"


def test_area_catalogue_names_are_unique():
    names = [name for name, _v in _MACRO_AREA + _MICRO_AREA]
    assert len(names) == len(set(names)), "duplicate area object name in the catalogue"


def test_capacity_catalogue_names_are_unique():
    names = [name for name, _v in _MACRO_CAPACITY + _MICRO_CAPACITY]
    assert len(names) == len(set(names)), "duplicate capacity object name in the catalogue"


def test_macro_area_values_are_at_least_one_square_metre():
    for name, value in _MACRO_AREA:
        assert value >= 1, f"{name}: {value} m2 is not macro-scale"


def test_micro_area_values_are_under_one_square_metre_in_cm2_terms():
    # micro values are stored in cm2; < 10000 cm2 == < 1 m2
    for name, value in _MICRO_AREA:
        assert value < 10000, f"{name}: {value} cm2 is not micro-scale"


def test_macro_capacity_values_are_at_least_one_litre():
    for name, value in _MACRO_CAPACITY:
        assert value >= 1, f"{name}: {value} L is not macro-scale"


def test_micro_capacity_values_are_under_1000_millilitres():
    for name, value in _MICRO_CAPACITY:
        assert value < 1000, f"{name}: {value} mL is not micro-scale"


# ===========================================================================
# Shared behaviour: area
# ===========================================================================


class TestEstimateAreaReference:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = _mk_area(seed, difficulty)
            q2 = _mk_area(seed, difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = _mk_area(seed, difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == 3
            assert len(q.distractors) == len(set(q.distractors))

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = _mk_area(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, str(correct_index)).correct
            for dist in q.distractors:
                dist_index = options.index(dist)
                assert not grade_mcq(q, str(dist_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_a_deliberately_wrong_index(self, difficulty):
        for seed in SEEDS:
            q = _mk_area(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            wrong_index = (correct_index + 1) % len(options)
            assert not grade_mcq(q, str(wrong_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_in_options(self, difficulty):
        for seed in SEEDS:
            q = _mk_area(seed, difficulty)
            assert q.answer_text in mcq_options(q)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions(self, difficulty):
        texts = {_mk_area(seed, difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_easy_only_draws_macro_objects_asked_in_square_metres(self):
        macro_names = {n for n, _v in _MACRO_AREA}
        for seed in SEEDS:
            q = _mk_area(seed, "easy")
            assert any(n in q.text for n in macro_names), q.text
            assert "in square metres." in q.text
            _, unit = _parse_area_option(q.answer_text)
            assert unit == "m²"

    def test_medium_only_draws_micro_objects_asked_in_square_centimetres(self):
        micro_names = {n for n, _v in _MICRO_AREA}
        for seed in SEEDS:
            q = _mk_area(seed, "medium")
            assert any(n in q.text for n in micro_names), q.text
            assert "in square centimetres." in q.text
            _, unit = _parse_area_option(q.answer_text)
            assert unit == "cm²"

    def test_hard_always_asks_in_the_non_natural_unit_and_reaches_both_halves(self):
        macro_names = {n for n, _v in _MACRO_AREA}
        micro_names = {n for n, _v in _MICRO_AREA}
        seen_macro = seen_micro = False
        for seed in SEEDS:
            q = _mk_area(seed, "hard")
            _, unit = _parse_area_option(q.answer_text)
            is_macro = any(n in q.text for n in macro_names)
            is_micro = any(n in q.text for n in micro_names)
            assert is_macro != is_micro, q.text
            if is_macro:
                assert unit == "cm²", f"macro area object asked in its OWN natural unit: {q.text}"
                seen_macro = True
            else:
                assert unit == "m²", f"micro area object asked in its OWN natural unit: {q.text}"
                seen_micro = True
        assert seen_macro and seen_micro, "area hard tier failed to reach both halves of the catalogue over 60 seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_options_share_the_same_decimal_place_count(self, difficulty):
        for seed in SEEDS:
            q = _mk_area(seed, difficulty)
            all_options = [q.answer_text] + list(q.distractors)
            places = [len(opt.split(" ")[0].split(".")[1]) if "." in opt else 0 for opt in all_options]
            assert len(set(places)) == 1, f"{difficulty} seed {seed}: mixed decimal-place shapes {all_options}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_are_exact_order_of_magnitude_multiples(self, difficulty):
        for seed in SEEDS:
            q = _mk_area(seed, difficulty)
            answer_val, answer_unit = _parse_area_option(q.answer_text)
            values = []
            for dist in q.distractors:
                val, unit = _parse_area_option(dist)
                assert unit == answer_unit
                values.append(val)
            expected = {answer_val * Fraction(10) ** k for k in (-2, -1, 1, 2, 3)}
            assert len(set(values)) == 3 and set(values) <= expected, (
                f"{difficulty} seed {seed}: {values} != {expected}"
            )

    def test_hard_tier_squared_factor_is_ten_thousand_not_a_hundred(self):
        """Confirm by construction that a macro object's cm2 figure is
        computed with the SQUARED length factor (10 000), never the
        linear x100 a learner might wrongly carry over from length."""
        macro_names = {n for n, _v in _MACRO_AREA}
        checked_macro = checked_micro = 0
        for seed in SEEDS:
            q = _mk_area(seed, "hard")
            is_macro = any(n in q.text for n in macro_names)
            answer_val, unit = _parse_area_option(q.answer_text)
            if is_macro:
                name = next(n for n in macro_names if n in q.text)
                canonical = dict(_MACRO_AREA)[name]
                assert answer_val == canonical * 10000
                checked_macro += 1
            else:
                micro_names_map = dict(_MICRO_AREA)
                name = next(n for n in micro_names_map if n in q.text)
                canonical = micro_names_map[name]
                assert answer_val == canonical / 10000
                checked_micro += 1
        assert checked_macro > 0 and checked_micro > 0

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-estimate-the-areas-of-different-shapes"), None)
        assert skill is not None
        assert skill.domain == "measurement"
        assert skill.answer == "mcq"
        assert skill.prerequisites == ["convert-metric-length"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 3


# ===========================================================================
# Shared behaviour: volume/capacity
# ===========================================================================


class TestEstimateVolumeCapacityReference:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = _mk_capacity(seed, difficulty)
            q2 = _mk_capacity(seed, difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = _mk_capacity(seed, difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == 3
            assert len(q.distractors) == len(set(q.distractors))

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = _mk_capacity(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, str(correct_index)).correct
            for dist in q.distractors:
                dist_index = options.index(dist)
                assert not grade_mcq(q, str(dist_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_a_deliberately_wrong_index(self, difficulty):
        for seed in SEEDS:
            q = _mk_capacity(seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            wrong_index = (correct_index + 1) % len(options)
            assert not grade_mcq(q, str(wrong_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_in_options(self, difficulty):
        for seed in SEEDS:
            q = _mk_capacity(seed, difficulty)
            assert q.answer_text in mcq_options(q)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions(self, difficulty):
        texts = {_mk_capacity(seed, difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_easy_only_draws_macro_containers_asked_in_litres(self):
        macro_names = {n for n, _v in _MACRO_CAPACITY}
        for seed in SEEDS:
            q = _mk_capacity(seed, "easy")
            assert any(n in q.text for n in macro_names), q.text
            assert "in litres." in q.text
            _, unit = _parse_capacity_option(q.answer_text)
            assert unit == "L"

    def test_medium_only_draws_micro_containers_asked_in_millilitres(self):
        micro_names = {n for n, _v in _MICRO_CAPACITY}
        for seed in SEEDS:
            q = _mk_capacity(seed, "medium")
            assert any(n in q.text for n in micro_names), q.text
            assert "in millilitres." in q.text
            _, unit = _parse_capacity_option(q.answer_text)
            assert unit == "mL"

    def test_hard_always_asks_in_the_non_natural_unit_and_reaches_both_halves(self):
        macro_names = {n for n, _v in _MACRO_CAPACITY}
        micro_names = {n for n, _v in _MICRO_CAPACITY}
        seen_macro = seen_micro = False
        for seed in SEEDS:
            q = _mk_capacity(seed, "hard")
            _, unit = _parse_capacity_option(q.answer_text)
            is_macro = any(n in q.text for n in macro_names)
            is_micro = any(n in q.text for n in micro_names)
            assert is_macro != is_micro, q.text
            if is_macro:
                assert unit == "mL", f"macro container asked in its OWN natural unit: {q.text}"
                seen_macro = True
            else:
                assert unit == "L", f"micro container asked in its OWN natural unit: {q.text}"
                seen_micro = True
        assert seen_macro and seen_micro, (
            "capacity hard tier failed to reach both halves of the catalogue over 60 seeds"
        )

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_options_share_the_same_decimal_place_count(self, difficulty):
        for seed in SEEDS:
            q = _mk_capacity(seed, difficulty)
            all_options = [q.answer_text] + list(q.distractors)
            places = [len(opt.split(" ")[0].split(".")[1]) if "." in opt else 0 for opt in all_options]
            assert len(set(places)) == 1, f"{difficulty} seed {seed}: mixed decimal-place shapes {all_options}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_are_exact_order_of_magnitude_multiples(self, difficulty):
        for seed in SEEDS:
            q = _mk_capacity(seed, difficulty)
            answer_val, answer_unit = _parse_capacity_option(q.answer_text)
            values = []
            for dist in q.distractors:
                val, unit = _parse_capacity_option(dist)
                assert unit == answer_unit
                values.append(val)
            expected = {answer_val * Fraction(10) ** k for k in (-2, -1, 1, 2, 3)}
            assert len(set(values)) == 3 and set(values) <= expected, (
                f"{difficulty} seed {seed}: {values} != {expected}"
            )

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_no_question_ever_says_volume(self, difficulty):
        for seed in SEEDS:
            q = _mk_capacity(seed, difficulty)
            assert "volume" not in q.text.lower(), q.text
            assert "capacity" in q.text.lower(), q.text

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-estimate-volume-and-capacity-of-various-objects"), None)
        assert skill is not None
        assert skill.domain == "measurement"
        assert skill.answer == "mcq"
        assert skill.prerequisites == ["convert-metric-length"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 3


# ===========================================================================
# Shared _format_options / _exact_places_needed helper
# ===========================================================================


class TestSharedFormatHelper:
    def test_exact_places_needed(self):
        assert _exact_places_needed(Fraction(18)) == 0
        assert _exact_places_needed(Fraction(18, 10)) == 1
        assert _exact_places_needed(Fraction(45, 100)) == 2

    def test_exact_places_needed_does_not_hang_on_real_area_and_capacity_values(self):
        for _name, canonical in _MACRO_AREA + _MICRO_AREA + _MACRO_CAPACITY + _MICRO_CAPACITY:
            for factor in (Fraction(1), Fraction(100), Fraction(1000), Fraction(10000)):
                tenth = canonical * factor / 10
                assert _exact_places_needed(tenth) <= 5


# ===========================================================================
# Independent-check plumbing, exercised directly
# ===========================================================================


class TestIndependentCheck:
    def test_area_conversion_check_fires_on_a_corrupted_factor_table(self, monkeypatch):
        import random

        monkeypatch.setattr(p11a_mod, "_CM2_PER_UNIT", {"cm2": Fraction(1), "m2": Fraction(999)})
        raised = False
        for seed in range(20):
            try:
                _build_area(random.Random(seed), "hard")
            except VerificationError:
                raised = True
                break
        assert raised, "no seed exercised the corrupted-area-conversion mismatch"

    def test_capacity_conversion_check_fires_on_a_corrupted_factor_table(self, monkeypatch):
        import random

        monkeypatch.setattr(p11a_mod, "_ML_PER_UNIT", {"mL": Fraction(1), "L": Fraction(999)})
        raised = False
        for seed in range(20):
            try:
                _build_capacity(random.Random(seed), "hard")
            except VerificationError:
                raised = True
                break
        assert raised, "no seed exercised the corrupted-capacity-conversion mismatch"

    def test_pairwise_distinct_fires_on_a_collapsed_distractor(self):
        from mathgen.core import check_pairwise_distinct, VerificationError as VE

        with pytest.raises(VE):
            check_pairwise_distinct(Fraction(5), Fraction(5), Fraction(6), Fraction(7))
