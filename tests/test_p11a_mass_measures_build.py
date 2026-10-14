"""Tests `estimate_mass_reference` and `review_metric_units_of_mass`: catalogue sizes, magnitude bands and
independent checks.

Mass measures build."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
import mathgen.generators.p11a_mass_measures_build as p11a_mod
from mathgen.generators.p11a_mass_measures_build import (
    _ADJACENT_PAIRS,
    _ALL_UNITS,
    _MACRO,
    _MEGA,
    _MICRO,
    _MILLI,
    _build_estimate_mass,
    _build_review_metric_units_of_mass,
    _exact_places_needed,
)

TOPIC_A = "estimate_mass_reference"
TOPIC_B = "review_metric_units_of_mass"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

_PARAMS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}


def _mk(topic, seed, difficulty):
    return make(topic, seed=seed, difficulty=difficulty, **_PARAMS[difficulty])


_OPTION_RE_MASS = re.compile(r"^(-?\d+(?:\.\d+)?) (kg|g)$")


def _parse_mass_option(opt: str) -> tuple[Fraction, str]:
    m = _OPTION_RE_MASS.match(opt)
    assert m, f"unparseable option: {opt!r}"
    return Fraction(m.group(1)), m.group(2)


# ===========================================================================
# Catalogue invariants
# ===========================================================================


def test_catalogue_sizes():
    assert len(_MACRO) == 22
    assert len(_MICRO) == 22
    assert len(_MEGA) == 12
    assert len(_MILLI) == 4


def test_catalogue_names_are_unique_within_each_table():
    for table in (_MACRO, _MICRO, _MEGA, _MILLI):
        names = [n for n, _v in table]
        assert len(names) == len(set(names))


def test_macro_canonical_values_are_at_least_one_kg():
    for name, value in _MACRO:
        assert value >= 1, f"{name}: {value} kg is not macro-scale"


def test_micro_canonical_values_are_under_1000_g():
    for name, value in _MICRO:
        assert value < 1000, f"{name}: {value} g is not micro-scale"


def test_mega_canonical_values_are_at_least_500_kg():
    for name, value in _MEGA:
        assert value >= 500, f"{name}: {value} kg is not mega-scale"


def test_milli_canonical_values_are_under_1_g():
    for name, value in _MILLI:
        assert value < 1, f"{name}: {value} g is not milli-scale"


# ===========================================================================
# Skill A: estimate_mass_reference
# ===========================================================================


class TestEstimateMassReference:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = _mk(TOPIC_A, seed, difficulty)
            q2 = _mk(TOPIC_A, seed, difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == 3
            assert len(q.distractors) == len(set(q.distractors))

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, correct_index).correct
            for dist in q.distractors:
                dist_index = options.index(dist)
                assert not grade_mcq(q, dist_index).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_a_deliberately_wrong_index(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            wrong_index = (correct_index + 1) % len(options)
            assert not grade_mcq(q, wrong_index).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_in_options(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, difficulty)
            assert q.answer_text in mcq_options(q)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions(self, difficulty):
        texts = {_mk(TOPIC_A, seed, difficulty).text for seed in SEEDS}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_easy_only_draws_macro_objects_asked_in_kg(self):
        macro_names = {n for n, _v in _MACRO}
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, "easy")
            assert any(n in q.text for n in macro_names), q.text
            assert "in kilograms." in q.text
            _, unit = _parse_mass_option(q.answer_text)
            assert unit == "kg"

    def test_medium_only_draws_micro_objects_asked_in_g(self):
        micro_names = {n for n, _v in _MICRO}
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, "medium")
            assert any(n in q.text for n in micro_names), q.text
            assert "in grams." in q.text
            _, unit = _parse_mass_option(q.answer_text)
            assert unit == "g"

    def test_hard_always_asks_in_the_non_natural_unit(self):
        macro_names = {n for n, _v in _MACRO}
        micro_names = {n for n, _v in _MICRO}
        seen_macro = seen_micro = False
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, "hard")
            _, unit = _parse_mass_option(q.answer_text)
            is_macro = any(n in q.text for n in macro_names)
            is_micro = any(n in q.text for n in micro_names)
            assert is_macro != is_micro, q.text
            if is_macro:
                assert unit == "g", f"macro object asked in its own natural unit: {q.text}"
                seen_macro = True
            else:
                assert unit == "kg", f"micro object asked in its own natural unit: {q.text}"
                seen_micro = True
        assert seen_macro and seen_micro

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_options_share_the_same_decimal_place_count(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, difficulty)
            all_options = [q.answer_text] + list(q.distractors)
            places = [len(opt.split(" ")[0].split(".")[1]) if "." in opt else 0 for opt in all_options]
            assert len(set(places)) == 1, f"{difficulty} seed {seed}: mixed decimal-place shapes {all_options}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_are_exact_order_of_magnitude_multiples(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_A, seed, difficulty)
            answer_val, answer_unit = _parse_mass_option(q.answer_text)
            values = []
            for dist in q.distractors:
                val, unit = _parse_mass_option(dist)
                assert unit == answer_unit
                values.append(val)
            expected = {answer_val * Fraction(10) ** k for k in (-2, -1, 1, 2, 3)}
            assert len(set(values)) == 3 and set(values) <= expected

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-estimate-the-mass-of-different-objects"), None)
        assert skill is not None
        assert skill.domain == "measurement"
        assert skill.answer == "mcq"
        assert skill.prerequisites == ["convert-metric-length"]
        assert skill.placement_eligible is False
        assert len([s for s in skill.sources if s.difficulty != "exam"]) == 3


# ===========================================================================
# Skill B: review_metric_units_of_mass
# ===========================================================================

_UNIT_CHOICE_Q_RE = re.compile(r"^Which unit is the most appropriate choice for measuring the mass of (.+)\?$")
_ADJACENT_Q_RE = re.compile(r"^How many (mg|g|kg) are there in (\d+) (g|kg|t)\?$")
_EQUIVALENCE_Q_RE = re.compile(r"^Which of these has the same mass as (.+)\?$")


class TestReviewMetricUnitsOfMass:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            q1 = _mk(TOPIC_B, seed, difficulty)
            q2 = _mk(TOPIC_B, seed, difficulty)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_B, seed, difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == 3
            assert len(q.distractors) == len(set(q.distractors))

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_B, seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, correct_index).correct
            for dist in q.distractors:
                dist_index = options.index(dist)
                assert not grade_mcq(q, dist_index).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_a_deliberately_wrong_index(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_B, seed, difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            wrong_index = (correct_index + 1) % len(options)
            assert not grade_mcq(q, wrong_index).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_in_options(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_B, seed, difficulty)
            assert q.answer_text in mcq_options(q)

    def test_never_contains_the_word_convert(self):
        """No rendered row of this skill may match the shape 'Convert ... to ...' --
        that is convert-metric-mass's own core shape."""
        for tier in TIERS:
            for seed in SEEDS:
                q = _mk(TOPIC_B, seed, tier)
                assert "Convert " not in q.text, f"{tier} seed {seed}: {q.text!r}"

    def test_easy_options_are_always_the_four_abbreviations(self):
        for seed in SEEDS:
            q = _mk(TOPIC_B, seed, "easy")
            m = _UNIT_CHOICE_Q_RE.match(q.text)
            assert m, q.text
            all_options = set(mcq_options(q))
            assert all_options == set(_ALL_UNITS)

    def test_easy_unit_choice_balance_at_400_seeds(self):
        """Construct-to-target: the correct unit is drawn uniformly from
        {mg, g, kg, t} -- assert this at 400 seeds, not 40."""
        counts = {u: 0 for u in _ALL_UNITS}
        for seed in range(400):
            q = _mk(TOPIC_B, seed, "easy")
            counts[q.answer_text] += 1
        for unit, count in counts.items():
            assert 60 <= count <= 140, f"{unit}: {count}/400 outside a fair-draw range: {counts}"

    def test_medium_adjacent_relation_shape(self):
        for seed in SEEDS:
            q = _mk(TOPIC_B, seed, "medium")
            m = _ADJACENT_Q_RE.match(q.text)
            assert m, q.text
            smaller, n_str, larger = m.groups()
            assert (smaller, larger) in _ADJACENT_PAIRS
            n = int(n_str)
            assert 1 <= n <= 9
            assert q.answer_text == f"{n * 1000} {smaller}"

    def test_medium_distractors_are_order_of_magnitude_family(self):
        for seed in SEEDS:
            q = _mk(TOPIC_B, seed, "medium")
            m = _ADJACENT_Q_RE.match(q.text)
            smaller = m.group(1)
            answer_val = Fraction(q.answer_text.split(" ")[0])
            values = set()
            for dist in q.distractors:
                val_str, unit = dist.split(" ")
                assert unit == smaller
                values.add(Fraction(val_str))
            assert values == {answer_val * 10, answer_val / 10, answer_val * 100}

    def test_hard_equivalence_shape_and_direction_both_seen(self):
        seen_small_given = seen_big_given = False
        for seed in SEEDS:
            q = _mk(TOPIC_B, seed, "hard")
            m = _EQUIVALENCE_Q_RE.match(q.text)
            assert m, q.text
            stated = m.group(1)
            answer_val_str, answer_unit = q.answer_text.split(" ")
            if re.match(r"^\d+ (mg|g)$", stated):
                # small-unit value stated in the question -> big-unit decimal answer
                seen_small_given = True
                assert answer_unit in ("kg", "t")
                assert "." in answer_val_str
            else:
                # big-unit decimal value stated in the question -> small integer answer
                seen_big_given = True
                assert answer_unit in ("mg", "g")
                assert "." not in answer_val_str
        assert seen_small_given and seen_big_given

    def test_at_least_20_distinct_questions_easy_and_hard(self):
        for tier in ("easy", "hard"):
            texts = {_mk(TOPIC_B, seed, tier).text for seed in SEEDS}
            assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_medium_pool_widened_to_at_least_20(self):
        """Widened to N in {1..9} (3 x 9 = 27 possible facts) by widening the generator's own draw space, not by
        padding this test to accept a thin pool."""
        texts = {_mk(TOPIC_B, seed, "medium").text for seed in SEEDS}
        assert len(texts) <= len(_ADJACENT_PAIRS) * 9 == 27
        assert len(texts) >= 20, f"medium: only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    @pytest.mark.parametrize("difficulty", ["medium", "hard"])
    def test_options_share_the_same_decimal_place_count(self, difficulty):
        for seed in SEEDS:
            q = _mk(TOPIC_B, seed, difficulty)
            all_options = [q.answer_text] + list(q.distractors)
            places = [len(opt.split(" ")[0].split(".")[1]) if "." in opt else 0 for opt in all_options]
            assert len(set(places)) == 1, f"{difficulty} seed {seed}: mixed decimal-place shapes {all_options}"

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-eview-metric-units-of-mass-and-weight-their"), None)
        assert skill is not None
        assert skill.domain == "measurement"
        assert skill.answer == "mcq"
        assert skill.prerequisites == ["convert-metric-length"]
        assert skill.placement_eligible is False
        assert len([s for s in skill.sources if s.difficulty != "exam"]) == 3


# ===========================================================================
# Independent-check plumbing, exercised directly
# ===========================================================================


class TestIndependentCheck:
    def test_conversion_check_fires_on_estimate_mass_with_a_corrupted_factor_table(self, monkeypatch):
        import random

        monkeypatch.setattr(
            p11a_mod,
            "_MG_PER_UNIT",
            {"mg": Fraction(1), "g": Fraction(999), "kg": Fraction(1_000_000), "t": Fraction(1_000_000_000)},
        )
        raised = False
        for seed in range(20):
            try:
                _build_estimate_mass(random.Random(seed), "hard")
            except VerificationError:
                raised = True
                break
        assert raised, "no seed exercised the corrupted-conversion mismatch"

    def test_conversion_check_fires_on_adjacent_relation_with_a_corrupted_factor_table(self, monkeypatch):
        import random

        monkeypatch.setattr(
            p11a_mod,
            "_MG_PER_UNIT",
            {"mg": Fraction(1), "g": Fraction(999), "kg": Fraction(1_000_000), "t": Fraction(1_000_000_000)},
        )
        raised = False
        for seed in range(20):
            try:
                _build_review_metric_units_of_mass(random.Random(seed), "medium")
            except VerificationError:
                raised = True
                break
        assert raised, "no seed exercised the corrupted-conversion mismatch"

    def test_conversion_check_fires_on_equivalence_with_a_corrupted_factor_table(self, monkeypatch):
        import random

        monkeypatch.setattr(
            p11a_mod,
            "_MG_PER_UNIT",
            {"mg": Fraction(1), "g": Fraction(1_000), "kg": Fraction(999), "t": Fraction(1_000_000_000)},
        )
        raised = False
        for seed in range(20):
            try:
                _build_review_metric_units_of_mass(random.Random(seed), "hard")
            except VerificationError:
                raised = True
                break
        assert raised, "no seed exercised the corrupted-conversion mismatch"

    def test_exact_places_needed(self):
        assert _exact_places_needed(Fraction(18)) == 0
        assert _exact_places_needed(Fraction(18, 10)) == 1
        assert _exact_places_needed(Fraction(45, 100)) == 2
