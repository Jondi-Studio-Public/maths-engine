"""Energy-unit conversions (senior-convert-from-one-unit-of-energy-to-another)."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_conversion
from mathgen.generators.p5a_measurement_energy_units_build import _FACTORS_J

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

TOPIC = "convert_energy_units"


# --- determinism / distractor hygiene ----------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generator_is_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        repeat = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85


def test_bare_call_default_has_enough_distractors():
    """Bare `make(topic, seed=s)` call with no `difficulty` argument."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(TOPIC, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_text_never_uses_scientific_notation(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "e" not in question.answer_text.lower()


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        if isinstance(question.answer, int):
            wrong = question.answer + 1
        else:
            wrong = question.answer + 1.0
        assert not grade_number(question, str(wrong)).correct


# --- convert_energy_units ---------------------------------------------------


def test_easy_j_kj_conversion_is_always_exact():
    """Easy tier (J<->kJ conversion) always produces exact integers."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text


def test_medium_kj_mj_terminates_within_three_decimal_places():
    """Medium tier (kJ<->MJ conversion) terminates within 3 decimal places."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="medium")
        if "." in question.answer_text:
            places = len(question.answer_text.split(".")[1])
            assert places <= 3


def test_hard_kwh_kj_conversion_is_always_exact():
    """Hard tier (kWh<->kJ conversion) always produces exact integers because
    we choose values such that kWh*3600 or kJ/3600 are always whole numbers."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="hard")
        assert "." not in question.answer_text


def test_independent_check_fires_on_corrupted_conversion():
    """check_conversion with joule pivot catches an error."""
    from fractions import Fraction

    check_conversion(
        Fraction(1000), "J", "kJ", Fraction(1), _FACTORS_J, label="conversion"
    )  # 1000 J = 1 kJ, must not raise
    with pytest.raises(VerificationError):
        check_conversion(Fraction(1000), "J", "kJ", Fraction(2), _FACTORS_J, label="conversion")


def test_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="unknown")


# --- skill loading -----------------------------------------------------------


def test_skill_loads_with_expected_properties():
    """senior-convert-from-one-unit-of-energy-to-another skill loads correctly."""
    _, skills = load_skills(ROOT / "skills")
    skill = next((s for s in skills if s.id == "senior-convert-from-one-unit-of-energy-to-another"), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert "convert-metric-length" in skill.prerequisites
    assert len(skill.sources) == 3  # easy, medium, hard
