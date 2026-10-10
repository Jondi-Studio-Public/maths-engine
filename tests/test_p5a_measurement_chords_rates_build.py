"""Chords theorem and rate conversions (senior-chords-of-equal-length-subtend-equal-angles-at,
senior-convert-between-units-for-rates-for-example-km)."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5a_measurement_chords_rates_build import (
    _verify_easy_chords_forward,
    _verify_medium_chords_converse,
    _verify_hard_chords_chained,
    _verify_easy_kmh_to_ms,
    _verify_medium_ms_to_kmh,
    _verify_hard_km_per_min_to_ms,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

CHORDS_TOPIC = "chords_equal_length_equal_angles"
RATES_TOPIC = "convert_units_for_rates"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [CHORDS_TOPIC, RATES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [CHORDS_TOPIC, RATES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85


@pytest.mark.parametrize("topic", [CHORDS_TOPIC, RATES_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Bare make(topic, seed=s) with no difficulty should have >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [CHORDS_TOPIC, RATES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [CHORDS_TOPIC, RATES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        if isinstance(question.answer, int):
            wrong = question.answer + 1
        else:
            wrong = question.answer + 1.0
        assert not grade_number(question, str(wrong)).correct


# --- chords_equal_length_equal_angles ----------------------------------------


def test_easy_chords_always_integer_degrees():
    """Easy-tier chord angle answers are always integers (degrees)."""
    for seed in SEEDS:
        question = make(CHORDS_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_medium_chords_always_integer_cm():
    """Medium-tier chord length answers are always integers (cm)."""
    for seed in SEEDS:
        question = make(CHORDS_TOPIC, seed=seed, difficulty="medium")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_hard_chords_always_integer_cm():
    """Hard-tier chord length answers are always integers (cm)."""
    for seed in SEEDS:
        question = make(CHORDS_TOPIC, seed=seed, difficulty="hard")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_easy_chords_excludes_90_degrees():
    """Easy tier should exclude theta=90 so distractors don't coincide."""
    for seed in range(100):
        question = make(CHORDS_TOPIC, seed=seed, difficulty="easy")
        # Parse the angle from the question text
        import re

        match = re.search(r"subtends an angle of (\d+) degrees", question.text)
        assert match
        theta = int(match.group(1))
        assert theta != 90


def test_easy_chords_independent_check_fires_on_corrupted_answer():
    """Independent check for easy tier should fire on mismatch."""
    _verify_easy_chords_forward(length_ab=8, angle_ab=50, length_cd=8, angle_cd=50, answer=50)
    with pytest.raises(VerificationError):
        _verify_easy_chords_forward(length_ab=8, angle_ab=50, length_cd=8, angle_cd=51, answer=50)


def test_medium_chords_independent_check_fires_on_corrupted_answer():
    """Independent check for medium tier should fire on mismatch."""
    _verify_medium_chords_converse(angle_ab=60, angle_cd=60, length_ab=10, length_cd=10, answer=10)
    with pytest.raises(VerificationError):
        _verify_medium_chords_converse(angle_ab=60, angle_cd=60, length_ab=10, length_cd=11, answer=10)


def test_hard_chords_independent_check_fires_on_corrupted_answer():
    """Independent check for hard tier should fire on mismatch."""
    _verify_hard_chords_chained(length_ab=8, angle_ab=45, length_cd=8, angle_cd=45, angle_ef=45, length_ef=8, answer=8)
    with pytest.raises(VerificationError):
        _verify_hard_chords_chained(
            length_ab=8, angle_ab=45, length_cd=8, angle_cd=45, angle_ef=45, length_ef=8, answer=9
        )


def test_chords_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(CHORDS_TOPIC, seed=1, mode="unknown")


# --- convert_units_for_rates -------------------------------------------------


def test_easy_kmh_to_ms_always_integer():
    """Easy-tier km/h to m/s conversion produces exact integer m/s."""
    for seed in SEEDS:
        question = make(RATES_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_medium_ms_to_kmh_has_one_decimal_place():
    """Medium-tier m/s to km/h conversion produces 1 decimal place or integer."""
    for seed in SEEDS:
        question = make(RATES_TOPIC, seed=seed, difficulty="medium")
        if "." in question.answer_text:
            places = len(question.answer_text.split(".")[1])
            assert places == 1, f"Expected 1 decimal place, got {places}: {question.answer_text}"


def test_hard_km_per_min_to_ms_always_integer():
    """Hard-tier km/min to m/s conversion produces exact integer."""
    for seed in SEEDS:
        question = make(RATES_TOPIC, seed=seed, difficulty="hard")
        assert "." not in question.answer_text
        assert question.answer_text.isdigit()


def test_easy_kmh_speed_is_multiple_of_3_6():
    """Easy tier km/h speed should be an exact multiple of 3.6."""
    for seed in range(50):
        question = make(RATES_TOPIC, seed=seed, difficulty="easy")
        import re

        match = re.search(r"travels at ([\d.]+) km/h", question.text)
        assert match
        speed_str = match.group(1)
        speed_float = float(speed_str)
        # Check if it's a multiple of 3.6 (allowing small floating-point error)
        quotient = speed_float / 3.6
        assert abs(quotient - round(quotient)) < 0.001


def test_easy_kmh_to_ms_independent_check_fires_on_corrupted():
    """Independent check for easy should fire on mismatch."""
    from fractions import Fraction

    _verify_easy_kmh_to_ms(speed_kmh=Fraction(36), answer_ms=10)
    with pytest.raises(VerificationError):
        _verify_easy_kmh_to_ms(speed_kmh=Fraction(36), answer_ms=11)


def test_medium_ms_to_kmh_independent_check_fires_on_corrupted():
    """Independent check for medium should fire on mismatch."""
    from fractions import Fraction

    _verify_medium_ms_to_kmh(speed_ms=10, answer_kmh=Fraction(36))
    with pytest.raises(VerificationError):
        _verify_medium_ms_to_kmh(speed_ms=10, answer_kmh=Fraction(37))


def test_hard_km_per_min_to_ms_independent_check_fires_on_corrupted():
    """Independent check for hard should fire on mismatch."""
    from fractions import Fraction

    # 3 km/min * 50/3 = 50 m/s
    _verify_hard_km_per_min_to_ms(speed_km_per_min=3, answer_ms=Fraction(50))
    with pytest.raises(VerificationError):
        _verify_hard_km_per_min_to_ms(speed_km_per_min=3, answer_ms=Fraction(51))


def test_hard_km_per_min_is_multiple_of_3():
    """Hard tier km/min speed should be a multiple of 3."""
    for seed in range(50):
        question = make(RATES_TOPIC, seed=seed, difficulty="hard")
        import re

        match = re.search(r"flying at (\d+) km per minute", question.text)
        assert match
        k = int(match.group(1))
        assert k % 3 == 0, f"k should be multiple of 3, got {k}"
        assert k in {3, 6, 9, 12}, f"k should be in {{3, 6, 9, 12}}, got {k}"


def test_rates_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(RATES_TOPIC, seed=1, mode="unknown")


# --- both skills load correctly through the API -----------


def test_both_skills_load_through_api():
    """Both skills should load correctly with correct prerequisites and answer type."""
    domains, skills = load_skills()
    skills_dict = {skill.id: skill for skill in skills}

    chords_skill = skills_dict.get("senior-chords-of-equal-length-subtend-equal-angles-at")
    assert chords_skill is not None
    assert "convert-metric-length" in chords_skill.prerequisites
    assert chords_skill.answer == "number"
    assert chords_skill.placement_eligible is False
    assert len(chords_skill.sources) == 3  # easy, medium, hard

    rates_skill = skills_dict.get("senior-convert-between-units-for-rates-for-example-km")
    assert rates_skill is not None
    assert "convert-metric-length" in rates_skill.prerequisites
    assert rates_skill.answer == "number"
    assert rates_skill.placement_eligible is False
    assert len(rates_skill.sources) == 3  # easy, medium, hard
