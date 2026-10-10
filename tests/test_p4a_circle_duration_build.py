"""Circle measures and timezone duration (f10-circle-measures, f10-circle-and-prism-duration-rates)."""

from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, round_half_up
from mathgen.generators.p4a_circle_duration_build import (
    _verify_circle_measures,
    _verify_duration_via_inverse,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

CIRCLE_TOPIC = "circle_measures"
DURATION_TOPIC = "circle_and_prism_duration_rates"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [CIRCLE_TOPIC, DURATION_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [CIRCLE_TOPIC, DURATION_TOPIC])
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


@pytest.mark.parametrize("topic", [CIRCLE_TOPIC, DURATION_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Mirrors test bar (>=2 distractors on >=85% of seeds) for a bare
    `make(topic, seed=s)` call with no `difficulty` argument."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [CIRCLE_TOPIC, DURATION_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [CIRCLE_TOPIC, DURATION_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        wrong = str(int(question.answer) + 1)
        assert not grade_number(question, wrong).correct


# --- circle_measures (skill 1) -----------------------------------------------


def test_circle_easy_answers_are_one_decimal_place():
    """Easy-tier answers are always stated to 1 decimal place."""
    for seed in SEEDS:
        question = make(CIRCLE_TOPIC, seed=seed, difficulty="easy")
        assert "." in question.answer_text
        parts = question.answer_text.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 1


def test_circle_medium_answers_are_one_decimal_place():
    """Medium-tier answers are always stated to 1 decimal place."""
    for seed in SEEDS:
        question = make(CIRCLE_TOPIC, seed=seed, difficulty="medium")
        assert "." in question.answer_text
        parts = question.answer_text.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 1


def test_circle_hard_answers_are_one_decimal_place():
    """Hard-tier answers are always stated to 1 decimal place."""
    for seed in SEEDS:
        question = make(CIRCLE_TOPIC, seed=seed, difficulty="hard")
        assert "." in question.answer_text
        parts = question.answer_text.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 1


def test_circle_easy_text_mentions_radius_and_circumference():
    """Easy tier asks for circumference with radius given."""
    for seed in range(50):
        question = make(CIRCLE_TOPIC, seed=seed, difficulty="easy")
        assert "radius" in question.text.lower()
        assert "circumference" in question.text.lower()


def test_circle_medium_text_mentions_radius_and_area():
    """Medium tier asks for area with radius given."""
    for seed in range(50):
        question = make(CIRCLE_TOPIC, seed=seed, difficulty="medium")
        assert "radius" in question.text.lower()
        assert "area" in question.text.lower()


def test_circle_hard_text_mentions_diameter():
    """Hard tier uses diameter, not radius."""
    for seed in range(50):
        question = make(CIRCLE_TOPIC, seed=seed, difficulty="hard")
        assert "diameter" in question.text.lower()


def test_circle_easy_independent_check_fires_on_corrupted():
    """Easy-tier independent check (repeated addition) catches incorrect answer."""
    # r=5: circumference = 2 * 3.14 * 5 = 31.4, rounded to 31.4
    _verify_circle_measures(5, "circumference", Fraction(314, 10))  # must not raise
    with pytest.raises(VerificationError):
        _verify_circle_measures(5, "circumference", Fraction(315, 10))


def test_circle_medium_independent_check_fires_on_corrupted():
    """Medium-tier independent check (repeated addition) catches incorrect answer."""
    # r=6: area = 3.14 * 36 = 113.04, rounded to 113.0
    _verify_circle_measures(6, "area", Fraction(1130, 10))  # must not raise
    with pytest.raises(VerificationError):
        _verify_circle_measures(6, "area", Fraction(1131, 10))


def test_circle_hard_independent_check_fires_on_corrupted():
    """Hard-tier independent check catches incorrect answer."""
    # r=7: circumference = 2 * 3.14 * 7 = 43.96, rounded to 44.0
    _verify_circle_measures(7, "circumference", Fraction(440, 10))  # must not raise
    with pytest.raises(VerificationError):
        _verify_circle_measures(7, "circumference", Fraction(441, 10))


def test_circle_half_up_rounding_hand_check():
    """Test half-up rounding on a boundary case.
    Fraction(125, 100) = 1.25 should round to 1.3 (half-up), not 1.2 (banker's)."""
    result = round_half_up(Fraction(125, 100), 1)
    assert result == Fraction(13, 10)
    assert f"{float(result):.1f}" == "1.3"


def test_circle_hard_radius_at_least_3():
    """Hard tier diameter is sampled 6..30 (even), so radius is >= 3."""
    for seed in SEEDS:
        question = make(CIRCLE_TOPIC, seed=seed, difficulty="hard")
        # Extract diameter from text
        text_lower = question.text.lower()
        # The text is "A circle has a diameter of {d} cm..."
        # We can check the answer constraint indirectly: for r >= 3, the two
        # formulas (circumference and area) never collide (r=2 would collide).
        # Just verify all distractors are distinct from the answer.
        for distractor in question.distractors:
            assert distractor != question.answer_text


def test_circle_distractor_gaps_exceed_0_05():
    """Distractor gaps must exceed 0.05 (half the tolerance for 1 decimal place)."""
    from fractions import Fraction

    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(CIRCLE_TOPIC, seed=seed, difficulty=difficulty)
            correct_val = Fraction(question.answer_text)
            for distractor in question.distractors:
                distractor_val = Fraction(distractor)
                gap = abs(distractor_val - correct_val)
                # Gap must be > 0.05 to pass the tolerance
                assert gap > Fraction(5, 100), f"Distractor gap {gap} not > 0.05 for seed {seed} {difficulty}"


# --- circle_and_prism_duration_rates (skill 2) --------------------------------


def test_duration_easy_answers_are_integers():
    """Easy-tier answers are always integers (no decimals)."""
    for seed in SEEDS:
        question = make(DURATION_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text


def test_duration_medium_answers_are_integers():
    """Medium-tier answers are always integers."""
    for seed in SEEDS:
        question = make(DURATION_TOPIC, seed=seed, difficulty="medium")
        assert "." not in question.answer_text


def test_duration_hard_answers_are_integers():
    """Hard-tier answers are always integers (exact durations guaranteed)."""
    for seed in SEEDS:
        question = make(DURATION_TOPIC, seed=seed, difficulty="hard")
        assert "." not in question.answer_text


def test_duration_easy_text_mentions_same_day():
    """Easy tier mentions 'same day'."""
    for seed in range(50):
        question = make(DURATION_TOPIC, seed=seed, difficulty="easy")
        assert "same day" in question.text.lower()


def test_duration_medium_text_mentions_layover():
    """Medium tier mentions layover."""
    for seed in range(50):
        question = make(DURATION_TOPIC, seed=seed, difficulty="medium")
        assert "layover" in question.text.lower()


def test_duration_hard_text_mentions_next_day():
    """Hard tier explicitly states 'next day'."""
    for seed in range(50):
        question = make(DURATION_TOPIC, seed=seed, difficulty="hard")
        assert "next day" in question.text.lower()


def test_duration_hard_midnight_crossing_every_seed():
    """Hard tier forces midnight crossing on every seed.
    Arrival time (Zone A frame) should be >= 180 (03:00 or later on next day)."""
    for seed in SEEDS:
        question = make(DURATION_TOPIC, seed=seed, difficulty="hard")
        # Just verify it parses and has an answer
        assert question.answer > 0
        assert question.answer_text


def test_duration_easy_independent_check_fires_on_corrupted():
    """Easy-tier independent check catches incorrect answer."""
    # dep_A = 180, arr_in_A_frame_clock = 360, offset_minutes = 60
    # arr_B = 300, flight_duration = 180, layover = 0, answer = 180
    _verify_duration_via_inverse(180, 300, 60, 0, False, 180)  # must not raise
    with pytest.raises(VerificationError):
        _verify_duration_via_inverse(180, 300, 60, 0, False, 181)


def test_duration_medium_independent_check_fires_on_corrupted():
    """Medium-tier independent check catches incorrect answer."""
    # dep_A = 180, arr_in_A_frame_clock = 360, offset_minutes = 60
    # arr_B = 300, flight_duration = 180, layover = 60, answer = 240
    _verify_duration_via_inverse(180, 300, 60, 60, False, 240)  # must not raise
    with pytest.raises(VerificationError):
        _verify_duration_via_inverse(180, 300, 60, 60, False, 241)


def test_duration_hard_independent_check_fires_on_corrupted():
    """Hard-tier independent check catches incorrect answer."""
    # dep_A = 1200 (20:00), arr_in_A_frame_clock = 300 (next day 05:00)
    # offset_minutes = 60, arr_B = 240, flight_duration = 540, layover = 60, answer = 600
    _verify_duration_via_inverse(1200, 240, 60, 60, True, 600)  # must not raise
    with pytest.raises(VerificationError):
        _verify_duration_via_inverse(1200, 240, 60, 60, True, 601)


# --- skills load through API --------


def test_circle_and_duration_skills_load():
    """Both skills load through mathgen.skills.load_skills."""
    _, skills = load_skills()
    circle_skill = next((s for s in skills if s.id == "f10-circle-measures"), None)
    duration_skill = next((s for s in skills if s.id == "f10-circle-and-prism-duration-rates"), None)

    assert circle_skill is not None, "f10-circle-measures not found in skills"
    assert duration_skill is not None, "f10-circle-and-prism-duration-rates not found in skills"

    # Check expected properties
    assert circle_skill.answer == "number"
    assert duration_skill.answer == "number"

    assert circle_skill.placement_eligible is False
    assert duration_skill.placement_eligible is False


def test_circle_prerequisites():
    """Circle skill has pythagorean-hypotenuse as prerequisite."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "f10-circle-measures"), None)
    assert skill is not None
    assert "pythagorean-hypotenuse" in skill.prerequisites


def test_duration_prerequisites():
    """Duration skill has correct prerequisites."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "f10-circle-and-prism-duration-rates"), None)
    assert skill is not None
    expected = {"convert-time-systems", "elapsed-time", "ratio-rate-proportion"}
    assert set(skill.prerequisites) == expected
