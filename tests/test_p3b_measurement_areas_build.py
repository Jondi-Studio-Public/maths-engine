"""Area formulas and irregular decomposition (senior-use-formulas-to-calculate-areas-of-regular-shapes,
senior-find-the-area-of-irregular-figures-by-decomposition)."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p3b_measurement_areas_build import (
    _verify_easy_decomposition,
    _verify_hard_decomposition,
    _verify_medium_decomposition,
    _verify_rectangle_or_square,
    _verify_trapezium,
    _verify_triangle_or_parallelogram,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

AREA_FORMULA_TOPIC = "use_formulas_to_calculate_areas"
IRREGULAR_TOPIC = "find_the_area_of_irregular_figures"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [AREA_FORMULA_TOPIC, IRREGULAR_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [AREA_FORMULA_TOPIC, IRREGULAR_TOPIC])
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


@pytest.mark.parametrize("topic", [AREA_FORMULA_TOPIC, IRREGULAR_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Mirrors `test_all_topics_usually_have_enough_distractors`'s own bar
    (>=2 distractors on >=85% of seeds) for a bare `make(topic, seed=s)`
    call with no `difficulty` argument."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [AREA_FORMULA_TOPIC, IRREGULAR_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [AREA_FORMULA_TOPIC, IRREGULAR_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        if isinstance(question.answer, int):
            wrong = question.answer + 1
        else:
            wrong = question.answer + 1.0
        assert not grade_number(question, str(wrong)).correct


# --- use_formulas_to_calculate_areas -----------------------------------------


def test_easy_rectangle_square_area_always_exact():
    """Easy-tier answers should always be integers (no decimals)."""
    for seed in SEEDS:
        question = make(AREA_FORMULA_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text


def test_medium_triangle_parallelogram_area_terminates():
    """Medium-tier answers (triangle has 1/2 factor, parallelogram is exact)
    terminate within 1 decimal place."""
    for seed in SEEDS:
        question = make(AREA_FORMULA_TOPIC, seed=seed, difficulty="medium")
        if "." in question.answer_text:
            places = len(question.answer_text.split(".")[1])
            assert places <= 1


def test_hard_trapezium_area_terminates():
    """Hard-tier trapezium answers (1/2 factor) terminate within 1 decimal place."""
    for seed in SEEDS:
        question = make(AREA_FORMULA_TOPIC, seed=seed, difficulty="hard")
        if "." in question.answer_text:
            places = len(question.answer_text.split(".")[1])
            assert places <= 1


def test_hard_trapezium_forces_parallel_sides_distinct():
    """The trapezium's two parallel sides must be distinct (a != b) so that
    the 'uses-one-parallel-side' misconception always produces a different answer."""
    for seed in range(100):
        question = make(AREA_FORMULA_TOPIC, seed=seed, difficulty="hard")
        # Extract the two parallel side lengths from the text.
        # Format: "parallel sides of X cm and Y cm"
        import re

        match = re.search(r"parallel sides of (\d+) cm and (\d+) cm", question.text)
        assert match, f"Could not parse trapezium sides from: {question.text}"
        a, b = int(match.group(1)), int(match.group(2))
        assert a != b, f"Trapezium sides should be distinct: {a} vs {b}"


def test_area_formula_rectangle_independent_check_fires_on_corrupted_answer():
    from fractions import Fraction

    _verify_rectangle_or_square(Fraction(5), Fraction(6), Fraction(30))  # must not raise
    with pytest.raises(VerificationError):
        _verify_rectangle_or_square(Fraction(5), Fraction(6), Fraction(31))


def test_area_formula_triangle_independent_check_fires_on_corrupted_answer():
    from fractions import Fraction

    _verify_triangle_or_parallelogram(Fraction(6), Fraction(4), is_triangle=True, expected=Fraction(12))
    with pytest.raises(VerificationError):
        _verify_triangle_or_parallelogram(Fraction(6), Fraction(4), is_triangle=True, expected=Fraction(13))


def test_area_formula_trapezium_independent_check_fires_on_corrupted_answer():
    from fractions import Fraction

    # (3 + 5) / 2 * 4 = 16
    _verify_trapezium(Fraction(3), Fraction(5), Fraction(4), Fraction(16))
    with pytest.raises(VerificationError):
        _verify_trapezium(Fraction(3), Fraction(5), Fraction(4), Fraction(17))


def test_area_formula_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(AREA_FORMULA_TOPIC, seed=1, mode="unknown")


# --- find_the_area_of_irregular_figures --------------------------------------


def test_easy_irregular_two_rectangles_area_terminates():
    """Easy-tier answers (sum of two exact rectangle areas) are always integers."""
    for seed in SEEDS:
        question = make(IRREGULAR_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text


def test_medium_irregular_rect_triangle_area_terminates():
    """Medium-tier answers (rectangle + triangle with 1/2 factor) terminate within 1 dp."""
    for seed in SEEDS:
        question = make(IRREGULAR_TOPIC, seed=seed, difficulty="medium")
        if "." in question.answer_text:
            places = len(question.answer_text.split(".")[1])
            assert places <= 1


def test_hard_irregular_notch_area_terminates():
    """Hard-tier answers (big area - cut area) terminate within 1 dp."""
    for seed in SEEDS:
        question = make(IRREGULAR_TOPIC, seed=seed, difficulty="hard")
        if "." in question.answer_text:
            places = len(question.answer_text.split(".")[1])
            assert places <= 1


def test_hard_irregular_notch_is_positive():
    """Hard-tier notch decomposition must have the cut-out strictly inside
    the big rectangle, so the answer is always positive."""
    for seed in range(100):
        question = make(IRREGULAR_TOPIC, seed=seed, difficulty="hard")
        answer = float(question.answer)
        assert answer > 0, f"Answer should be positive: {answer}"


def test_easy_irregular_independent_check_fires_on_corrupted_answer():
    from fractions import Fraction

    _verify_easy_decomposition(Fraction(12), Fraction(8), Fraction(20))
    with pytest.raises(VerificationError):
        _verify_easy_decomposition(Fraction(12), Fraction(8), Fraction(21))


def test_medium_irregular_independent_check_fires_on_corrupted_answer():
    from fractions import Fraction

    # Rectangle 20, Triangle 10
    _verify_medium_decomposition(Fraction(20), Fraction(10), Fraction(30))
    with pytest.raises(VerificationError):
        _verify_medium_decomposition(Fraction(20), Fraction(10), Fraction(31))


def test_hard_irregular_independent_check_fires_on_corrupted_answer():
    from fractions import Fraction

    # Big 100, Cut 30
    _verify_hard_decomposition(Fraction(100), Fraction(30), Fraction(70))
    with pytest.raises(VerificationError):
        _verify_hard_decomposition(Fraction(100), Fraction(30), Fraction(71))


def test_irregular_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(IRREGULAR_TOPIC, seed=1, mode="unknown")


# --- skill contracts ---------------------------------------------------------


def test_p3b_areas_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    area_formula_skill = by_id["senior-use-formulas-to-calculate-areas-of-regular-shapes"]
    irregular_skill = by_id["senior-find-the-area-of-irregular-figures-by-decomposition"]

    # Area formula skill checks
    assert area_formula_skill.prerequisites == ["convert-metric-length"]
    assert area_formula_skill.answer == "number"
    assert area_formula_skill.placement_eligible is False
    assert area_formula_skill.status == "active"
    assert area_formula_skill.learner_band == "general"
    assert [source.difficulty for source in area_formula_skill.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in area_formula_skill.sources} == {AREA_FORMULA_TOPIC}
    assert [source.stages for source in area_formula_skill.sources] == [[6], [6], [6, "maintenance"]]

    # Irregular skill checks
    assert irregular_skill.prerequisites == ["convert-metric-length"]
    assert irregular_skill.answer == "number"
    assert irregular_skill.placement_eligible is False
    assert irregular_skill.status == "active"
    assert irregular_skill.learner_band == "general"
    assert [source.difficulty for source in irregular_skill.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in irregular_skill.sources} == {IRREGULAR_TOPIC}
    assert [source.stages for source in irregular_skill.sources] == [[6], [6], [6, "maintenance"]]
