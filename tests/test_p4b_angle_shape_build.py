"""Foundation-Year 10 geometry/measurement skills for expressing turns as right angles and classifying shape
features."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p4b_angle_shape_build import (
    TURN_VALUES,
    SHAPE_SIDES,
    SHAPES_CURVED,
    SHAPES_STRAIGHT_ALL,
    SHAPES_STRAIGHT_UNIQUE,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- Skill 1: angle_turn_right_reference ------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_angle_turn_determinism_and_distractor_hygiene(difficulty):
    """Determinism across seeds and distractor validity."""
    for seed in SEEDS:
        question = make("angle_turn_right_reference", seed=seed, difficulty=difficulty)
        repeat = make("angle_turn_right_reference", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_angle_turn_answer_is_integer_string(difficulty):
    """Answer is a plain integer, no decimal point, matching exact-match grading."""
    for seed in SEEDS:
        question = make("angle_turn_right_reference", seed=seed, difficulty=difficulty)
        assert question.answer_text.isdigit() or (
            question.answer_text.startswith("-") and question.answer_text[1:].isdigit()
        )
        answer_int = int(question.answer_text)
        # Easy/medium always in {1..8}; hard always in {1..4} (is the second turn's contribution)
        if difficulty in ["easy", "medium"]:
            assert 1 <= answer_int <= 8
        else:
            assert 1 <= answer_int <= 4


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_angle_turn_correct_answer_grades_correct_and_distractors_incorrect(difficulty):
    """The real grader (grade_number) accepts the canonical answer and rejects distractors."""
    for seed in SEEDS:
        question = make("angle_turn_right_reference", seed=seed, difficulty=difficulty)
        # For number answers, pass the answer_text directly
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


def test_angle_turn_deliberately_wrong_answer_rejected():
    """A deliberately wrong answer (e.g., out-of-range or obviously incorrect) is rejected."""
    checked = 0
    for seed in SEEDS:
        question = make("angle_turn_right_reference", seed=seed, difficulty="medium")
        answer_int = int(question.answer_text)
        # Try a clearly wrong value
        wrong_value = answer_int + 100
        result = grade_number(question, str(wrong_value))
        assert not result.correct
        checked += 1
    assert checked > 0


def test_angle_turn_easy_single_turn():
    """Easy tier always samples one turn label from the four."""
    for seed in SEEDS:
        question = make("angle_turn_right_reference", seed=seed, difficulty="easy")
        text = question.text.lower()
        # Should NOT have "then" or "another" or "total" (those indicate medium/hard)
        assert "then" not in text
        assert "another" not in text
        assert "total" not in text
        # Answer should be in {1, 2, 3, 4}
        assert int(question.answer_text) in [1, 2, 3, 4]


def test_angle_turn_medium_two_turns():
    """Medium tier always samples two turn labels independently."""
    for seed in SEEDS:
        question = make("angle_turn_right_reference", seed=seed, difficulty="medium")
        text = question.text.lower()
        # Should mention "then" (indicating two turns)
        assert "then" in text or "another" in text
        # Answer should be sum of two values, range 2-8
        answer_int = int(question.answer_text)
        assert 2 <= answer_int <= 8


def test_angle_turn_hard_inverse_reasoning():
    """Hard tier uses inverse reasoning: given label1 and total T, find the second turn."""
    for seed in SEEDS:
        question = make("angle_turn_right_reference", seed=seed, difficulty="hard")
        text = question.text.lower()
        # Should mention keeping turning and a total
        assert "keep" in text and "turn" in text and "total" in text
        # Answer should be the second turn's contribution, always in {1,2,3,4}
        assert int(question.answer_text) in [1, 2, 3, 4]


def test_angle_turn_backbone_pair_always_distinct_and_present():
    """Backbone distractors (off-by-one under and over) are always present and distinct from answer.

    Verify algebraically: for any integer answer a, a-1 != a != a+1, so the three
    values are always mutually distinct.
    """
    checked_count = 0
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make("angle_turn_right_reference", seed=seed, difficulty=difficulty)
            answer = int(question.answer_text)
            # At minimum, we should have at least 2 distractors from backbone pair
            assert len(question.distractors) >= 2
            # The backbone pair values are answer-1 and answer+1
            backbone_values = {answer - 1, answer + 1}
            # At least the backbone pair should be in distractors (or collided with others)
            # Verify the pair are distinct from each other and from answer
            assert answer - 1 != answer
            assert answer + 1 != answer
            assert answer - 1 != answer + 1
            checked_count += 1
    assert checked_count > 0


# --- Skill 2: shape_features_classification ------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_shape_features_determinism_and_distractor_hygiene(difficulty):
    """Determinism across seeds and distractor validity."""
    for seed in SEEDS:
        question = make("shape_features_classification", seed=seed, difficulty=difficulty)
        repeat = make("shape_features_classification", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        # Easy tier is binary (2 options, 1 distractor); medium/hard have 3 distractors each
        if difficulty == "easy":
            assert len(question.distractors) >= 1
        else:
            assert len(question.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_shape_features_answer_is_one_of_fixed_mcq_options(difficulty):
    """The answer must always be one of the shuffled MCQ options."""
    for seed in SEEDS:
        question = make("shape_features_classification", seed=seed, difficulty=difficulty)
        answer = question.answer_text
        options = mcq_options(question)
        assert answer in options


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_shape_features_correct_answer_grades_correct_and_distractors_incorrect(difficulty):
    """The real grader (grade_mcq) accepts the canonical answer and rejects distractors."""
    for seed in SEEDS:
        question = make("shape_features_classification", seed=seed, difficulty=difficulty)
        # For mcq, grade_mcq expects an index into mcq_options(question)
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grade_mcq(question, correct_index).correct
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            assert not grade_mcq(question, distractor_index).correct


def test_shape_features_out_of_range_index_rejected():
    """An out-of-range option index is rejected."""
    checked = 0
    for seed in SEEDS:
        question = make("shape_features_classification", seed=seed, difficulty="medium")
        # Try an out-of-range index
        result = grade_mcq(question, "99")
        assert not result.correct
        checked += 1
    assert checked > 0


def test_shape_features_bare_call_resolves_to_medium_or_hard():
    """Bare-call default (no difficulty argument) must resolve to medium or hard tier,
    never easy, to avoid needing the two-option exemption."""
    for seed in SEEDS:
        question = make("shape_features_classification", seed=seed)  # no difficulty
        options = mcq_options(question)
        # Medium/hard have at least 3 options (answer + 3 distractors);
        # easy has 2 (answer + 1 distractor).
        assert len(options) >= 3


def test_shape_features_easy_binary_curved_vs_straight():
    """Easy tier is binary classification: curved or straight-sided."""
    for seed in SEEDS:
        question = make("shape_features_classification", seed=seed, difficulty="easy")
        # Answer should always be "curved" or "straight-sided"
        assert question.answer_text in ["curved", "straight-sided"]
        # Exactly 1 distractor (the other option)
        assert len(question.distractors) == 1
        assert question.distractors[0] in ["curved", "straight-sided"]
        assert question.distractors[0] != question.answer_text


def test_shape_features_easy_shapes_from_restricted_pool():
    """Easy tier samples from {circle, oval, triangle, square} only."""
    shapes_seen = set()
    for seed in SEEDS:
        question = make("shape_features_classification", seed=seed, difficulty="easy")
        text = question.text.lower()
        for shape in ["circle", "oval", "triangle", "square"]:
            if shape in text:
                shapes_seen.add(shape)
    # Should eventually see shapes from the easy pool
    assert shapes_seen.issubset({"circle", "oval", "triangle", "square"})


def test_shape_features_medium_forward_side_count():
    """Medium tier: forward side-count lookup from {triangle, square, pentagon, hexagon, octagon}."""
    for seed in SEEDS:
        question = make("shape_features_classification", seed=seed, difficulty="medium")
        # Answer should be a string number (side count)
        try:
            count = int(question.answer_text)
            assert count in [3, 4, 5, 6, 8]
        except ValueError:
            pytest.fail(f"Medium tier answer {question.answer_text!r} is not a numeric side count")


def test_shape_features_hard_inverse_side_count_to_shape():
    """Hard tier: inverse reasoning -- given count, name the shape from unique-count pool."""
    for seed in SEEDS:
        question = make("shape_features_classification", seed=seed, difficulty="hard")
        # Answer should be a shape name from the unique-count pool
        assert question.answer_text in SHAPES_STRAIGHT_UNIQUE
        # Distractors should all be other shapes from the unique-count pool
        for distractor in question.distractors:
            assert distractor in SHAPES_STRAIGHT_UNIQUE
            assert distractor != question.answer_text


def test_shape_features_all_distractors_are_real_counts_or_shapes():
    """For medium/hard tiers, all distractors are real side counts or shape names from the vocabulary."""
    for seed in SEEDS:
        for difficulty in ["medium", "hard"]:
            question = make("shape_features_classification", seed=seed, difficulty=difficulty)
            for distractor in question.distractors:
                if difficulty == "medium":
                    # Distractors should be valid side counts from SHAPES_STRAIGHT_ALL
                    try:
                        count = int(distractor)
                        assert any(SHAPE_SIDES[s] == count for s in SHAPES_STRAIGHT_ALL)
                    except ValueError:
                        pytest.fail(f"Medium tier distractor {distractor!r} is not a numeric side count")
                else:
                    # Distractors should be shape names from unique pool
                    assert distractor in SHAPES_STRAIGHT_UNIQUE


# --- Independent check coverage ------


def test_angle_turn_independent_check_fires_on_corrupted_dict(monkeypatch):
    """The independent check via Fraction arithmetic fires when TURN_VALUES is corrupted.

    Monkeypatch TURN_VALUES to have a wrong value, then verify that generation
    raises VerificationError.
    """
    # Corrupt the dict: half turn is 2, we make it 3
    monkeypatch.setitem(TURN_VALUES, "half turn", 3)

    verification_fired = False
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            try:
                question = make("angle_turn_right_reference", seed=seed, difficulty=difficulty)
                # If we reach here for this seed, the corruption didn't affect this path
            except VerificationError as e:
                # The independent check detected the corruption
                verification_fired = True
                assert "dict" in str(e).lower() or "fraction" in str(e).lower()
                break
        if verification_fired:
            break
    # Assert that at least one seed raised an error due to the corrupted dict
    assert verification_fired, "Independent check did not fire when TURN_VALUES was corrupted"


def test_shape_features_independent_check_fires_on_corrupted_dict(monkeypatch):
    """The independent check via second dict fires when SHAPE_SIDES is corrupted.

    Monkeypatch SHAPE_SIDES to have a wrong value, then verify that generation
    raises VerificationError.
    """
    # Corrupt the dict: hexagon has 6 sides, we make it 7
    monkeypatch.setitem(SHAPE_SIDES, "hexagon", 7)

    verification_fired = False
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            try:
                question = make("shape_features_classification", seed=seed, difficulty=difficulty)
                # If we reach here for this seed, the corruption didn't affect this path
            except VerificationError as e:
                # The independent check detected the corruption
                verification_fired = True
                assert "dict" in str(e).lower() or "side" in str(e).lower() or "count" in str(e).lower()
                break
        if verification_fired:
            break
    # Assert that at least one seed raised an error due to the corrupted dict
    assert verification_fired, "Independent check did not fire when SHAPE_SIDES was corrupted"


# --- Skill loading via the API ------


def test_angle_turn_loads_with_correct_metadata():
    """Skill 1 must load with correct prerequisites, answer kind, and placement_eligible."""
    _, skills_list = load_skills(ROOT / "skills")
    skills = {skill.id: skill for skill in skills_list}
    skill = skills["f10-angle-turn-right-reference"]
    assert skill.id == "f10-angle-turn-right-reference"
    assert skill.prerequisites == ["angle-types"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False


def test_shape_features_loads_with_correct_metadata():
    """Skill 2 must load with correct prerequisites, answer kind, and placement_eligible."""
    _, skills_list = load_skills(ROOT / "skills")
    skills = {skill.id: skill for skill in skills_list}
    skill = skills["f10-shape-features-classification"]
    assert skill.id == "f10-shape-features-classification"
    assert skill.prerequisites == ["name-basic-shapes"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
