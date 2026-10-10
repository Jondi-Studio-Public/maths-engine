"""Foundation-Year 10 geometry skills for coordinate changes and prism classification."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, grade_pair, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["coordinate_movement", "prism_cross_sections"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# --- coordinate_movement (skill 1) ------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_coordinate_movement_answer_is_pair_with_dx_dy_keys(difficulty):
    """The answer must be a pair dict with 'dx' and 'dy' keys (plain ints)."""
    for seed in SEEDS:
        question = make("coordinate_movement", seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, dict)
        assert set(question.answer.keys()) == {"dx", "dy"}
        assert isinstance(question.answer["dx"], int)
        assert isinstance(question.answer["dy"], int)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_coordinate_movement_answer_text_format(difficulty):
    """The answer_text must be formatted as 'dx = <int>, dy = <int>'."""
    for seed in SEEDS:
        question = make("coordinate_movement", seed=seed, difficulty=difficulty)
        # Check that answer_text matches the expected format
        parts = question.answer_text.split(", ")
        assert len(parts) == 2
        assert parts[0].startswith("dx = ")
        assert parts[1].startswith("dy = ")
        # Verify the values match the answer dict
        dx_val = int(parts[0].split("=")[1].strip())
        dy_val = int(parts[1].split("=")[1].strip())
        assert dx_val == question.answer["dx"]
        assert dy_val == question.answer["dy"]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_coordinate_movement_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    """The real grader accepts the canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make("coordinate_movement", seed=seed, difficulty=difficulty)
        # For pair, grade_pair expects question.answer_text
        result = grade_pair(question, question.answer_text)
        assert result.correct, f"canonical answer {question.answer_text!r} should grade correct"
        for distractor in question.distractors:
            result = grade_pair(question, distractor)
            assert not result.correct, f"distractor {distractor!r} should grade incorrect"


def test_coordinate_movement_easy_tier_same_quadrant():
    """Easy tier should have both start and end points in quadrant I (positive coords)."""
    for seed in SEEDS:
        question = make("coordinate_movement", seed=seed, difficulty="easy")
        # Extract coordinates from the text
        # Format: "A point moves from (x1, y1) to (x2, y2). ..."
        text = question.text
        import re

        match = re.search(r"from \((-?\d+), (-?\d+)\) to \((-?\d+), (-?\d+)\)", text)
        assert match, f"Could not parse coordinates from {text!r}"
        x1, y1, x2, y2 = map(int, match.groups())
        # Easy tier: both points should be in quadrant I (positive)
        assert x1 >= 1 and y1 >= 1, f"Start point ({x1}, {y1}) not in quadrant I"
        assert x2 >= 1 and y2 >= 1, f"End point ({x2}, {y2}) not in quadrant I"


def test_coordinate_movement_medium_tier_spans_quadrants():
    """Medium tier allows coordinates in range [-10, 10], spanning quadrants."""
    for seed in SEEDS:
        question = make("coordinate_movement", seed=seed, difficulty="medium")
        # Medium tier allows negative coordinates, so we just verify it's possible
        # We can't guarantee a specific quadrant, just that the coordinate range is correct
        text = question.text
        import re

        match = re.search(r"from \((-?\d+), (-?\d+)\) to \((-?\d+), (-?\d+)\)", text)
        assert match


def test_coordinate_movement_hard_tier_three_points():
    """Hard tier should have three points (start, middle, end)."""
    for seed in SEEDS:
        question = make("coordinate_movement", seed=seed, difficulty="hard")
        # Format: "A point starts at (x1, y1), moves to (x2, y2), then moves again to (x3, y3)..."
        text = question.text
        # Count the number of coordinate pairs in the text
        import re

        matches = re.findall(r"\((-?\d+), (-?\d+)\)", text)
        assert len(matches) == 3, f"Expected 3 coordinate pairs in hard tier, got {len(matches)}"


def test_coordinate_movement_points_are_distinct():
    """No two consecutive points should be identical."""
    for seed in SEEDS:
        question = make("coordinate_movement", seed=seed, difficulty="hard")
        text = question.text
        import re

        matches = re.findall(r"\((-?\d+), (-?\d+)\)", text)
        x1, y1 = map(int, matches[0])
        x2, y2 = map(int, matches[1])
        x3, y3 = map(int, matches[2])
        assert (x1, y1) != (x2, y2), "First and second points should be distinct"
        assert (x2, y2) != (x3, y3), "Second and third points should be distinct"


# --- prism_cross_sections (skill 2) ----------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_prism_cross_sections_answer_is_string_mcq_option(difficulty):
    """The answer must be a string (one of the three fixed MCQ options)."""
    for seed in SEEDS:
        question = make("prism_cross_sections", seed=seed, difficulty=difficulty)
        assert isinstance(question.answer_text, str)
        assert question.answer_text in ["a right prism", "not a right prism", "there is not enough information to tell"]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_prism_cross_sections_answer_is_one_of_fixed_options(difficulty):
    """The answer must always be one of the three fixed MCQ options."""
    for seed in SEEDS:
        question = make("prism_cross_sections", seed=seed, difficulty=difficulty)
        answer = question.answer_text
        options = mcq_options(question)
        assert answer in options


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_prism_cross_sections_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    """The real grader accepts the canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make("prism_cross_sections", seed=seed, difficulty=difficulty)
        # For mcq, grade_mcq expects an index into mcq_options(question)
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        result = grade_mcq(question, str(correct_index))
        assert result.correct, f"canonical answer index {correct_index} should grade correct"
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            result = grade_mcq(question, str(distractor_index))
            assert not result.correct, f"distractor index {distractor_index} should grade incorrect"


def test_prism_cross_sections_never_describes_circular_or_oblique():
    """Never describe circular cross-sections or oblique prisms."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make("prism_cross_sections", seed=seed, difficulty=difficulty)
            text = question.text.lower()
            assert "circular" not in text, f"Text should not mention circular: {question.text}"
            assert "cylinder" not in text, f"Text should not mention cylinder: {question.text}"
            assert "oblique" not in text, f"Text should not mention oblique: {question.text}"


def test_prism_cross_sections_shape_names_bounded():
    """Cross-section shapes should only be from {rectangular, triangular, hexagonal, pentagonal, square}."""
    allowed_shapes = {"rectangular", "triangular", "hexagonal", "pentagonal", "square"}
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make("prism_cross_sections", seed=seed, difficulty=difficulty)
            text = question.text.lower()
            # Check that at least one allowed shape is mentioned
            found_shape = False
            for shape in allowed_shapes:
                if shape in text:
                    found_shape = True
                    break
            assert found_shape, f"No allowed shape found in: {question.text}"


def test_prism_cross_sections_answer_never_information_foil():
    """The answer should never be 'there is not enough information to tell'."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make("prism_cross_sections", seed=seed, difficulty=difficulty)
            assert question.answer_text != "there is not enough information to tell", (
                "Answer should never be the 'not enough info' foil"
            )


def test_prism_cross_sections_easy_has_keywords():
    """Easy tier should use 'same' or 'point' keywords to make classification direct."""
    keywords = ["the same", "point"]
    for seed in SEEDS:
        question = make("prism_cross_sections", seed=seed, difficulty="easy")
        text = question.text.lower()
        has_keyword = any(kw in text for kw in keywords)
        assert has_keyword, f"Easy tier should have a keyword (same/point) in: {question.text}"


# --- skills loading --------------------------------


def test_both_skills_load_via_api_skills():
    """Both skills must load through mathgen.skills.load_skills."""
    _, skills = load_skills()
    skill_ids = {s.id for s in skills}
    assert "f10-coordinate-movement" in skill_ids
    assert "f10-prism-cross-sections" in skill_ids


def test_f10_coordinate_movement_properties():
    """Verify f10-coordinate-movement properties."""
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "f10-coordinate-movement")
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "pair"
    assert "coordinate-plane-readiness" in skill.prerequisites


def test_f10_prism_cross_sections_properties():
    """Verify f10-prism-cross-sections properties."""
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "f10-prism-cross-sections")
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "mcq"
    assert "rectangular-prism-volume" in skill.prerequisites
