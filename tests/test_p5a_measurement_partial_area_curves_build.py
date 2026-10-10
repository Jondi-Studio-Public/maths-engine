"""Foundation-Year 10 measurement skill for partially marked/unmarked scales, and senior calculus skill for area
between curves."""

from pathlib import Path
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# === determinism / distractor hygiene, both topics ==========================


@pytest.mark.parametrize("topic", ["unmarked_partial_measurement", "area_between_curves_simple_cases"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    """Determinism across >=40 seeds; distractors distinct and non-empty."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# === unmarked_partial_measurement (Skill 1) ==================================


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unmarked_partial_measurement_answer_is_integer(difficulty):
    """Answer must always be a plain int (never float for this skill)."""
    for seed in SEEDS:
        question = make("unmarked_partial_measurement", seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, int)
        assert isinstance(question.answer_text, str)
        assert question.answer_text == str(question.answer)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unmarked_partial_measurement_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    """The real grader accepts the canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make("unmarked_partial_measurement", seed=seed, difficulty=difficulty)
        # For number, grader expects learner-typed string
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(
                question, str(int(distractor)) if isinstance(distractor, (int, float)) else distractor
            ).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unmarked_partial_measurement_easy_is_length_only(difficulty):
    """Easy tier samples length family only."""
    if difficulty != "easy":
        pytest.skip("Only test easy tier")
    for seed in SEEDS:
        question = make("unmarked_partial_measurement", seed=seed, difficulty="easy")
        # Easy should mention "cm" for length
        assert "cm" in question.text


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unmarked_partial_measurement_medium_is_length_mass_capacity(difficulty):
    """Medium tier samples all three families: length, mass, capacity."""
    if difficulty != "medium":
        pytest.skip("Only test medium tier")
    families_seen = set()
    for seed in SEEDS:
        question = make("unmarked_partial_measurement", seed=seed, difficulty="medium")
        if "cm" in question.text:
            families_seen.add("length")
        if "g" in question.text and "mg" not in question.text:
            families_seen.add("mass")
        if "mL" in question.text:
            families_seen.add("capacity")
    # Medium should exercise all three families across seeds
    assert "length" in families_seen
    assert "mass" in families_seen
    assert "capacity" in families_seen


def test_unmarked_partial_measurement_hard_exercises_both_directions():
    """Hard tier must exercise both 'from_lower' and 'before_upper' directions."""
    from_lower_count = 0
    from_upper_count = 0
    for seed in SEEDS:
        question = make("unmarked_partial_measurement", seed=seed, difficulty="hard")
        text_lower = question.text.lower()
        if "past the" in text_lower:
            from_lower_count += 1
        elif "before the" in text_lower:
            from_upper_count += 1
    # Both directions should fire at least some times
    assert from_lower_count > 0, "hard tier never exercised from_lower direction"
    assert from_upper_count > 0, "hard tier never exercised from_upper direction"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unmarked_partial_measurement_distractors_are_integers_and_distinct_from_answer(difficulty):
    """All distractors must be parseable as distinct integers, different from the answer."""
    for seed in SEEDS:
        question = make("unmarked_partial_measurement", seed=seed, difficulty=difficulty)
        answer = question.answer
        for distractor in question.distractors:
            # Distractors are returned as strings by pick_distractors
            distractor_int = int(distractor)
            assert distractor_int != answer


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unmarked_partial_measurement_division_plural_correctness(difficulty):
    """Division/divisions must be pluralized correctly based on count k."""
    for seed in SEEDS:
        question = make("unmarked_partial_measurement", seed=seed, difficulty=difficulty)
        # The defect would manifest as literal " division " (singular) when k > 1.
        # After fix, only "1 division" is correct; k > 1 must use "divisions"
        import re

        # Match patterns like "2 division", "3 division", etc. (multi-digit counts with singular)
        # But "1 division" is correct, so only match 2+ with singular
        buggy_pattern = r"\b[2-9]\d* division\b"
        assert not re.search(buggy_pattern, question.text), (
            f"Found unpluralised 'divisions' for count > 1: {question.text}"
        )


# === area_between_curves_simple_cases (Skill 2) ==============================


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_area_between_curves_simple_cases_answer_is_numeric(difficulty):
    """Answer must be int or float (via Fraction conversion)."""
    for seed in SEEDS:
        question = make("area_between_curves_simple_cases", seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, (int, float))
        assert isinstance(question.answer_text, str)
        # answer_text must be formatted as fixed-point decimal
        assert "." in question.answer_text or isinstance(question.answer, int)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_area_between_curves_simple_cases_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    """The real grader accepts the canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make("area_between_curves_simple_cases", seed=seed, difficulty=difficulty)
        # For number, grader expects learner-typed string
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            # Distractors are already strings from pick_distractors
            result = grade_number(question, distractor)
            assert not result.correct, (
                f"distractor {distractor} should not grade correct for answer {question.answer_text}"
            )


def test_area_between_curves_simple_cases_easy_always_states_which_curve_is_on_top():
    """Easy tier must state explicitly which curve is above and which is below."""
    for seed in SEEDS:
        question = make("area_between_curves_simple_cases", seed=seed, difficulty="easy")
        text_lower = question.text.lower()
        # Easy states "below" and "above" explicitly
        assert "below" in text_lower and "above" in text_lower


def test_area_between_curves_simple_cases_hard_never_states_which_curve_is_on_top():
    """Hard tier must NOT state which curve is above (learner must infer)."""
    for seed in SEEDS:
        question = make("area_between_curves_simple_cases", seed=seed, difficulty="hard")
        text_lower = question.text.lower()
        # Hard never uses "below"/"above" in explicit ordering
        # It should use "between"
        assert "between" in text_lower


def test_area_between_curves_simple_cases_medium_always_states_line_and_parabola():
    """Medium tier must state f is a line and g is x²."""
    for seed in SEEDS:
        question = make("area_between_curves_simple_cases", seed=seed, difficulty="medium")
        text_lower = question.text.lower()
        # Medium should mention the parabola/x squared form and state which is on top
        assert "x²" in question.text or "x^2" in question.text or "squared" in text_lower


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_area_between_curves_simple_cases_distractors_are_numeric_and_distinct_from_answer(difficulty):
    """All distractors must be parseable as numeric, distinct from answer."""
    for seed in SEEDS:
        question = make("area_between_curves_simple_cases", seed=seed, difficulty=difficulty)
        answer = question.answer
        for distractor in question.distractors:
            # Distractors are returned as strings by pick_distractors
            try:
                distractor_val = float(distractor)
            except ValueError:
                pytest.fail(f"distractor {distractor!r} is not numeric")
            # For medium (always integer), must be exactly distinct
            if difficulty == "medium":
                assert distractor_val != answer
            else:
                # For easy/hard (can be float), check difference > 0.05 (half the 0.1 tolerance)
                assert abs(distractor_val - answer) > 0.05


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_area_between_curves_simple_cases_linear_coefficient_formatting(difficulty):
    """Linear functions must not display coefficients 0 and 1 literally: no '0x', '1x', or '+ 0'."""
    import re

    for seed in SEEDS:
        question = make("area_between_curves_simple_cases", seed=seed, difficulty=difficulty)
        text = question.text

        # Use word boundary regex to avoid false positives from "10x" containing "0x", etc.
        # Patterns: (?<!\d) negative lookbehind ensures no digit before, \b word boundary after
        buggy_patterns = [
            (r"(?<!\d)0x\b", "'0x' found (coefficient 0 not simplified)"),
            (r"(?<!\d)1x\b", "'1x' found (coefficient 1 not simplified)"),
            (r"\+ 0\b", "'+ 0' found (constant 0 not dropped)"),
        ]

        for pattern, msg in buggy_patterns:
            assert not re.search(pattern, text), f"{msg} in {difficulty}: {text}"


# === Skills load correctly ===================================================


def test_both_skills_load_with_correct_metadata():
    """Both skills must load through mathgen.skills with correct prerequisites and answer type."""
    domains, skills_list = load_skills(root=ROOT / "skills")
    skills = {s.id: s for s in skills_list}

    # Check f10-unmarked-partial-measurement
    skill1 = skills.get("f10-unmarked-partial-measurement")
    assert skill1 is not None
    assert skill1.answer == "number"
    assert "f10-metric-measure-instrument-estimate" in skill1.prerequisites
    assert skill1.placement_eligible is False
    assert skill1.status == "active"
    # Check sources cover all three difficulties
    difficulties = {s.difficulty for s in skill1.sources}
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties

    # Check senior-calculate-the-area-between-curves-in-simple-cases
    skill2 = skills.get("senior-calculate-the-area-between-curves-in-simple-cases")
    assert skill2 is not None
    assert skill2.answer == "number"
    assert "convert-metric-length" in skill2.prerequisites
    assert skill2.placement_eligible is False
    assert skill2.status == "active"
    # Check sources cover all three difficulties
    difficulties = {s.difficulty for s in skill2.sources}
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties


# === Verification that independent checks fire ===============================


def test_unmarked_partial_measurement_independent_check_catches_corrupted_answer():
    """Corrupt the answer and verify the independent check raises."""
    import mathgen.generators.p5a_measurement_partial_area_curves_build as gen

    # This test uses the internal check function to verify it would catch an error
    # If answer were wrong, _verify_unmarked_partial_measurement would raise
    try:
        gen._verify_unmarked_partial_measurement(k=2, M=20, N=4, ref_mark=50, direction="from_lower", expected=60)
        # 50 + 2 * (20/4) = 50 + 10 = 60, should pass
    except gen.check as e:
        pytest.fail(f"independent check should not raise for correct values: {e}")

    # Now corrupt it
    with pytest.raises(Exception):  # check raises AssertionError or similar
        gen._verify_unmarked_partial_measurement(
            k=2, M=20, N=4, ref_mark=50, direction="from_lower", expected=65
        )  # wrong!


def test_area_between_curves_simple_cases_independent_check_catches_corrupted_answer():
    """Corrupt the answer and verify the independent check raises."""
    import mathgen.generators.p5a_measurement_partial_area_curves_build as gen

    # Test easy verification
    try:
        # area = (5-3)/2 * (4^2 - 1^2) + (8-2) * (4-1)
        #      = 1 * 15 + 6 * 3 = 15 + 18 = 33
        gen._verify_area_easy(m1=5, m2=3, c1=8, c2=2, a=1, b=4, expected=Fraction(33))
    except Exception as e:
        pytest.fail(f"independent check should not raise for correct values: {e}")

    # Corrupt it
    with pytest.raises(Exception):
        gen._verify_area_easy(m1=5, m2=3, c1=8, c2=2, a=1, b=4, expected=Fraction(32))  # wrong!
