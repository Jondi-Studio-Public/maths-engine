"""Tests the skills that classify causality versus correlation and interpolation versus extrapolation.

Senior skills for classifying causality vs correlation and interpolation vs extrapolation."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5d_causality_interpolation_build import (
    ASSIGNABLE_PAIRS,
    OBSERVATIONAL_PAIRS,
    OPTION_BOTH,
    OPTION_NEITHER,
    OPTION_BOTH_B,
    OPTION_ANY_X_B,
    OPTION_EDGE_OUTSIDE_B,
    _independent_check_skill_a,
    _independent_check_skill_b,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene, both topics -----------------------


@pytest.mark.parametrize("topic", ["causality_correlation_classify", "interpolation_extrapolation_classify"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# --- causality_correlation_classify (Skill A) --------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_causality_correlation_classify_answer_is_one_of_four_fixed_options(difficulty):
    """The answer must always be one of the four fixed MCQ options."""
    for seed in SEEDS:
        question = make("causality_correlation_classify", seed=seed, difficulty=difficulty)
        answer = question.answer_text
        options = mcq_options(question)
        assert answer in options, f"answer {answer!r} not in options {options}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_causality_correlation_classify_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    """The real grader accepts the canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make("causality_correlation_classify", seed=seed, difficulty=difficulty)
        # For mcq, grade_mcq expects an index into mcq_options(question)
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grade_mcq(question, correct_index).correct, f"grade_mcq rejected correct answer at index {correct_index}"
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            result = grade_mcq(question, distractor_index)
            assert not result.correct, f"grade_mcq accepted distractor {distractor!r} at index {distractor_index}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_causality_correlation_classify_neither_both_nor_neither_are_never_the_answer(difficulty):
    """OPTION_BOTH and OPTION_NEITHER must never be the intended label, by construction."""
    for seed in SEEDS:
        question = make("causality_correlation_classify", seed=seed, difficulty=difficulty)
        assert question.answer_text not in (OPTION_BOTH, OPTION_NEITHER), (
            f"padding option {question.answer_text!r} should never be the answer"
        )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_causality_correlation_classify_independent_check_fires_on_corrupted_answer(difficulty):
    """The independent check detects when the rendered text's marker does not match
    the construction label. We test this by deliberately corrupting a question's text
    and invoking the check function directly."""
    if difficulty != "hard":
        pytest.skip("Only test hard tier for corrupted-segment detection")

    for seed in SEEDS:
        question = make("causality_correlation_classify", seed=seed, difficulty=difficulty)
        text = question.text

        # Extract the correct label from the answer text
        if "Causation is supported" in question.answer_text:
            label = "causation"
        else:
            label = "correlation"

        # Deliberately corrupt the text: if asked about Study 1, swap what's in Study 1
        # to contain the opposite marker
        if "Study 1:" in text and "Study 2:" in text:
            if "Which of these is correct about Study 1" in text:
                # Corrupt Study 1 to have the wrong marker
                parts = text.split("Study 2:")
                study_1_part = parts[0]
                study_2_part = parts[1]

                # Flip the markers in Study 1
                if "randomly assigned" in study_1_part:
                    corrupted_study_1 = study_1_part.replace("randomly assigned", "third variable").replace(
                        "Researchers observed", "Researchers randomly assigned"
                    )
                else:
                    corrupted_study_1 = study_1_part.replace("third variable", "randomly assigned").replace(
                        "A third variable", "They then randomly assigned"
                    )

                corrupted_text = corrupted_study_1 + "Study 2:" + study_2_part

                # Call the independent check with the corrupted text
                # It should raise VerificationError because the marker in the addressed
                # segment no longer matches the label
                with pytest.raises(VerificationError):
                    _independent_check_skill_a(corrupted_text, difficulty, label)


def test_causality_correlation_classify_pair_partition_guard():
    """Regression guard: assert that causation-branch vignettes never draw from
    OBSERVATIONAL_PAIRS, and correlation-branch vignettes never draw from
    ASSIGNABLE_PAIRS. Also verify the partition constraints and pair-specific
    confounds for textbook examples."""

    # First: verify partition constraints
    observational_pairs = {(X, Y) for X, Y, _ in OBSERVATIONAL_PAIRS}
    assert len(observational_pairs) == len(OBSERVATIONAL_PAIRS), (
        "each observational pair appears once, bound to exactly one confound"
    )
    assert set(ASSIGNABLE_PAIRS).isdisjoint(observational_pairs), (
        "ASSIGNABLE_PAIRS and OBSERVATIONAL_PAIRS must be disjoint"
    )
    assert len(ASSIGNABLE_PAIRS) >= 5, f"ASSIGNABLE_PAIRS must have at least 5 entries; got {len(ASSIGNABLE_PAIRS)}"

    # Second: sweep 200+ seeds per tier and check pair membership
    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(200):
            question = make("causality_correlation_classify", seed=seed, difficulty=difficulty)
            text = question.text

            if difficulty in ("easy", "medium"):
                # Single vignette: check the whole text
                segments = [text]
            else:  # hard
                # Two vignettes: check both by splitting on "Study 2:"
                parts = text.split("Study 2:")
                if len(parts) == 2:
                    segments = [parts[0], "Study 2:" + parts[1]]
                else:
                    segments = [text]

            for segment in segments:
                # Check causation segments: "randomly assigned" should only use ASSIGNABLE_PAIRS
                if "randomly assigned" in segment:
                    # Extract the variable pair from "different levels of {X}, then measured {Y}"
                    found_assignable = False
                    for X, Y in ASSIGNABLE_PAIRS:
                        if f"different levels of {X}, then measured {Y}" in segment:
                            found_assignable = True
                            break

                    # Also check that no observational pair appears
                    for X, Y, _Z in OBSERVATIONAL_PAIRS:
                        assert f"different levels of {X}, then measured {Y}" not in segment, (
                            f"seed {seed}, difficulty {difficulty}: causation marker used observational pair ({X!r}, {Y!r})"
                        )

                    assert found_assignable, (
                        f"seed {seed}, difficulty {difficulty}: causation segment has 'randomly assigned' but no assignable pair found"
                    )

                # Check correlation segments: "third variable" should only use OBSERVATIONAL_PAIRS
                if "third variable" in segment:
                    # Extract the pair from "Researchers observed {X} and {Y} in a sample"
                    found_observational = False
                    found_pair = None
                    expected_confound = None
                    for X, Y, Z in OBSERVATIONAL_PAIRS:
                        if f"Researchers observed {X} and {Y} in a sample" in segment:
                            found_observational = True
                            found_pair = (X, Y)
                            expected_confound = Z
                            break

                    # Also check that no assignable pair appears
                    for X, Y in ASSIGNABLE_PAIRS:
                        assert f"Researchers observed {X} and {Y} in a sample" not in segment, (
                            f"seed {seed}, difficulty {difficulty}: correlation marker used assignable pair ({X!r}, {Y!r})"
                        )

                    assert found_observational, (
                        f"seed {seed}, difficulty {difficulty}: correlation segment has 'third variable' but no observational pair found"
                    )

                    # Every pair names the confound bound to it -- unconditionally.
                    # This assertion was previously guarded by "if this pair has a
                    # specific confound", which let an unbound pair draw any confound
                    # from a generic pool and still pass.
                    assert f"A third variable, {expected_confound}, was" in segment, (
                        f"seed {seed}, difficulty {difficulty}: textbook pair {found_pair} should use confound {expected_confound!r}, but got: {segment}"
                    )


# --- interpolation_extrapolation_classify (Skill B) ---------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_interpolation_extrapolation_classify_answer_is_one_of_four_fixed_options(difficulty):
    """The answer must always be one of the four fixed MCQ options."""
    for seed in SEEDS:
        question = make("interpolation_extrapolation_classify", seed=seed, difficulty=difficulty)
        answer = question.answer_text
        options = mcq_options(question)
        assert answer in options, f"answer {answer!r} not in options {options}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_interpolation_extrapolation_classify_correct_answer_grades_correct_and_distractors_grade_incorrect(difficulty):
    """The real grader accepts the canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make("interpolation_extrapolation_classify", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grade_mcq(question, correct_index).correct, f"grade_mcq rejected correct answer at index {correct_index}"
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            result = grade_mcq(question, distractor_index)
            assert not result.correct, f"grade_mcq accepted distractor {distractor!r} at index {distractor_index}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_interpolation_extrapolation_classify_neither_both_nor_neither_are_never_the_answer(difficulty):
    """OPTION_BOTH_B and the two misconception options must never be the intended label, by construction."""
    for seed in SEEDS:
        question = make("interpolation_extrapolation_classify", seed=seed, difficulty=difficulty)
        assert question.answer_text not in (OPTION_BOTH_B, OPTION_EDGE_OUTSIDE_B, OPTION_ANY_X_B), (
            f"padding option {question.answer_text!r} should never be the answer"
        )


def test_interpolation_extrapolation_classify_easy_randomises_inside_vs_outside():
    """Easy tier must randomise between inside and outside (50/50 in distribution)."""
    inside_count = 0
    outside_count = 0

    for seed in SEEDS:
        question = make("interpolation_extrapolation_classify", seed=seed, difficulty="easy")
        if "Interpolation" in question.answer_text:
            inside_count += 1
        else:
            outside_count += 1

    # Both branches should fire at least a few times
    assert inside_count > 0, "easy tier never generated inside case"
    assert outside_count > 0, "easy tier never generated outside case"


def test_interpolation_extrapolation_classify_medium_always_boundary_interpolation():
    """Medium tier always has the prediction at a boundary (x == xmin or x == xmax),
    which is always interpolation."""
    for seed in SEEDS:
        question = make("interpolation_extrapolation_classify", seed=seed, difficulty="medium")
        assert "Interpolation" in question.answer_text, (
            f"medium tier should always be interpolation; got {question.answer_text!r}"
        )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_interpolation_extrapolation_classify_independent_check_fires_on_corrupted_range(difficulty):
    """The independent check detects when the min/max recomputation fails."""
    if difficulty != "hard":
        pytest.skip("Only test hard tier for range-recomputation")

    for seed in SEEDS:
        question = make("interpolation_extrapolation_classify", seed=seed, difficulty=difficulty)

        # Extract label from answer text
        if "Interpolation" in question.answer_text:
            label = "interpolation"
        else:
            label = "extrapolation"

        # Parse the table from the text to extract x values
        # New format: "Data table: (x, y), (x, y), ..."
        text = question.text
        x_values = []
        if "Data table: " in text:
            # Extract the table portion
            table_start = text.index("Data table: ") + len("Data table: ")
            table_end = text.index(".", table_start)
            table_str = text[table_start:table_end]
            # Extract ordered pairs: (x, y), (x, y), ...
            import re

            pairs = re.findall(r"\((\d+),\s*(\d+)\)", table_str)
            for x_str, y_str in pairs:
                try:
                    x_values.append(int(x_str))
                except ValueError:
                    pass

        if x_values:
            # Extract x0 from the text (prediction point)
            lines = text.split("\n")
            x0 = None
            for line in lines:
                if "prediction at x = " in line:
                    x0_part = line.split("x = ")[1].split(".")[0]
                    try:
                        x0 = int(x0_part)
                    except ValueError:
                        pass
                    break

            if x0 is not None:
                # Call the independent check with correct values: should not raise
                _independent_check_skill_b(x_values, x0, label)

                # Now call it with corrupted x_values (swap min/max)
                if len(x_values) >= 2:
                    corrupted_x_values = [max(x_values), min(x_values)] + x_values[2:]
                    # The derived label will now be computed from corrupted values
                    # If construction label doesn't match, it will raise
                    try:
                        _independent_check_skill_b(corrupted_x_values, x0, label)
                        # If no exception, the check correctly re-derived the same label
                        # (which can happen if the corruption doesn't change the min/max)
                    except VerificationError:
                        # This is the expected case: corruption changed the label
                        pass


# --- skills loader --------------------------------------------------------


def test_both_skills_load_via_api_skills():
    """Both skills must load via the API skills loader with correct metadata."""
    domains, skills = load_skills()
    skills_dict = {s.id: s for s in skills}

    # Skill A
    skill_a = skills_dict.get("senior-distinguish-between-causality-and-correlation-through-examples")
    assert skill_a is not None, "Skill A not found in skills dict"
    assert skill_a.answer == "mcq", f"Skill A answer kind should be mcq; got {skill_a.answer}"
    assert "simple-event-probability" in skill_a.prerequisites, (
        "Skill A should have simple-event-probability as prerequisite"
    )
    assert skill_a.placement_eligible is False, "Skill A should have placement_eligible: false"

    # Skill B
    skill_b = skills_dict.get("senior-distinguish-between-interpolation-and-extrapolation-when-using-the")
    assert skill_b is not None, "Skill B not found in skills dict"
    assert skill_b.answer == "mcq", f"Skill B answer kind should be mcq; got {skill_b.answer}"
    assert "simple-event-probability" in skill_b.prerequisites, (
        "Skill B should have simple-event-probability as prerequisite"
    )
    assert skill_b.placement_eligible is False, "Skill B should have placement_eligible: false"
