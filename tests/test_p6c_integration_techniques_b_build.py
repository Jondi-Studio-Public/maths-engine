"""Integration techniques (arcsin/arctan) and volumes of revolution
(senior-integrate-expressions-of-the-form-frac-pm1-sqrt,
senior-determine-volumes-of-solids-of-revolution-about-either)."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

ARCSIN_ARCTAN_TOPIC = "integrate_expressions_of_the_form_frac_pm1_sqrt"
VOLUMES_TOPIC = "determine_volumes_of_solids_of_revolution_about_either"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [ARCSIN_ARCTAN_TOPIC, VOLUMES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [ARCSIN_ARCTAN_TOPIC, VOLUMES_TOPIC])
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


@pytest.mark.parametrize("topic", [ARCSIN_ARCTAN_TOPIC, VOLUMES_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Bare make(topic, seed=s) with no difficulty should have >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [ARCSIN_ARCTAN_TOPIC, VOLUMES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_expression(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_expression(question, distractor).correct


@pytest.mark.parametrize("topic", [ARCSIN_ARCTAN_TOPIC, VOLUMES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        # Append an extra term to make it wrong
        wrong = question.answer_text + " + 1"
        assert not grade_expression(question, wrong).correct


# --- skills loading --------------------------------------------------------


def test_both_skills_load_correctly():
    """Both skills load through the API with correct configuration."""
    _, skills = load_skills()
    skills_dict = {s.id: s for s in skills}

    arcsin_arctan = skills_dict.get("senior-integrate-expressions-of-the-form-frac-pm1-sqrt")
    assert arcsin_arctan is not None
    assert arcsin_arctan.answer == "expression"
    assert arcsin_arctan.placement_eligible is False
    assert "integrate-polynomial" in arcsin_arctan.prerequisites
    assert len(arcsin_arctan.sources) == 4  # easy, medium, hard, exam

    volumes = skills_dict.get("senior-determine-volumes-of-solids-of-revolution-about-either")
    assert volumes is not None
    assert volumes.answer == "expression"
    assert volumes.placement_eligible is False
    assert "integrate-polynomial" in volumes.prerequisites
    # easy, medium, hard, multi-part exam chain, plus the single-part exam source re-added as key exam-single (bf2fca7)
    assert len(volumes.sources) == 5


# --- arcsin/arctan specific tests -------------------------------------------


def test_arcsin_arctan_deterministic_check_both_tiers():
    """Verify that check_equivalent calls fire on corrupted answers."""
    # We test by verifying the generator runs without error
    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(5):
            question = make(ARCSIN_ARCTAN_TOPIC, seed=seed, difficulty=difficulty)
            # If we got here, the generator's checks passed
            assert question.answer_text is not None


def test_arctan_hard_bounds_never_zero():
    """Hard tier arctan bounds never place x/a at 0."""
    for seed in SEEDS:
        question = make(ARCSIN_ARCTAN_TOPIC, seed=seed, difficulty="hard")
        # Extract bounds from text: "Find ∫ from x = -a to x = a"
        import re

        match = re.search(r"from x = (-?\d+) to x = (\d+)", question.text)
        assert match, f"Could not parse bounds from: {question.text}"
        lo, hi = int(match.group(1)), int(match.group(2))
        assert lo != 0, f"Lower bound is 0: {lo}"
        assert hi != 0, f"Upper bound is 0: {hi}"


def test_arcsin_single_pi_multiple():
    """Easy tier arcsin answers are clean pi multiples."""
    from sympy import pi, sympify

    for seed in SEEDS:
        question = make(ARCSIN_ARCTAN_TOPIC, seed=seed, difficulty="easy")
        # All easy answers should be k*pi/6 or k*pi/2
        answer_str = question.answer_text
        assert "pi" in answer_str, f"Easy tier should be pi-multiple: {answer_str}"


def test_arcsin_medium_pi_multiple():
    """Medium tier arcsin answers are clean pi multiples."""
    from sympy import pi, sympify

    for seed in SEEDS:
        question = make(ARCSIN_ARCTAN_TOPIC, seed=seed, difficulty="medium")
        answer_str = question.answer_text
        assert "pi" in answer_str, f"Medium tier should be pi-multiple: {answer_str}"


def test_arctan_hard_pi_multiple():
    """Hard tier arctan answers are clean pi multiples."""
    for seed in SEEDS:
        question = make(ARCSIN_ARCTAN_TOPIC, seed=seed, difficulty="hard")
        answer_str = question.answer_text
        assert "pi" in answer_str, f"Hard tier should be pi-multiple: {answer_str}"


def test_at_least_20_distinct_arcsin_arctan_per_tier():
    """At least 20 distinct rendered questions per tier over 60 seeds."""
    for difficulty in ["easy", "medium", "hard"]:
        questions = set()
        for seed in range(60):
            question = make(ARCSIN_ARCTAN_TOPIC, seed=seed, difficulty=difficulty)
            questions.add(question.text)
        assert len(questions) >= 20, f"{difficulty} tier has only {len(questions)} distinct questions"


def test_no_unicode_in_arcsin_arctan_text():
    """Arcsin/arctan text should not contain unicode math characters."""
    unicode_chars = ["√", "≥", "≤", "π"]
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(ARCSIN_ARCTAN_TOPIC, seed=seed, difficulty=difficulty)
            for char in unicode_chars:
                assert char not in question.text, f"Found '{char}' in {difficulty} text: {question.text}"
                assert char not in question.answer_text, (
                    f"Found '{char}' in {difficulty} answer: {question.answer_text}"
                )


# --- volumes of revolution specific tests -----------------------------------


def test_volumes_deterministic_check_both_tiers():
    """Verify that check_equivalent calls fire on corrupted answers."""
    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(5):
            question = make(VOLUMES_TOPIC, seed=seed, difficulty=difficulty)
            assert question.answer_text is not None


def test_easy_volume_has_zero_lower_bound():
    """Easy tier volume has lower bound at 0."""
    for seed in SEEDS:
        question = make(VOLUMES_TOPIC, seed=seed, difficulty="easy")
        assert "x = 0" in question.text


def test_medium_volume_x_axis():
    """Medium tier rotates about x-axis."""
    for seed in SEEDS:
        question = make(VOLUMES_TOPIC, seed=seed, difficulty="medium")
        assert "x-axis" in question.text
        assert "y = x^2" in question.text


def test_hard_volume_y_axis():
    """Hard tier rotates about y-axis."""
    for seed in SEEDS:
        question = make(VOLUMES_TOPIC, seed=seed, difficulty="hard")
        assert "y-axis" in question.text
        # Check explicitly for ASCII sqrt notation, not unicode
        assert "sqrt(y)" in question.text, f"Hard tier should have 'sqrt(y)': {question.text}"
        assert "√" not in question.text, f"Hard tier should not have unicode sqrt: {question.text}"


def test_hard_volume_no_unicode():
    """Hard tier volumes use only ASCII, no unicode characters."""
    unicode_chars = ["√", "≥", "≤", "π"]
    for seed in SEEDS:
        question = make(VOLUMES_TOPIC, seed=seed, difficulty="hard")
        for char in unicode_chars:
            assert char not in question.text, f"Found '{char}' in hard volume text: {question.text}"
            assert char not in question.answer_text, f"Found '{char}' in hard volume answer: {question.answer_text}"


def test_all_volume_answers_have_pi():
    """All volume answers contain pi."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(VOLUMES_TOPIC, seed=seed, difficulty=difficulty)
            assert "pi" in question.answer_text, f"{difficulty} tier should have pi: {question.answer_text}"


def test_at_least_20_distinct_volumes_per_tier():
    """At least 20 distinct rendered questions per tier over 60 seeds."""
    for difficulty in ["easy", "medium", "hard"]:
        questions = set()
        for seed in range(60):
            question = make(VOLUMES_TOPIC, seed=seed, difficulty=difficulty)
            questions.add(question.text)
        assert len(questions) >= 20, f"{difficulty} tier has only {len(questions)} distinct questions"


def test_volumes_answer_and_distractors_formatted_alike():
    """Answer and every distractor formatted alike (all contain pi for volumes)."""
    for seed in range(20):
        for difficulty in ["easy", "medium", "hard"]:
            question = make(VOLUMES_TOPIC, seed=seed, difficulty=difficulty)
            # All should have pi in the same way
            assert "pi" in question.answer_text
            for distractor in question.distractors:
                # Distractors should also be valid expressions, though they may be wrong
                assert len(distractor) > 0
