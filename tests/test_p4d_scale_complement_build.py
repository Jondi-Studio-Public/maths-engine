"""Tests for scale and complement/union probability generators."""

from __future__ import annotations

import pytest
from fractions import Fraction

from mathgen.grading.typed import grade_mcq, mcq_options, grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p4d_scale_complement_build import (
    classify_scale,
    OPTION_IMPOSSIBLE,
    OPTION_CERTAIN,
    OPTION_BETWEEN,
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- Skill C: probability_scale_classify (MCQ) ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_scale_deterministic_across_seeds(difficulty):
    """Scale generator is deterministic: same seed produces same question."""
    for seed in SEEDS:
        question = make("probability_scale_classify", seed=seed, difficulty=difficulty)
        repeat = make("probability_scale_classify", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_scale_distractors_are_distinct_and_not_equal_answer(difficulty):
    """Scale: distractors are distinct from each other and from the answer."""
    for seed in SEEDS:
        question = make("probability_scale_classify", seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == 2
        assert len(set(question.distractors)) == 2  # all distinct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_scale_correct_answer_grades_correct(difficulty):
    """Scale: the canonical answer grades correct."""
    for seed in SEEDS:
        question = make("probability_scale_classify", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        answer_index = options.index(question.answer_text)
        result = grade_mcq(question, str(answer_index))
        assert result.correct, f"seed {seed}: canonical answer did not grade correct"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_scale_every_distractor_grades_incorrect(difficulty):
    """Scale: every distractor grades incorrect."""
    for seed in SEEDS:
        question = make("probability_scale_classify", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            result = grade_mcq(question, str(distractor_index))
            assert not result.correct, f"seed {seed}: distractor '{distractor}' should grade incorrect"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_scale_wrong_index_grades_incorrect(difficulty):
    """Scale: a deliberately wrong index grades incorrect."""
    for seed in SEEDS:
        question = make("probability_scale_classify", seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        answer_index = options.index(question.answer_text)
        # Pick a wrong index
        wrong_index = (answer_index + 1) % len(options)
        result = grade_mcq(question, str(wrong_index))
        assert not result.correct, f"seed {seed}: wrong index {wrong_index} should grade incorrect"


def test_scale_independent_check_fires_on_corruption():
    """The classify_scale function is an independent check that fires on corrupted answers."""
    from mathgen.core import check as mathgen_check

    n = 6
    correct_list = [2, 4]
    correct_label = classify_scale(n, correct_list)
    assert correct_label == OPTION_BETWEEN

    # Corrupt: empty list should be IMPOSSIBLE
    corrupted_list = []
    corrupted_label = classify_scale(n, corrupted_list)
    assert corrupted_label == OPTION_IMPOSSIBLE

    # Verify that the check would fire if we asserted the wrong classification
    with pytest.raises(VerificationError):
        mathgen_check(classify_scale(n, corrupted_list) == OPTION_BETWEEN, "classification mismatch")


def test_scale_hard_categories_all_occur():
    """Hard tier must exercise all three categories across sufficient seeds."""
    categories_seen = set()
    for seed in range(100):
        question = make("probability_scale_classify", seed=seed, difficulty="hard")
        # The answer label encodes the category
        categories_seen.add(question.answer_text)

    # All three answer types must appear
    assert OPTION_IMPOSSIBLE in categories_seen, "impossible category never occurred"
    assert OPTION_CERTAIN in categories_seen, "certain category never occurred"
    assert OPTION_BETWEEN in categories_seen, "between category never occurred"


# --- Skill D: probability_complement_union (fraction) ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_complement_union_deterministic_across_seeds(difficulty):
    """Complement/union generator is deterministic: same seed produces same question."""
    for seed in SEEDS:
        question = make("probability_complement_union", seed=seed, difficulty=difficulty)
        repeat = make("probability_complement_union", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_complement_union_distractors_are_distinct_and_not_equal_answer(difficulty):
    """Complement/union: distractors are distinct from each other and from the answer."""
    for seed in SEEDS:
        question = make("probability_complement_union", seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == 2
        assert len(set(question.distractors)) == 2  # all distinct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_complement_union_correct_answer_grades_correct(difficulty):
    """Complement/union: the canonical answer grades correct."""
    for seed in SEEDS:
        question = make("probability_complement_union", seed=seed, difficulty=difficulty)
        result = grade_fraction(question, question.answer_text)
        assert result.correct, f"seed {seed}: canonical answer did not grade correct"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_complement_union_every_distractor_grades_incorrect(difficulty):
    """Complement/union: every distractor grades incorrect."""
    for seed in SEEDS:
        question = make("probability_complement_union", seed=seed, difficulty=difficulty)
        for distractor in question.distractors:
            result = grade_fraction(question, distractor)
            assert not result.correct, f"seed {seed}: distractor '{distractor}' should grade incorrect"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_complement_union_wrong_answer_grades_incorrect(difficulty):
    """Complement/union: a deliberately wrong answer is REJECTED by the grader."""
    for seed in SEEDS:
        question = make("probability_complement_union", seed=seed, difficulty=difficulty)
        # Try an obviously wrong answer
        result = grade_fraction(question, "1/3")  # Unlikely to be correct
        # (could be correct for some seed, but very unlikely; this is just a sanity check)
        if question.answer_text != "1/3":
            assert not result.correct


def test_complement_union_accepts_unreduced_fractions():
    """Grade_fraction accepts unreduced equivalent fractions."""
    for seed in range(20):
        question = make("probability_complement_union", seed=seed, difficulty="easy")
        answer = Fraction(question.answer["p"], question.answer["q"])
        # Test with doubled form
        equiv_num = answer.numerator * 2
        equiv_den = answer.denominator * 2
        result = grade_fraction(question, f"{equiv_num}/{equiv_den}")
        assert result.correct, (
            f"seed {seed}: unreduced fraction {equiv_num}/{equiv_den} should equal {question.answer_text}"
        )


# --- Skills loading and metadata ---


def test_skill_c_loads_correctly():
    """Skill C loads with correct metadata."""
    _, skills_list = load_skills(ROOT / "skills")
    skills = {skill.id: skill for skill in skills_list}
    skill = skills.get("senior-review-the-probability-scale-0-leq-p-a")
    assert skill is not None, "Skill C not found in loaded skills"
    assert skill.domain == "statistics-probability"
    assert skill.answer == "mcq"
    assert skill.prerequisites == [
        "senior-review-probability-as-a-measure-of-the-likelihood",
        "f10-probability-scale-and-expectation",
    ]
    assert skill.placement_eligible is False


def test_skill_d_loads_correctly():
    """Skill D loads with correct metadata."""
    _, skills_list = load_skills(ROOT / "skills")
    skills = {skill.id: skill for skill in skills_list}
    skill = skills.get("senior-review-the-rules-p-left-overline-a-right")
    assert skill is not None, "Skill D not found in loaded skills"
    assert skill.domain == "statistics-probability"
    assert skill.answer == "fraction"
    assert skill.prerequisites == [
        "senior-review-the-concepts-and-language-of-outcomes-sample",
        "senior-review-the-probability-scale-0-leq-p-a",
        "f10-probability-complements-compounds",
    ]
    assert skill.placement_eligible is False
