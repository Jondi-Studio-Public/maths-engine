"""Tests the vector notation skills for ordered-pair and column-vector notation (ACMSM014) and scalar multiples
(ACMSM012).

Vectors notation cluster.

senior-use-ordered-pair-notation-and-column-vector-notation (ACMSM014)
senior-represent-a-scalar-multiple-of-a-vector (ACMSM012)
"""

from pathlib import Path
import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p4c_vectors_notation_build import (
    _verify_forward_simple,
    _verify_inverse_simple,
    _verify_forward_multiplication,
    _verify_inverse_division,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(500)  # 500-seed sweep for distractor safety

TOPIC_NOTATION = "use_ordered_pair_notation_and_column_vector_notation"
TOPIC_SCALAR = "represent_a_scalar_multiple_of_a_vector"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [TOPIC_NOTATION, TOPIC_SCALAR])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    """Determinism across 500 seeds at every difficulty."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [TOPIC_NOTATION, TOPIC_SCALAR])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    """Distractors are distinct and never equal the answer, exact counts verified."""
    if topic == TOPIC_NOTATION:
        expected_count = 3  # 3 misconceptions: reversed-order, wrong-axis, endpoint-only
    else:  # TOPIC_SCALAR
        expected_count = 3 if difficulty in ["easy", "medium"] else 2  # hard tier has 2

    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors)), (
            f"Distractors have duplicates: {question.distractors}"
        )
        assert len(question.distractors) == expected_count, (
            f"Seed {seed}: expected {expected_count} distractors, got {len(question.distractors)}: {question.distractors}"
        )


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [TOPIC_NOTATION, TOPIC_SCALAR])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    """Real grader accepts canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        result = grade_number(question, question.answer_text)
        assert result.correct, f"Seed {seed}: canonical answer {question.answer_text} was rejected"
        for distractor in question.distractors:
            result = grade_number(question, distractor)
            assert not result.correct, f"Seed {seed}: distractor {distractor} was accepted as correct"


@pytest.mark.parametrize("topic", [TOPIC_NOTATION, TOPIC_SCALAR])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_deliberately_wrong_answer(topic, difficulty):
    """Real grader rejects a deliberately wrong answer (answer + 7 as string)."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        wrong_value = int(question.answer_text) + 7
        wrong_str = str(wrong_value)
        result = grade_number(question, wrong_str)
        assert not result.correct, f"Seed {seed}: deliberately wrong answer {wrong_str} was accepted"


# --- use_ordered_pair_notation_and_column_vector_notation -------------------


def test_notation_easy_all_integers_no_decimal():
    """Easy tier answers are plain integers (no decimal point)."""
    for seed in SEEDS:
        question = make(TOPIC_NOTATION, seed=seed, difficulty="easy")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_notation_medium_all_integers_no_decimal():
    """Medium tier answers are plain integers (no decimal point)."""
    for seed in SEEDS:
        question = make(TOPIC_NOTATION, seed=seed, difficulty="medium")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_notation_hard_all_integers_no_decimal():
    """Hard tier answers are plain integers (no decimal point)."""
    for seed in SEEDS:
        question = make(TOPIC_NOTATION, seed=seed, difficulty="hard")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_notation_easy_independent_check_fires_on_corrupted_answer():
    """Independent check for easy tier (forward_positive) fires on wrong expected value."""
    # correct: Ax=1, Bx=3, diff=2, so adding back: 1 + 2 == 3
    _verify_forward_simple(1, 3, 2)  # must not raise
    with pytest.raises(VerificationError):
        _verify_forward_simple(1, 3, 1)  # wrong expected


def test_notation_medium_independent_check_fires_on_corrupted_answer():
    """Independent check for medium tier (forward_signed) fires on wrong expected value."""
    # correct: A=-2, B=3, diff=5, so adding back: -2 + 5 == 3
    _verify_forward_simple(-2, 3, 5)  # must not raise
    with pytest.raises(VerificationError):
        _verify_forward_simple(-2, 3, 4)  # wrong expected


def test_notation_hard_independent_check_fires_on_corrupted_answer():
    """Independent check for hard tier (inverse) fires on wrong expected value."""
    # correct: A=3, v=2, result=5, so subtracting back: 5 - 3 == 2
    _verify_inverse_simple(3, 2, 5)  # must not raise
    with pytest.raises(VerificationError):
        _verify_inverse_simple(3, 2, 4)  # wrong expected


def test_notation_hard_both_axes_asked_over_sweep():
    """Hard tier: both x and y axes are actually asked (not always one)."""
    x_asked = False
    y_asked = False
    for seed in SEEDS:
        question = make(TOPIC_NOTATION, seed=seed, difficulty="hard")
        if "x-component" in question.text or "x-coordinate" in question.text:
            x_asked = True
        if "y-component" in question.text or "y-coordinate" in question.text:
            y_asked = True
        if x_asked and y_asked:
            break
    assert x_asked, "Hard tier never asked for x-component"
    assert y_asked, "Hard tier never asked for y-component"


def test_notation_raises_on_unknown_variant():
    with pytest.raises(ValueError):
        make(TOPIC_NOTATION, seed=1, variant="unknown")


# --- represent_a_scalar_multiple_of_a_vector --------------------------------


def test_scalar_easy_all_integers_no_decimal():
    """Easy tier answers are plain integers (no decimal point)."""
    for seed in SEEDS:
        question = make(TOPIC_SCALAR, seed=seed, difficulty="easy")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_scalar_medium_all_integers_no_decimal():
    """Medium tier answers are plain integers (no decimal point)."""
    for seed in SEEDS:
        question = make(TOPIC_SCALAR, seed=seed, difficulty="medium")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_scalar_hard_all_integers_no_decimal():
    """Hard tier answers are plain integers (no decimal point)."""
    for seed in SEEDS:
        question = make(TOPIC_SCALAR, seed=seed, difficulty="hard")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_scalar_easy_independent_check_fires_on_corrupted_answer():
    """Independent check for easy tier (forward_positive) fires on wrong expected value."""
    # correct: component=3, k=2, product=6, so repeated-addition: 3+3 == 6
    _verify_forward_multiplication(3, 2, 6)  # must not raise
    with pytest.raises(VerificationError):
        _verify_forward_multiplication(3, 2, 5)  # wrong expected


def test_scalar_medium_independent_check_fires_on_corrupted_answer():
    """Independent check for medium tier (forward_signed) fires on wrong expected value."""
    # correct: component=-2, k=3, product=-6, so repeated-addition: (-2)+(-2)+(-2) == -6
    _verify_forward_multiplication(-2, 3, -6)  # must not raise
    with pytest.raises(VerificationError):
        _verify_forward_multiplication(-2, 3, -5)  # wrong expected


def test_scalar_hard_independent_check_fires_on_corrupted_answer():
    """Independent check for hard tier (inverse) fires on wrong expected value."""
    # correct: P=10, k=5, a=2, so multiply back: 5 * 2 == 10
    _verify_inverse_division(10, 5, 2)  # must not raise
    with pytest.raises(VerificationError):
        _verify_inverse_division(10, 5, 1)  # wrong expected


def test_scalar_easy_no_collision_added_instead_of_multiplied():
    """Easy tier (k in {2,3,4,5}, a,b in 1..9) with k==2 exclusion prevents
    added-instead-of-multiplied collision (a+k == a*k only when a==2,k==2)."""
    for seed in SEEDS:
        question = make(TOPIC_SCALAR, seed=seed, difficulty="easy")
        # The added-instead-of-multiplied candidate should not equal answer
        # This is guaranteed by the a==2,k==2 exclusion in the generator
        answer = int(question.answer_text)
        for distractor in question.distractors:
            assert int(distractor) != answer


def test_scalar_hard_hard_tier_has_2_distractors():
    """Hard tier has exactly 2 distractors (added-instead-of-divided, used-product-directly)."""
    for seed in SEEDS:
        question = make(TOPIC_SCALAR, seed=seed, difficulty="hard")
        assert len(question.distractors) == 2, f"Seed {seed}: expected 2 distractors, got {len(question.distractors)}"


def test_scalar_raises_on_unknown_variant():
    with pytest.raises(ValueError):
        make(TOPIC_SCALAR, seed=1, variant="unknown")


# --- skills load_skills checks -----------------------------------------


def test_notation_skill_loads_with_correct_properties():
    """Skill senior-use-ordered-pair-notation-and-column-vector-notation loads correctly."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-use-ordered-pair-notation-and-column-vector-notation"), None)
    assert skill is not None
    assert skill.prerequisites == ["vector-addition"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    # Check sources cover all three difficulties
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == {"easy", "medium", "hard"}


def test_scalar_skill_loads_with_correct_properties():
    """Skill senior-represent-a-scalar-multiple-of-a-vector loads correctly."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-represent-a-scalar-multiple-of-a-vector"), None)
    assert skill is not None
    assert skill.prerequisites == ["vector-addition"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    # Check sources cover all three difficulties
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == {"easy", "medium", "hard"}
