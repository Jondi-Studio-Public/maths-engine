"""Focused contract tests for the recursive-sequence-generation skills."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p3c_recursion_build import (
    _has_zero_crossing,
    _verify_arithmetic_recursion,
    _verify_geometric_recursion_integer,
    _verify_geometric_recursion_fraction,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- Determinism and distractor hygiene ---


@pytest.mark.parametrize("topic", ["arithmetic_recursion_generate", "geometric_recursion_generate"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    """Same seed produces identical questions."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", ["arithmetic_recursion_generate", "geometric_recursion_generate"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_is_list_of_ints(topic, difficulty):
    """Answer is a plain list of ints."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, list)
        assert all(isinstance(t, int) for t in question.answer)


@pytest.mark.parametrize("topic", ["arithmetic_recursion_generate", "geometric_recursion_generate"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_are_distinct_and_non_empty(topic, difficulty):
    """Distractors are all distinct from each other and the answer."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors, (
            f"answer_text {question.answer_text!r} appears in distractors"
        )
        assert len(question.distractors) == len(set(question.distractors)), (
            f"distractors have duplicates: {question.distractors}"
        )
        assert len(question.distractors) == 3, (
            f"expected exactly 3 distractors, got {len(question.distractors)}: {question.distractors}"
        )


@pytest.mark.parametrize("topic", ["arithmetic_recursion_generate", "geometric_recursion_generate"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_correct_answer_grades_correct_and_distractors_grade_incorrect(topic, difficulty):
    """Real grader accepts canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        # Canonical answer should grade correct
        result = grade_ordered_list(question, question.answer_text)
        assert result.correct, f"seed {seed}: canonical {question.answer_text!r} did not grade correct"
        # Every distractor should grade incorrect
        for i, distractor in enumerate(question.distractors):
            result = grade_ordered_list(question, distractor)
            assert not result.correct, f"seed {seed}: distractor {i} {distractor!r} graded correct (should not)"


@pytest.mark.parametrize("topic", ["arithmetic_recursion_generate", "geometric_recursion_generate"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_reversed_order_distractor_is_present_and_rejects(topic, difficulty):
    """The reversed-order permutation is a distractor and grade_ordered_list rejects it.

    This is the load-bearing test that proves the grader is order-sensitive and
    the defect in permutation-composition (using answer: list) is avoided here.
    """
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        items = question.answer
        if len(items) < 2:
            continue
        # Build the reversed-order answer string
        reversed_items = list(reversed(items))
        reversed_text = ", ".join(str(t) for t in reversed_items)

        # Assert the reversed-order string is in the distractors
        assert reversed_text in question.distractors, (
            f"seed {seed}: reversed order {reversed_text!r} is NOT in distractors {question.distractors}"
        )

        # Assert grade_ordered_list rejects the reversed order
        result = grade_ordered_list(question, reversed_text)
        assert not result.correct, (
            f"seed {seed}: grade_ordered_list accepted reversed order {reversed_text!r}, should reject"
        )


def test_deliberately_wrong_answer_is_rejected():
    """A deliberately wrong answer (e.g., extra term) is rejected by the grader."""
    for seed in range(20):
        for topic in ["arithmetic_recursion_generate", "geometric_recursion_generate"]:
            for difficulty in ["easy", "medium", "hard"]:
                question = make(topic, seed=seed, difficulty=difficulty)
                # Create a wrong answer by appending an extra term
                wrong_items = question.answer + [question.answer[-1] + 1]
                wrong_text = ", ".join(str(t) for t in wrong_items)
                result = grade_ordered_list(question, wrong_text)
                assert not result.correct, f"seed {seed}/{topic}/{difficulty}: deliberately wrong answer graded correct"


@pytest.mark.parametrize("topic", ["arithmetic_recursion_generate", "geometric_recursion_generate"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_independent_check_fires_on_corrupted_answer(topic, difficulty):
    """Independent check raises VerificationError when answer is corrupted."""
    # Arithmetic: easy example
    if topic == "arithmetic_recursion_generate":
        with pytest.raises(VerificationError):
            _verify_arithmetic_recursion(t1=1, d=2, k=3, terms=[3, 5, 6])  # corrupted (should be [3, 5, 7])
    else:  # geometric
        if difficulty == "hard":
            with pytest.raises(VerificationError):
                _verify_geometric_recursion_fraction(t1=24, r_denom=2, k=4, terms=[12, 6, 3, 1])  # changed last term
        else:
            with pytest.raises(VerificationError):
                _verify_geometric_recursion_integer(
                    t1=2, r=3, k=3, terms=[6, 18, 55]
                )  # corrupted (should be [6, 18, 54])


# --- Arithmetic-specific tests ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_arithmetic_easy_has_three_terms(difficulty):
    """Arithmetic easy tier generates k=3 terms."""
    for seed in SEEDS:
        question = make("arithmetic_recursion_generate", seed=seed, difficulty="easy")
        assert len(question.answer) == 3, f"seed {seed}: expected 3 terms, got {len(question.answer)}"


def test_arithmetic_easy_has_positive_common_difference():
    """Arithmetic easy tier uses d in [1, 9] (always positive)."""
    for seed in SEEDS:
        question = make("arithmetic_recursion_generate", seed=seed, difficulty="easy")
        terms = question.answer
        diffs = [terms[i + 1] - terms[i] for i in range(len(terms) - 1)]
        assert all(1 <= d <= 9 for d in diffs), f"seed {seed}: not all differences in [1, 9]: {diffs}"


def test_arithmetic_medium_has_four_terms():
    """Arithmetic medium tier generates k=4 terms."""
    for seed in SEEDS:
        question = make("arithmetic_recursion_generate", seed=seed, difficulty="medium")
        assert len(question.answer) == 4


def test_arithmetic_medium_allows_negative_common_difference():
    """Arithmetic medium tier d may be negative."""
    saw_negative = False
    for seed in range(60):
        question = make("arithmetic_recursion_generate", seed=seed, difficulty="medium")
        terms = question.answer
        d = terms[1] - terms[0]
        if d < 0:
            saw_negative = True
            break
    assert saw_negative, "never saw a negative common difference in 60 seeds"


def test_arithmetic_hard_has_five_terms():
    """Arithmetic hard tier generates k=5 terms."""
    for seed in SEEDS:
        question = make("arithmetic_recursion_generate", seed=seed, difficulty="hard")
        assert len(question.answer) == 5


def test_arithmetic_hard_has_zero_crossing():
    """Arithmetic hard tier enforces zero-crossing: the generated sequence
    (including t1) must cross or touch zero."""
    for seed in SEEDS:
        question = make("arithmetic_recursion_generate", seed=seed, difficulty="hard")
        # Extract t1 and d from the question text
        match_t1 = re.search(r"t1 = (-?\d+)", question.text)
        match_d = re.search(r"\+ \(?(-?\d+)\)?\.", question.text)
        assert match_t1, f"could not extract t1 from: {question.text}"
        assert match_d, f"could not extract d from: {question.text}"
        t1 = int(match_t1.group(1))
        d = int(match_d.group(1))
        # Construct full sequence including t1
        k = len(question.answer)
        full_sequence = [t1 + m * d for m in range(k + 1)]
        # Verify zero-crossing
        assert _has_zero_crossing(full_sequence), f"seed {seed}: sequence {full_sequence} does not cross zero"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_arithmetic_answer_contains_only_generated_terms_not_t1(difficulty):
    """The answer contains only the k generated terms, never t1 itself."""
    for seed in SEEDS:
        question = make("arithmetic_recursion_generate", seed=seed, difficulty=difficulty)
        # Extract t1
        match = re.search(r"t1 = (-?\d+)", question.text)
        assert match
        t1 = int(match.group(1))
        # t1 should not be in the answer
        assert t1 not in question.answer, f"seed {seed}: t1 = {t1} appears in answer {question.answer}"


# --- Geometric-specific tests ---


def test_geometric_easy_has_three_terms():
    """Geometric easy tier generates k=3 terms."""
    for seed in SEEDS:
        question = make("geometric_recursion_generate", seed=seed, difficulty="easy")
        assert len(question.answer) == 3


def test_geometric_easy_has_integer_growth():
    """Geometric easy tier uses r in [2, 4] (integer growth)."""
    for seed in SEEDS:
        question = make("geometric_recursion_generate", seed=seed, difficulty="easy")
        # Extract r and verify it's in [2, 4]
        match = re.search(r"t\(n\+1\) = t\(n\) × (\d+)", question.text)
        assert match, f"could not extract r from: {question.text}"
        r = int(match.group(1))
        assert 2 <= r <= 4


def test_geometric_medium_has_four_terms():
    """Geometric medium tier generates k=4 terms."""
    for seed in SEEDS:
        question = make("geometric_recursion_generate", seed=seed, difficulty="medium")
        assert len(question.answer) == 4


def test_geometric_medium_allows_negative_ratio():
    """Geometric medium tier r may be negative (alternating sign)."""
    saw_negative = False
    for seed in range(60):
        question = make("geometric_recursion_generate", seed=seed, difficulty="medium")
        # Extract r
        match = re.search(r"t\(n\+1\) = t\(n\) × \((-\d+)\)", question.text)
        if match:
            r = int(match.group(1))
            assert r < 0
            saw_negative = True
            break
    assert saw_negative, "never saw a negative ratio in 60 seeds"


def test_geometric_hard_has_four_terms():
    """Geometric hard tier generates k=4 terms (fractional decay)."""
    for seed in SEEDS:
        question = make("geometric_recursion_generate", seed=seed, difficulty="hard")
        assert len(question.answer) == 4


def test_geometric_hard_is_fractional_decay():
    """Geometric hard tier uses division (fractional decay, r = 1/r_denom where r_denom in {2, 3})."""
    for seed in SEEDS:
        question = make("geometric_recursion_generate", seed=seed, difficulty="hard")
        # Extract r_denom from question text (should have "/ 2" or "/ 3")
        match = re.search(r"t\(n\+1\) = t\(n\) / (\d+)", question.text)
        assert match, f"could not extract r_denom from: {question.text}"
        r_denom = int(match.group(1))
        assert r_denom in {2, 3}
        # Verify the answer is a decreasing sequence
        terms = question.answer
        assert all(terms[i] > terms[i + 1] for i in range(len(terms) - 1)), (
            f"seed {seed}: geometric hard answer not decreasing: {terms}"
        )


def test_geometric_hard_exact_divisibility():
    """Geometric hard tier: every step is exactly divisible (no truncation).
    This test verifies t1 % (r_denom**m) == 0 for all m in [1, k]."""
    for seed in SEEDS:
        question = make("geometric_recursion_generate", seed=seed, difficulty="hard")
        # Extract t1 and r_denom
        match_t1 = re.search(r"t1 = (\d+)", question.text)
        match_denom = re.search(r"/ (\d+)", question.text)
        assert match_t1 and match_denom
        t1 = int(match_t1.group(1))
        r_denom = int(match_denom.group(1))
        k = len(question.answer)
        # Verify divisibility at each step
        for m in range(1, k + 1):
            assert t1 % (r_denom**m) == 0, f"seed {seed}: t1={t1} not exactly divisible by r_denom^{m}={r_denom**m}"


def test_geometric_hard_rejects_divisibility_failure():
    """The independent check fires when exact divisibility fails."""
    with pytest.raises(VerificationError):
        # t1 = 7, r_denom = 2, k = 4: 7 // 2 is not exact
        _verify_geometric_recursion_fraction(t1=7, r_denom=2, k=4, terms=[3, 1, 0, 0])


# --- Skill contract loading tests ---


def test_both_skills_load_with_answer_ordered_list():
    """Both skills load with answer: ordered_list."""
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    arith_skill = by_id.get("senior-use-recursion-to-generate-an-arithmetic-sequence")
    geom_skill = by_id.get("senior-use-recursion-to-generate-a-geometric-sequence")

    assert arith_skill is not None, "arithmetic skill not found"
    assert geom_skill is not None, "geometric skill not found"

    assert arith_skill.answer == "ordered_list", (
        f"arithmetic skill answer is {arith_skill.answer!r}, expected 'ordered_list'"
    )
    assert geom_skill.answer == "ordered_list", (
        f"geometric skill answer is {geom_skill.answer!r}, expected 'ordered_list'"
    )


def test_both_skills_have_correct_prerequisites():
    """Each skill builds on function notation and on the recursive definition of its own sequence."""
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    arith_skill = by_id["senior-use-recursion-to-generate-an-arithmetic-sequence"]
    geom_skill = by_id["senior-use-recursion-to-generate-a-geometric-sequence"]

    assert arith_skill.prerequisites == ["function-notation", "senior-recognise-and-use-the-recursive-definition-of-an"]
    assert geom_skill.prerequisites == ["function-notation", "senior-recognise-and-use-the-recursive-definition-of-a"]


def test_both_skills_are_active_and_not_placement_eligible():
    """Both skills have status: active and placement_eligible: false."""
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    arith_skill = by_id["senior-use-recursion-to-generate-an-arithmetic-sequence"]
    geom_skill = by_id["senior-use-recursion-to-generate-a-geometric-sequence"]

    assert arith_skill.status == "active"
    assert arith_skill.placement_eligible is False
    assert geom_skill.status == "active"
    assert geom_skill.placement_eligible is False


def test_both_skills_have_sources_covering_all_difficulties():
    """Both skills have sources for easy, medium, hard."""
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    arith_skill = by_id["senior-use-recursion-to-generate-an-arithmetic-sequence"]
    geom_skill = by_id["senior-use-recursion-to-generate-a-geometric-sequence"]

    arith_difficulties = [source.difficulty for source in arith_skill.sources]
    geom_difficulties = [source.difficulty for source in geom_skill.sources]

    assert set(arith_difficulties) == {"easy", "medium", "hard", "exam"}
    assert set(geom_difficulties) == {"easy", "medium", "hard", "exam"}


# --- Helper function tests ---


def test_has_zero_crossing():
    """Test the zero-crossing detection helper.

    The zero-crossing detection uses adjacent-pair sign changes:
    one value < 0 and the next >= 0, or one >= 0 and the next < 0.
    Note: a sequence like [0, 1, 2] contains a zero but has no qualifying pair
    and is NOT detected as a crossing by this test."""
    # No crossing
    assert not _has_zero_crossing([1, 2, 3])
    assert not _has_zero_crossing([-3, -2, -1])
    assert not _has_zero_crossing([0, 1, 2])  # contains zero but no sign change pair
    assert not _has_zero_crossing([2, 1, 0])  # contains zero but no sign change pair

    # Sign change cases (adjacent pair with one < 0 and next >= 0, or vice versa)
    assert _has_zero_crossing([1, 0, -1])  # pair (0, -1): 0 >= 0 and -1 < 0
    assert _has_zero_crossing([-1, 0, 1])  # pair (-1, 0): -1 < 0 and 0 >= 0
    assert _has_zero_crossing([1, -1])  # pair (1, -1): 1 >= 0 and -1 < 0
    assert _has_zero_crossing([-1, 1])  # pair (-1, 1): -1 < 0 and 1 >= 0
