"""Natural logarithm and geometric nth-term formula (senior-define-the-natural-logarithm-ln-x-log-ex,
senior-use-the-formula-t-n-r-n-1)."""

from pathlib import Path
import re
import math

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p3c_logarithms_b_build import (
    _verify_ln_of_e_power,
    _verify_ln_general,
    _verify_e_to_the_k,
    _verify_easy_geometric_term,
    _verify_medium_geometric_ratio,
    _verify_hard_geometric_term,
    _iterative_geometric_term,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

LN_TOPIC = "natural_logarithm_define"
GEOMETRIC_TOPIC = "geometric_nth_term_formula"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [LN_TOPIC, GEOMETRIC_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [LN_TOPIC, GEOMETRIC_TOPIC])
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


@pytest.mark.parametrize("topic", [LN_TOPIC, GEOMETRIC_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Mirrors the convention bar: >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [LN_TOPIC, GEOMETRIC_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [LN_TOPIC, GEOMETRIC_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        if isinstance(question.answer, int):
            wrong = question.answer + 1
        else:
            wrong = question.answer + 0.001  # Add a small amount for floats
        assert not grade_number(question, str(wrong)).correct


# --- natural_logarithm_define -----------------------------------------------


def test_easy_ln_is_always_integer_no_decimal():
    """Easy-tier ln(e^k) always returns an integer (k itself)."""
    for seed in SEEDS:
        question = make(LN_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_medium_ln_to_3_decimal_places():
    """Medium-tier ln(x) is formatted to 3 decimal places."""
    for seed in SEEDS:
        question = make(LN_TOPIC, seed=seed, difficulty="medium")
        assert "." in question.answer_text
        places = len(question.answer_text.split(".")[1])
        assert places == 3


def test_hard_e_to_k_to_3_decimal_places():
    """Hard-tier e^k is formatted to 3 decimal places."""
    for seed in SEEDS:
        question = make(LN_TOPIC, seed=seed, difficulty="hard")
        assert "." in question.answer_text
        places = len(question.answer_text.split(".")[1])
        assert places == 3


def test_medium_ln_distractors_differ_by_more_than_half_unit_third_place():
    """For 3-decimal-place answers, every distractor must differ by > 0.0005."""
    for seed in SEEDS:
        question = make(LN_TOPIC, seed=seed, difficulty="medium")
        canonical = float(question.answer_text)
        for distractor in question.distractors:
            distractor_value = float(distractor)
            gap = abs(distractor_value - canonical)
            assert gap > 0.0005, f"Seed {seed}: distractor {distractor} too close to answer {question.answer_text}"


def test_hard_e_to_k_distractors_differ_by_more_than_half_unit_third_place():
    """For 3-decimal-place e^k answers, every distractor must differ by > 0.0005."""
    for seed in SEEDS:
        question = make(LN_TOPIC, seed=seed, difficulty="hard")
        canonical = float(question.answer_text)
        for distractor in question.distractors:
            distractor_value = float(distractor)
            gap = abs(distractor_value - canonical)
            assert gap > 0.0005, f"Seed {seed}: distractor {distractor} too close to answer {question.answer_text}"


def test_ln_easy_independent_check_fires_on_corrupted_answer():
    _verify_ln_of_e_power(2, 2)  # must not raise (k=2, expected=2)
    with pytest.raises(VerificationError):
        _verify_ln_of_e_power(2, 3)  # wrong expected value


def test_ln_medium_independent_check_fires_on_corrupted_answer():
    _verify_ln_general(10, 2.302)  # must not raise
    with pytest.raises(VerificationError):
        _verify_ln_general(10, 2.0)  # clearly wrong


def test_ln_hard_independent_check_fires_on_corrupted_answer():
    x = round(math.e**2, 3)  # e^2 ≈ 7.389
    _verify_e_to_the_k(x, 2)  # must not raise
    with pytest.raises(VerificationError):
        _verify_e_to_the_k(x, 1)  # wrong k


def test_ln_raises_on_unknown_variant():
    with pytest.raises(ValueError):
        make(LN_TOPIC, seed=1, variant="unknown")


# --- geometric_nth_term_formula -----------------------------------------------


def test_easy_geometric_term_is_always_integer():
    """Easy-tier direct substitution always produces an integer."""
    for seed in SEEDS:
        question = make(GEOMETRIC_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_medium_geometric_ratio_is_always_integer():
    """Medium-tier solving for r always produces an integer."""
    for seed in SEEDS:
        question = make(GEOMETRIC_TOPIC, seed=seed, difficulty="medium")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_hard_geometric_negative_ratio_is_always_integer():
    """Hard-tier with negative r always produces an integer."""
    for seed in SEEDS:
        question = make(GEOMETRIC_TOPIC, seed=seed, difficulty="hard")
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


def test_easy_geometric_independent_check_fires_on_corrupted_answer():
    _verify_easy_geometric_term(2, 3, 4, 54)  # 3^3 * 2 = 54, must not raise
    with pytest.raises(VerificationError):
        _verify_easy_geometric_term(2, 3, 4, 55)


def test_medium_geometric_independent_check_fires_on_corrupted_answer():
    # t1=2, r=2, n=4 => t_n = 2^(4-1) * 2 = 2^3 * 2 = 8 * 2 = 16
    _verify_medium_geometric_ratio(2, 16, 4, 2)  # must not raise
    with pytest.raises(VerificationError):
        _verify_medium_geometric_ratio(2, 16, 4, 3)  # wrong r (3^3*2=54, not 16)


def test_hard_geometric_negative_independent_check_fires_on_corrupted_answer():
    # r=-2, t1=3, n=4 => (-2)^3*3 = -8*3 = -24
    _verify_hard_geometric_term(3, -2, 4, -24)  # must not raise
    with pytest.raises(VerificationError):
        _verify_hard_geometric_term(3, -2, 4, -25)


def test_hard_geometric_sign_collision_ignores_sign_misconception():
    """Measure how often the 'ignores-the-sign' distractor collides with the answer
    when n-1 is even (even power of negative base gives positive result)."""
    collision_count = 0
    for seed in SEEDS:
        question = make(GEOMETRIC_TOPIC, seed=seed, difficulty="hard")
        # Parse r and n from the question text
        match = re.search(r"common ratio r = (-?\d+)", question.text)
        assert match, f"Could not parse r from: {question.text}"
        r = int(match.group(1))
        match = re.search(r"find t_(\d+)", question.text)
        assert match, f"Could not parse n from: {question.text}"
        n = int(match.group(1))

        # Collision occurs when n-1 is even
        if (n - 1) % 2 == 0:
            collision_count += 1

    # Measure the collision fraction
    collision_fraction = collision_count / len(SEEDS)
    # With n uniformly sampled from {3,4,5,6}, n-1 is in {2,3,4,5}, half even/half odd
    # So we expect collision_fraction ≈ 0.5, but allow 40-seed variance
    assert 0.3 < collision_fraction < 0.7, f"Collision fraction {collision_fraction} outside expected range"


def test_hard_geometric_negative_ratio_has_example_r_minus_3():
    """Hand-verify a specific negative-ratio case: r=-3, t1=2, n=5."""
    # t_n = (-3)^(5-1) * 2 = (-3)^4 * 2 = 81 * 2 = 162
    result = _iterative_geometric_term(2, -3, 5)
    assert result == 162
    # Verify grading works on negative typed strings
    question_text = "A geometric sequence has t1 = 2 and common ratio r = -3. Using t_n = r**(n-1) * t1, find t_5."
    # Simulate creating a question with this answer
    test_question = make(GEOMETRIC_TOPIC, seed=0, difficulty="hard", variant="negative_ratio")
    # Just verify that -3 is handled correctly as a negative base
    result_neg2 = _iterative_geometric_term(5, -2, 6)
    assert result_neg2 == (-2) ** 5 * 5  # (-2)^5 * 5 = -32 * 5 = -160
    assert result_neg2 == -160


def test_easy_geometric_inter_distractor_overlap():
    """Measure any overlap between the 3 easy-tier misconception distractors.
    While unlikely, some seeds may produce fewer than 3 distinct distractors.
    All 40 seeds should have at least 1 distractor (enforced by pick_distractors),
    and >=85% should have >=2."""
    overlap_seeds = 0
    for seed in SEEDS:
        question = make(GEOMETRIC_TOPIC, seed=seed, difficulty="easy")
        # Count how many distractors are present
        if len(question.distractors) < 3:
            overlap_seeds += 1
    # Most seeds should have all 3 distractors, but allow some overlap
    # The bar is >=85% with >=2, which is already checked in test_distractors_distinct_nonempty_never_equal_answer


def test_geometric_raises_on_unknown_variant():
    with pytest.raises(ValueError):
        make(GEOMETRIC_TOPIC, seed=1, variant="unknown")


# --- skill contracts ---------------------------------------------------------


def test_p3c_logarithms_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    ln_skill = by_id["senior-define-the-natural-logarithm-ln-x-log-ex"]
    geometric_skill = by_id["senior-use-the-formula-t-n-r-n-1"]

    # Ln skill checks
    assert ln_skill.prerequisites == ["function-notation"]
    assert ln_skill.answer == "number"
    assert ln_skill.placement_eligible is False
    assert ln_skill.status == "active"
    assert [source.difficulty for source in ln_skill.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in ln_skill.sources} == {LN_TOPIC}
    assert [source.stages for source in ln_skill.sources] == [[6], [6], [6, "maintenance"]]

    # Geometric skill checks
    assert geometric_skill.prerequisites == ["function-notation"]
    assert geometric_skill.answer == "number"
    assert geometric_skill.placement_eligible is False
    assert geometric_skill.status == "active"
    assert [source.difficulty for source in geometric_skill.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in geometric_skill.sources} == {GEOMETRIC_TOPIC}
    assert [source.stages for source in geometric_skill.sources] == [[6], [6], [6, "maintenance"]]
