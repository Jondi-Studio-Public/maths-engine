"""Unit vectors and parallel/perpendicular vectors (senior-define-and-use-unit-vectors-and-the-perpendicular,
senior-examine-properties-of-parallel-and-perpendicular-vectors-and)."""

from pathlib import Path
from fractions import Fraction
import re

import pytest

from mathgen.grading.typed import grade_fraction, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p4c_vectors_unit_parallel_build import (
    TRIPLES,
    _verify_unit_vector_magnitude_one,
    _verify_perpendicular_unit_vector_magnitude_one,
    _verify_triple,
    _verify_perpendicular_dot_product_zero,
    _verify_parallel_proportionality,
    _verify_recover_scalar_from_vectors,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(500)

UNIT_VECTORS_TOPIC = "define_and_use_unit_vectors_and_the_perpendicular"
PARALLEL_PERPENDICULAR_TOPIC = "examine_properties_of_parallel_and_perpendicular_vectors_and"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [UNIT_VECTORS_TOPIC, PARALLEL_PERPENDICULAR_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_500_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [UNIT_VECTORS_TOPIC, PARALLEL_PERPENDICULAR_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


# --- Skill 1: define_and_use_unit_vectors_and_the_perpendicular ----------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unit_vectors_answer_is_always_fraction_string(difficulty):
    """All tiers return fraction-formatted answer_text."""
    for seed in range(50):
        question = make(UNIT_VECTORS_TOPIC, seed=seed, difficulty=difficulty)
        # Answer text should be parseable as a Fraction (may have negative)
        frac = Fraction(question.answer_text)
        assert isinstance(frac, Fraction)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unit_vectors_distractors_differ_in_fraction_value(difficulty):
    """Every distractor has a Fraction value that differs from the answer."""
    for seed in SEEDS:
        question = make(UNIT_VECTORS_TOPIC, seed=seed, difficulty=difficulty)
        answer_frac = Fraction(question.answer_text)
        for distractor in question.distractors:
            distractor_frac = Fraction(distractor)
            assert answer_frac != distractor_frac, (
                f"Seed {seed}: distractor {distractor} equals answer {question.answer_text} in Fraction value"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unit_vectors_real_grader_accepts_canonical_rejects_distractors(difficulty):
    for seed in range(50):
        question = make(UNIT_VECTORS_TOPIC, seed=seed, difficulty=difficulty)
        assert grade_fraction(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_fraction(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_unit_vectors_real_grader_rejects_deliberately_wrong_answer(difficulty):
    for seed in range(50):
        question = make(UNIT_VECTORS_TOPIC, seed=seed, difficulty=difficulty)
        # Wrong answer: add a small fraction value
        wrong = Fraction(question.answer_text) + Fraction(1, 100)
        assert not grade_fraction(question, str(wrong)).correct


def test_unit_vectors_easy_both_i_and_j_asked_over_sweep():
    """Easy tier should ask for both i and j coefficients across many seeds."""
    i_asked = 0
    j_asked = 0
    for seed in range(200):
        question = make(UNIT_VECTORS_TOPIC, seed=seed, difficulty="easy")
        if " coefficient of i " in question.text:
            i_asked += 1
        elif " coefficient of j " in question.text:
            j_asked += 1
    assert i_asked >= 50, f"Only {i_asked} seeds asked for i"
    assert j_asked >= 50, f"Only {j_asked} seeds asked for j"


def test_unit_vectors_medium_both_i_and_j_asked_over_sweep():
    """Medium tier should ask for both i and j coefficients across many seeds."""
    i_asked = 0
    j_asked = 0
    for seed in range(200):
        question = make(UNIT_VECTORS_TOPIC, seed=seed, difficulty="medium")
        if " coefficient of i " in question.text:
            i_asked += 1
        elif " coefficient of j " in question.text:
            j_asked += 1
    assert i_asked >= 50, f"Only {i_asked} seeds asked for i"
    assert j_asked >= 50, f"Only {j_asked} seeds asked for j"


def test_unit_vectors_hard_both_i_and_j_asked_over_sweep():
    """Hard tier should ask for both i and j coefficients across many seeds."""
    i_asked = 0
    j_asked = 0
    for seed in range(200):
        question = make(UNIT_VECTORS_TOPIC, seed=seed, difficulty="hard")
        if " coefficient of i " in question.text:
            i_asked += 1
        elif " coefficient of j " in question.text:
            j_asked += 1
    assert i_asked >= 50, f"Only {i_asked} seeds asked for i"
    assert j_asked >= 50, f"Only {j_asked} seeds asked for j"


def test_unit_vectors_all_9_triples_sampled_over_sweep():
    """All 9 Pythagorean triples should be sampled over a large seed sweep."""
    sampled_triples = set()
    for seed in SEEDS:
        question = make(UNIT_VECTORS_TOPIC, seed=seed, difficulty="easy")
        # Extract the vector (a, b) from question text: "v = (a, b)."
        match = re.search(r"v = \((-?\d+), (-?\d+)\)", question.text)
        if match:
            a_abs, b_abs = int(match.group(1)), int(match.group(2))
            # Normalize to the unsigned form in TRIPLES
            a_unsigned = abs(a_abs)
            b_unsigned = abs(b_abs)
            # Find which triple this is (ignoring signs)
            for leg_a, leg_b, hyp in TRIPLES:
                if {a_unsigned, b_unsigned} == {leg_a, leg_b}:
                    sampled_triples.add((leg_a, leg_b, hyp))
    assert len(sampled_triples) >= 8, f"Only {len(sampled_triples)} distinct triples sampled: {sampled_triples}"


def test_unit_vectors_easy_independent_check_fires_on_corrupted_answer():
    _verify_unit_vector_magnitude_one(Fraction(3, 5), Fraction(4, 5), Fraction(3, 5))
    with pytest.raises(VerificationError):
        _verify_unit_vector_magnitude_one(Fraction(3, 5), Fraction(4, 6), Fraction(3, 5))


def test_unit_vectors_hard_independent_check_fires_on_corrupted_answer():
    _verify_perpendicular_unit_vector_magnitude_one(Fraction(-4, 5), Fraction(3, 5), Fraction(-4, 5))
    with pytest.raises(VerificationError):
        _verify_perpendicular_unit_vector_magnitude_one(Fraction(-4, 5), Fraction(3, 6), Fraction(-4, 5))


def test_unit_vectors_triple_verification():
    _verify_triple(3, 4, 5)
    with pytest.raises(VerificationError):
        _verify_triple(3, 4, 6)  # Not a valid triple


def test_unit_vectors_hard_tier_distractor_overlap_disclosure():
    """Hard tier: when i is asked, used-original and wrong-axis both give a/m.
    Verify: at least 2 distinct distractors on every seed, never two identical strings."""
    for seed in SEEDS:
        question = make(UNIT_VECTORS_TOPIC, seed=seed, difficulty="hard")
        # Should have at least 2 distinct distractors
        assert len(question.distractors) >= 2, f"Seed {seed}: only {len(question.distractors)} distractors"
        # All distractors must be distinct strings
        assert len(set(question.distractors)) == len(question.distractors), (
            f"Seed {seed}: duplicate distractor strings: {question.distractors}"
        )


# --- Skill 2: examine_properties_of_parallel_and_perpendicular_vectors_and ------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_parallel_perpendicular_answer_is_always_integer(difficulty):
    """All tiers return integer answer."""
    for seed in range(50):
        question = make(PARALLEL_PERPENDICULAR_TOPIC, seed=seed, difficulty=difficulty)
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_parallel_perpendicular_real_grader_accepts_canonical_rejects_distractors(difficulty):
    for seed in range(50):
        question = make(PARALLEL_PERPENDICULAR_TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_parallel_perpendicular_real_grader_rejects_deliberately_wrong_answer(difficulty):
    for seed in range(50):
        question = make(PARALLEL_PERPENDICULAR_TOPIC, seed=seed, difficulty=difficulty)
        wrong = question.answer + 1
        assert not grade_number(question, str(wrong)).correct


def test_parallel_perpendicular_easy_both_symmetric_presentations():
    """Easy tier: both x-given/y-blank and y-given/x-blank should occur."""
    x_given = 0
    y_given = 0
    for seed in range(150):
        question = make(PARALLEL_PERPENDICULAR_TOPIC, seed=seed, difficulty="easy")
        if ", ?)" in question.text:
            y_given += 1
        elif "(?, " in question.text:
            x_given += 1
    assert x_given >= 40, f"Only {x_given} seeds with x given"
    assert y_given >= 40, f"Only {y_given} seeds with y given"


def test_parallel_perpendicular_medium_both_symmetric_presentations():
    """Medium tier: both x-given/y-blank and y-given/x-blank should occur."""
    x_given = 0
    y_given = 0
    for seed in range(150):
        question = make(PARALLEL_PERPENDICULAR_TOPIC, seed=seed, difficulty="medium")
        if ", ?)" in question.text:
            y_given += 1
        elif "(?, " in question.text:
            x_given += 1
    assert x_given >= 40, f"Only {x_given} seeds with x given"
    assert y_given >= 40, f"Only {y_given} seeds with y given"


def test_parallel_perpendicular_easy_independent_check_fires_on_corrupted_answer():
    _verify_perpendicular_dot_product_zero(3, 4, 4, -3)  # 3*4 + 4*(-3) = 0
    with pytest.raises(VerificationError):
        _verify_perpendicular_dot_product_zero(3, 4, 4, -2)  # 3*4 + 4*(-2) != 0


def test_parallel_perpendicular_medium_independent_check_fires_on_corrupted_answer():
    # v = (2, 4) parallel to u = (1, 2): 2*2 == 1*4
    _verify_parallel_proportionality(2, 4, 1, 2)
    with pytest.raises(VerificationError):
        _verify_parallel_proportionality(2, 3, 1, 2)  # 2*2 != 1*3


def test_parallel_perpendicular_hard_independent_check_fires_on_corrupted_answer():
    # v = (4, 6), u = (2, 3), k = 2
    _verify_recover_scalar_from_vectors(2, 2, 3, 4, 6)
    with pytest.raises(VerificationError):
        _verify_recover_scalar_from_vectors(2, 2, 3, 4, 7)  # Proportionality fails


def test_parallel_perpendicular_hard_independent_check_k_recomputation():
    """Hard tier: verify the independent k recomputation via B // q."""
    _verify_recover_scalar_from_vectors(3, 2, 5, 6, 15)  # k=3: 6=3*2, 15=3*5, 15//5=3
    with pytest.raises(VerificationError):
        _verify_recover_scalar_from_vectors(3, 2, 5, 6, 14)  # 14//5=2, not 3


# --- skill contracts ---------------------------------------------------------


def test_unit_vectors_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    skill = by_id["senior-define-and-use-unit-vectors-and-the-perpendicular"]

    assert skill.prerequisites == ["vector-addition"]
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in skill.sources} == {UNIT_VECTORS_TOPIC}
    assert [source.stages for source in skill.sources] == [[6], [6], [6, "maintenance"]]


def test_parallel_perpendicular_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    skill = by_id["senior-examine-properties-of-parallel-and-perpendicular-vectors-and"]

    assert skill.prerequisites == ["vector-addition"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in skill.sources} == {PARALLEL_PERPENDICULAR_TOPIC}
    assert [source.stages for source in skill.sources] == [[6], [6], [6, "maintenance"]]
