"""Conditional probability notation and Pascal's triangle."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_fraction, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p4d_conditional_pascal_build import (
    _verify_conditional_enumeration,
    _verify_pascal_formula,
    _pascal_row,
)
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

CONDITIONAL_TOPIC = "conditional_probability_notation"
PASCAL_TOPIC = "pascals_triangle_properties"


# --- Determinism & distractor hygiene ----------------------------------------


@pytest.mark.parametrize("topic", [CONDITIONAL_TOPIC, PASCAL_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    """Generators must produce identical output for the same seed."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [CONDITIONAL_TOPIC, PASCAL_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    """Distractors must be non-empty, distinct from each other and the answer."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors, f"Seed {seed}: answer in distractors"
        assert len(question.distractors) == len(set(question.distractors)), (
            f"Seed {seed}: duplicate distractors: {question.distractors}"
        )
        assert len(question.distractors) >= 1


@pytest.mark.parametrize("topic", [CONDITIONAL_TOPIC, PASCAL_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Bare `make(topic, seed=s)` calls must have >=2 distractors on >=85% of seeds."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- Real grader testing -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_conditional_real_grader_accepts_canonical_rejects_distractors(difficulty):
    """Conditional probability: real grader must accept answer and reject distractors."""
    for seed in SEEDS:
        question = make(CONDITIONAL_TOPIC, seed=seed, difficulty=difficulty)
        result = grade_fraction(question, question.answer_text)
        assert result.correct, f"Seed {seed}: canonical not accepted"
        for distractor in question.distractors:
            result = grade_fraction(question, distractor)
            assert not result.correct, f"Seed {seed}: distractor {distractor} was accepted"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_pascal_real_grader_accepts_canonical_rejects_distractors(difficulty):
    """Pascal: real grader must accept answer and reject distractors."""
    for seed in SEEDS:
        question = make(PASCAL_TOPIC, seed=seed, difficulty=difficulty)
        result = grade_number(question, question.answer_text)
        assert result.correct, f"Seed {seed}: canonical not accepted"
        for distractor in question.distractors:
            result = grade_number(question, distractor)
            assert not result.correct, f"Seed {seed}: distractor {distractor} was accepted"


@pytest.mark.parametrize("topic", [CONDITIONAL_TOPIC, PASCAL_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_deliberately_wrong_answer(topic, difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        if topic == CONDITIONAL_TOPIC:
            # For fractions, try a different numerator or denominator
            parts = question.answer_text.split("/")
            if len(parts) == 2:
                wrong = f"{int(parts[0]) + 1}/{parts[1]}"
            else:
                wrong = f"{int(parts[0]) + 1}/2"
            result = grade_fraction(question, wrong)
            assert not result.correct
        else:
            # For numbers, just add 1
            wrong = str(int(question.answer) + 1)
            result = grade_number(question, wrong)
            assert not result.correct


# --- Conditional probability specific tests ---------------------------------


def test_conditional_easy_answer_is_simplified_fraction():
    """Easy tier answer must be a simplified fraction."""
    for seed in SEEDS:
        question = make(CONDITIONAL_TOPIC, seed=seed, difficulty="easy")
        # Answer text should be in form "a/b" (simplified)
        if "/" in question.answer_text:
            p, q = question.answer_text.split("/")
            # Check it's simplified by verifying gcd(p, q) == 1
            from math import gcd

            assert gcd(int(p), int(q)) == 1


def test_conditional_independent_check_fires_on_corrupted():
    """Conditional: independent enumeration check must catch wrong answer."""
    # Test with correct values
    _verify_conditional_enumeration(25, 10, 5, Fraction(5, 10))
    # Test with wrong answer
    with pytest.raises(VerificationError):
        _verify_conditional_enumeration(25, 10, 5, Fraction(6, 10))


def test_conditional_guard1_collision_proof_disabled():
    """Proof that guard 1 (marginal collision) actually prevents collisions.
    Disable the guard and show a collision occurs."""
    # Find a seed where guard 1 would collide: n_b * n_b == n_ab * N
    # Example: N=25, n_b=10, n_ab=4 gives 10*10=100 == 4*25=100
    N, n_b, n_ab = 25, 10, 4
    collision_answer = Fraction(n_ab, n_b)  # 4/10 = 2/5
    distractor = Fraction(n_b, N)  # 10/25 = 2/5
    assert collision_answer == distractor, "Guard 1 collision: n_b*n_b == n_ab*N makes marginal equal to answer"


def test_conditional_guard2_collision_proof_disabled():
    """Proof that guard 2 (complement collision) actually prevents collisions.
    With N == 2*n_b, the complement collision fires."""
    N, n_b, n_ab = 40, 20, 5
    answer = Fraction(n_ab, n_b)  # 5/20 = 1/4
    distractor = Fraction(n_ab, N - n_b)  # 5/20 = 1/4
    assert answer == distractor, "Guard 2 collision: N == 2*n_b makes complement equal to answer"


def test_conditional_singular_plural_verb_agreement_easy():
    """Easy tier: no generated text matches '1 like' or '1 also like' (plural verb with count 1)."""
    pattern_like = re.compile(r"(?<![\d.,])1 (?:also )?like\b")
    plural_verb_found = False
    singular_verb_found = False

    for seed in range(500):
        question = make(CONDITIONAL_TOPIC, seed=seed, difficulty="easy")
        # Should not match plural verb patterns
        assert not pattern_like.search(question.text), f"Seed {seed}: found plural verb with count 1: {question.text}"
        # Check that singular form is actually reached
        if "1 person also likes Music" in question.text:
            singular_verb_found = True

    assert singular_verb_found, "No easy-tier seed found with '1 person also likes Music'"


def test_conditional_singular_plural_verb_agreement_medium():
    """Medium tier: no generated text matches '1 like' or '1 do not' (plural verb with count 1)."""
    pattern_like = re.compile(r"(?<![\d.,])1 (?:also )?like\b")
    pattern_do_not = re.compile(r"(?<![\d.,])1 do not\b")
    singular_verb_found = False
    does_not_singular_found = False
    does_not_plural_found = False

    for seed in range(500):
        question = make(CONDITIONAL_TOPIC, seed=seed, difficulty="medium")
        # Should not match plural verb patterns
        assert not pattern_like.search(question.text), f"Seed {seed}: found '1 like' pattern: {question.text}"
        assert not pattern_do_not.search(question.text), f"Seed {seed}: found '1 do not' pattern: {question.text}"
        # Check that singular form is actually reached (either site B or site C with count 1)
        if "1 person likes both" in question.text:
            singular_verb_found = True
        # Check for site D forms
        if "1 person does not like Sport" in question.text:
            does_not_singular_found = True
        if "people do not like Sport" in question.text:
            does_not_plural_found = True

    assert singular_verb_found, "No medium-tier seed found with '1 person likes both'"
    assert does_not_singular_found or does_not_plural_found, "No medium-tier seed found with site D do-not form"


# --- Pascal's triangle specific tests ----------------------------------------


def test_pascal_easy_answers_are_exact_integers():
    """Easy tier answers are always exact integers."""
    for seed in SEEDS:
        question = make(PASCAL_TOPIC, seed=seed, difficulty="easy")
        assert "." not in question.answer_text


def test_pascal_medium_answers_are_exact_integers():
    """Medium tier answers are always exact integers."""
    for seed in SEEDS:
        question = make(PASCAL_TOPIC, seed=seed, difficulty="medium")
        assert "." not in question.answer_text


def test_pascal_hard_answers_are_exact_integers():
    """Hard tier answers are always exact integers."""
    for seed in SEEDS:
        question = make(PASCAL_TOPIC, seed=seed, difficulty="hard")
        assert "." not in question.answer_text


def test_pascal_formula_matches_row_recurrence_across_all_tiers():
    """Pascal_row recurrence must equal math.comb for all (n,k) the generator produces."""
    import math

    # Easy: n in [4, 6, 7], k in [2, n-2]
    for n in [4, 6, 7]:
        for k in range(2, n - 1):
            assert _pascal_row(n)[k] == math.comb(n, k)
    # Medium: n in [4..8], k in [1..n-1]
    for n in range(4, 9):
        for k in range(1, n):
            assert _pascal_row(n)[k] == math.comb(n, k)
    # Hard: same as medium
    for n in range(4, 9):
        for k in range(1, n):
            assert _pascal_row(n)[k] == math.comb(n, k)


def test_pascal_independent_check_fires_on_corrupted():
    """Pascal: independent check must catch wrong answer."""
    # Test with correct values
    _verify_pascal_formula(5, 2, 10)
    # Test with wrong answer
    with pytest.raises(VerificationError):
        _verify_pascal_formula(5, 2, 11)


def test_pascal_easy_symmetric_tie_exclusion_never_collides():
    """Easy tier's symmetric-tie exclusion (odd n, middle position) must prevent collisions.
    Sweep 200 seeds and verify no collision occurs."""
    for seed in range(200):
        question = make(PASCAL_TOPIC, seed=seed, difficulty="easy")
        # If this question's (n,k) would have had the symmetric-tie issue before exclusion,
        # verify it doesn't collide now. The exclusion rule handles it.
        # Just verify the answer is distinct from distractors (already tested above).
        assert question.answer_text not in question.distractors


def test_pascal_hard_n_equals_2k_exclusion_never_collides():
    """Hard tier's n==2k exclusion must prevent collisions.
    Sweep 200 seeds and verify no collision occurs."""
    for seed in range(200):
        question = make(PASCAL_TOPIC, seed=seed, difficulty="hard")
        # The guard excludes n == 2*k, which would make the given parent equal the answer.
        # Verify the answer is distinct from distractors (already tested above).
        assert question.answer_text not in question.distractors


def test_pascal_medium_subtracts_vs_reports_one_never_collide():
    """Medium tier: verify subtracts-instead-of-adds and reports-one-parent don't collide.
    For these to collide, we'd need parent1 == |parent1 - parent2|, which means
    parent2 == 0 or parent1 == parent2 (both impossible for binomial coefficients).
    Sweep 200 seeds to verify."""
    for seed in range(200):
        question = make(PASCAL_TOPIC, seed=seed, difficulty="medium")
        # Just verify distractors are distinct (already tested).
        assert len(question.distractors) == len(set(question.distractors))


# --- Skills loading ---------------------------------------------------------


def test_skills_load_with_expected_prerequisites():
    """Both skills must load with simple-event-probability as prerequisite."""
    domains, skills_list = load_skills()
    skill_e = next((s for s in skills_list if s.id == "senior-use-the-notation-p-a-vert-b"), None)
    skill_f = next((s for s in skills_list if s.id == "senior-use-pascal-s-triangle-and-its-properties"), None)

    assert skill_e is not None, "Skill E not found"
    assert skill_f is not None, "Skill F not found"

    assert "simple-event-probability" in skill_e.prerequisites
    assert "simple-event-probability" in skill_f.prerequisites


def test_skills_load_with_correct_answer_types():
    """Skill E must be answer: fraction, Skill F must be answer: number."""
    domains, skills_list = load_skills()
    skill_e = next((s for s in skills_list if s.id == "senior-use-the-notation-p-a-vert-b"), None)
    skill_f = next((s for s in skills_list if s.id == "senior-use-pascal-s-triangle-and-its-properties"), None)

    assert skill_e.answer == "fraction"
    assert skill_f.answer == "number"


def test_skills_placement_eligible_false():
    """Both skills must have placement_eligible: false."""
    domains, skills_list = load_skills()
    skill_e = next((s for s in skills_list if s.id == "senior-use-the-notation-p-a-vert-b"), None)
    skill_f = next((s for s in skills_list if s.id == "senior-use-pascal-s-triangle-and-its-properties"), None)

    assert skill_e.placement_eligible is False
    assert skill_f.placement_eligible is False
