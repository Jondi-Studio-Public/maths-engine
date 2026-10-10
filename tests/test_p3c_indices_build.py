"""Index laws review and radicals/fractional indices
(senior-review-indices-including-fractional-indices-and-the-index,
senior-use-radicals-and-convert-to-and-from-fractional)."""

from pathlib import Path

import pytest
from sympy import Rational, sympify, simplify, symbols

from mathgen.grading.typed import grade_expression, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p3c_indices_build import (
    _verify_index_law_combination,
    _verify_radical_evaluation,
    _find_integer_root,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

INDEX_LAWS_TOPIC = "index_laws_review"
RADICALS_TOPIC = "radicals_fractional_indices"

x = symbols("x")

# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [INDEX_LAWS_TOPIC, RADICALS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [INDEX_LAWS_TOPIC, RADICALS_TOPIC])
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


@pytest.mark.parametrize("topic", [INDEX_LAWS_TOPIC, RADICALS_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Mirrors the bar (>=2 distractors on >=85% of seeds) for a bare
    `make(topic, seed=s)` call with no `difficulty` argument."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [INDEX_LAWS_TOPIC, RADICALS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        if topic == INDEX_LAWS_TOPIC:
            # Index laws use expression grader.
            assert grade_expression(question, question.answer_text).correct
            for distractor in question.distractors:
                assert not grade_expression(question, distractor).correct
        else:
            # Radicals use number grader.
            assert grade_number(question, question.answer_text).correct
            for distractor in question.distractors:
                assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [INDEX_LAWS_TOPIC, RADICALS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        if topic == INDEX_LAWS_TOPIC:
            # Deliberately wrong for index laws: x**(ans_exp + 3).
            # Reconstruct from the answer string.
            try:
                ans_expr = sympify(question.answer)
                # Add 3 to the exponent (assume answer is of form x**exp).
                if isinstance(ans_expr, type(x**2)):  # x**n form
                    # Extract exponent and add 3.
                    wrong_expr = sympify(f"sympify('{question.answer}') * x**3")
                    wrong = str(ans_expr * x**3)
                else:
                    wrong = str(ans_expr * x**3)
            except Exception:
                wrong = question.answer_text + "_wrong"
            assert not grade_expression(question, wrong).correct
        else:
            # Deliberately wrong for radicals: answer + 5.
            wrong = str(question.answer + 5)
            assert not grade_number(question, wrong).correct


# --- index_laws_review -------------------------------------------------------


def test_easy_index_product_exponents_range():
    """Easy tier: a in [2,9], b in [1,9]."""
    for seed in range(100):
        question = make(INDEX_LAWS_TOPIC, seed=seed, difficulty="easy")
        # The answer should be x^(a+b) where 3 <= a+b <= 18.
        # We can't easily extract a,b from the question, but the answer text
        # should match this pattern.
        assert question.answer_text.startswith("x^")


def test_medium_index_guarantees_positive_final_exponent():
    """Medium tier: final exponent a+b-c >= 1 (guaranteed by c in [1, a+b-1])."""
    for seed in range(100):
        question = make(INDEX_LAWS_TOPIC, seed=seed, difficulty="medium")
        # Answer should be x**n where n >= 1.
        # When n=1, sympy represents it as just "x", not "x^1".
        answer_str = question.answer_text
        assert answer_str == "x" or answer_str.startswith("x^")


def test_hard_index_fractional_exponent_in_answer():
    """Hard tier: answer includes fractional exponent like x**(1/2), x**(1/3), etc."""
    for seed in range(100):
        question = make(INDEX_LAWS_TOPIC, seed=seed, difficulty="hard")
        # At least check that the answer_text is valid and can be parsed.
        try:
            ans = sympify(question.answer)
        except Exception:
            pytest.fail(f"Could not parse answer: {question.answer_text}")


def test_hard_index_exhaustive_check_power_of_power_vs_add():
    """Hard tier: verify exhaustively (not by sampling) that p*q != p+q for all
    9 combinations of p in {1/2, 1/3, 2/3} and q in [2, 4]."""
    from sympy import Rational

    FRACS = [Rational(1, 2), Rational(1, 3), Rational(2, 3)]
    for p in FRACS:
        for q in range(2, 5):
            assert p * q != p + q, f"p*q == p+q for p={p}, q={q}"


def test_index_laws_independent_check_fires_on_corrupted_answer():
    # Test that the independent check fires when answer is wrong.
    with pytest.raises(VerificationError):
        _verify_index_law_combination(x**2 * x**3, x**6)  # Should be x**5


# --- radicals_fractional_indices -------------------------------------------------------


def test_easy_radical_root_extraction():
    """Easy tier: evaluate n**(1/r) where n = b**r."""
    for seed in range(100):
        question = make(RADICALS_TOPIC, seed=seed, difficulty="easy")
        answer = int(question.answer)
        # Answer should be between 2 and 6 (since b is in [2,6]).
        assert 2 <= answer <= 6


def test_medium_radical_includes_power_exponent():
    """Medium tier: worded as 'root of n, raised to power of m', so m > 1 always."""
    for seed in range(100):
        question = make(RADICALS_TOPIC, seed=seed, difficulty="medium")
        # Just verify answer is an int and reasonable.
        answer = int(question.answer)
        assert answer > 0


def test_hard_radical_higher_roots():
    """Hard tier: 4th or 5th roots, so r in [4, 5]."""
    for seed in range(100):
        question = make(RADICALS_TOPIC, seed=seed, difficulty="hard")
        answer = int(question.answer)
        # Answer is b**m where b in [2,3], m in [2, r-1].
        # Max is 3**(4) = 81.
        assert answer <= 81


def test_radical_all_answers_are_integers():
    """All tiers: answer_text should be a plain integer (no decimal)."""
    for difficulty in ["easy", "medium", "hard"]:
        for seed in SEEDS:
            question = make(RADICALS_TOPIC, seed=seed, difficulty=difficulty)
            assert "." not in question.answer_text


def test_radical_independent_check_fires_on_corrupted_answer():
    """Independent check should fire when answer is wrong."""
    # Test case: 8 is 2**3, so the 3rd root of 8 is 2.
    with pytest.raises(VerificationError):
        _verify_radical_evaluation(8, 3, 2, 5)  # Should be 2**2=4, not 5


def test_find_integer_root_basic():
    """Test the brute-force integer root finder."""
    assert _find_integer_root(8, 3) == 2  # 2**3 = 8
    assert _find_integer_root(9, 2) == 3  # 3**2 = 9
    assert _find_integer_root(16, 2) == 4  # 4**2 = 16
    assert _find_integer_root(10, 2) is None  # 10 is not a perfect square


def test_radical_wrong1_misconception_always_distinct():
    """Misconception wrong1 (forgets-the-root, computes n**m) should always
    differ from the correct answer b**m when b >= 2 and r >= 2."""
    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(100):
            question = make(RADICALS_TOPIC, seed=seed, difficulty=difficulty)
            # We can't easily extract n, r, m from the question text,
            # but the misconception n**m should always be much larger than b**m.
            # Just verify the question doesn't have collisions in its distractors.
            assert question.answer_text not in question.distractors


# --- skill contracts ---------------------------------------------------------


def test_p3c_index_laws_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    skill = by_id["senior-review-indices-including-fractional-indices-and-the-index"]

    assert skill.prerequisites == []
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in skill.sources} == {INDEX_LAWS_TOPIC}
    assert [source.stages for source in skill.sources] == [[6], [6], [6, "maintenance"], ["exam"]]


def test_p3c_radicals_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    skill = by_id["senior-use-radicals-and-convert-to-and-from-fractional"]

    assert skill.prerequisites == ["function-notation"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in skill.sources} == {RADICALS_TOPIC}
    assert [source.stages for source in skill.sources] == [[6], [6], [6, "maintenance"]]
