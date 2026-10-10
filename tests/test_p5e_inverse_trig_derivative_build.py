"""Derivative of the inverse trigonometric functions."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from sympy import sqrt, symbols, asin, acos, atan, diff, simplify, sec, sin

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

TOPIC = "find_and_use_the_derivative_of_the_inverse"


# --- Determinism & distractor hygiene ----------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_from_answer_sympy_level(difficulty):
    """Distractors must be symbolically distinct from the answer."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        from sympy import sympify

        answer_expr = sympify(question.answer.replace("^", "**"))
        for distractor in question.distractors:
            # Convert ^ back to ** for sympy parsing
            distractor_expr = sympify(distractor.replace("^", "**"))
            diff_result = simplify(answer_expr - distractor_expr)
            assert diff_result != 0, f"Seed {seed}/{difficulty}: distractor {distractor} equals answer symbolically"


# --- Real grader testing -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_rejects_distractors(difficulty):
    """Real grader must accept answer and reject distractors."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        result = grade_expression(question, question.answer_text)
        assert result.correct, f"Seed {seed}/{difficulty}: canonical not accepted"
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, f"Seed {seed}/{difficulty}: distractor {distractor} was accepted"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_deliberately_wrong_answer(difficulty):
    """Real grader must reject a deliberately wrong answer."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        # For expression answers, add 1 to make it wrong
        wrong = question.answer_text + "+1"
        result = grade_expression(question, wrong)
        assert not result.correct, f"Seed {seed}/{difficulty}: wrong answer '{wrong}' was accepted"


def test_branch_coverage_easy():
    """All three function kinds (asin, acos, atan) exercised at least once in easy tier across 40 seeds."""
    kinds_seen = set()
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="easy")
        # Determine kind from the answer text
        if "sqrt(1 -" in question.answer_text:
            if question.answer_text.startswith("-"):
                kinds_seen.add("acos")
            else:
                kinds_seen.add("asin")
        elif "x^2 + 1)" in question.answer_text:
            kinds_seen.add("atan")
    assert len(kinds_seen) == 3, f"Not all kinds covered in easy: {kinds_seen}"


def test_branch_coverage_medium():
    """All three function kinds exercised at least once in medium tier across 40 seeds."""
    kinds_seen = set()
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="medium")
        # Determine kind from the answer text
        if "sqrt(1 -" in question.answer_text:
            if question.answer_text.startswith("-"):
                kinds_seen.add("acos")
            else:
                kinds_seen.add("asin")
        elif "^2 + 1)" in question.answer_text:
            kinds_seen.add("atan")
    assert len(kinds_seen) == 3, f"Not all kinds covered in medium: {kinds_seen}"


def test_branch_coverage_hard():
    """All three function kinds exercised at least once in hard tier across 40 seeds."""
    kinds_seen = set()
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="hard")
        # Determine kind from the answer text
        if "sqrt(1 -" in question.answer_text:
            if question.answer_text.startswith("-"):
                kinds_seen.add("acos")
            else:
                kinds_seen.add("asin")
        elif "^2 + 1)" in question.answer_text:
            kinds_seen.add("atan")
    assert len(kinds_seen) == 3, f"Not all kinds covered in hard: {kinds_seen}"


def test_no_arcsin_arccos_arctan_in_answer_or_distractors():
    """Assert 'arcsin(', 'arccos(', 'arctan(' never appear in answer_text or distractors."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            for pattern in ["arcsin(", "arccos(", "arctan("]:
                assert pattern not in question.answer_text, f"Seed {seed}/{difficulty}: '{pattern}' in answer_text"
                for distractor in question.distractors:
                    assert pattern not in distractor, (
                        f"Seed {seed}/{difficulty}: '{pattern}' in distractor {distractor}"
                    )


# --- Skill loading -----


def test_skill_loads_with_correct_properties():
    """Inverse trig derivative skill must load with correct properties."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-find-and-use-the-derivative-of-the-inverse"), None)
    assert skill is not None, "Inverse trig derivative skill not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible == False
    assert "differentiate-polynomial" in skill.prerequisites


def test_skill_sources_cover_correct_stages():
    """Skill must have three sources covering stages [0,1,2], [3,4], [5,6,maintenance]."""
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-find-and-use-the-derivative-of-the-inverse"), None)
    assert skill is not None
    assert len(skill.sources) == 3
    # Check that all sources have the correct topic
    for source in skill.sources:
        assert source.topic == "find_and_use_the_derivative_of_the_inverse"
    # Check stages for each difficulty
    assert set(skill.sources[0].stages) == {0, 1, 2}, "easy stages"
    assert set(skill.sources[1].stages) == {3, 4}, "medium stages"
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}, "hard stages"
