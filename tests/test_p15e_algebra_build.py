"""Focused contract tests for: ucv1-quotient-ring-operation (answer: expression)."""

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "ucv1-quotient-ring-operation"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources(SKILL_ID)


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_double_star_in_any_learner_facing_field(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert "**" not in q.answer_text
        assert "**" not in q.text
        for d in q.distractors:
            assert "**" not in d


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        wrong = f"({q.answer_text}) + 1000"
        assert not grade_expression(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


def test_easy_tier_is_addition_not_multiplication():
    for seed in SEEDS:
        q = _make("easy", seed)
        assert "a(x) + b(x)" in q.text or "add a(x)" in q.text


def test_exam_tier_states_prime_and_residue_language():
    for seed in SEEDS:
        q = _make("exam", seed)
        assert "Z_" in q.text
        assert "residue" in q.text or "residues" in q.text


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.domain == "abstract-algebra"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "expression"
    assert SKILL.prerequisites == ["integer-polynomial-quotient-remainder"]
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = SOURCES["exam"]
    assert exam_src.stages == ["exam"]
