"""Tests `p19f_derivative_linearity` (ACMMM090): linearity of the derivative, answered as an expression."""

import pytest

from mathgen.grading.typed import GRADERS, grade_expression
from mathgen.skills import load_skills
from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p19f_derivative_linearity"
SKILL = "senior-recognise-and-use-linearity-properties-of-the-derivative"


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_and_rejects(tier):
    assert GRADERS["expression"] is grade_expression
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_expression(q, q.answer_text)
        assert result.correct, f"seed={seed}: canonical {q.answer_text!r} not accepted"
        for d in q.distractors:
            result = grade_expression(q, d)
            assert not result.correct, f"seed={seed}: distractor {d!r} accepted"
        wrong = q.answer_text + " + 7"
        result = grade_expression(q, wrong)
        assert not result.correct, f"seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Answer kind -----------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_an_expression_string(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, str)
        assert isinstance(q.answer_text, str) and q.answer_text


# --- Distinct rendered questions -------------------------------------------


# --- Answer/distractor shape parity -----------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_formatted_like_the_answer(tier):
    """Every option (answer + distractors) is a polynomial-in-x string built
    by the same `_fmt_expr_terms`/`join_terms` route -- no option should be
    identifiable by carrying a decimal point, a fraction slash or a `+ C`
    the others lack."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        options = [q.answer_text, *q.distractors]
        for opt in options:
            assert "." not in opt, f"seed={seed}: {opt!r} has a decimal point"
            assert "/" not in opt, f"seed={seed}: {opt!r} has a fraction slash"
            assert "C" not in opt, f"seed={seed}: {opt!r} carries a constant-of-integration C"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    from mathgen.core import VerificationError, check_equivalent

    with pytest.raises(VerificationError):
        check_equivalent("x + 1", "x + 2", "should fail")


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["differentiate-polynomial"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
