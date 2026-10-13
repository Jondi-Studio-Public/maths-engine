"""Focused contract tests for: ucv1-fourier-transform-gaussian and ucv1-wave-equation-separation (both answer:
expression)."""

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from tests.exd_x_r2h_support import legacy_exam_sources
from mathgen import make

FT_ID = "ucv1-fourier-transform-gaussian"
WAVE_ID = "ucv1-wave-equation-separation"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, legacy_exam_sources(skill.id, {src.difficulty: src for src in skill.sources})


FT_SKILL, FT_SOURCES = _skill_sources(FT_ID)
WAVE_SKILL, WAVE_SOURCES = _skill_sources(WAVE_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-fourier-transform-gaussian
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_ft_prompt_and_answer_use_exp_never_e_caret(difficulty):
    for seed in SEEDS:
        q = _make(FT_SOURCES, difficulty, seed)
        assert "e^" not in q.text
        assert "e^" not in q.answer_text
        assert "exp(" in q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_ft_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(FT_SOURCES, difficulty, seed)
        wrong = f"({q.answer_text}) + 1000"
        assert not grade_expression(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


def test_ft_hard_tier_states_unitary_convention():
    for seed in SEEDS:
        q = _make(FT_SOURCES, "hard", seed)
        assert "UNITARY" in q.text or "unitary" in q.text


def test_ft_exam_tier_uses_frequency_variable_xi():
    """The exam tier names the transform variable `xi`, never `f` -- `f` is already the FUNCTION's own name (f(x)
    =...) in the same sentence, so reusing it for the frequency variable is a name collision."""
    for seed in SEEDS:
        q = _make(FT_SOURCES, "exam", seed)
        assert "F(xi)" in q.text
        assert "F(f)" not in q.text
        assert "xi" in q.answer_text


def test_ft_skill_loads_through_api_with_correct_properties():
    assert FT_SKILL.domain == "fourier-analysis-and-pdes"
    assert FT_SKILL.status == "active"
    assert FT_SKILL.placement_eligible is False
    assert FT_SKILL.answer == "expression"
    assert FT_SKILL.prerequisites == ["fourier-coefficient-simple"]
    assert {src.difficulty for src in FT_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = FT_SOURCES["exam"]
    assert exam_src.stages == ["exam"]


# ==========================================================================
# ucv1-wave-equation-separation
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_wave_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(WAVE_SOURCES, difficulty, seed)
        wrong = f"({q.answer_text}) + 1000"
        assert not grade_expression(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


def test_wave_easy_uses_dirichlet_and_rest_only():
    for seed in SEEDS:
        q = _make(WAVE_SOURCES, "easy", seed)
        assert "u(0,t) = u(pi,t) = 0" in q.text
        assert "starts from rest" in q.text
        assert "cos" in q.answer_text


def test_wave_hard_uses_neumann_and_zero_displacement_only():
    for seed in SEEDS:
        q = _make(WAVE_SOURCES, "hard", seed)
        assert "Neumann" in q.text or "free ends" in q.text
        assert "zero displacement" in q.text
        assert "sin" in q.answer_text


def test_wave_exam_states_mixed_boundary_with_no_candidates():
    """The exam tier must not print two candidate mode strings, one of which is the verbatim key. It instead
    states a MIXED boundary condition whose eigenfunction the learner must construct."""
    for seed in SEEDS:
        q = _make(WAVE_SOURCES, "exam", seed)
        assert "Candidate 1" not in q.text and "Candidate 2" not in q.text
        assert "mixed" in q.text.lower()
        assert "u(0,t) = 0" in q.text
        assert q.answer_text not in q.text


def test_wave_skill_loads_through_api_with_correct_properties():
    assert WAVE_SKILL.domain == "fourier-analysis-and-pdes"
    assert WAVE_SKILL.status == "active"
    assert WAVE_SKILL.placement_eligible is False
    assert WAVE_SKILL.answer == "expression"
    assert WAVE_SKILL.prerequisites == ["classify-constant-coefficient-pde"]
    assert {src.difficulty for src in WAVE_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = WAVE_SOURCES["exam"]
    assert exam_src.stages == ["exam"]
