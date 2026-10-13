"""Focused contract tests for: ucv1-residue-simple-pole and ucv1-heat-equation-mode (both answer: expression)."""

import dataclasses
import re

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make

RESIDUE_ID = "ucv1-residue-simple-pole"
HEAT_ID = "ucv1-heat-equation-mode"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]


_SINGLE_ANSWER_EXAM = {"ucv1-heat-equation-mode": "p13e_heat_equation_mode"}


def _by_difficulty(skill):
    """Sources by difficulty. The exam tier of a skill listed above now serves a multi-part chain; these tests
    cover the single-answer generator it replaced, so that tier is pointed back at it."""
    sources = {src.difficulty: src for src in skill.sources}
    if skill.id in _SINGLE_ANSWER_EXAM:
        sources["exam"] = dataclasses.replace(sources["exam"], topic=_SINGLE_ANSWER_EXAM[skill.id], params={})
    return sources


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, _by_difficulty(skill)


RESIDUE_SKILL, RESIDUE_SOURCES = _skill_sources(RESIDUE_ID)
HEAT_SKILL, HEAT_SOURCES = _skill_sources(HEAT_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-residue-simple-pole
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_residue_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(RESIDUE_SOURCES, difficulty, seed)
        wrong = f"({q.answer_text}) + 1000"
        assert not grade_expression(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


def test_residue_exam_tier_states_two_candidates():
    # Not every seed's answer_text carries an "I" token: when the drawn
    # numerator root r happens to equal the true pole's real part p, the
    # residue (w_true - r)/(2*I*q) collapses to the purely real 1/2 -- a
    # legitimate value, not a defect (verified by hand: r=p seed 11).
    for seed in SEEDS:
        q = _make(RESIDUE_SOURCES, "exam", seed)
        assert "Candidate 1" in q.text and "Candidate 2" in q.text


def test_residue_skill_loads_through_api_with_correct_properties():
    assert RESIDUE_SKILL.domain == "complex-analysis"
    assert RESIDUE_SKILL.status == "active"
    assert RESIDUE_SKILL.placement_eligible is False
    assert RESIDUE_SKILL.answer == "expression"
    assert RESIDUE_SKILL.prerequisites == ["multiply-complex-numbers"]
    assert {src.difficulty for src in RESIDUE_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = RESIDUE_SOURCES["exam"]
    assert exam_src.stages == ["exam"]


# ==========================================================================
# ucv1-heat-equation-mode
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_heat_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(HEAT_SOURCES, difficulty, seed)
        wrong = f"({q.answer_text}) + 1000"
        assert not grade_expression(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


def test_heat_easy_uses_pi_length_and_unit_coefficient():
    for seed in SEEDS:
        q = _make(HEAT_SOURCES, "easy", seed)
        assert "B = 1" in q.text
        assert "sin" in q.answer_text and "cos" not in q.answer_text


def test_heat_hard_is_neumann_family():
    for seed in SEEDS:
        q = _make(HEAT_SOURCES, "hard", seed)
        assert "insulated" in q.text
        assert "cos" in q.answer_text and "sin" not in q.answer_text


def test_heat_exam_states_two_candidates_and_boundary_words():
    for seed in SEEDS:
        q = _make(HEAT_SOURCES, "exam", seed)
        assert "Candidate 1" in q.text and "Candidate 2" in q.text
        assert ("fixed ends" in q.text) or ("insulated ends" in q.text)


def test_heat_skill_loads_through_api_with_correct_properties():
    assert HEAT_SKILL.domain == "fourier-analysis-and-pdes"
    assert HEAT_SKILL.status == "active"
    assert HEAT_SKILL.placement_eligible is False
    assert HEAT_SKILL.answer == "expression"
    assert HEAT_SKILL.prerequisites == ["fourier-coefficient-simple"]
    assert {src.difficulty for src in HEAT_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = HEAT_SOURCES["exam"]
    assert exam_src.stages == ["exam"]
