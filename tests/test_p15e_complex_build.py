"""Focused contract tests for: ucv1-contour-integral-residue (answer: expression) and ucv1-cr-equations-check
(answer: mcq)."""

import pytest

from mathgen.grading.typed import GRADERS, grade_expression, mcq_options
from mathgen.skills import load_skills
from mathgen import make

CONTOUR_ID = "ucv1-contour-integral-residue"
CR_ID = "ucv1-cr-equations-check"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


CONTOUR_SKILL, CONTOUR_SOURCES = _skill_sources(CONTOUR_ID)
CR_SKILL, CR_SOURCES = _skill_sources(CR_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-contour-integral-residue
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_contour_answer_never_needs_literal_i(difficulty):
    # Every design keeps the enclosed pole's residue purely imaginary, so
    # 2*pi*i*Res is always real: the graded KEY never carries "I". A
    # distractor MAY still carry "I" (e.g. "forgets-2-pi-i" offers the bare,
    # purely-imaginary residue itself) -- that is a legitimate wrong answer,
    # not a defect, and is already proven to clear the grader as a reject by
    # test_contour_real_grader_accepts_canonical_and_rejects_every_distractor.
    for seed in SEEDS:
        q = _make(CONTOUR_SOURCES, difficulty, seed)
        assert "I" not in q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_contour_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(CONTOUR_SOURCES, difficulty, seed)
        wrong = f"({q.answer_text}) + 1000"
        assert not grade_expression(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


def test_contour_exam_tier_requires_solving_the_quadratic():
    """The exam tier must not print a solved pole; it states the quadratic and the geometric half-plane condition
    only, so the learner (and the oracle) must solve for both roots and keep the one with Im(z) > 0."""
    for seed in SEEDS:
        q = _make(CONTOUR_SOURCES, "exam", seed)
        assert "Candidate 1" not in q.text and "Candidate 2" not in q.text
        assert "upper half-plane" in q.text
        assert "quadratic" in q.text.lower() or "= 0" in q.text  # the poles are to be solved, not read off


def test_contour_skill_loads_through_api_with_correct_properties():
    assert CONTOUR_SKILL.domain == "complex-analysis"
    assert CONTOUR_SKILL.status == "active"
    assert CONTOUR_SKILL.placement_eligible is False
    assert CONTOUR_SKILL.answer == "expression"
    assert CONTOUR_SKILL.prerequisites == ["ucv1-residue-simple-pole"]
    assert {src.difficulty for src in CONTOUR_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = CONTOUR_SOURCES["exam"]
    assert exam_src.stages == ["exam"]


# ==========================================================================
# ucv1-cr-equations-check
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_cr_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(CR_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3  # four-way MCQ, one correct + 3 distractors


@pytest.mark.parametrize("difficulty", TIERS)
def test_cr_real_grader_accepts_correct_option_index(difficulty):
    grader = GRADERS["mcq"]
    for seed in SEEDS:
        q = _make(CR_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grader(q, str(correct_index)).correct
        for i in range(len(options)):
            if i != correct_index:
                assert not grader(q, str(i)).correct


def test_cr_all_four_classes_appear_over_many_seeds():
    for difficulty in TIERS:
        labels = {_make(CR_SOURCES, difficulty, seed).answer_text for seed in SWEEP_SEEDS}
        assert labels == {
            "Both Cauchy-Riemann equations hold",
            "Only u_x = v_y fails",
            "Only u_y = -v_x fails",
            "Both Cauchy-Riemann equations fail",
        }, f"{difficulty}: only saw {labels}"


def test_cr_skill_loads_through_api_with_correct_properties():
    assert CR_SKILL.domain == "complex-analysis"
    assert CR_SKILL.status == "active"
    assert CR_SKILL.placement_eligible is False
    assert CR_SKILL.answer == "mcq"
    assert CR_SKILL.prerequisites == ["differentiate-polynomial"]
    assert {src.difficulty for src in CR_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = CR_SOURCES["exam"]
    assert exam_src.stages == ["exam"]
