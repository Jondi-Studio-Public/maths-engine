"""Pure antidifferentiation with an initial condition.

senior-determine-f-left-x-right-given-f-x (ACMMM122, topic
p10a_antiderivative_given_fprime, answer: expression). See the generator
module docstring for the easy-tier sign-error-on-constant applicability
finding.
"""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

TOPIC = "p10a_antiderivative_given_fprime"
SKILL = "senior-determine-f-left-x-right-given-f-x"
GRADE_EXPRESSION = GRADERS["expression"]


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractor_counts_match_tier_design(tier):
    """easy never offers sign-error-on-constant (see module docstring); medium
    offers exactly 3 (forgot-constant, power-rule-no-divide, sign-error);
    hard's 4th candidate (ignored-negative-base-sign) is capped out by
    pick_distractors's n=3 -- exercised directly in the oracle test instead."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        if tier == "easy":
            assert len(q.distractors) == 2, f"seed={seed}: easy expected 2 distractors, got {q.distractors}"
        else:
            assert len(q.distractors) == 3, f"seed={seed}: {tier} expected 3 distractors, got {q.distractors}"


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors_and_wrong_value(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = GRADE_EXPRESSION(q, q.answer_text)
        assert result.correct, f"seed={seed}/{tier}: canonical not accepted"
        for d in q.distractors:
            result = GRADE_EXPRESSION(q, d)
            assert not result.correct, f"seed={seed}/{tier}: distractor {d!r} accepted"
        wrong = q.answer_text + " + 999"
        result = GRADE_EXPRESSION(q, wrong)
        assert not result.correct, f"seed={seed}/{tier}: deliberately wrong {wrong!r} accepted"


# --- Answer kind -------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_a_polynomial_expression_in_x(tier):
    import sympy

    x = sympy.symbols("x")
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        expr = sympy.sympify(q.answer_text.replace("^", "**"))
        assert expr.free_symbols <= {x}, f"seed={seed}/{tier}: answer has unexpected symbols: {expr}"
        assert sympy.Poly(expr, x).degree() in (1, 2, 3), f"seed={seed}/{tier}: unexpected degree"


# --- Formatted alike ----------------------------------------------------


_TRAILING_CONST_RE = re.compile(r"(^|[+\-]\s*)\d+(/\d+)?\s*$")


def _has_constant_term(expr_text: str) -> bool:
    return bool(_TRAILING_CONST_RE.search(expr_text.strip()))


@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_all_polynomial_in_x(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for opt in [q.answer_text] + q.distractors:
            assert "x" in opt, f"seed={seed}/{tier}: option {opt!r} has no x term"


@pytest.mark.parametrize("tier", TIERS)
def test_forgot_constant_is_the_shape_outlier_never_the_answer(tier):
    """`forgot-constant` (G(x), C = 0) is BUILT to lack a trailing constant term, and a 2000-seed sweep found it
    lacks one on every single draw at every tier -- so it genuinely IS visually distinguishable by shape from
    an option that carries one. What matters for validity is which option that marks as the outlier: the TRUE
    ANSWER always carries a constant term (asserted directly here, every seed) and is therefore never itself
    the shape outlier -- a shape-based guess can only point a learner AWAY from the correct answer (toward
    forgot-constant), never toward it."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert _has_constant_term(q.answer_text), (
            f"seed={seed}/{tier}: true answer {q.answer_text!r} unexpectedly lacks a constant term"
        )
        lacking = [d for d in q.distractors if not _has_constant_term(d)]
        assert len(lacking) >= 1, f"seed={seed}/{tier}: forgot-constant distractor missing from options"


# --- Distinct rendered questions -----------------------------------------


# --- Independent-check tripwire -------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []
    assert len(skill.sources) == 4  # easy, medium, hard and exam
    for source in skill.sources[:3]:  # the exam source is covered by this skill's tests/test_exd_x_* file
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
