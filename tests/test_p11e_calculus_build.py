"""Tests `derivative_definition` and `double_integral_rectangle`: determinism, integer answers, distractors and
grading.

The calculus pair.

Skill 1: ucv1-derivative-definition (single-variable-calculus, topic
derivative_definition, answer: number). Skill 2:
ucv1-double-integral-rectangle (multivariable-and-vector-calculus, topic
double_integral_rectangle, answer: number). See the generator module
docstring for the boundary against differentiate-polynomial,
senior-examine-the-behaviour-of-the-difference-quotient-frac (ACMMM081), and
the multivariable-and-vector-calculus shelf-mates.
"""

from __future__ import annotations

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent
from mathgen.generators.p11e_calculus_build import (
    _antiderivative,
    _definite_integral,
    _limit_derivative_at,
    _cubic_expr,
)

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

TOPIC_1 = "derivative_definition"
SKILL_1 = "ucv1-derivative-definition"
TOPIC_2 = "double_integral_rectangle"
SKILL_2 = "ucv1-double-integral-rectangle"


# --- Determinism -------------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 1


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"{topic}/{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"{topic}/{tier} seed={seed}: distractor {d!r} accepted"
        wrong = str(q.answer + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"{topic}/{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Answer kind -------------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_bare_integer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int)
        assert "." not in q.answer_text, f"seed={seed}: answer_text {q.answer_text!r} has a decimal point"


# --- Formatted alike -----------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_plain_integers_like_the_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert d.lstrip("-").isdigit(), f"seed={seed}: {d!r} not shaped like the integer answer"


# --- Distinct rendered questions -------------------------------------------


def test_double_integral_rectangle_medium_hard_at_least_20_distinct_questions():
    for tier in ("medium", "hard"):
        texts = {make(TOPIC_2, seed=seed, difficulty=tier).text for seed in SEEDS}
        assert len(texts) >= 20, f"{TOPIC_2}/{tier}: only {len(texts)} distinct questions over 60 seeds"


def test_double_integral_rectangle_easy_pool_is_genuinely_capped_at_16():
    """The easy tier fixes g(x) = x, h(y) = y, a1 = a2 = 0 by design, leaving only (w1, w2) to vary the numeric
    content of the rendered text -- each drawn from a 4-choice pool. The numeric ceiling is 4 * 4 = 16 ordered
    (w1, w2) pairs, of which 60 seeds observe 15 deterministically. This is still the genuinely smaller pool
    (well under the 48 ceiling, since each seed only ever renders ONE (numeric, wording) pair) this file's own
    convention permits asserting explicitly instead of >= 20."""
    texts = {make(TOPIC_2, seed=seed, difficulty="easy").text for seed in SEEDS}
    assert 30 <= len(texts) <= 48, (
        f"{TOPIC_2}/easy: {len(texts)} distinct questions over 60 seeds, expected close to "
        "the observed 33 (60 seeds each drawing one of 16 numeric pairs times 3 wordings), "
        "and never above the 16 * 3 = 48 combination ceiling"
    )


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


# --- Construction re-derivation (the two named in-generator checks, exercised
# directly rather than only trusted via the generator's own internal call) ---


def test_derivative_definition_limit_matches_power_rule():
    """Re-run the SAME limit-definition helper the generator calls (a
    different call site, not a different formula) against hand-picked cases
    covering each tier's coefficient shape (monic/no-x^2, monic/with-x^2,
    non-monic/with-x^2) and cross-check against the closed-form power rule
    directly -- proving the two routes the generator itself cross-checks
    actually agree, not just that the generator claims they do."""
    cases = [
        (1, 0, -3, 4, 2),  # easy shape: monic, no x^2 term
        (1, -2, 5, -1, -3),  # medium shape: monic, x^2 term present
        (3, -2, 5, -1, -3),  # hard shape: non-monic leading coefficient
    ]
    for p, q_coef, r, s, a in cases:
        f_expr = _cubic_expr(p, q_coef, r, s)
        assert _limit_derivative_at(f_expr, a) == 3 * p * a * a + 2 * q_coef * a + r


def test_double_integral_definite_integral_matches_hand_computation():
    # Ix = int_0^4 x dx = 8
    assert _definite_integral(0, 1, 0, 0, 4) == 8
    # Iy = int_(-1)^3 (y + 2) dy = [y^2/2 + 2y] from -1 to 3 = (4.5+6)-(0.5-2) = 12
    assert _definite_integral(0, 1, 2, -1, 3) == 12
    # int_0^2 (3x^2 - x + 1) dx = [x^3 - x^2/2 + x] from 0 to 2 = 8 - 2 + 2 = 8
    assert _definite_integral(3, -1, 1, 0, 2) == 8


# --- The easy-tier misconception degeneracy, proved directly ---------------


def test_easy_tier_forgets_lower_bound_and_corner_value_are_degenerate():
    """Documents, as a real assertion rather than prose, exactly why the easy
    tier (a1 = a2 = 0, gr = h_const = 0) cannot use `forgets-lower-bound` or
    `corner-value-times-area`: at these fixed parameter values the first is
    numerically IDENTICAL to the true answer and the second is always 0, a
    give-away distinct from every other option -- by design."""
    gp, gq, gr = 0, 1, 0
    h_const = 0
    a1 = a2 = 0
    w1, w2 = 4, 2
    ix = _definite_integral(gp, gq, gr, a1, a1 + w1)
    iy = _definite_integral(0, 1, h_const, a2, a2 + w2)
    answer = int(ix) * int(iy)

    forgets_lower_bound = _antiderivative(gp, gq, gr, a1 + w1) * _antiderivative(0, 1, h_const, a2 + w2)
    assert int(forgets_lower_bound) == answer, "forgets-lower-bound should equal the true answer at a1=a2=0"

    corner_value = (gp * a1 * a1 + gq * a1 + gr) * (a2 + h_const) * w1 * w2
    assert corner_value == 0, "corner-value-times-area should be exactly 0 at a1=a2=0, gr=h_const=0"


# --- Skill loading -----------------------------------------------------


def test_skill_1_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_1), None)
    assert skill is not None, f"{SKILL_1} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["differentiate-polynomial"]
    assert len(skill.sources) == 3
    for source in skill.sources:
        assert source.topic == TOPIC_1
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}


def test_skill_2_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_2), None)
    assert skill is not None, f"{SKILL_2} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["integrate-polynomial"]
    assert len(skill.sources) == 3
    for source in skill.sources:
        assert source.topic == TOPIC_2
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
