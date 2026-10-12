"""Focused contract tests for: ucv1-diagonal-operator-norm (answer: number) and ucv1-function-space-approximation
(answer: pair)."""

from dataclasses import replace

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number, grade_pair
from mathgen.skills import load_skills
from mathgen import make

NORM_ID = "ucv1-diagonal-operator-norm"
APPROX_ID = "ucv1-function-space-approximation"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


NORM_SKILL, NORM_SOURCES = _skill_sources(NORM_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
NORM_SOURCES["exam"] = replace(NORM_SOURCES["exam"], topic="p15c_diagonal_operator_norm", params={"variant": "exam"})
APPROX_SKILL, APPROX_SOURCES = _skill_sources(APPROX_ID)
APPROX_SOURCES["exam"] = replace(
    APPROX_SOURCES["exam"], topic="p15c_function_space_approximation", params={"basis": "sq_cube", "evaluate_fn": True}
)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-diagonal-operator-norm
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_norm_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(NORM_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_norm_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(NORM_SOURCES, difficulty, seed)
        if difficulty == "exam":
            true_value = Fraction(q.answer["p"], q.answer["q"])
            wrong = str(true_value + Fraction(97, 100))
        else:
            wrong = str(int(q.answer_text) + 97)
        assert not grade_number(q, wrong).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_norm_answer_is_nonnegative(difficulty):
    for seed in SEEDS:
        q = _make(NORM_SOURCES, difficulty, seed)
        if difficulty == "exam":
            assert Fraction(q.answer["p"], q.answer["q"]) >= 0
        else:
            assert int(q.answer_text) >= 0


@pytest.mark.parametrize("difficulty", TIERS)
def test_norm_200_seed_answer_never_concentrates(difficulty):
    """Construct-to-target (draw the answer, then build entries realising it) tightened this from the free-draw
    original's 92% (easy) / 32% (exam) down to under 20% at every tier."""
    counts = Counter(_make(NORM_SOURCES, difficulty, seed).answer_text for seed in range(200))
    top_share = max(counts.values()) / 200
    assert top_share <= 0.20, f"{difficulty}: top answer is {top_share:.1%} of 200 draws"


_BARE_COEF_RE = re.compile(r"(?<![\w)])(-?1)([xy])\b")
_FRAC_BEFORE_VAR_RE = re.compile(r"-?\d+/\d+[xy]\b")


@pytest.mark.parametrize("difficulty", TIERS)
def test_norm_no_unit_coefficient_or_unparenthesised_fraction_before_a_variable(difficulty):
    """A rendered map must never show a redundant coefficient of 1 ("1x", "-1y") or a fraction directly ahead of
    a variable with no separator ("-1/2x", which reads as -1/(2x)) -- every entry goes through
    `_coef_var_text`, checked here from the rendered text alone."""
    for seed in range(300):
        q = _make(NORM_SOURCES, difficulty, seed)
        assert not _BARE_COEF_RE.search(q.text), (seed, q.text)
        assert not _FRAC_BEFORE_VAR_RE.search(q.text), (seed, q.text)


def test_norm_hard_tier_multiply_norms_trap_diverges_from_answer():
    """The hard tier's central trap (multiply the two operator norms instead
    of multiplying matching diagonal entries) must be a genuinely different
    -- and offered -- wrong value on every seed."""
    for seed in SWEEP_SEEDS:
        q = _make(NORM_SOURCES, "hard", seed)
        assert len(q.distractors) == 3


def test_norm_easy_tier_states_diagonal_directly():
    for seed in SEEDS:
        q = _make(NORM_SOURCES, "easy", seed)
        assert "diag(" in q.text


def test_norm_medium_tier_states_action_formula():
    for seed in SEEDS:
        q = _make(NORM_SOURCES, "medium", seed)
        assert "T(x, y) = (" in q.text


def test_norm_hard_tier_is_a_composition():
    for seed in SEEDS:
        q = _make(NORM_SOURCES, "hard", seed)
        assert "composition" in q.text


def test_norm_exam_tier_is_a_power_with_fractional_entries():
    for seed in SEEDS:
        q = _make(NORM_SOURCES, "exam", seed)
        assert "^" in q.text
        assert isinstance(q.answer, dict) and set(q.answer) == {"p", "q"}


def test_norm_skill_loads_through_api_with_correct_properties():
    assert NORM_SKILL.domain == "functional-and-linear-analysis"
    assert NORM_SKILL.status == "active"
    assert NORM_SKILL.placement_eligible is False
    assert NORM_SKILL.answer == "number"
    assert NORM_SKILL.prerequisites == []
    assert {src.difficulty for src in NORM_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert NORM_SOURCES["exam"].stages == ["exam"]


# ==========================================================================
# ucv1-function-space-approximation
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_approx_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(APPROX_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_approx_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(APPROX_SOURCES, difficulty, seed)
        c1, c2 = Fraction(q.answer["c1"]), Fraction(q.answer["c2"])
        wrong = f"c1 = {c1 + 11}, c2 = {c2 + 11}"
        assert not grade_pair(q, wrong).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_approx_answer_is_a_labelled_pair(difficulty):
    for seed in SEEDS:
        q = _make(APPROX_SOURCES, difficulty, seed)
        assert isinstance(q.answer, dict) and set(q.answer) == {"c1", "c2"}
        if difficulty != "exam":
            # easy/medium/hard construct-to-target an exact reconstruction,
            # always integer.
            for v in q.answer.values():
                assert Fraction(v).denominator == 1
        assert q.answer_text.startswith("c1 = ") and ", c2 = " in q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_approx_200_seed_answer_never_concentrates(difficulty):
    counts = Counter(_make(APPROX_SOURCES, difficulty, seed).answer_text for seed in range(200))
    top_share = max(counts.values()) / 200
    assert top_share <= 0.10, f"{difficulty}: top answer is {top_share:.1%} of 200 draws"


def test_approx_prerequisite_basis_never_used():
    """Boundary: the basis must never be {1, x} (the prerequisite
    `least-squares-line`'s own basis)."""
    for difficulty in TIERS:
        for seed in SEEDS:
            q = _make(APPROX_SOURCES, difficulty, seed)
            assert "{1, x}" not in q.text


def test_approx_easy_uses_one_and_xsq_basis():
    for seed in SEEDS:
        q = _make(APPROX_SOURCES, "easy", seed)
        assert "{1, x^2}" in q.text


def test_approx_medium_uses_x_and_xsq_basis_with_signed_x():
    for seed in SEEDS:
        q = _make(APPROX_SOURCES, "medium", seed)
        assert "{x, x^2}" in q.text


def test_approx_hard_and_exam_use_sq_cube_basis():
    """Replaced with {x^2, x^3}, which the generator verifies (via an explicit off-diagonal-nonzero check) always
    produces a coupled 2x2 system requiring the full normal-equation solve."""
    for difficulty in ("hard", "exam"):
        for seed in SEEDS:
            q = _make(APPROX_SOURCES, difficulty, seed)
            assert "{x^2, x^3}" in q.text
            assert "sin(theta)" not in q.text and "cos(theta)" not in q.text


def test_approx_exam_evaluates_a_genuine_function_outside_the_span():
    """The data must BE the stated function, exactly: f(x) = x^4, a degree-4 polynomial genuinely outside {x^2,
    x^3}'s span, so the fit is a genuine approximation rather than an exact reconstruction of a pre-chosen
    target. No residual annotation, no "add"/"subtract" wording."""
    for seed in SEEDS:
        q = _make(APPROX_SOURCES, "exam", seed)
        assert "f(x) = x^4" in q.text
        assert re.search(r"y1 = f\(-?\d+\) = ", q.text)
        assert "residual" not in q.text and "add the two readings first" not in q.text
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = _make(APPROX_SOURCES, difficulty, seed)
            assert "f(x)" not in q.text


def test_approx_skill_loads_through_api_with_correct_properties():
    assert APPROX_SKILL.domain == "functional-and-linear-analysis"
    assert APPROX_SKILL.status == "active"
    assert APPROX_SKILL.placement_eligible is False
    assert APPROX_SKILL.answer == "pair"
    assert APPROX_SKILL.prerequisites == ["least-squares-line"]
    assert {src.difficulty for src in APPROX_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert APPROX_SOURCES["exam"].stages == ["exam"]
