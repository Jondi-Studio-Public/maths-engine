"""Focused contract tests for: ucv1-gram-schmidt-2d and ucv1-hessian-2x2 (both ``answer: ordered_list``)."""

from dataclasses import replace
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make

GRAM_ID = "ucv1-gram-schmidt-2d"
HESS_ID = "ucv1-hessian-2x2"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_FRACTION_ITEM_RE = r"-?\d+(?:/\d+)?"
_GRAM_ANSWER_RE = re.compile(rf"{_FRACTION_ITEM_RE}(?:, {_FRACTION_ITEM_RE}){{3}}")
_HESS_ANSWER_RE = re.compile(r"-?\d+(?:, -?\d+){3}")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


GRAM_SKILL, GRAM_SOURCES = _skill_sources(GRAM_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
GRAM_SOURCES["exam"] = replace(GRAM_SOURCES["exam"], topic="p12f_gram_schmidt_2d", params={})
HESS_SKILL, HESS_SOURCES = _skill_sources(HESS_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-gram-schmidt-2d
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_gram_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(GRAM_SOURCES, difficulty, seed)
        parts = [str(Fraction(v) + 5) for v in q.answer]
        wrong = ", ".join(parts)
        assert not grade_ordered_list(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


@pytest.mark.parametrize("difficulty", TIERS)
def test_gram_answer_is_four_exact_fractions_in_lowest_terms(difficulty):
    for seed in SEEDS:
        q = _make(GRAM_SOURCES, difficulty, seed)
        assert isinstance(q.answer, list) and len(q.answer) == 4
        for v in q.answer:
            assert type(v) in (int, Fraction), f"{v!r} is {type(v)}, not int/Fraction"
            f = Fraction(v)
            assert f == Fraction(f.numerator, f.denominator)  # already lowest terms
        assert _GRAM_ANSWER_RE.fullmatch(q.answer_text), q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_gram_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(GRAM_SOURCES, difficulty, seed)
        for d in q.distractors:
            assert _GRAM_ANSWER_RE.fullmatch(d), d


# This test asserts the same 200-seed concentration directly, at the tighter <=10% bar the fix
# requires, and a floor of 20 distinct answers on every tier -- proven RED against the unwidened
# pool before the fix.
@pytest.mark.parametrize("difficulty", TIERS)
def test_gram_200_seed_answer_pool_is_wide_enough(difficulty):
    from collections import Counter

    counts = Counter(_make(GRAM_SOURCES, difficulty, seed).answer_text for seed in range(200))
    distinct = len(counts)
    top_share = max(counts.values()) / 200
    assert distinct >= 20, f"{difficulty}: only {distinct} distinct answers over 200 seeds"
    assert top_share <= 0.10, (
        f"{difficulty}: top answer {counts.most_common(1)[0][0]!r} is "
        f"{top_share:.1%} of 200 draws, over the 10% ceiling"
    )


def test_gram_hard_tier_u_has_a_negative_component_easy_medium_do_not():
    for seed in SEEDS:
        easy_q = _make(GRAM_SOURCES, "easy", seed)
        medium_q = _make(GRAM_SOURCES, "medium", seed)
        hard_q = _make(GRAM_SOURCES, "hard", seed)
        assert "-" not in easy_q.text.split("u = (")[1].split(")")[0]
        assert "-" not in medium_q.text.split("u = (")[1].split(")")[0]
        # hard: exactly one of u's two components is negative (never both, never neither)
        u_part = hard_q.text.split("u = (")[1].split(")")[0]
        ux_s, uy_s = [p.strip() for p in u_part.split(",")]
        assert (ux_s.startswith("-")) != (uy_s.startswith("-"))


def test_gram_exam_tier_states_two_candidate_vectors_and_a_criterion():
    for seed in SEEDS:
        q = _make(GRAM_SOURCES, "exam", seed)
        assert "v_A" in q.text and "v_B" in q.text
        assert "positive x-component" in q.text


def test_gram_skill_loads_through_api_with_correct_properties():
    assert GRAM_SKILL.domain == "linear-algebra"
    assert GRAM_SKILL.status == "active"
    assert GRAM_SKILL.placement_eligible is False
    assert GRAM_SKILL.answer == "ordered_list"
    assert GRAM_SKILL.prerequisites == ["vector-projection"]
    assert {src.difficulty for src in GRAM_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert GRAM_SOURCES["exam"].stages == ["exam"]


# ==========================================================================
# ucv1-hessian-2x2
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_hessian_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(HESS_SOURCES, difficulty, seed)
        wrong = ", ".join(str(int(v) + 1000) for v in q.answer)
        assert not grade_ordered_list(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"
        )


@pytest.mark.parametrize("difficulty", TIERS)
def test_hessian_answer_is_four_exact_integers(difficulty):
    for seed in SEEDS:
        q = _make(HESS_SOURCES, difficulty, seed)
        assert isinstance(q.answer, list) and len(q.answer) == 4
        for v in q.answer:
            assert type(v) is int, f"{v!r} is {type(v)}, not int"
        assert _HESS_ANSWER_RE.fullmatch(q.answer_text), q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_hessian_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(HESS_SOURCES, difficulty, seed)
        for d in q.distractors:
            assert _HESS_ANSWER_RE.fullmatch(d), d


@pytest.mark.parametrize("difficulty", TIERS)
def test_hessian_fxy_equals_fyx_every_seed(difficulty):
    for seed in SEEDS:
        q = _make(HESS_SOURCES, difficulty, seed)
        fxx, fxy, fyx, fyy = q.answer
        assert fxy == fyx


def test_hessian_diagonal_entries_never_collide():
    """fxx != fyy is required by construction so diagonal-swap is always a
    genuine distractor; assert it holds across a wide sweep."""
    for seed in SWEEP_SEEDS:
        for difficulty in TIERS:
            q = _make(HESS_SOURCES, difficulty, seed)
            fxx, _fxy, _fyx, fyy = q.answer
            assert fxx != fyy, f"{difficulty} seed {seed}: fxx == fyy == {fxx}"


def test_hessian_easy_tier_hessian_is_point_independent():
    """Honestly: at easy, the Hessian is CONSTANT -- the same
    (fxx, fxy, fyx, fyy) regardless of which point is stated, for a fixed
    f. Evaluate the same f at two different points and confirm equality."""
    import mathgen.generators.p12f_geometry_calculus_build as gen

    for seed in range(10):
        A, B, C = 2, 1, -3
        _f, fxx1, fxy1, fyx1, fyy1 = gen._hessian_sympy(A, B, C, 0, 0, 3, -2)
        _f, fxx2, fxy2, fyx2, fyy2 = gen._hessian_sympy(A, B, C, 0, 0, -5, 7)
        assert (fxx1, fxy1, fyx1, fyy1) == (fxx2, fxy2, fyx2, fyy2)


def test_hessian_medium_and_hard_tier_hessian_depends_on_the_point():
    import mathgen.generators.p12f_geometry_calculus_build as gen

    A, B, C, D = 2, 1, -3, 2
    _f, fxx1, *_ = gen._hessian_sympy(A, B, C, D, 0, 3, -2)
    _f, fxx2, *_ = gen._hessian_sympy(A, B, C, D, 0, -5, 7)
    assert fxx1 != fxx2


def test_hessian_hard_tier_x0_always_negative():
    for seed in SWEEP_SEEDS:
        q = _make(HESS_SOURCES, "hard", seed)
        assert re.search(r"\(-\d+, -?\d+\)", q.text), q.text


def test_hessian_exam_tier_states_unknown_coefficient_and_one_hessian_fact():
    for seed in SEEDS:
        q = _make(HESS_SOURCES, "exam", seed)
        assert "E" in q.text and "unknown" in q.text
        assert "f_xy" in q.text


def test_hessian_skill_loads_through_api_with_correct_properties():
    assert HESS_SKILL.domain == "multivariable-and-vector-calculus"
    assert HESS_SKILL.status == "active"
    assert HESS_SKILL.placement_eligible is False
    assert HESS_SKILL.answer == "ordered_list"
    assert HESS_SKILL.prerequisites == ["jacobian-two-variable"]
    assert {src.difficulty for src in HESS_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert HESS_SOURCES["exam"].stages == ["exam"]


# ==========================================================================
# Grader-format edge cases (both skills)
# ==========================================================================


def test_gram_grader_accepts_bracketed_whole_answer():
    q = _make(GRAM_SOURCES, "easy", 0)
    assert grade_ordered_list(q, f"({q.answer_text})").correct


def test_gram_grader_accepts_no_space_after_comma_when_no_three_digit_run():
    q = _make(GRAM_SOURCES, "easy", 0)
    assert grade_ordered_list(q, q.answer_text.replace(" ", "")).correct


def test_hessian_grader_accepts_bracketed_whole_answer():
    q = _make(HESS_SOURCES, "easy", 0)
    assert grade_ordered_list(q, f"[{q.answer_text}]").correct
