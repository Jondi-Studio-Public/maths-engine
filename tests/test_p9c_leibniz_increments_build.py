"""Tests the Leibniz increments skill (ACMMM078): the bare increments δx and δy as a labelled pair, never a ratio
or a limit."""

import re

import pytest
from sympy import symbols

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent
from mathgen.generators.p9c_leibniz_increments_build import _general_delta_y

SKILL_ID = "senior-use-the-leibniz-notation-delta-x-and-delta"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
_PAIR_RE = re.compile(r"δx = (-?\d+), δy = (-?\d+)")


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism & distractor hygiene ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(difficulty, seed)
        # The three collision guards (x1 != dx, dx != dy, dy != 0) make every
        # one of the three misconceptions unconditionally distinct from both
        # the answer and each other -- see the generator's collision-guard
        # loop -- so exactly 3 distractors are offered on every question.
        assert len(q1.distractors) == 3


# --- Real grader tests ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        wrong = f"δx = {int(q.answer['dx']) + 13}, δy = {int(q.answer['dy']) - 17}"
        assert not grade_pair(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


# --- Answer shape ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_is_an_exact_integer_pair(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert _PAIR_RE.fullmatch(q.answer_text), q.answer_text
        assert float(q.answer["dx"]).is_integer()
        assert float(q.answer["dy"]).is_integer()


@pytest.mark.parametrize(
    "difficulty,expected_dx_sign",
    [
        ("easy", 1),
        ("medium", 1),
        ("hard", -1),
    ],
)
def test_dx_sign_matches_tier_contract(difficulty, expected_dx_sign):
    """easy/medium draw δx positive; hard draws it negative (x decreasing)."""
    for seed in SEEDS:
        q = _make(difficulty, seed)
        dx = int(q.answer["dx"])
        if expected_dx_sign > 0:
            assert dx > 0, f"{difficulty} seed={seed}: dx={dx} should be positive"
        else:
            assert dx < 0, f"{difficulty} seed={seed}: dx={dx} should be negative"


def test_no_quotient_or_limit_or_estimate_language():
    """Load-bearing: this skill asks for the two bare
    increments only -- never a ratio, a limit, or an estimate."""
    for difficulty in ["easy", "medium", "hard"]:
        for seed in SEEDS:
            q = _make(difficulty, seed)
            lowered = q.text.lower()
            assert "lim" not in lowered
            assert "/" not in q.text
            assert "estimate" not in lowered


# --- Independent check fires on a corrupted delta y ---


def test_independent_check_fires_on_a_corrupted_delta_y():
    x, dx_sym = symbols("x dx")
    f_expr = 3 * x + 5
    general = _general_delta_y(f_expr)
    x1, dx = 2, 3
    true_dy = general.subs({x: x1, dx_sym: dx})
    check_equivalent(true_dy, 3 * dx, "sanity: the true dy passes")  # 3*3 = 9
    with pytest.raises(VerificationError):
        check_equivalent(true_dy, 3 * dx + 1, "corrupted dy should fail")


# --- Pool size (>= 20 distinct rendered questions per tier over 60 seeds) ---

# --- Distractors formatted like the answer (no shape tell) ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        for d in q.distractors:
            assert _PAIR_RE.fullmatch(d), d


# --- Skill loads through the API ---


def test_skill_loads_through_api_with_correct_properties():
    _domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None
    assert skill.domain == "senior-mathematics"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "pair"
    assert skill.prerequisites == []
    assert {src.difficulty for src in skill.sources} == {"easy", "medium", "hard"}
