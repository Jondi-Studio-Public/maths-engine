"""Focused contract tests for: senior-midpoints-of-the-sides-of-a-quadrilateral-join (Varignon's theorem)."""

import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p8d_midpoint_parallelogram_build import _verify_relation

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TOPIC = "midpoints_of_the_sides_of_a_quadrilateral_join"
SKILL_ID = "senior-midpoints-of-the-sides-of-a-quadrilateral-join"


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism & distractor hygiene ---

# --- Real grader tests ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        wrong = f"x = {int(q.answer['x']) + 13}, y = {int(q.answer['y']) - 17}"
        assert not grade_pair(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


# --- Answer shape ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_is_an_exact_integer_pair(difficulty):
    """An exact integer pair is required (no fractions) at every tier."""
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert re.fullmatch(r"x = -?\d+, y = -?\d+", q.answer_text), q.answer_text
        assert float(q.answer["x"]).is_integer()
        assert float(q.answer["y"]).is_integer()


# --- Independent check fires on a corrupted reconstruction ---


def test_independent_check_fires_on_corrupted_reconstruction():
    P, Q, R = (0, 0), (2, 0), (2, 2)
    S = (0, 2)  # correct: Q - P == R - S
    _verify_relation(P, Q, R, S)  # sanity: the true parallelogram passes
    with pytest.raises(VerificationError):
        _verify_relation(P, Q, R, (1, 2))  # corrupted S


# --- Boundary against midpoint-two-points / midpoint-coordinate-scalar ---
# (load-bearing: no tier's answer may ever be obtainable
# from a single pair of given, adjacent quadrilateral vertices alone)

_COORD_RE = re.compile(r"([A-D])\((-?\d+), (-?\d+)\)")


def _given_vertices(text):
    """Which of A, B, C, D appear with explicit coordinates in the question."""
    return {m.group(1) for m in _COORD_RE.finditer(text)}


@pytest.mark.parametrize(
    "difficulty,adjacent_pair",
    [
        ("easy", ("A", "B")),
        ("easy", ("B", "C")),
        ("medium", ("A", "B")),
        ("medium", ("A", "D")),
        ("hard", ("A", "B")),
    ],
)
def test_boundary_answer_never_obtainable_from_a_single_given_adjacent_pair(difficulty, adjacent_pair):
    """Every adjacent pair of quadrilateral vertices that IS given in a
    question is never, by itself, the pair whose plain midpoint the question
    asks for -- the requested letter (S/R/Q) always names a midpoint whose
    own vertex pair includes the withheld vertex."""
    target_letter = {"easy": "S", "medium": "R", "hard": "Q"}[difficulty]
    side_of = {"P": {"A", "B"}, "Q": {"B", "C"}, "R": {"C", "D"}, "S": {"D", "A"}}
    for seed in SEEDS:
        q = _make(difficulty, seed)
        given = _given_vertices(q.text)
        assert set(adjacent_pair) <= given
        # The requested midpoint's own side never equals a fully-given adjacent pair.
        assert side_of[target_letter] != set(adjacent_pair)
        # And at least one vertex of the requested midpoint's side is withheld.
        assert not side_of[target_letter] <= given, (
            f"seed {seed} {difficulty}: target {target_letter}'s side "
            f"{side_of[target_letter]} is fully given ({given}) -- it would "
            "collapse to the plain midpoint formula"
        )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_boundary_question_never_gives_all_four_vertices(difficulty):
    """A sanity backstop: if all four vertices were ever given, EVERY
    midpoint would be directly computable and the theorem would not be
    needed at all."""
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert len(_given_vertices(q.text)) <= 3


# --- Pool size (>= 20 distinct rendered questions per tier over 60 seeds) ---

# --- Distractors formatted like the answer (no shape tell) ---


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        for d in q.distractors:
            assert re.fullmatch(r"x = -?\d+, y = -?\d+", d), d


# --- Skill loads through the API ---


def test_skill_loads_through_api_with_correct_properties():
    _domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None
    assert skill.domain == "senior-mathematics"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "pair"
    assert skill.prerequisites == ["f10-quadrilateral-properties", "midpoint-two-points"]
    assert {src.difficulty for src in skill.sources} == {"easy", "medium", "hard", "exam"}
