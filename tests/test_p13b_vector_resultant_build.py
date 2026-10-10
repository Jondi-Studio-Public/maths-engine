"""Resultant of vectors at a non-right angle.

Tests for senior-solve-problems-involving-displacement-force-and-velocity-involving
(ACMSM023, topic ``p13b_vector_resultant_angle``).
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance

TOPIC = "p13b_vector_resultant_angle"
SKILL_ID = "senior-solve-problems-involving-displacement-force-and-velocity-involving"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
DP_TOL = stated_precision_tolerance(1)  # easy/medium/exam: 1 d.p.
INT_TOL = stated_precision_tolerance(0)  # hard: exact integer


def _tol_for(difficulty: str) -> Fraction:
    return INT_TOL if difficulty == "hard" else DP_TOL


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, difficulty, seed):
    """Generate through the skill's own `sources` entry, params-driven, not a
    bare `make(topic, seed, difficulty)` call."""
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene -------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_gap_exceeds_stated_precision_tolerance(skills_dict, difficulty):
    tol = _tol_for(difficulty)
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        values = [Fraction(q.answer_text)] + [Fraction(d) for d in q.distractors]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                gap = abs(values[i] - values[j])
                assert gap > tol, (difficulty, seed, q.answer_text, q.distractors, gap)


# --- The real grader accepts the canonical and rejects every distractor -----


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_answer_rejects_distractors(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert grade_number(q, q.answer_text).correct is True, (difficulty, seed, q.text)
        for d in q.distractors:
            assert grade_number(q, d).correct is False, (difficulty, seed, q.text, d)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_a_deliberately_wrong_answer_is_rejected(skills_dict, difficulty):
    tol = _tol_for(difficulty)
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        wrong = Fraction(q.answer_text) + tol * 10 + 1
        if difficulty == "hard":
            wrong_text = str(int(wrong))
        else:
            wrong_text = str(float(wrong))
        assert grade_number(q, wrong_text).correct is False, (difficulty, seed, q.text, wrong_text)


# --- Answer shape: precision, formatting, no rounding-boundary sit ---------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_kind_and_precision(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        if difficulty == "hard":
            assert "." not in q.answer_text, (difficulty, seed, q.answer_text)
            assert q.answer_text == str(int(q.answer_text))
        else:
            assert "." in q.answer_text, (difficulty, seed, q.answer_text)
            dp = len(q.answer_text.split(".")[1])
            assert dp == 1, (difficulty, seed, q.answer_text)
        for d in q.distractors:
            if difficulty == "hard":
                assert "." not in d, (difficulty, seed, d)
            else:
                assert "." in d and len(d.split(".")[1]) == 1, (difficulty, seed, d)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_easy_and_medium_never_use_a_right_angle(skills_dict, difficulty):
    if difficulty not in ("easy", "medium"):
        pytest.skip("no two-vector angle in this tier's prompt")
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert "90 degrees" not in q.text, (difficulty, seed, q.text)


# --- At least 20 distinct rendered questions per tier over 60 seeds --------

# --- Independent in-generator checks fire on a corrupted answer ------------


def test_independent_check_fires_on_corruption(monkeypatch):
    """`check_equivalent`/`check` inside the generator must raise when the
    construction is deliberately corrupted -- exercised directly by
    monkeypatching `math.cos` to return a wrong value mid-construction."""
    import math as math_mod

    import mathgen.generators.p13b_vector_resultant_build as gen_mod
    from mathgen.core import VerificationError

    real_cos = math_mod.cos
    monkeypatch.setattr(gen_mod.math, "cos", lambda x: real_cos(x) + 0.5)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=0, difficulty="easy")


# --- Skill loading ------------------------------------------------------


def test_skill_loads_with_expected_shape(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill.prerequisites == ["solve-ode-exponential"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    difficulties_covered = {src.difficulty for src in skill.sources}
    assert difficulties_covered == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]
