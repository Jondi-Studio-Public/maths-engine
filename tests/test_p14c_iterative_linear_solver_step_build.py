"""Tests `p14c_iterative_linear_solver_step` (`ucv1-iterative-linear-solver-step`): one iterative-solver step,
answered as a pair."""

from __future__ import annotations

import re

import pytest

from mathgen.grading import typed as grading
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p14c_iterative_linear_solver_step"
SKILL = "ucv1-iterative-linear-solver-step"

_PAIR_RE = re.compile(r"^x = -?\d+(/\d+)?, y = -?\d+(/\d+)?$")
# Tolerant of either the "bare x-coef / parenthesised y-coef" style the
# lower tiers use (and the un-fixed exam tier inherits) or a uniform
# "both terms parenthesised" style -- reads either format so the RED run
# below fails on the ASSERTION, not on a regex miss against the old text.
_UNIFORM_EQ_RE = re.compile(r"\(?(-?\d+)\)?x \+ \(?(-?\d+)\)?y = (-?\d+)")


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grading.grade("pair", q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grading.grade("pair", q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_pair_dict_and_formatted_alike(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and set(q.answer) == {"x", "y"}
        assert _PAIR_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not x=.., y=.. shaped"
        for d in q.distractors:
            assert _PAIR_RE.match(d), f"seed={seed}: distractor {d!r} not x=.., y=.. shaped"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_states_gauss_seidel_and_update_order(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "Gauss-Seidel" in q.text, f"{tier} seed={seed}: prompt does not name Gauss-Seidel"
        assert "just-computed" in q.text, f"{tier} seed={seed}: prompt does not state the update order"


def test_exam_tier_starting_point_is_a_genuine_fraction():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert re.search(r"\(-?\d+/\d+, -?\d+/\d+\)", q.text) or re.search(r"\(-?\d+/\d+,\s*-?\d+/\d+\)", q.text), (
            f"exam seed={seed}: starting point is not stated as a fraction pair: {q.text!r}"
        )


def test_hard_tier_asks_for_x2_y2_not_x1_y1():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert "(x2, y2)" in q.text, f"hard seed={seed}: does not ask for the second iterate: {q.text!r}"


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct((1, 2), (1, 2), msg="should fail")


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_lower_tiers_always_print_x_dominant_equation_first(tier):
    """Control: every lower tier always prints the x-dominant equation
    (|x-coef| > |y-coef| within that equation) first -- this is what the
    CURRENT code already does at every tier, so it should already pass."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        eqs = _UNIFORM_EQ_RE.findall(q.text)
        assert len(eqs) == 2, f"{tier} seed={seed}: expected 2 equations in {q.text!r}"
        (cx1, cy1, _), _ = [(int(a), int(b), int(c)) for a, b, c in eqs]
        assert abs(cx1) > abs(cy1), f"{tier} seed={seed}: first equation is not x-dominant: {q.text!r}"


def test_exam_tier_randomizes_which_equation_prints_first():
    """The exam tier must NOT always print the x-dominant equation first --
    unlike every lower tier (see the control test above). Over 60 seeds,
    both display orders must appear a genuine number of times (>= 10 each),
    proving the equation-identification step is a real reasoning demand and
    not a coincidence of a handful of seeds."""
    dominant_first = 0
    dominant_second = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        eqs = _UNIFORM_EQ_RE.findall(q.text)
        assert len(eqs) == 2, f"exam seed={seed}: expected 2 equations in {q.text!r}"
        (cx1, cy1, _), (cx2, cy2, _) = [(int(a), int(b), int(c)) for a, b, c in eqs]
        if abs(cx1) > abs(cy1):
            dominant_first += 1
        else:
            assert abs(cx2) > abs(cy2), f"exam seed={seed}: neither equation is x-dominant: {q.text!r}"
            dominant_second += 1
    assert dominant_first >= 10, f"exam tier: x-dominant equation almost never printed first ({dominant_first}/60)"
    assert dominant_second >= 10, f"exam tier: x-dominant equation almost never printed second ({dominant_second}/60)"


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "pair"
    assert skill.prerequisites == ["jacobi-iteration-2x2"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        # the exam source moved to the multi-part chain exd_x_r3b_gauss_seidel
        assert source.topic == ("exd_x_r3b_gauss_seidel" if source.difficulty == "exam" else TOPIC)
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
