"""Tests `p14c_inverse_regularised_solve` (`ucv1-inverse-regularised-solve`): a regularised inverse solve,
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
TOPIC = "p14c_inverse_regularised_solve"
SKILL = "ucv1-inverse-regularised-solve"

_PAIR_RE = re.compile(r"^x1 = -?\d+(/\d+)?, x2 = -?\d+(/\d+)?$")
_LAMBDA_RE = re.compile(r"^lambda = -?\d+(/\d+)?$")


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grading.grade("pair", q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grading.grade("pair", q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_compute_tiers_answer_shape(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and set(q.answer) == {"x1", "x2"}
        assert _PAIR_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not x1=.., x2=.. shaped"
        for d in q.distractors:
            assert _PAIR_RE.match(d), f"seed={seed}: distractor {d!r} not x1=.., x2=.. shaped"


def test_exam_tier_answer_shape():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert isinstance(q.answer, dict) and set(q.answer) == {"lambda"}
        assert _LAMBDA_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not lambda=.. shaped"
        for d in q.distractors:
            assert _LAMBDA_RE.match(d), f"seed={seed}: distractor {d!r} not lambda=.. shaped"


def test_hard_tier_answer_has_a_genuine_fraction_component():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert q.answer["x1"].denominator != 1 or q.answer["x2"].denominator != 1, (
            f"hard seed={seed}: both components integer: {q.answer_text!r}"
        )


def test_easy_tier_matrix_is_square_2x2():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        rows = re.findall(r"\((-?\d+), (-?\d+)\)", q.text)
        assert len(rows) == 2, f"easy seed={seed}: expected 2 rows in {q.text!r}"


def test_medium_tier_matrix_has_three_rows():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        rows = re.findall(r"\((-?\d+), (-?\d+)\)", q.text)
        assert len(rows) == 3, f"medium seed={seed}: expected 3 rows in {q.text!r}"


def test_hard_tier_matrix_has_four_rows():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        rows = re.findall(r"\((-?\d+), (-?\d+)\)", q.text)
        assert len(rows) == 4, f"hard seed={seed}: expected 4 rows in {q.text!r}"


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct((1, 2), (1, 2), msg="should fail")


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "pair"
    assert skill.prerequisites == ["matrix-rank-small"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        # The exam tier is now a multi-part chain (tests/test_exd_x_r2h_ridge_chain.py).
        assert source.topic == ("exd_x_r2h_ridge_chain" if source.difficulty == "exam" else TOPIC)
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
