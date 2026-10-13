"""Tests `p13f_coordinate_basis` (`ucv1-coordinate-basis`): coordinates of a vector relative to a basis, answered
as a pair."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_pairwise_distinct

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
SOLVE_TIERS = ("easy", "medium", "hard")

TOPIC = "p13f_coordinate_basis"
SKILL = "ucv1-coordinate-basis"

_SOLVE_ANSWER_RE = re.compile(r"^c1 = -?\d+(/\d+)?, c2 = -?\d+(/\d+)?$")
_EXAM_ANSWER_RE = re.compile(r"^\(-?\d+, -?\d+\)$")


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_pair(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_pair(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"
        # a deliberately wrong value: bump every component by 1
        values = list(q.answer.values())
        wrong = ", ".join(str(v + 1) for v in values)
        result = grade_pair(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong value accepted"


def test_grader_accepts_unreduced_equivalent_fraction():
    """`grade_pair` compares component VALUES via `extract_values`: an
    un-simplified but equivalent fraction component must still be accepted."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        c1, c2 = q.answer["c1"], q.answer["c2"]
        if Fraction(c1).denominator == 1 or Fraction(c2).denominator == 1:
            continue
        unreduced_text = f"c1 = {c1.numerator * 3}/{c1.denominator * 3}, c2 = {c2}"
        result = grade_pair(q, unreduced_text)
        assert result.correct, f"seed={seed}: unreduced-equivalent {unreduced_text!r} rejected"
        return
    pytest.skip("no fractional hard-tier draw found in 60 seeds")


# --- Answer kind / formatted alike ----------------------------------------


@pytest.mark.parametrize("tier", SOLVE_TIERS)
def test_solve_tiers_answer_shape(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and set(q.answer) == {"c1", "c2"}
        assert _SOLVE_ANSWER_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not c1/c2 shaped"
        for d in q.distractors:
            assert _SOLVE_ANSWER_RE.match(d), f"seed={seed}: distractor {d!r} not c1/c2 shaped"


def test_exam_tier_answer_shape():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert isinstance(q.answer, dict) and set(q.answer) == {"vx", "vy"}
        assert _EXAM_ANSWER_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not (vx, vy) shaped"
        for d in q.distractors:
            assert _EXAM_ANSWER_RE.match(d), f"seed={seed}: distractor {d!r} not (vx, vy) shaped"


def test_easy_medium_are_always_integer():
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert Fraction(q.answer["c1"]).denominator == 1, f"{tier} seed={seed}: c1 not integer"
            assert Fraction(q.answer["c2"]).denominator == 1, f"{tier} seed={seed}: c2 not integer"


def test_hard_tier_D_is_never_unit():
    """The headline property of the hard tier: |D| > 1 on every draw."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        b1x, b1y = map(int, re.search(r"b1 = \((-?\d+), (-?\d+)\)", q.text).groups())
        b2x, b2y = map(int, re.search(r"b2 = \((-?\d+), (-?\d+)\)", q.text).groups())
        d = b1x * b2y - b1y * b2x
        assert abs(d) > 1, f"hard seed={seed}: |D|={abs(d)} is not > 1"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_distinct((Fraction(1), Fraction(2)), (Fraction(1), Fraction(2)), msg="should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct((Fraction(1), Fraction(2)), (Fraction(1), Fraction(2)), msg="should fail")


# --- Distinct rendered questions -------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_200_seed_answer_concentration_under_25_percent(tier):
    counts = Counter(make(TOPIC, seed=seed, difficulty=tier).answer_text for seed in range(200))
    top_share = counts.most_common(1)[0][1] / 200
    assert top_share <= 0.25, f"{tier}: top answer concentration {top_share:.1%} exceeds 25% ({counts.most_common(3)})"


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["matrix-multiplication"]
    assert len(skill.sources) == 4
    for source in skill.sources[:3]:
        assert source.topic == TOPIC
    assert skill.sources[3].topic == "exd_x_r2g_basis_map"  # the exam tier is a multi-part chain now
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
