"""Tests `p13f_initial_value_constant` (`ucv1-initial-value-constant`): the constant of an ODE solution from an
initial value."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_pairwise_distinct

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p13f_initial_value_constant"
SKILL = "ucv1-initial-value-constant"

_FRACTION_RE = re.compile(r"^-?\d+(/\d+)?$")


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_fraction(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_fraction(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"
        expected = Fraction(q.answer["p"], q.answer["q"])
        wrong = expected + 1
        result = grade_fraction(q, f"{wrong.numerator}/{wrong.denominator}")
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong value accepted"


def test_grader_accepts_unreduced_equivalent_fraction():
    q = make(TOPIC, seed=0, difficulty="medium")
    p, qd = q.answer["p"], q.answer["q"]
    unreduced = f"{p * 3}/{qd * 3}"
    result = grade_fraction(q, unreduced)
    assert result.correct, f"unreduced-but-equivalent {unreduced!r} should be accepted"


# --- Answer kind / formatted alike ----------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_fraction_dict_and_formatted_alike(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, dict) and set(q.answer) == {"p", "q"}
        assert _FRACTION_RE.match(q.answer_text), f"seed={seed}: {q.answer_text!r} not p or p/q shaped"
        for d in q.distractors:
            assert _FRACTION_RE.match(d), f"seed={seed}: distractor {d!r} not p or p/q shaped"


# --- Tier-specific structural properties -----------------------------------


def test_easy_tier_x0_is_always_zero():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert "y(0)" in q.text, f"seed={seed}: easy tier text does not fix x0=0: {q.text!r}"


def test_medium_tier_x0_is_small_positive():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = re.search(r"y\((-?\d+)\)", q.text)
        assert m is not None
        x0 = int(m.group(1))
        assert 1 <= x0 <= 4, f"seed={seed}: medium x0={x0} out of range"


def test_hard_tier_x0_is_always_negative():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = re.search(r"y\((-?\d+)\)", q.text)
        assert m is not None
        x0 = int(m.group(1))
        assert x0 < 0, f"seed={seed}: hard x0={x0} is not negative"


_FUNCTION_NOTATION_CONDITION_RE = re.compile(r"y\(-?\d+\)\s*=")


def test_exam_tier_uses_a_verbal_condition_never_function_notation_for_the_condition():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert not _FUNCTION_NOTATION_CONDITION_RE.search(q.text), (
            f"seed={seed}: exam text should not restate the condition in y(x0)=y0 function notation: {q.text!r}"
        )
        assert ("crosses the x-axis" in q.text) or ("passes through the point" in q.text), (
            f"seed={seed}: exam text missing a recognised verbal condition: {q.text!r}"
        )


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_distinct(Fraction(5, 7), Fraction(5, 7), msg="should fail")
    with pytest.raises(VerificationError):
        check_pairwise_distinct(Fraction(5, 7), Fraction(5, 7), msg="should fail")


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
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["solve-ode-separable-poly"]
    assert len(skill.sources) == 4
    for source in skill.sources[:3]:
        assert source.topic == TOPIC
    assert skill.sources[3].topic == "exd_x_r3f_initial_value_chain"  # the exam source is the multi-part chain
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"


# --- Working-text sign hygiene (regression) --------------------


@pytest.mark.parametrize("tier", TIERS)
def test_working_text_never_shows_a_double_sign(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for line in q.working:
            assert "- -" not in line, f"{tier} seed={seed}: double-sign working line: {line!r}"
            assert "+ -" not in line, f"{tier} seed={seed}: double-sign working line: {line!r}"
